import os
import re
import time
from io import BytesIO
import uuid
from dataclasses import dataclass
from glob import iglob
import argparse
from einops import rearrange
from fire import Fire
from PIL import ExifTags, Image, ImageChops


import torch
import torch.nn.functional as F
import gradio as gr
import numpy as np
from transformers import pipeline

from flux.sampling import denoise_fireflow, get_schedule, prepare, unpack
from flux.util import (configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5)

from transformers import CLIPProcessor, CLIPModel

import lpips
import torchvision.transforms as transforms

import timm

@dataclass
class SamplingOptions:
    source_prompt: str
    target_prompt: str
    # prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None

@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(torch_device)
    with torch.no_grad():
        init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image


class FluxEditor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.offload = args.offload
        self.name = args.name
        self.is_schnell = args.name == "flux-schnell"

        self.feature_path = 'feature'
        self.output_dir = 'result'
        self.add_sampling_metadata = True

        if self.name not in configs:
            available = ", ".join(configs.keys())
            raise ValueError(f"Got unknown model name: {self.name}, chose from {available}")

        # init all components
        self.t5 = load_t5(self.device, max_length=256 if self.name == "flux-schnell" else 512)
        self.clip = load_clip(self.device)
        self.model = load_flow_model(self.name, device="cpu" if self.offload else self.device)
        self.ae = load_ae(self.name, device="cpu" if self.offload else self.device)
        self.t5.eval()
        self.clip.eval()
        self.ae.eval()
        self.model.eval()
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336") # it's lightweighted so it can live on CPU
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        
        self.lpips = lpips.LPIPS(net='vgg').to('cuda')
        self.lpips_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),  # resize for consistency (optional)
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # LPIPS expects inputs in [-1, 1]
        ])
        
        self.dino = timm.create_model('vit_small_patch16_224.dino', pretrained=True, num_classes=0).to('cuda')
        self.dino.eval()
        self.dino_transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def dino_dist(self, img1, img2):
        img1 = self.dino_transform(img1).unsqueeze(0)
        img2 = self.dino_transform(img2).unsqueeze(0)
        with torch.no_grad():
            emb1 = F.normalize(self.dino(img1.to('cuda')), dim=1)
            emb2 = F.normalize(self.dino(img2.to('cuda')), dim=1)
        return F.cosine_similarity(emb1, emb2).item()

    def print_clip_score(self, image, prompt):
        clip_inputs = self.clip_processor(
            text=[prompt,],
            images=image,
            return_tensors="pt",
            padding=True,
        )
        clip_outputs = self.clip_model(**clip_inputs)
        print("CLIP score: ", clip_outputs.logits_per_image.detach().item())

    @torch.inference_mode()
    def edit(self, init_image, source_prompt, target_prompt, editing_strategy, num_steps, inject_step, guidance, seed):
        torch.cuda.empty_cache()
        seed = int(seed) if seed else int(torch.Generator(device="cpu").seed())
        torch.manual_seed(seed)

        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.encoder.to(self.device)

        init_original = Image.fromarray(np.uint8(init_image))

        shape = init_image.shape
        new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
        init_image = init_image[:new_h, :new_w, :]
        width, height = init_image.shape[0], init_image.shape[1]
        init_image = encode(init_image, self.device, self.ae)

        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )
        
        print(f"Generating with seed {opts.seed}:\n{opts.source_prompt}")
        t0 = time.perf_counter()

        opts.seed = None
        if self.offload:
            self.ae = self.ae.cpu()
            torch.cuda.empty_cache()
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

        #############inverse#######################
        info = {}
        info['feature'] = {}
        info['inject_step'] = inject_step
        info['editing_strategy']= " ".join(editing_strategy)
        info['start_layer_index'] = 0
        info['end_layer_index'] = 37
        info['reuse_v']= False
        qkv_ratio = '1.0,1.0,1.0'
        info['qkv_ratio'] = list(map(float, qkv_ratio.split(',')))

        if not os.path.exists(self.feature_path):
            os.mkdir(self.feature_path)

        with torch.no_grad():
            inp = prepare(self.t5, self.clip, init_image, prompt=opts.source_prompt)
            inp_target = prepare(self.t5, self.clip, init_image, prompt=opts.target_prompt)
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))

        # offload TEs to CPU, load model to gpu
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)

        # inversion initial noise
        with torch.no_grad():
            z, info = denoise_fireflow(self.model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)
        inp_target["img"] = z

        timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(self.name != "flux-schnell"))

        # denoise initial noise
        x, _ = denoise_fireflow(self.model, **inp_target, timesteps=timesteps, guidance=guidance, inverse=False, info=info)

        # offload model, load autoencoder to gpu
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), opts.width, opts.height)

        output_name = os.path.join(self.output_dir, "img_{idx}.jpg")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            idx = 0
        else:
            fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
            if len(fns) > 0:
                idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
            else:
                idx = 0

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        fn = output_name.format(idx=idx)
        print(f"Done in {t1 - t0:.1f}s. Saving {fn}")
        # bring into PIL format and save
        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        exif_data[ExifTags.Base.Model] = self.name
        if self.add_sampling_metadata:
            exif_data[ExifTags.Base.ImageDescription] = source_prompt
        # img.save(fn, exif=exif_data, quality=95, subsampling=0)
        init_resized = init_original.convert("RGB").resize(
            img.size,                                 # match W×H
            resample=Image.Resampling.LANCZOS,        # high-quality down/up-sampling
        )
        diff_img = ImageChops.difference(
            init_resized.convert("RGB"),   # make sure both are RGB
            img.convert("RGB")
        )
        
        print("target prompt vs target image: ")
        self.print_clip_score(img, target_prompt)
        print("target prompt vs source image: ")
        self.print_clip_score(init_resized, target_prompt)
        print("source prompt vs target image: ")
        self.print_clip_score(img, source_prompt)
        print("source prompt vs source image: ")
        self.print_clip_score(init_resized, source_prompt)

        a1 = np.asarray(init_resized, dtype=np.float32)
        a2 = np.asarray(img, dtype=np.float32)
        # per‑channel difference
        diff_np = a1 - a2
        # average across the 3 channels so l1/l2 match image (H, W)
        l1 = np.mean(np.abs(diff_np), axis=-1)      # shape (H, W)
        l2 = np.mean(diff_np ** 2,  axis=-1)        # shape (H, W)

        print("L1 Distance:", l1.mean(), "median:", np.median(l1))
        print("L2 Distance:", l2.mean(), "median:", np.median(l2))

        
        with torch.no_grad():
            dist = self.lpips(self.lpips_transform(init_resized).to("cuda"), self.lpips_transform(img).to("cuda"))
        print("LPIPS distance: ", dist.item())
        
        with torch.no_grad():
            dist = self.dino_dist(init_resized, img)
        print("DINO distance: ", 1.0-dist)

        torch.cuda.empty_cache()
        print("End Edit\n\n")
        return img, diff_img



def create_demo(model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", offload: bool = False):
    editor = FluxEditor(args)
    is_schnell = model_name == "flux-schnell"

    with gr.Blocks() as demo:
        gr.Markdown(f"# FireFlow Demo (FLUX for image editing)")
        
        with gr.Row():
            with gr.Column():
                source_prompt = gr.Textbox(label="Source Prompt", value="(Optional) Describe the content of the uploaded image.")
                target_prompt = gr.Textbox(label="Target Prompt", value="(Required) Describe the desired content of the edited image.")
                editing_strategy = gr.CheckboxGroup(
                    label="Editing Technique",
                    choices=['replace_v', 'add_q', 'add_k', 'add_v', 'replace_q', 'replace_k'],
                    value=['replace_v'],  # Default: none selected
                    interactive=True
                )
                generate_btn = gr.Button("Generate")
                with gr.Accordion("Advanced Options", open=True):
                    num_steps = gr.Slider(1, 30, 8, step=1, label="Number of steps")
                    inject_step = gr.Slider(1, 15, 1, step=1, label="Number of inject steps")
                    guidance = gr.Slider(1.0, 10.0, 2, step=0.1, label="Guidance", interactive=not is_schnell)
                    seed = gr.Textbox(None, label="Seed")

            with gr.Column():
                init_image = gr.Image(label="Input Image", visible=True)
            
            with gr.Column():
                output_image = gr.Image(label="Generated Image", format='jpg')

            with gr.Column():
                diff_image   = gr.Image(label="Difference (|input - output|)", format='jpg')

        generate_btn.click(
            fn=editor.edit,
            inputs=[init_image, source_prompt, target_prompt, editing_strategy, num_steps, inject_step, guidance, seed],
            outputs=[output_image, diff_image]
        )


    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flux")
    parser.add_argument("--name", type=str, default="flux-dev", choices=list(configs.keys()), help="Model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--share", action="store_true", help="Create a public link to your demo")
    parser.add_argument("--port", type=int, default=42035)
    args = parser.parse_args()

    demo = create_demo(args.name, args.device, args.offload)
    demo.launch(server_name='0.0.0.0', share=args.share, server_port=args.port)
