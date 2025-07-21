#!/usr/bin/env python3
"""
Run FireFlow semantic‑editing experiments defined in a JSON dataset file.

The script reproduces the exact editing path of the official Gradio demo:
  https://github.com/Black-Forest-Labs/flux  (FireFlow branch)

Metrics reported per example
  • CLIP similarity (target prompt ↔ edited image)
  • CLIP similarity (target prompt ↔ source image) — sanity baseline
  • LPIPS distance (edited image ↔ source image)
  • Pixelwise L1 mean & median
  • Pixelwise L2 mean & median
  • Paths to edited & diff images

Results are printed to stdout and also written to a CSV file.
"""

import argparse
import csv
import json
import os
import re
import time
from glob import iglob
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import ExifTags, Image, ImageChops
from einops import rearrange
from transformers import CLIPModel, CLIPProcessor

# FireFlow / FLUX imports —
# install via:  pip install git+https://github.com/Black-Forest-Labs/flux.git
from flux.sampling import denoise_fireflow, get_schedule, prepare, unpack
from flux.util import (
    configs,
    embed_watermark,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)

import lpips

# ------------------------------------------------------------
# 1.  Metric helpers (identical to FLUX‑SDE runner)
# ------------------------------------------------------------

class MetricComputer:
    def __init__(self, device: torch.device):
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.lpips_fn = lpips.LPIPS(net="vgg").to(device)
        self.lpips_transform = T.Compose(
            [T.Resize((1024, 1024)), T.ToTensor(), T.Normalize((0.5,), (0.5,))]
        )

    @torch.inference_mode()
    def clip_similarity(self, image: Image.Image, text: str) -> float:
        inputs = self.clip_processor(
            text=[text], images=image, return_tensors="pt", padding=True
        ).to(self.clip_model.device)
        return self.clip_model(**inputs).logits_per_image.item()

    @torch.inference_mode()
    def lpips_distance(self, img1: Image.Image, img2: Image.Image) -> float:
        t1 = self.lpips_transform(img1).unsqueeze(0).to(self.device)
        t2 = self.lpips_transform(img2).unsqueeze(0).to(self.device)
        return self.lpips_fn(t1, t2).item()

    @staticmethod
    def pixel_distances(img1: Image.Image, img2: Image.Image) -> Tuple[float, float, float, float]:
        a1 = np.asarray(img1, dtype=np.float32)
        a2 = np.asarray(img2, dtype=np.float32)
        # per‑channel difference
        diff = a1 - a2
        # average across the 3 channels so l1/l2 match image (H, W)
        l1 = np.mean(np.abs(diff), axis=-1)      # shape (H, W)
        l2 = np.mean(diff ** 2,  axis=-1)        # shape (H, W)
        return l1.mean(), np.median(l1), l2.mean(), np.median(l2)


# ------------------------------------------------------------
# 2.  FireFlow editor (logic ported 1‑to‑1 from the demo)
# ------------------------------------------------------------
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
        init_image = np.array(init_image)
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
        # print(info['feature']['1.0_True_37_single_V'].sum())
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

        return img



# ------------------------------------------------------------
# 3.  Per‑example runner (saves edited & diff, returns metrics)
# ------------------------------------------------------------

def run_edit(
    editor: FluxEditor,
    metric: MetricComputer,
    image_path: Path,
    source_prompt: str,
    target_prompt: str,
    *,
    save_dir: Path,
    idx: int,
    num_steps: int,
    inject_step: int,
    guidance: float,
    editing_strategy: List[str],
    seed: int,
) -> Dict[str, float]:
    init_pil = Image.open(image_path).convert("RGB")
    edited   = editor.edit(
        init_pil,
        source_prompt,
        target_prompt,
        num_steps=num_steps,
        inject_step=inject_step,
        guidance=guidance,
        editing_strategy=editing_strategy,
        seed=seed,
    )

    # --- save ---
    stem, ext = image_path.stem, image_path.suffix or ".png"
    save_dir.mkdir(parents=True, exist_ok=True)
    edit_path = save_dir / f"{idx:04d}_{stem}_edited{ext}"
    diff_path = save_dir / f"{idx:04d}_{stem}_diff{ext}"
    edited.save(edit_path)
    ImageChops.difference(init_pil.resize(edited.size, Image.Resampling.LANCZOS), edited).save(diff_path)

    # --- metrics ---
    m_clip_edit = metric.clip_similarity(edited,        target_prompt)
    m_clip_src  = metric.clip_similarity(init_pil,      target_prompt)
    m_lpips     = metric.lpips_distance(init_pil,       edited)
    l1_mean, l1_med, l2_mean, l2_med = metric.pixel_distances(init_pil.resize(edited.size), edited)

    return {
        "clip_target_edit": m_clip_edit,
        "clip_target_src":  m_clip_src,
        "lpips":            m_lpips,
        "l1_mean":          l1_mean,
        "l1_median":        l1_med,
        "l2_mean":          l2_mean,
        "l2_median":        l2_med,
        "edited_path":      str(edit_path),
        "diff_path":        str(diff_path),
        "seed":             seed,
    }


# ------------------------------------------------------------
# 4.  CLI & main loop
# ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("FireFlow Experiment Runner")
    p.add_argument("--data",  type=str, default="dataset.json")
    p.add_argument("--name", type=str, default="flux-dev", choices=list(configs.keys()))
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--offload", action="store_true", help="Keep large weights on CPU when possible")

    # experiment hyper‑params (Gradio defaults)
    p.add_argument("--num_steps",     type=int,   default=8)
    p.add_argument("--inject_step",   type=int,   default=1)
    p.add_argument("--guidance",      type=float, default=2.0)
    p.add_argument(
        "--editing_strategy",
        type=str,
        default="replace_v",
        help="Comma‑separated list (choices: replace_v, add_q, add_k, add_v, replace_q, replace_k)",
    )

    # misc
    p.add_argument("--save_dir", type=str, default="edited_fireflow")
    p.add_argument("--csv_out",  type=str, default="results_fireflow.csv")
    p.add_argument("--seed",     type=int, default=None, help="Global seed; if omitted, a fresh seed per sample")

    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    editor  = FluxEditor(args)
    metric  = MetricComputer(device)

    with open(args.data) as f:
        dataset: List[Dict] = json.load(f)

    rows = []
    for idx, entry in enumerate(dataset):
        print(f"\n=== [{idx+1}/{len(dataset)}] {entry['image_path']} ===")
        seed = (args.seed if args.seed is not None
                else int(torch.Generator(device="cpu").seed()) )

        stats = run_edit(
            editor,
            metric,
            Path(entry["image_path"]),
            entry["source_prompt"],
            entry["target_prompt"],
            save_dir=Path(args.save_dir),
            idx=idx,
            num_steps=args.num_steps,
            inject_step=args.inject_step,
            guidance=args.guidance,
            editing_strategy=[s.strip() for s in args.editing_strategy.split(",") if s.strip()],
            seed=seed,
        )
        for k, v in stats.items():
            print(f"{k:>20}: {v}")
        rows.append({**entry, **stats})

    # write CSV
    out = Path(args.csv_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nAll done — detailed metrics written to {out.resolve()}")


if __name__ == "__main__":
    main()
