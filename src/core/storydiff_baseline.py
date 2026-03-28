"""
StoryDiffusion Baseline Wrapper for MSR-50 Benchmark.

Wraps HVision-NKU/StoryDiffusion's consistent self-attention
mechanism as a multi-shot generation baseline.

Uses the original SpatialAttnProcessor2_0 from predict.py (embedded here
to avoid cog/gradio import chain).

Reference:
  Zhou et al., "StoryDiffusion: Consistent Self-Attention for
  Long-Range Image and Video Generation", NeurIPS 2024 Spotlight.
"""

from __future__ import annotations

import copy
import json
import random
import sys
import types as _types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# Add StoryDiffusion to path and mock gradio
STORYDIFF_ROOT = Path("/home/dongwoo44/papers/paper_DIRECTOR/related_papers/StoryDiffusion")
sys.path.insert(0, str(STORYDIFF_ROOT))

if "gradio" not in sys.modules:
    sys.modules["gradio"] = _types.ModuleType("gradio")

from diffusers import StableDiffusionXLPipeline, DDIMScheduler

# Import mask computation from gradio_utils (no cog dependency)
from utils.gradio_utils import cal_attn_mask_xl


# ═══════════════════════════════════════════════════════════════════════
# Global state (StoryDiffusion uses module-level globals)
# ═══════════════════════════════════════════════════════════════════════

total_count = 0
attn_count = 0
cur_step = 0
mask1024 = None
mask4096 = None
write = True
sa32 = 0.5
sa64 = 0.5
height = 512
width = 512


# ═══════════════════════════════════════════════════════════════════════
# Attention Processors (copied from StoryDiffusion predict.py)
# ═══════════════════════════════════════════════════════════════════════

class AttnProcessor(nn.Module):
    """Standard attention processor (for non-consistent layers)."""
    def __init__(self):
        super().__init__()

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, h, w = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, h * w).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class SpatialAttnProcessor2_0(torch.nn.Module):
    """StoryDiffusion's consistent self-attention processor.

    Copied verbatim from predict.py to avoid cog import chain.
    Uses module-level globals for cross-processor coordination.
    """

    def __init__(self, hidden_size=None, cross_attention_dim=None,
                 id_length=4, device="cuda", dtype=torch.float16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None):
        global total_count, attn_count, cur_step, mask1024, mask4096
        global sa32, sa64, write, height, width

        if write:
            self.id_bank[cur_step] = [
                hidden_states[:self.id_length],
                hidden_states[self.id_length:],
            ]
        else:
            encoder_hidden_states = torch.cat((
                self.id_bank[cur_step][0].to(self.device),
                hidden_states[:1],
                self.id_bank[cur_step][1].to(self.device),
                hidden_states[1:],
            ))

        if cur_step < 5:
            hidden_states = self.__call2__(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb
            )
        else:
            random_number = random.random()
            rand_num = 0.3 if cur_step < 20 else 0.1
            if random_number > rand_num:
                if not write:
                    if hidden_states.shape[1] == (height // 32) * (width // 32):
                        attention_mask = mask1024[
                            mask1024.shape[0] // self.total_length * self.id_length:
                        ]
                    else:
                        attention_mask = mask4096[
                            mask4096.shape[0] // self.total_length * self.id_length:
                        ]
                else:
                    if hidden_states.shape[1] == (height // 32) * (width // 32):
                        attention_mask = mask1024[
                            :mask1024.shape[0] // self.total_length * self.id_length,
                            :mask1024.shape[0] // self.total_length * self.id_length,
                        ]
                    else:
                        attention_mask = mask4096[
                            :mask4096.shape[0] // self.total_length * self.id_length,
                            :mask4096.shape[0] // self.total_length * self.id_length,
                        ]
                hidden_states = self.__call1__(
                    attn, hidden_states, encoder_hidden_states, attention_mask, temb
                )
            else:
                hidden_states = self.__call2__(
                    attn, hidden_states, None, attention_mask, temb
                )

        attn_count += 1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            mask1024, mask4096 = cal_attn_mask_xl(
                self.total_length, self.id_length,
                sa32, sa64, height, width,
                device=self.device, dtype=self.dtype,
            )

        return hidden_states

    def __call1__(self, attn, hidden_states, encoder_hidden_states=None,
                  attention_mask=None, temb=None):
        """Paired attention with cross-image feature mixing."""
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                total_batch_size, channel, height * width
            ).transpose(1, 2)
        total_batch_size, nums_token, channel = hidden_states.shape
        img_nums = total_batch_size // 2
        hidden_states = hidden_states.view(
            -1, img_nums, nums_token, channel
        ).reshape(-1, img_nums * nums_token, channel)

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            encoder_hidden_states = encoder_hidden_states.view(
                -1, self.id_length + 1, nums_token, channel
            ).reshape(-1, (self.id_length + 1) * nums_token, channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            total_batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                total_batch_size, channel, height, width
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    def __call2__(self, attn, hidden_states, encoder_hidden_states=None,
                  attention_mask=None, temb=None):
        """Standard attention with optional cross-image KV."""
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, channel = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            encoder_hidden_states = encoder_hidden_states.view(
                -1, self.id_length + 1, sequence_length, channel
            ).reshape(-1, (self.id_length + 1) * sequence_length, channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ═══════════════════════════════════════════════════════════════════════
# Setup Attention Processors
# ═══════════════════════════════════════════════════════════════════════

def setup_consistent_attention(unet, id_length, device="cuda"):
    """Patch UNet self-attention on up_blocks with consistent processors."""
    global total_count
    total_count = 0
    attn_procs = {}

    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if cross_attention_dim is None and name.startswith("up_blocks"):
            attn_procs[name] = SpatialAttnProcessor2_0(
                id_length=id_length, device=device, dtype=torch.float16,
            )
            total_count += 1
        else:
            attn_procs[name] = AttnProcessor()

    unet.set_attn_processor(copy.deepcopy(attn_procs))
    print(f"  [StoryDiff] Consistent self-attention: {total_count} processors")


# ═══════════════════════════════════════════════════════════════════════
# StoryDiffusion Generator
# ═══════════════════════════════════════════════════════════════════════

class StoryDiffusionGenerator:
    """Wraps StoryDiffusion for multi-shot keyframe generation."""

    def __init__(
        self,
        device: str = "cuda:3",
        num_steps: int = 25,
        guidance_scale: float = 5.0,
        sa32_strength: float = 0.5,
        sa64_strength: float = 0.5,
        num_ids: int = 3,
        seed: int = 42,
        img_size: int = 512,
    ):
        self.device = device
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.sa32_strength = sa32_strength
        self.sa64_strength = sa64_strength
        self.num_ids = num_ids
        self.seed = seed
        self.img_size = img_size
        self.pipe = None

    def load_pipeline(self):
        """Load SDXL pipeline."""
        if self.pipe is not None:
            print("[StoryDiff] Pipeline already loaded, reusing.")
            return
        print("[StoryDiff] Loading SDXL pipeline...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.enable_vae_slicing()
        print("[StoryDiff] Pipeline loaded.")

    def _setup_globals(self):
        """Set module-level globals for attention processors."""
        global sa32, sa64, height, width, attn_count, cur_step
        sa32 = self.sa32_strength
        sa64 = self.sa64_strength
        height = self.img_size
        width = self.img_size
        attn_count = 0
        cur_step = 0

    def _reset_step_state(self):
        """Reset per-generation attention state."""
        global attn_count, cur_step, mask1024, mask4096
        attn_count = 0
        cur_step = 0
        mask1024, mask4096 = cal_attn_mask_xl(
            self.num_ids + 1, self.num_ids,
            self.sa32_strength, self.sa64_strength,
            self.img_size, self.img_size,
            device=self.device, dtype=torch.float16,
        )

    def generate_scenario(
        self,
        scenario: dict,
        out_dir: Path,
    ) -> dict[str, Image.Image]:
        """Generate all 10 shots for a scenario using StoryDiffusion."""
        global write

        out_dir.mkdir(parents=True, exist_ok=True)

        if self.pipe is None:
            self.load_pipeline()

        self._setup_globals()

        # Install consistent attention processors
        setup_consistent_attention(
            self.pipe.unet, self.num_ids, device=self.device
        )

        shots = scenario["shots"]
        neg_prompt = scenario.get("negative_prompt", "blurry, low quality")

        # Phase 1: Generate reference frames (write=True)
        print(f"\n  [StoryDiff] Phase 1: Generating {self.num_ids} reference frames...")
        write = True
        self._reset_step_state()

        ref_prompts = [shots[i]["keyframe_prompt"] for i in range(self.num_ids)]

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        gen = torch.Generator(device=self.device).manual_seed(self.seed)

        ref_images = self.pipe(
            prompt=ref_prompts,
            negative_prompt=[neg_prompt] * self.num_ids,
            num_inference_steps=self.num_steps,
            guidance_scale=self.guidance_scale,
            generator=gen,
            height=self.img_size,
            width=self.img_size,
        ).images

        keyframes = {}
        for i, img in enumerate(ref_images):
            sid = shots[i]["shot_id"]
            keyframes[sid] = img
            img.save(out_dir / f"shot_{sid}.png")
            print(f"    {sid}: saved (reference frame)")

        # Phase 2: Generate remaining frames one-by-one (write=False)
        # id_bank from Phase 1 provides identity features.
        print(f"\n  [StoryDiff] Phase 2: Generating {len(shots) - self.num_ids} consistent frames...")

        for i in range(self.num_ids, len(shots)):
            write = False
            self._reset_step_state()

            prompt = shots[i]["keyframe_prompt"]
            sid = shots[i]["shot_id"]

            gen = torch.Generator(device=self.device).manual_seed(
                self.seed + i
            )

            img = self.pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=self.num_steps,
                guidance_scale=self.guidance_scale,
                generator=gen,
                height=self.img_size,
                width=self.img_size,
            ).images[0]

            keyframes[sid] = img
            img.save(out_dir / f"shot_{sid}.png")
            print(f"    {sid}: saved (consistent generation)")

        # Contact sheet
        self._make_contact_sheet(keyframes, shots, out_dir)

        print(f"\n  [StoryDiff] Done -> {out_dir.resolve()}")
        return keyframes

    def _make_contact_sheet(self, keyframes, shots, out_dir):
        """Create a contact sheet of all generated keyframes."""
        real_shots = [s for s in shots if not s.get("is_bridge", False)]
        n = len(real_shots)
        thumb = 256
        cols = min(n, 5)
        rows = (n + cols - 1) // cols

        sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 30)), "white")
        draw = ImageDraw.Draw(sheet)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        for idx, shot in enumerate(real_shots):
            sid = shot["shot_id"]
            if sid not in keyframes:
                continue
            img = keyframes[sid].resize((thumb, thumb), Image.LANCZOS)
            r, c = divmod(idx, cols)
            x, y = c * thumb, r * (thumb + 30)
            sheet.paste(img, (x, y))
            draw.text((x + 5, y + thumb + 5), sid, fill="black", font=font)

        sheet.save(out_dir / "contact_sheet.png")

    def cleanup(self):
        """Release GPU memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# Batch Runner for MSR-50
# ═══════════════════════════════════════════════════════════════════════

def run_storydiff_on_msr50(
    dataset_dir: Path | str = "datasets/MSR-50",
    output_dir: Path | str = "outputs/msr50/storydiff",
    device: str = "cuda:3",
    max_scenarios: int | None = None,
):
    """Run StoryDiffusion baseline on MSR-50 benchmark."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    combined = dataset_dir / "MSR-50.json"
    with open(combined) as f:
        data = json.load(f)

    scenarios = data["scenarios"]
    if max_scenarios:
        scenarios = scenarios[:max_scenarios]

    print(f"\n{'=' * 70}")
    print(f"StoryDiffusion Baseline on MSR-50")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Device: {device}")
    print(f"{'=' * 70}")

    gen = StoryDiffusionGenerator(device=device)
    gen.load_pipeline()

    for i, scenario in enumerate(scenarios):
        sid = scenario["scenario_id"]
        domain = scenario["domain"]
        s_out = output_dir / sid

        if (s_out / "shot_S10.png").exists():
            print(f"\n  [{i+1}/{len(scenarios)}] {sid} ({domain}) — already done, skipping")
            continue

        print(f"\n  [{i+1}/{len(scenarios)}] {sid} ({domain})")
        try:
            gen.generate_scenario(scenario, s_out)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        torch.cuda.empty_cache()

    gen.cleanup()
    print(f"\n{'=' * 70}")
    print(f"StoryDiffusion baseline complete -> {output_dir.resolve()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/MSR-50")
    parser.add_argument("--output", default="outputs/msr50/storydiff")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()

    run_storydiff_on_msr50(
        dataset_dir=args.dataset,
        output_dir=args.output,
        device=args.device,
        max_scenarios=args.max,
    )
