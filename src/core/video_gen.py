"""
Phase 4: Image-to-Video Generation via I2VGen-XL.

Animates keyframes produced by the routing pipeline.
Generates per-shot clips and comparison grids across ablation conditions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ═══════════════════════════════════════════════════════════════════════
# Motion Prompts — action-enhanced versions for I2V
# ═══════════════════════════════════════════════════════════════════════

FANTASY_MOTION_PROMPTS = {
    "S1": "A silver-armored knight is slowly walking in front of a peaceful royal castle at sunset, cinematic, high quality",
    "S2": "A robed mage walks up next to a silver-armored knight and converses with hand gestures at a royal castle, cinematic, high quality",
    "S3": "A giant fire dragon flies down from the sky, lands in front of a royal castle and breathes fire, cinematic, high quality",
    "S4": "A silver-armored knight and a robed mage are looking around cautiously, hiding inside a dark ice cave with frozen stalactites, cinematic, high quality",
    "S5": "A silver-armored knight confidently walks forward toward the front of a peaceful royal castle at sunset, cinematic, high quality",
}

FANTASY_NEGATIVE = "blurry, low quality, distorted, static, no motion, watermark, text"


# ═══════════════════════════════════════════════════════════════════════
# I2V Pipeline
# ═══════════════════════════════════════════════════════════════════════

class VideoGenerator:

    def __init__(self, device: str = "cuda:3"):
        self.device = device
        self.dtype = torch.float16
        self.pipe = None

    def load_pipeline(self):
        from diffusers import I2VGenXLPipeline

        print("[I2V] Loading I2VGen-XL pipeline...")
        self.pipe = I2VGenXLPipeline.from_pretrained(
            "ali-vilab/i2vgen-xl",
            torch_dtype=self.dtype,
            variant="fp16",
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        print("[I2V] Ready.\n")

    def animate_keyframe(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = FANTASY_NEGATIVE,
        num_steps: int = 50,
        guidance_scale: float = 9.0,
        seed: int = 42,
    ) -> list[Image.Image]:
        """Generate video frames from a single keyframe image."""
        gen = torch.Generator(device=self.device).manual_seed(seed)

        # I2VGen-XL expects specific image sizes
        image = image.resize((512, 512))

        output = self.pipe(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=gen,
        )
        return output.frames[0]

    def animate_scenario(
        self,
        keyframe_dir: Path,
        motion_prompts: dict[str, str],
        shot_ids: list[str],
        out_dir: Path,
        label: str = "",
    ) -> dict[str, list[Image.Image]]:
        """Animate all keyframes from a directory."""
        from diffusers.utils import export_to_gif

        out_dir.mkdir(parents=True, exist_ok=True)
        all_clips: dict[str, list[Image.Image]] = {}

        for sid in shot_ids:
            kf_path = keyframe_dir / f"shot_{sid}.png"
            if not kf_path.exists():
                print(f"  [SKIP] {sid}: keyframe not found at {kf_path}")
                continue

            keyframe = Image.open(kf_path)
            prompt = motion_prompts.get(sid, "")

            tag = f"{label} " if label else ""
            print(f"  [{tag}{sid}] Animating: {prompt[:60]}...")

            frames = self.animate_keyframe(keyframe, prompt)
            all_clips[sid] = frames

            # Save individual clip as GIF
            gif_path = out_dir / f"{sid}.gif"
            export_to_gif(frames, str(gif_path))
            print(f"    Saved -> {gif_path} ({len(frames)} frames)")

            # Also save as mp4
            try:
                from diffusers.utils import export_to_video
                mp4_path = out_dir / f"{sid}.mp4"
                export_to_video(frames, str(mp4_path), fps=8)
                print(f"    Saved -> {mp4_path}")
            except Exception as e:
                print(f"    MP4 export failed: {e}")

        return all_clips


# ═══════════════════════════════════════════════════════════════════════
# Comparison Builder — frame-by-frame grid
# ═══════════════════════════════════════════════════════════════════════

def _load_font(size: int):
    from PIL import ImageFont
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def make_comparison_gif(
    all_results: dict[str, dict[str, list[Image.Image]]],
    shot_id: str,
    out_path: Path,
    fps: int = 8,
):
    """Create side-by-side GIF comparing one shot across conditions."""
    from diffusers.utils import export_to_gif

    conditions = list(all_results.keys())
    clips = [all_results[c].get(shot_id, []) for c in conditions]

    if not any(clips):
        print(f"  No clips for {shot_id}, skipping comparison")
        return

    # Find min frame count
    n_frames = min(len(c) for c in clips if c)
    n_conds = len(conditions)

    w, h = 512, 512
    pad = 4
    label_h = 36
    font = _load_font(24)

    grid_w = n_conds * (w + pad) + pad
    grid_h = h + pad + label_h

    comparison_frames = []

    for f_idx in range(n_frames):
        frame = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
        draw = ImageDraw.Draw(frame)

        for c_idx, (cond_name, clip) in enumerate(zip(conditions, clips)):
            x = pad + c_idx * (w + pad)
            if clip and f_idx < len(clip):
                img = clip[f_idx].resize((w, h))
                frame.paste(img, (x, pad))
            draw.text(
                (x + w // 2 - 40, pad + h + 4),
                cond_name,
                fill=(200, 200, 200),
                font=font,
            )

        comparison_frames.append(frame)

    export_to_gif(comparison_frames, str(out_path), fps=fps)
    print(f"  Comparison GIF -> {out_path} ({n_frames} frames)")


def make_full_comparison_gif(
    all_results: dict[str, dict[str, list[Image.Image]]],
    shot_ids: list[str],
    out_path: Path,
    fps: int = 8,
):
    """Create grid GIF: rows=conditions, cols=shots, animated."""
    from diffusers.utils import export_to_gif

    conditions = list(all_results.keys())
    n_conds = len(conditions)
    n_shots = len(shot_ids)

    # Find min frame count across all clips
    n_frames = 999
    for cond_clips in all_results.values():
        for sid in shot_ids:
            clip = cond_clips.get(sid, [])
            if clip:
                n_frames = min(n_frames, len(clip))
    if n_frames == 999:
        print("  No clips found, skipping full comparison")
        return

    w, h = 256, 256  # smaller for grid
    pad = 3
    label_h = 28
    header_w = 160
    font_label = _load_font(18)
    font_header = _load_font(20)

    grid_w = header_w + n_shots * (w + pad) + pad
    grid_h = pad + label_h + n_conds * (h + pad + label_h)

    comparison_frames = []

    for f_idx in range(n_frames):
        frame = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
        draw = ImageDraw.Draw(frame)

        # Column headers
        for j, sid in enumerate(shot_ids):
            x = header_w + pad + j * (w + pad)
            draw.text((x + w // 2 - 14, pad + 2), sid, fill=(255, 255, 100), font=font_header)

        # Rows
        for i, cond_name in enumerate(conditions):
            y = pad + label_h + i * (h + pad + label_h)
            draw.text((pad + 4, y + h // 2 - 10), cond_name, fill=(200, 200, 200), font=font_label)

            for j, sid in enumerate(shot_ids):
                x = header_w + pad + j * (w + pad)
                clip = all_results[cond_name].get(sid, [])
                if clip and f_idx < len(clip):
                    img = clip[f_idx].resize((w, h))
                    frame.paste(img, (x, y))

        comparison_frames.append(frame)

    export_to_gif(comparison_frames, str(out_path), fps=fps)
    print(f"  Full comparison GIF -> {out_path} ({n_frames} frames, {n_conds}×{n_shots} grid)")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ablation_base = Path("outputs/ablation_killer/fantasy")
    video_base = Path("outputs/video_comparison")
    video_base.mkdir(parents=True, exist_ok=True)

    shot_ids = ["S1", "S2", "S3", "S4", "S5"]

    conditions = {
        "Ours":          ablation_base / "ours",
        "Markovian":     ablation_base / "exp1_markovian",
        "No Bridge":     ablation_base / "exp2_no_bridge",
        "Global Inject": ablation_base / "exp3_global_inject",
    }

    vgen = VideoGenerator(device="cuda:3")
    vgen.load_pipeline()

    all_results: dict[str, dict[str, list[Image.Image]]] = {}

    for cond_name, kf_dir in conditions.items():
        print(f"\n{'='*60}")
        print(f"  Animating: {cond_name}")
        print(f"{'='*60}")

        clips_dir = video_base / cond_name.lower().replace(" ", "_")
        clips = vgen.animate_scenario(
            keyframe_dir=kf_dir,
            motion_prompts=FANTASY_MOTION_PROMPTS,
            shot_ids=shot_ids,
            out_dir=clips_dir,
            label=cond_name,
        )
        all_results[cond_name] = clips

    # ── Comparison outputs ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Generating comparison GIFs")
    print(f"{'='*60}")

    # Per-shot side-by-side
    for sid in shot_ids:
        make_comparison_gif(
            all_results, sid,
            video_base / f"compare_{sid}.gif",
        )

    # Full grid (all conditions × all shots)
    make_full_comparison_gif(
        all_results, shot_ids,
        video_base / "comparison_full.gif",
    )

    # Key shots focused
    make_comparison_gif(
        {"Ours": all_results["Ours"], "Markovian": all_results["Markovian"]},
        "S4",
        video_base / "compare_S4_routing.gif",
    )
    make_comparison_gif(
        {"Ours": all_results["Ours"], "Global Inject": all_results["Global Inject"]},
        "S2",
        video_base / "compare_S2_chimera.gif",
    )

    print(f"\n{'='*60}")
    print(f"  VIDEO GENERATION COMPLETE")
    print(f"  Output: {video_base.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
