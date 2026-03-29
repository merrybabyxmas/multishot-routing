"""
Full Cyberpunk Pipeline — Keyframe Generation + I2V + Ablation Comparison.

5 conditions:
  1. Ours (Full Pipeline)
  2. Markovian Baseline
  3. No Bridge Baseline
  4. Global Injection Baseline
  5. Naive T2I Baseline (no K/V injection, IP-Adapter only)

Features:
  - Resume logic: skips already-generated shots
  - HuggingFace upload
  - Storage monitoring
  - Comprehensive logging
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.routing import ShotNode, RoutingGraph, distance
from src.core.generator import KeyframeGenerator, _stable_hash
from src.core.ablation import (
    MarkovianGenerator,
    NoBridgeGenerator,
    GlobalInjectGenerator,
    make_comparison_sheet,
)
from src.core.cyberpunk_scenario import (
    ENTITY_PROMPTS, BG_PROMPTS, MOTION_PROMPTS, NEGATIVE_PROMPT,
    SHOT_IDS, build_scenario,
)


# ═══════════════════════════════════════════════════════════════════════
# 5th Baseline: Naive T2I (no K/V injection, only IP-Adapter)
# ═══════════════════════════════════════════════════════════════════════

class NaiveT2IGenerator(KeyframeGenerator):
    """Each shot generated independently — no K/V injection, no routing.
    Only IP-Adapter provides identity conditioning. Represents the simplest
    approach without any of our contributions."""

    def run(self, scenario, entity_prompts, bg_prompts, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("BASELINE: Naive T2I (IP-Adapter only, no K/V injection)")
        print("=" * 70)

        self.load_pipeline()
        self.build_anchor_cache(entity_prompts, bg_prompts, out_dir)

        print("\n" + "=" * 70)
        print("KEYFRAME GENERATION (Independent T2I)")
        print("=" * 70)

        keyframes: dict[str, Image.Image] = {}

        for node in scenario:
            prompt = self.build_prompt(node, entity_prompts, bg_prompts)
            ip_embeds = self._compose_ip_embeds(node)
            gen = torch.Generator(device=self.device).manual_seed(
                self.seed + _stable_hash(node.shot_id)
            )

            print(f"\n  {'─'*60}")
            print(f"  [Shot] {node.shot_id}  Mode: INDEPENDENT T2I")
            print(f"  Prompt: {prompt[:75]}...")

            # Pure T2I — no K/V injection at all
            self.attn_ctrl.set_mode_bypass()

            img = self.pipe(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted, deformed",
                ip_adapter_image_embeds=ip_embeds,
                num_inference_steps=self.num_steps,
                guidance_scale=self.guidance_scale,
                generator=gen,
                width=self.width, height=self.height,
            ).images[0]

            keyframes[node.shot_id] = img
            path = out_dir / f"shot_{node.shot_id}.png"
            img.save(path)
            print(f"  Saved -> {path}")

        real_shots = [n for n in scenario if not n.is_bridge]
        self._make_contact_sheet(real_shots, keyframes, out_dir)
        print(f"\nNaive T2I done -> {out_dir.resolve()}")
        return keyframes


# ═══════════════════════════════════════════════════════════════════════
# Storage Monitor
# ═══════════════════════════════════════════════════════════════════════

def check_disk_usage(path: str = "/home/dongwoo44") -> dict:
    total, used, free = shutil.disk_usage(path)
    gb = 1024**3
    info = {
        "total_gb": total / gb,
        "used_gb": used / gb,
        "free_gb": free / gb,
        "usage_pct": used / total * 100,
    }
    print(f"  [Disk] {info['used_gb']:.1f}/{info['total_gb']:.1f} GB used "
          f"({info['usage_pct']:.1f}%), {info['free_gb']:.1f} GB free")
    return info


# ═══════════════════════════════════════════════════════════════════════
# HuggingFace Upload
# ═══════════════════════════════════════════════════════════════════════

def upload_to_hf(
    local_dir: Path,
    repo_id: str = "merrybabyxmas/multishot-routing-results",
    hf_token: str | None = None,
):
    """Upload results to HuggingFace dataset repo."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("  [HF] huggingface_hub not installed, skipping upload")
        return

    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("  [HF] No token provided, skipping upload")
        return

    api = HfApi(token=hf_token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True, token=hf_token)
    except Exception as e:
        print(f"  [HF] Repo creation note: {e}")

    print(f"  [HF] Uploading {local_dir} -> {repo_id}")
    try:
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
        )
        print(f"  [HF] Upload complete!")
        return True
    except Exception as e:
        print(f"  [HF] Upload failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════
# Resume-aware Generation Runner
# ═══════════════════════════════════════════════════════════════════════

def check_completed_shots(out_dir: Path, shot_ids: list[str]) -> set[str]:
    """Return set of shot IDs that already have generated keyframes."""
    completed = set()
    for sid in shot_ids:
        if (out_dir / f"shot_{sid}.png").exists():
            completed.add(sid)
    return completed


def run_condition(
    label: str,
    gen_class: type,
    out_dir: Path,
    shot_ids: list[str],
    force: bool = False,
) -> dict[str, Image.Image]:
    """Run one condition with resume logic."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check existing results
    completed = check_completed_shots(out_dir, shot_ids)
    if completed == set(shot_ids) and not force:
        print(f"\n  [{label}] All {len(shot_ids)} shots already exist, loading from disk")
        kf = {}
        for sid in shot_ids:
            kf[sid] = Image.open(out_dir / f"shot_{sid}.png")
        return kf

    if completed and not force:
        print(f"\n  [{label}] {len(completed)}/{len(shot_ids)} shots exist, regenerating all for consistency")

    print(f"\n{'█' * 70}")
    print(f"  GENERATING: {label}")
    print(f"{'█' * 70}\n")

    gen = gen_class(device="cuda:3", num_steps=25, max_blend=0.7, inject_pct=0.6, guidance_scale=7.5)
    gen.run(
        scenario=build_scenario(),
        entity_prompts=ENTITY_PROMPTS,
        bg_prompts=BG_PROMPTS,
        out_dir=out_dir,
    )

    del gen
    torch.cuda.empty_cache()

    kf = {}
    for sid in shot_ids:
        p = out_dir / f"shot_{sid}.png"
        if p.exists():
            kf[sid] = Image.open(p)
    return kf


# ═══════════════════════════════════════════════════════════════════════
# I2V Generation (with resume)
# ═══════════════════════════════════════════════════════════════════════

def run_i2v_for_condition(
    label: str,
    keyframe_dir: Path,
    video_dir: Path,
    shot_ids: list[str],
    force: bool = False,
) -> dict[str, list[Image.Image]]:
    """Run I2V on keyframes with resume logic."""
    from src.core.video_gen import VideoGenerator

    video_dir.mkdir(parents=True, exist_ok=True)

    # Check existing
    completed = set()
    for sid in shot_ids:
        if (video_dir / f"{sid}.gif").exists():
            completed.add(sid)

    if completed == set(shot_ids) and not force:
        print(f"\n  [I2V {label}] All clips exist, skipping")
        return {}

    remaining = [sid for sid in shot_ids if sid not in completed or force]
    print(f"\n  [I2V {label}] Generating {len(remaining)} clips...")

    vgen = VideoGenerator(device="cuda:3")
    vgen.load_pipeline()

    clips = vgen.animate_scenario(
        keyframe_dir=keyframe_dir,
        motion_prompts=MOTION_PROMPTS,
        shot_ids=remaining,
        out_dir=video_dir,
        label=label,
    )

    del vgen
    torch.cuda.empty_cache()

    return clips


# ═══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════

def main():
    base_dir = Path("outputs/cyberpunk")
    base_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN", "")

    # ── Step 0: Validate prompts ──────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 0: PROMPT VALIDATION")
    print("=" * 70)
    from src.core.prompt_validator import PromptValidator
    validator = PromptValidator()
    if not validator.run_all():
        print("ABORT: Prompt validation failed!")
        return

    check_disk_usage()

    # ── Step 1: Keyframe Generation (5 conditions) ────────────────
    print("\n" + "=" * 70)
    print("STEP 1: KEYFRAME GENERATION (5 conditions × 10 shots)")
    print("=" * 70)

    conditions = [
        ("Ours",           KeyframeGenerator,    base_dir / "keyframes" / "ours"),
        ("Markovian",      MarkovianGenerator,   base_dir / "keyframes" / "markovian"),
        ("No Bridge",      NoBridgeGenerator,    base_dir / "keyframes" / "no_bridge"),
        ("Global Inject",  GlobalInjectGenerator, base_dir / "keyframes" / "global_inject"),
        ("Naive T2I",      NaiveT2IGenerator,    base_dir / "keyframes" / "naive_t2i"),
    ]

    all_keyframes: dict[str, dict[str, Image.Image]] = {}

    for label, gen_class, out_dir in conditions:
        kf = run_condition(label, gen_class, out_dir, SHOT_IDS)
        all_keyframes[label] = kf
        check_disk_usage()

    # ── Step 2: Comparison Sheets ─────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: COMPARISON SHEETS")
    print("=" * 70)

    comp_dir = base_dir / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Full 5×10 grid
    make_comparison_sheet(
        all_keyframes, SHOT_IDS,
        comp_dir / "keyframe_comparison_full.png",
    )

    # Per-ablation focused
    ours = all_keyframes["Ours"]
    for label in ["Markovian", "No Bridge", "Global Inject", "Naive T2I"]:
        make_comparison_sheet(
            {"Ours": ours, label: all_keyframes[label]},
            SHOT_IDS,
            comp_dir / f"keyframe_vs_{label.lower().replace(' ', '_')}.png",
        )

    # Key shots comparison (chimera + routing + bridge tests)
    make_comparison_sheet(
        all_keyframes,
        ["S2", "S4", "S7", "S8", "S9"],
        comp_dir / "keyframe_key_shots.png",
    )

    # ── Step 3: I2V Video Generation ──────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: I2V VIDEO GENERATION")
    print("=" * 70)

    for label, _, kf_dir in conditions:
        vid_dir = base_dir / "videos" / label.lower().replace(" ", "_")
        run_i2v_for_condition(label, kf_dir, vid_dir, SHOT_IDS)
        check_disk_usage()

    # ── Step 4: Upload to HuggingFace ─────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: HUGGINGFACE UPLOAD")
    print("=" * 70)

    if hf_token:
        upload_to_hf(base_dir, hf_token=hf_token)
    else:
        print("  No HF_TOKEN set, skipping upload")

    # ── Step 5: Final Summary ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    total_files = sum(1 for _ in base_dir.rglob("*") if _.is_file())
    print(f"  Total files: {total_files}")
    check_disk_usage()

    print(f"\n  Output: {base_dir.resolve()}")
    print(f"  Structure:")
    print(f"    keyframes/  — 5 conditions × 10 shots = 50 keyframes")
    print(f"    comparisons/ — comparison grids")
    print(f"    videos/     — I2V clips per condition")
    print("=" * 70)


if __name__ == "__main__":
    main()
