"""
MSR-50 Benchmark Evaluation Pipeline.

Two-phase architecture to avoid OOM:
  Phase 1 — GENERATE: Run each pipeline on each scenario sequentially.
            Generator loads → generates → saves PNGs → unloads. Full GPU.
  Phase 2 — EVALUATE: Load CLIP + DINOv2 scorers, iterate saved PNGs.
            Compute CLIP-I, DINO Identity, CLIP-T. No generator on GPU.

Metrics:
  1. Background Consistency (CLIP-I): cosine sim of CLIP image embeddings
     between shots sharing the same background symbol.
  2. Identity Preservation (DINO): DINOv2 embedding similarity of the
     same entity across different shots.
  3. Prompt Alignment (CLIP-T): CLIP text-image cosine similarity
     between each frame and its keyframe prompt.

Usage:
    # Quick sanity check (2 scenarios)
    python -m src.core.evaluate_benchmark --max-scenarios 2

    # Full MSR-50 benchmark
    python -m src.core.evaluate_benchmark

    # Evaluate only (skip generation, use existing frames)
    python -m src.core.evaluate_benchmark --eval-only
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.routing import ShotNode


# ═══════════════════════════════════════════════════════════════════════
# Metric Computers (Phase 2 only)
# ═══════════════════════════════════════════════════════════════════════

class CLIPScorer:
    """CLIP-based image-image (CLIP-I) and text-image (CLIP-T) similarity."""

    def __init__(self, device: str = "cuda:3"):
        import open_clip

        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device,
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, img: Image.Image) -> torch.Tensor:
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer([text]).to(self.device)
        feat = self.model.encode_text(tokens)
        return F.normalize(feat, dim=-1)

    def image_similarity(self, img_a: Image.Image, img_b: Image.Image) -> float:
        ea = self.encode_image(img_a)
        eb = self.encode_image(img_b)
        return (ea @ eb.T).item()

    def text_image_similarity(self, text: str, img: Image.Image) -> float:
        et = self.encode_text(text)
        ei = self.encode_image(img)
        return (et @ ei.T).item()

    def unload(self):
        del self.model
        torch.cuda.empty_cache()
        gc.collect()


class DINOScorer:
    """DINOv2-based identity preservation scoring."""

    def __init__(self, device: str = "cuda:3"):
        self.device = device
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", verbose=False,
        ).to(device).eval()

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def encode_image(self, img: Image.Image) -> torch.Tensor:
        x = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        feat = self.model(x)
        return F.normalize(feat, dim=-1)

    def similarity(self, img_a: Image.Image, img_b: Image.Image) -> float:
        ea = self.encode_image(img_a)
        eb = self.encode_image(img_b)
        return (ea @ eb.T).item()

    def unload(self):
        del self.model
        torch.cuda.empty_cache()
        gc.collect()


# ═══════════════════════════════════════════════════════════════════════
# Scenario → ShotNode conversion (for our pipeline)
# ═══════════════════════════════════════════════════════════════════════

def scenario_to_shot_nodes(scenario: dict) -> list[ShotNode]:
    """Convert MSR-50 JSON scenario to list of ShotNode for our generators."""
    nodes = []
    for shot in scenario["shots"]:
        nodes.append(ShotNode(
            shot_id=shot["shot_id"],
            entities=set(shot["target_entities"]),
            bg=shot["target_bg"],
            action=shot.get("motion_prompt", ""),
        ))
    return nodes


# ═══════════════════════════════════════════════════════════════════════
# Per-Scenario Metric Calculation
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(
    scenario: dict,
    keyframes: dict[str, Image.Image],
    clip_scorer: CLIPScorer,
    dino_scorer: DINOScorer,
) -> dict[str, float]:
    """Compute all 3 metrics for one scenario's generated keyframes."""
    shots = scenario["shots"]

    # ── 1. Background Consistency (CLIP-I) ──────────────────────────
    bg_groups: dict[str, list[str]] = {}
    for s in shots:
        bg_groups.setdefault(s["target_bg"], []).append(s["shot_id"])

    clip_i_scores = []
    for bg_sym, sids in bg_groups.items():
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                a, b = sids[i], sids[j]
                if a in keyframes and b in keyframes:
                    sim = clip_scorer.image_similarity(keyframes[a], keyframes[b])
                    clip_i_scores.append(sim)

    clip_i = float(np.mean(clip_i_scores)) if clip_i_scores else 0.0

    # ── 2. Identity Preservation (DINO) ─────────────────────────────
    entity_groups: dict[str, list[str]] = {}
    for s in shots:
        for ent in s["target_entities"]:
            entity_groups.setdefault(ent, []).append(s["shot_id"])

    dino_scores = []
    for ent, sids in entity_groups.items():
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                a, b = sids[i], sids[j]
                if a in keyframes and b in keyframes:
                    sim = dino_scorer.similarity(keyframes[a], keyframes[b])
                    dino_scores.append(sim)

    dino_id = float(np.mean(dino_scores)) if dino_scores else 0.0

    # ── 3. Prompt Alignment (CLIP-T) ────────────────────────────────
    clip_t_scores = []
    for s in shots:
        sid = s["shot_id"]
        if sid in keyframes:
            prompt = s["keyframe_prompt"]
            if len(prompt) > 300:
                prompt = prompt[:300]
            sim = clip_scorer.text_image_similarity(prompt, keyframes[sid])
            clip_t_scores.append(sim)

    clip_t = float(np.mean(clip_t_scores)) if clip_t_scores else 0.0

    return {"clip_i": clip_i, "dino_id": dino_id, "clip_t": clip_t}


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Generation
# ═══════════════════════════════════════════════════════════════════════

PIPELINE_LABELS = ["Ours", "StoryDiffusion", "Markovian", "NoBridge", "GlobalInject"]


def _all_shots_exist(out_dir: Path, shot_ids: list[str]) -> bool:
    return all((out_dir / f"shot_{sid}.png").exists() for sid in shot_ids)


def _load_keyframes(out_dir: Path, shot_ids: list[str]) -> dict[str, Image.Image]:
    return {sid: Image.open(out_dir / f"shot_{sid}.png") for sid in shot_ids}



def _create_generator(pipe_label: str, device: str):
    """Create a generator instance for the given pipeline label."""
    if pipe_label == "Ours":
        from src.core.generator import KeyframeGenerator
        return KeyframeGenerator(device=device, num_steps=8, max_blend=0.7, inject_pct=0.6)
    elif pipe_label == "StoryDiffusion":
        from src.core.storydiff_baseline import StoryDiffusionGenerator
        return StoryDiffusionGenerator(device=device, num_steps=25, guidance_scale=5.0)
    elif pipe_label in ("Markovian", "NoBridge", "GlobalInject"):
        from src.core.ablation import (
            MarkovianGenerator, NoBridgeGenerator, GlobalInjectGenerator,
        )
        cls_map = {
            "Markovian": MarkovianGenerator,
            "NoBridge": NoBridgeGenerator,
            "GlobalInject": GlobalInjectGenerator,
        }
        return cls_map[pipe_label](device=device, num_steps=8, max_blend=0.7, inject_pct=0.6)
    else:
        raise ValueError(f"Unknown pipeline: {pipe_label}")


def _run_generator(gen, pipe_label: str, scenario: dict, out_dir: Path):
    """Run a generator on one scenario."""
    if pipe_label == "StoryDiffusion":
        gen.generate_scenario(scenario, out_dir)
    else:
        gen.run(
            scenario=scenario_to_shot_nodes(scenario),
            entity_prompts=scenario["entities"],
            bg_prompts=scenario["backgrounds"],
            out_dir=out_dir,
        )


def phase1_generate(
    scenarios: list[dict],
    pipelines: list[str],
    base_dir: Path,
    device: str,
):
    """Phase 1: Generate all frames. One pipeline at a time — model loads once."""
    total = len(scenarios) * len(pipelines)
    done = 0

    for pipe_label in pipelines:
        print(f"\n{'━' * 70}")
        print(f"  PIPELINE: {pipe_label} — loading model...")
        print(f"{'━' * 70}")

        # Load generator ONCE per pipeline
        gen = _create_generator(pipe_label, device)

        for sc_idx, scenario in enumerate(scenarios):
            done += 1
            sc_id = scenario["scenario_id"]
            shot_ids = [s["shot_id"] for s in scenario["shots"]]
            pipe_dir = base_dir / "frames" / pipe_label / sc_id
            pipe_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  [{done}/{total}] {pipe_label} / {sc_id}")

            # Resume check
            if _all_shots_exist(pipe_dir, shot_ids):
                print(f"      Cached ({len(shot_ids)} frames)")
                continue

            t0 = time.time()
            try:
                _run_generator(gen, pipe_label, scenario, pipe_dir)
                elapsed = time.time() - t0
                print(f"      OK ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"      FAIL ({elapsed:.1f}s): {e}")
                import traceback
                traceback.print_exc()

        # Unload generator, free GPU
        del gen
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  {pipe_label} done, GPU freed.")

    print(f"\n  Phase 1 complete: frames in {base_dir / 'frames'}")


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Evaluation
# ═══════════════════════════════════════════════════════════════════════

def phase2_evaluate(
    scenarios: list[dict],
    pipelines: list[str],
    base_dir: Path,
    device: str,
) -> list[dict]:
    """Phase 2: Load scorers, compute metrics on saved frames."""
    print("\n" + "=" * 70)
    print("PHASE 2: METRIC EVALUATION")
    print("=" * 70)

    print("  Loading CLIP scorer...")
    clip_scorer = CLIPScorer(device=device)
    print("  Loading DINOv2 scorer...")
    dino_scorer = DINOScorer(device=device)
    print("  Scorers ready.\n")

    all_results: list[dict] = []
    total = len(scenarios) * len(pipelines)
    done = 0

    for sc_idx, scenario in enumerate(scenarios):
        sc_id = scenario["scenario_id"]
        domain = scenario.get("domain", "unknown")
        shot_ids = [s["shot_id"] for s in scenario["shots"]]

        print(f"\n  SCENARIO {sc_idx + 1}/{len(scenarios)}: {sc_id}")

        for pipe_label in pipelines:
            done += 1
            pipe_dir = base_dir / "frames" / pipe_label / sc_id

            if not _all_shots_exist(pipe_dir, shot_ids):
                missing = sum(1 for s in shot_ids
                              if not (pipe_dir / f"shot_{s}.png").exists())
                print(f"    [{done}/{total}] {pipe_label}: SKIP "
                      f"({missing}/{len(shot_ids)} frames missing)")
                continue

            keyframes = _load_keyframes(pipe_dir, shot_ids)

            t0 = time.time()
            metrics = compute_metrics(scenario, keyframes, clip_scorer, dino_scorer)
            elapsed = time.time() - t0

            result = {
                "scenario_id": sc_id,
                "domain": domain,
                "pipeline": pipe_label,
                **metrics,
            }
            all_results.append(result)

            print(f"    [{done}/{total}] {pipe_label}: "
                  f"CLIP-I={metrics['clip_i']:.4f}  "
                  f"DINO={metrics['dino_id']:.4f}  "
                  f"CLIP-T={metrics['clip_t']:.4f}  "
                  f"({elapsed:.1f}s)")

    # Cleanup scorers
    clip_scorer.unload()
    dino_scorer.unload()

    return all_results


# ═══════════════════════════════════════════════════════════════════════
# Result Aggregation & Formatting
# ═══════════════════════════════════════════════════════════════════════

def aggregate_results(
    all_results: list[dict],
) -> dict[str, dict[str, float]]:
    """Aggregate per-scenario results into per-pipeline averages."""
    from collections import defaultdict
    accum: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in all_results:
        pipe = r["pipeline"]
        for metric in ["clip_i", "dino_id", "clip_t"]:
            accum[pipe][metric].append(r[metric])

    summary = {}
    for pipe, metrics in accum.items():
        summary[pipe] = {
            k: float(np.mean(v)) for k, v in metrics.items()
        }
    return summary


def print_markdown_table(summary: dict[str, dict[str, float]]):
    """Print NeurIPS-ready markdown table."""
    metrics = ["clip_i", "dino_id", "clip_t"]
    best = {}
    for m in metrics:
        vals = {p: s[m] for p, s in summary.items()}
        best[m] = max(vals, key=lambda p: vals[p])

    header_map = {
        "clip_i": "BG Consist. (CLIP-I) ↑",
        "dino_id": "Identity (DINO) ↑",
        "clip_t": "Prompt Align (CLIP-T) ↑",
    }

    print("\n" + "=" * 78)
    print("EVALUATION RESULTS — MSR-50 Benchmark")
    print("=" * 78)

    header = "| Method | " + " | ".join(header_map[m] for m in metrics) + " |"
    sep = "|" + "|".join(["---"] * (len(metrics) + 1)) + "|"
    print(header)
    print(sep)

    pipe_order = ["Ours", "StoryDiffusion", "Markovian", "NoBridge", "GlobalInject"]
    for pipe in pipe_order:
        if pipe not in summary:
            continue
        s = summary[pipe]
        cells = []
        for m in metrics:
            val = f"{s[m]:.4f}"
            if pipe == best[m]:
                val = f"**{val}**"
            cells.append(val)
        row = f"| {pipe} | " + " | ".join(cells) + " |"
        print(row)

    print()


def save_csv(all_results: list[dict], path: Path):
    """Save per-scenario results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scenario_id", "domain", "pipeline", "clip_i", "dino_id", "clip_t"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  Saved CSV -> {path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MSR-50 Benchmark Evaluation")
    parser.add_argument("--dataset", type=str, default="datasets/MSR-50/MSR-50.json",
                        help="Path to MSR-50 combined JSON")
    parser.add_argument("--output-dir", type=str, default="outputs/msr50_eval",
                        help="Output directory for generated frames and results")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--max-scenarios", type=int, default=0,
                        help="Limit number of scenarios (0=all)")
    parser.add_argument("--pipelines", type=str, default="all",
                        help="Comma-separated pipeline names, or 'all'")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip generation, only compute metrics on existing frames")
    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ────────────────────────────────────────────────
    print("=" * 70)
    print("MSR-50 BENCHMARK EVALUATION")
    print("=" * 70)

    dataset_path = Path(args.dataset)
    with open(dataset_path) as f:
        dataset = json.load(f)

    scenarios = dataset["scenarios"] if "scenarios" in dataset else dataset
    if isinstance(scenarios, dict):
        scenarios = scenarios.get("scenarios", [scenarios])

    if args.max_scenarios > 0:
        scenarios = scenarios[:args.max_scenarios]

    # ── Select pipelines ────────────────────────────────────────────
    if args.pipelines == "all":
        pipelines = PIPELINE_LABELS
    else:
        pipelines = [p.strip() for p in args.pipelines.split(",")]

    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Pipelines: {pipelines}")
    print(f"  Output: {base_dir.resolve()}")
    print(f"  Device: {args.device}")
    print(f"  Mode: {'eval-only' if args.eval_only else 'generate + evaluate'}")

    # ── Phase 1: Generate ───────────────────────────────────────────
    if not args.eval_only:
        print("\n" + "=" * 70)
        print("PHASE 1: KEYFRAME GENERATION")
        print("=" * 70)
        phase1_generate(scenarios, pipelines, base_dir, args.device)

    # ── Phase 2: Evaluate ───────────────────────────────────────────
    all_results = phase2_evaluate(scenarios, pipelines, base_dir, args.device)

    # ── Output ──────────────────────────────────────────────────────
    if not all_results:
        print("\n  No results collected. Check errors above.")
        return

    save_csv(all_results, base_dir / "evaluation_results.csv")

    summary = aggregate_results(all_results)
    print_markdown_table(summary)

    # Save summary JSON
    summary_path = base_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary -> {summary_path}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
