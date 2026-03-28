"""
Ablation Study — 5-Shot "Killer Scenario" for maximum baseline failure.

Designed to stress-test all 3 core ideas simultaneously:
  S1 → S2: +entity (knight + mage)
  S2 → S3: entity swap (D=3: remove A,B + add C = catastrophic for Markovian)
  S3 → S4: non-Markovian jump back to S2's entities + bg change (D=3 without routing)
  S4 → S5: -entity + bg change (D=2: needs bridge)

Each baseline should fail visibly on at least one transition.
"""

from __future__ import annotations

import sys
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


# ═══════════════════════════════════════════════════════════════════════
# Killer Scenario — Entities & Backgrounds
# ═══════════════════════════════════════════════════════════════════════

KILLER_ENTITY_PROMPTS = {
    "A": "a silver-armored knight with a longsword, heroic, full body, standing, white background",
    "B": "a robed mage with a glowing staff, wise, full body, standing, white background",
    "C": "a giant fire dragon with red scales, menacing, full body, white background",
}

KILLER_BG_PROMPTS = {
    "D": "a peaceful royal castle with towers and banners at sunset, cinematic, no people",
    "E": "a dark ice cave with frozen stalactites and blue glow, cinematic, no people",
}


def _killer_scenario():
    return [
        ShotNode(
            shot_id="S1", entities={"A"}, bg="D",
            action="knight stands guard in front of the castle",
        ),
        ShotNode(
            shot_id="S2", entities={"A", "B"}, bg="D",
            action="mage joins the knight for conversation at the castle",
        ),
        ShotNode(
            shot_id="S3", entities={"C"}, bg="D",
            action="a giant fire dragon descends in front of the castle",
        ),
        ShotNode(
            shot_id="S4", entities={"A", "B"}, bg="E",
            action="knight and mage hiding inside the dark ice cave after fleeing",
        ),
        ShotNode(
            shot_id="S5", entities={"A"}, bg="D",
            action="knight returns alone to the peaceful castle",
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    base_dir = Path("outputs/ablation_killer")
    base_dir.mkdir(parents=True, exist_ok=True)
    shot_ids = ["S1", "S2", "S3", "S4", "S5"]

    all_results: dict[str, dict[str, Image.Image]] = {}

    # ── Ours ───────────────────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  KILLER SCENARIO: Ours (Full Pipeline)")
    print("█" * 70 + "\n")
    ours_dir = base_dir / "ours"
    gen = KeyframeGenerator(device="cuda:3", num_steps=8, max_blend=0.7, inject_pct=0.6)
    gen.run(
        scenario=_killer_scenario(),
        entity_prompts=KILLER_ENTITY_PROMPTS,
        bg_prompts=KILLER_BG_PROMPTS,
        out_dir=ours_dir,
    )
    ours_kf = {}
    for sid in shot_ids:
        p = ours_dir / f"shot_{sid}.png"
        if p.exists():
            ours_kf[sid] = Image.open(p)
    all_results["Ours"] = ours_kf
    del gen; torch.cuda.empty_cache()

    # ── Exp 1: Markovian ───────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  KILLER SCENARIO: Exp 1 — Markovian Baseline")
    print("█" * 70 + "\n")
    exp1_dir = base_dir / "exp1_markovian"
    gen1 = MarkovianGenerator(device="cuda:3", num_steps=8, max_blend=0.7, inject_pct=0.6)
    gen1.run(
        scenario=_killer_scenario(),
        entity_prompts=KILLER_ENTITY_PROMPTS,
        bg_prompts=KILLER_BG_PROMPTS,
        out_dir=exp1_dir,
    )
    exp1_kf = {}
    for sid in shot_ids:
        p = exp1_dir / f"shot_{sid}.png"
        if p.exists():
            exp1_kf[sid] = Image.open(p)
    all_results["Markovian"] = exp1_kf
    del gen1; torch.cuda.empty_cache()

    # ── Exp 2: No Bridge ──────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  KILLER SCENARIO: Exp 2 — No Bridge")
    print("█" * 70 + "\n")
    exp2_dir = base_dir / "exp2_no_bridge"
    gen2 = NoBridgeGenerator(device="cuda:3", num_steps=8, max_blend=0.7, inject_pct=0.6)
    gen2.run(
        scenario=_killer_scenario(),
        entity_prompts=KILLER_ENTITY_PROMPTS,
        bg_prompts=KILLER_BG_PROMPTS,
        out_dir=exp2_dir,
    )
    exp2_kf = {}
    for sid in shot_ids:
        p = exp2_dir / f"shot_{sid}.png"
        if p.exists():
            exp2_kf[sid] = Image.open(p)
    all_results["No Bridge"] = exp2_kf
    del gen2; torch.cuda.empty_cache()

    # ── Exp 3: Global Injection ───────────────────────────────────
    print("\n" + "█" * 70)
    print("  KILLER SCENARIO: Exp 3 — Global Injection")
    print("█" * 70 + "\n")
    exp3_dir = base_dir / "exp3_global_inject"
    gen3 = GlobalInjectGenerator(device="cuda:3", num_steps=8, max_blend=0.7, inject_pct=0.6)
    gen3.run(
        scenario=_killer_scenario(),
        entity_prompts=KILLER_ENTITY_PROMPTS,
        bg_prompts=KILLER_BG_PROMPTS,
        out_dir=exp3_dir,
    )
    exp3_kf = {}
    for sid in shot_ids:
        p = exp3_dir / f"shot_{sid}.png"
        if p.exists():
            exp3_kf[sid] = Image.open(p)
    all_results["Global Inject"] = exp3_kf
    del gen3; torch.cuda.empty_cache()

    # ── Comparison Sheets ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING KILLER COMPARISON SHEETS")
    print("=" * 70)

    # Full 4×5 comparison
    make_comparison_sheet(
        all_results, shot_ids,
        base_dir / "comparison_full.png",
    )

    # Per-experiment focused views
    # Exp 1: S3→S4 is the killer — Markovian uses dragon K/V for knight+mage
    make_comparison_sheet(
        {"Ours": ours_kf, "Markovian": exp1_kf},
        ["S2", "S3", "S4", "S5"],
        base_dir / "comparison_exp1_routing.png",
    )

    # Exp 2: S3→S4 has D=3 (no bridge = catastrophic jump)
    make_comparison_sheet(
        {"Ours": ours_kf, "No Bridge": exp2_kf},
        ["S2", "S3", "S4", "S5"],
        base_dir / "comparison_exp2_bridge.png",
    )

    # Exp 3: S2 and S4 are +entity shots — chimera expected
    make_comparison_sheet(
        {"Ours": ours_kf, "Global Inject": exp3_kf},
        ["S1", "S2", "S4", "S5"],
        base_dir / "comparison_exp3_chimera.png",
    )

    print("\n" + "=" * 70)
    print("KILLER ABLATION COMPLETE")
    print(f"Results: {base_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
