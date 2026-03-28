"""
Ablation Study — "Killer Scenarios" designed so every baseline fails visibly.

Scenario A (Fantasy 5-shot): Entity swap forces Markovian + Global Inject failure.
Scenario B (Bridge Killer 6-shot): Every shot after S1 has D≥2 from all real shots,
  so bridge nodes are the ONLY way to achieve D=1 transitions.

Routing analysis for Scenario B:
  S1:{A} D → S2:{B} E  (D=3 from S1, needs 2 bridges)
  S1:{A} D → S3:{C} F  (D=3 from all, needs 1 bridge via B3)
  S2:{B} E → S4:{A,B} F (D=2 from S2, needs 1 bridge)
  B2:{B} D → S5:{B,C} D (D=1 from bridge B2! No Bridge: D=2)
  S3:{C} F → S6:{A,C} E (D=2 from S3, needs 1 bridge)

  Markovian: EVERY transition after S1 is D=3 (catastrophic).
  No Bridge: S2,S3 jump at D=3; S4,S5,S6 jump at D=2.
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
# Scenario A — Fantasy 5-Shot (Markovian + Global Inject killer)
# ═══════════════════════════════════════════════════════════════════════

FANTASY_ENTITIES = {
    "A": "a silver-armored knight with a longsword, heroic, full body, standing, white background",
    "B": "a robed mage with a glowing staff, wise, full body, standing, white background",
    "C": "a giant fire dragon with red scales, menacing, full body, white background",
}

FANTASY_BGS = {
    "D": "a peaceful royal castle with towers and banners at sunset, cinematic, no people",
    "E": "a dark ice cave with frozen stalactites and blue glow, cinematic, no people",
}


def _fantasy_scenario():
    return [
        ShotNode(shot_id="S1", entities={"A"}, bg="D",
                 action="knight stands guard in front of the castle"),
        ShotNode(shot_id="S2", entities={"A", "B"}, bg="D",
                 action="mage joins the knight for conversation at the castle"),
        ShotNode(shot_id="S3", entities={"C"}, bg="D",
                 action="a giant fire dragon descends in front of the castle"),
        ShotNode(shot_id="S4", entities={"A", "B"}, bg="E",
                 action="knight and mage hiding inside the dark ice cave after fleeing"),
        ShotNode(shot_id="S5", entities={"A"}, bg="D",
                 action="knight returns alone to the peaceful castle"),
    ]


# ═══════════════════════════════════════════════════════════════════════
# Scenario B — Bridge Killer 6-Shot (ALL baselines fail)
#
# Key design: 3 entities × 3 backgrounds, every pair of real shots
# has D≥2, so bridges are the ONLY way to get D=1 transitions.
# ═══════════════════════════════════════════════════════════════════════

BRIDGE_ENTITIES = {
    "A": "a silver-armored knight with a longsword, heroic, full body, standing, white background",
    "B": "a robed mage with a glowing staff, wise, full body, standing, white background",
    "C": "a giant fire dragon with red scales, menacing, full body, white background",
}

BRIDGE_BGS = {
    "D": "a peaceful royal castle with towers and banners at sunset, cinematic, no people",
    "E": "a dark ice cave with frozen stalactites and blue glow, cinematic, no people",
    "F": "a dense enchanted forest with sunlight filtering through, cinematic, no people",
}


def _bridge_scenario():
    """Every consecutive pair of real shots has D≥2.
    Without bridge nodes, every transition is a destructive jump.
    """
    return [
        ShotNode(shot_id="S1", entities={"A"}, bg="D",
                 action="knight guards the castle gate alone"),
        ShotNode(shot_id="S2", entities={"B"}, bg="E",
                 action="mage explores the dark ice cave alone"),
        ShotNode(shot_id="S3", entities={"C"}, bg="F",
                 action="dragon lurks in the enchanted forest"),
        ShotNode(shot_id="S4", entities={"A", "B"}, bg="F",
                 action="knight and mage regroup in the forest"),
        ShotNode(shot_id="S5", entities={"B", "C"}, bg="D",
                 action="mage confronts dragon at the castle"),
        ShotNode(shot_id="S6", entities={"A", "C"}, bg="E",
                 action="knight battles dragon in the ice cave"),
    ]


# ═══════════════════════════════════════════════════════════════════════
# Runner — execute one scenario across all 4 conditions
# ═══════════════════════════════════════════════════════════════════════

def run_scenario(
    name: str,
    scenario_fn,
    entity_prompts: dict,
    bg_prompts: dict,
    base_dir: Path,
    shot_ids: list[str],
):
    """Run Ours + 3 ablation baselines for a given scenario."""
    out = base_dir / name
    out.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, Image.Image]] = {}

    configs = [
        ("Ours",           KeyframeGenerator,    out / "ours"),
        ("Markovian",      MarkovianGenerator,   out / "exp1_markovian"),
        ("No Bridge",      NoBridgeGenerator,    out / "exp2_no_bridge"),
        ("Global Inject",  GlobalInjectGenerator, out / "exp3_global_inject"),
    ]

    for label, GenClass, exp_dir in configs:
        print("\n" + "█" * 70)
        print(f"  {name.upper()} — {label}")
        print("█" * 70 + "\n")

        gen = GenClass(device="cuda:3", num_steps=8, max_blend=0.7, inject_pct=0.6)
        gen.run(
            scenario=scenario_fn(),
            entity_prompts=entity_prompts,
            bg_prompts=bg_prompts,
            out_dir=exp_dir,
        )

        kf = {}
        for sid in shot_ids:
            p = exp_dir / f"shot_{sid}.png"
            if p.exists():
                kf[sid] = Image.open(p)
        all_results[label] = kf

        del gen
        torch.cuda.empty_cache()

    # ── Comparison sheets ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"COMPARISON SHEETS — {name}")
    print("=" * 70)

    make_comparison_sheet(
        all_results, shot_ids,
        out / "comparison_full.png",
    )

    # Per-experiment focused comparisons
    ours = all_results["Ours"]

    make_comparison_sheet(
        {"Ours": ours, "Markovian": all_results["Markovian"]},
        shot_ids,
        out / "comparison_vs_markovian.png",
    )
    make_comparison_sheet(
        {"Ours": ours, "No Bridge": all_results["No Bridge"]},
        shot_ids,
        out / "comparison_vs_no_bridge.png",
    )
    make_comparison_sheet(
        {"Ours": ours, "Global Inject": all_results["Global Inject"]},
        shot_ids,
        out / "comparison_vs_global_inject.png",
    )

    print(f"\n{name} complete -> {out.resolve()}")
    return all_results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    base_dir = Path("outputs/ablation_killer")

    # ── Scenario A: Fantasy (5-shot) ──────────────────────────────
    run_scenario(
        name="fantasy",
        scenario_fn=_fantasy_scenario,
        entity_prompts=FANTASY_ENTITIES,
        bg_prompts=FANTASY_BGS,
        base_dir=base_dir,
        shot_ids=["S1", "S2", "S3", "S4", "S5"],
    )

    # ── Scenario B: Bridge Killer (6-shot) ────────────────────────
    run_scenario(
        name="bridge_killer",
        scenario_fn=_bridge_scenario,
        entity_prompts=BRIDGE_ENTITIES,
        bg_prompts=BRIDGE_BGS,
        base_dir=base_dir,
        shot_ids=["S1", "S2", "S3", "S4", "S5", "S6"],
    )

    print("\n" + "=" * 70)
    print("ALL KILLER ABLATIONS COMPLETE")
    print(f"Results: {base_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
