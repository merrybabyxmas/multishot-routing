"""
Test script: verify the 3 improvement strategies on scifi_01.

Strategy 1: Steep Sigmoid Spatial Mask (vs old linear gradient)
Strategy 2: Semantic Promotion + IP-Adapter skip for abstract entities
Strategy 3: K/V-Reliance Prompt Stripping (short prompts when K/V injected)

Generates keyframes and saves comparison contact sheet.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.routing import ShotNode, RoutingGraph, distance
from src.core.generator import KeyframeGenerator


def load_scenario(json_path: str) -> tuple[list[ShotNode], dict, dict]:
    with open(json_path) as f:
        data = json.load(f)
    nodes = []
    for shot in data["shots"]:
        nodes.append(ShotNode(
            shot_id=shot["shot_id"],
            entities=set(shot["target_entities"]),
            bg=shot["target_bg"],
        ))
    return nodes, data["entities"], data["backgrounds"]


def main():
    scenario_path = "datasets/MSR-50/scifi_01.json"
    out_dir = Path("outputs/three_strategies_test/scifi_01")

    nodes, entity_prompts, bg_prompts = load_scenario(scenario_path)

    # ── Verify Strategy 3: prompt stripping ──
    print("=" * 70)
    print("STRATEGY 3 VERIFICATION: Prompt Stripping")
    print("=" * 70)
    # Build routing to get parents
    graph = RoutingGraph()
    graph.build_from_shots(nodes)
    gen_order = graph.topological_order()

    for node in gen_order:
        has_parent = node.parent_node is not None
        full = KeyframeGenerator.build_prompt(node, entity_prompts, bg_prompts, is_kv_injected=False)
        stripped = KeyframeGenerator.build_prompt(node, entity_prompts, bg_prompts, is_kv_injected=True)
        kind = "Bridge" if node.is_bridge else "Shot"
        print(f"\n  [{kind}] {node.shot_id} (will use {'STRIPPED' if has_parent else 'FULL'}):")
        print(f"    FULL    ({len(full):3d} chars): {full[:90]}...")
        print(f"    STRIPPED({len(stripped):3d} chars): {stripped[:90]}...")

    # ── Verify Strategy 2: abstract entity detection ──
    print("\n" + "=" * 70)
    print("STRATEGY 2 VERIFICATION: Abstract Entity Detection")
    print("=" * 70)
    # Test with stylized_05 which has storm cloud
    with open("datasets/MSR-50/stylized_05.json") as f:
        s05 = json.load(f)
    for sym, prompt in s05["entities"].items():
        is_abs = KeyframeGenerator._is_abstract_entity(prompt)
        promoted = KeyframeGenerator._promote_entity_prompt(prompt.split(",")[0])
        print(f"  Entity {sym}: abstract={is_abs}")
        print(f"    Original : {prompt.split(',')[0]}")
        print(f"    Promoted : {promoted}")

    # ── Verify Strategy 1: sigmoid mask shape ──
    print("\n" + "=" * 70)
    print("STRATEGY 1 VERIFICATION: Sigmoid Mask Shape")
    print("=" * 70)
    import torch
    size = 64  # 64x64 = 4096 tokens (typical for 512x512 with VAE)
    steepness = 12.0
    x = torch.linspace(-1.0, 1.0, size)
    sigmoid_1d = torch.sigmoid(x * steepness)
    print(f"  Mask values across width (64 positions):")
    print(f"    Left edge  [0]:  {sigmoid_1d[0]:.6f}")
    print(f"    1/4 point [16]:  {sigmoid_1d[16]:.6f}")
    print(f"    Center    [31]:  {sigmoid_1d[31]:.6f}")
    print(f"    Center    [32]:  {sigmoid_1d[32]:.6f}")
    print(f"    3/4 point [48]:  {sigmoid_1d[48]:.6f}")
    print(f"    Right edge[63]:  {sigmoid_1d[63]:.6f}")
    transition_zone = ((sigmoid_1d > 0.05) & (sigmoid_1d < 0.95)).sum().item()
    print(f"  Transition zone (0.05-0.95): {transition_zone}/{size} positions "
          f"({transition_zone/size*100:.1f}% of width)")

    # ── Run generation ──
    print("\n" + "=" * 70)
    print("GENERATING KEYFRAMES (scifi_01)")
    print("=" * 70)

    nodes2, _, _ = load_scenario(scenario_path)
    gen = KeyframeGenerator(device="cuda:3", num_steps=8, max_blend=0.7, inject_pct=0.6)
    gen.run(
        scenario=nodes2,
        entity_prompts=entity_prompts,
        bg_prompts=bg_prompts,
        out_dir=out_dir,
    )

    print(f"\nDone! Results at: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
