"""
Strict unit tests for the Non-Markovian Graph Router.

Uses the 8-shot scenario derived from iea.txt with the following
Global Legend and expected routing behaviour.

Global Legend:
  Entities: A (hero), B (sidekick), C (villain)
  Backgrounds: D (city), E (forest), F (cave)

Expected strict conditions:
  1. Shot 3→4: path must be  Shot 3 -> Bridge (E={'B'}, BG='D') -> Shot 4
  2. Shot 6 must route back to Shot 3 (D=1), NOT Shot 5
  3. Shot 7 must route back to Shot 3 (D=0)
"""

import sys, os

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.routing import ShotNode, RoutingGraph, distance


# ── Global Legend ──────────────────────────────────────────────────────────
# Entities: A = hero, B = sidekick, C = villain
# Backgrounds: D = city, E = forest, F = cave

# ── 8-Shot Scenario ──────────────────────────────────────────────────────
# Designed so that the routing constraints are exercised.
#
# Shot 1: {A}      bg=D   "A walks through city"
# Shot 2: {A,B}    bg=D   "B joins A"                      → parent=S1 (D=1, +B)
# Shot 3: {A,B}    bg=D   "A and B talk"                   → parent=S2 (D=0)
# Shot 4: {B}      bg=E   "B alone in forest"              → needs bridge from S3 (D=2: -A, bg D→E)
#         Bridge:  {B}     bg=D  (subtract A first)         → D=1 from S3
#         then S4 parent = Bridge (D=1: bg D→E)
# Shot 5: {C}      bg=F   "C in cave"                      → far from everything, bridge needed
# Shot 6: {A,B}    bg=E   "A and B regroup in forest"      → S3 has {A,B},D → D=1 (bg diff only)
#                                                              S4 has {B},E → D=1 (+A)
#                                                              Pick S3? Both D=1. S3 is earliest min.
#                                                              Actually let me reconsider...
#
# Wait — Shot 6 must route to Shot 3 specifically. Let me adjust so S3 is
# the unique argmin.
#
# Revised so Shot 6 = {B, A} bg=D but action differs.
# No — the user says Shot 6 routes to Shot 3 with D=1. And Shot 7 routes
# to Shot 3 with D=0. Shot 7 = {A,B} bg=D → same as Shot 3 → D=0. Good.
# Shot 6 must have D=1 to Shot 3 and D>1 to all others.
#
# Let me re-derive carefully:
#   Shot 3 = {A,B}, D
#   Shot 6 routes to Shot 3 at D=1
#   → Shot 6 differs from Shot 3 by exactly 1 (either +/- one entity OR bg change)
#   Shot 7 = {A,B}, D  (same as Shot 3, D=0)
#
# For Shot 6 to prefer Shot 3 over Shot 4's bridge:
#   Shot 6 = {A,B}, E  → D(S6,S3)=1 (bg diff), D(S6,S4={B},E)=1 (+A)
#   Two candidates at D=1! We need S3 to win. Since both are D=1, the
#   algorithm picks the first one found. S3 appears before S4/bridges
#   in all_nodes, so the iteration picks S3 first (best_d=1 already).
#   Actually the loop updates only on strict <, so the FIRST node with the
#   minimum distance wins. S3 (index 2) comes before Bridge B1 (index 4) and
#   S4 (index 5). So S3 wins. ✓

scenario_8_shots = [
    ShotNode(shot_id="S1", entities={"A"},      bg="D", action="A walks through city"),
    ShotNode(shot_id="S2", entities={"A", "B"}, bg="D", action="B joins A in city"),
    ShotNode(shot_id="S3", entities={"A", "B"}, bg="D", action="A and B have conversation"),
    ShotNode(shot_id="S4", entities={"B"},      bg="E", action="B alone in forest"),
    ShotNode(shot_id="S5", entities={"C"},      bg="F", action="C lurks in cave"),
    ShotNode(shot_id="S6", entities={"A", "B"}, bg="E", action="A and B regroup in forest"),
    ShotNode(shot_id="S7", entities={"A", "B"}, bg="D", action="A and B return to city"),
    ShotNode(shot_id="S8", entities={"A", "C"}, bg="F", action="A confronts C in cave"),
]


def run_test():
    graph = RoutingGraph()
    graph.build_from_shots(scenario_8_shots)

    # ── Print full routing info ──
    graph.print_routing_table()
    graph.print_detailed_edges()
    graph.print_topological_path()

    # ══════════════════════════════════════════════════════════════════════
    # STRICT ASSERTIONS
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("RUNNING STRICT ASSERTIONS")
    print("=" * 70)

    # Helper: find node by id
    def find(sid: str) -> ShotNode:
        for n in graph.all_nodes:
            if n.shot_id == sid:
                return n
        raise ValueError(f"Node {sid} not found")

    s3 = find("S3")
    s4 = find("S4")
    s6 = find("S6")
    s7 = find("S7")

    # ── Condition 1: Shot 4 transition via bridge ──
    # S4's parent must be a bridge with entities={'B'} and bg='D'
    bridge_before_s4 = s4.parent_node
    assert bridge_before_s4 is not None, "S4 must have a parent"
    assert bridge_before_s4.is_bridge, (
        f"S4's parent must be a BRIDGE, got {bridge_before_s4}"
    )
    assert bridge_before_s4.entities == {"B"}, (
        f"Bridge before S4 must have entities={{'B'}}, got {bridge_before_s4.entities}"
    )
    assert bridge_before_s4.bg == "D", (
        f"Bridge before S4 must have bg='D', got {bridge_before_s4.bg}"
    )
    # And the bridge's parent must be S3
    assert bridge_before_s4.parent_node is s3, (
        f"Bridge's parent must be S3, got {bridge_before_s4.parent_node}"
    )
    d_s3_bridge = distance(s3, bridge_before_s4)
    d_bridge_s4 = distance(bridge_before_s4, s4)
    assert d_s3_bridge == 1, f"D(S3, Bridge) must be 1, got {d_s3_bridge}"
    assert d_bridge_s4 == 1, f"D(Bridge, S4) must be 1, got {d_bridge_s4}"
    print(
        f"  ✓ Condition 1 PASSED: S3 --[D=1]--> Bridge {bridge_before_s4.shot_id} "
        f"(E={bridge_before_s4.entities}, BG={bridge_before_s4.bg}) --[D=1]--> S4"
    )

    # ── Condition 2: Shot 6 routes back to Shot 3 (D=1) ──
    assert s6.parent_node is s3, (
        f"S6 must route to S3, got {s6.parent_node}"
    )
    d_s6_s3 = distance(s6, s3)
    assert d_s6_s3 == 1, f"D(S6, S3) must be 1, got {d_s6_s3}"
    print(f"  ✓ Condition 2 PASSED: S6 routes to S3 with D={d_s6_s3}")

    # ── Condition 3: Shot 7 routes back to Shot 3 (D=0) ──
    assert s7.parent_node is s3, (
        f"S7 must route to S3, got {s7.parent_node}"
    )
    d_s7_s3 = distance(s7, s3)
    assert d_s7_s3 == 0, f"D(S7, S3) must be 0, got {d_s7_s3}"
    print(f"  ✓ Condition 3 PASSED: S7 routes to S3 with D={d_s7_s3}")

    print("\n  ★ ALL STRICT ASSERTIONS PASSED ★\n")


if __name__ == "__main__":
    run_test()
