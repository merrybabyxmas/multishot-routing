"""
Tests for both Forward and Reverse routing strategies.

8-shot scenario:
  S1: {A},    D     S5: {C},    F
  S2: {A,B},  D     S6: {A,B},  E
  S3: {A,B},  D     S7: {A,B},  D
  S4: {B},    E     S8: {A,C},  F
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.routing import ShotNode, RoutingGraph, ReverseRoutingGraph, distance


def make_scenario():
    return [
        ShotNode(shot_id="S1", entities={"A"},      bg="D", action="A walks through city"),
        ShotNode(shot_id="S2", entities={"A", "B"}, bg="D", action="B joins A in city"),
        ShotNode(shot_id="S3", entities={"A", "B"}, bg="D", action="A and B have conversation"),
        ShotNode(shot_id="S4", entities={"B"},      bg="E", action="B alone in forest"),
        ShotNode(shot_id="S5", entities={"C"},      bg="F", action="C lurks in cave"),
        ShotNode(shot_id="S6", entities={"A", "B"}, bg="E", action="A and B regroup in forest"),
        ShotNode(shot_id="S7", entities={"A", "B"}, bg="D", action="A and B return to city"),
        ShotNode(shot_id="S8", entities={"A", "C"}, bg="F", action="A confronts C in cave"),
    ]


# ══════════════════════════════════════════════════════════════════════
# Forward Routing Tests (original)
# ══════════════════════════════════════════════════════════════════════

def test_forward():
    print("\n" + "=" * 70)
    print("FORWARD ROUTING TEST")
    print("=" * 70)

    graph = RoutingGraph()
    graph.build_from_shots(make_scenario())
    graph.print_routing_table()
    graph.print_detailed_edges()

    def find(sid):
        for n in graph.all_nodes:
            if n.shot_id == sid:
                return n
        raise ValueError(f"Node {sid} not found")

    s3, s4, s6, s7 = find("S3"), find("S4"), find("S6"), find("S7")

    # S4 must go through a bridge from S3
    bridge = s4.parent_node
    assert bridge.is_bridge, f"S4 parent must be bridge, got {bridge}"
    assert bridge.entities == {"B"}, f"Bridge entities must be {{B}}, got {bridge.entities}"
    assert bridge.parent_node is s3, f"Bridge parent must be S3"
    print("  ✓ S4 transitions via bridge from S3")

    # S6 routes to S3 (D=1)
    assert s6.parent_node is s3
    print("  ✓ S6 routes to S3 (D=1)")

    # S7 routes to S3 (D=0)
    assert s7.parent_node is s3
    assert distance(s7, s3) == 0
    print("  ✓ S7 routes to S3 (D=0)")

    print("\n  ★ FORWARD ROUTING TESTS PASSED ★\n")


# ══════════════════════════════════════════════════════════════════════
# Reverse Routing Tests
# ══════════════════════════════════════════════════════════════════════

def test_reverse():
    print("\n" + "=" * 70)
    print("REVERSE ROUTING TEST")
    print("=" * 70)

    graph = ReverseRoutingGraph()
    graph.build_from_shots(make_scenario())
    graph.print_routing_table()
    graph.print_detailed_edges()
    graph.print_topological_path()

    def find(sid):
        for n in graph.all_nodes:
            if n.shot_id == sid:
                return n
        raise ValueError(f"Node {sid} not found")

    # ── 1. Universal Anchor exists with ALL entities ──
    u0 = find("U0")
    assert u0.entities == {"A", "B", "C"}, f"Anchor must have all entities, got {u0.entities}"
    assert u0.parent_node is None, "Anchor must be root"
    print(f"  ✓ Universal Anchor: entities={u0.entities}, bg={u0.bg}")

    # ── 2. NO +entity transitions anywhere ──
    for node in graph.all_nodes:
        if node.parent_node is None:
            continue
        added = node.entities - node.parent_node.entities
        assert len(added) == 0, (
            f"+entity detected: {node.parent_node.shot_id}→{node.shot_id}, "
            f"added={added}"
        )
    print("  ✓ No +entity transitions in entire graph")

    # ── 3. All edges have D ≤ 1 ──
    for node in graph.all_nodes:
        if node.parent_node is None:
            continue
        d = distance(node, node.parent_node)
        assert d <= 1, (
            f"D={d} > 1: {node.parent_node.shot_id}→{node.shot_id}"
        )
    print("  ✓ All edges have D ≤ 1")

    # ── 4. S8={A,C},F is derived via subtraction path ──
    s8 = find("S8")
    # Trace ancestry: S8 → ... → U0
    path = []
    node = s8
    while node is not None:
        path.append(node)
        node = node.parent_node
    path.reverse()
    print(f"  S8 ancestry: {' → '.join(f'{n.shot_id}(E={n.entities},BG={n.bg})' for n in path)}")

    # S8's path must go through a node with {A,C} (B was subtracted, not added)
    # The key check is: at no point was C added — it was always present
    entity_sets = [frozenset(n.entities) for n in path]
    assert frozenset({"A", "B", "C"}) in entity_sets, "Path must start from {A,B,C}"
    # C must be present in every ancestor until it's no longer needed
    for i, (node_in_path, ent_set) in enumerate(zip(path, entity_sets)):
        if "C" in ent_set:
            # Once C appears, check it was inherited, not added
            if i > 0:
                prev_ent = entity_sets[i - 1]
                assert "C" in prev_ent, f"C was added at {node_in_path.shot_id}!"
    print("  ✓ S8 derived by subtraction (C never added)")

    # ── 5. D=0 temporal chaining for same-state shots ──
    s2, s3, s7 = find("S2"), find("S3"), find("S7")
    # S3 should parent to S2 (D=0), S7 should parent to S3 (D=0)
    assert distance(s3, s2) == 0
    assert s3.parent_node is s2, f"S3 should chain to S2, got {s3.parent_node.shot_id}"
    assert distance(s7, s3) == 0
    assert s7.parent_node is s3, f"S7 should chain to S3, got {s7.parent_node.shot_id}"
    print("  ✓ S2→S3→S7 temporal chaining (D=0)")

    # ── 6. All real shots are reachable ──
    gen_order = graph.topological_order()
    real_shot_ids = {n.shot_id for n in gen_order if not n.is_bridge}
    expected = {"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"}
    assert real_shot_ids == expected, f"Missing shots: {expected - real_shot_ids}"
    print("  ✓ All 8 shots reachable in topological order")

    print(f"\n  ★ ALL REVERSE ROUTING TESTS PASSED ★\n")


if __name__ == "__main__":
    test_forward()
    test_reverse()
