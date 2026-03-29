"""
Non-Markovian Graph Routing for Multi-Shot Video Generation.

Implements the DAG construction algorithm from iea.txt:
  - ShotNode / BridgeNode data structures
  - Distance metric D(S_t, S_k) = |E_t Δ E_k| + I(B_t ≠ B_k)
  - Non-Markovian routing with bridge injection when D ≥ 2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ShotNode:
    """A single shot (or bridge) in the routing graph."""

    shot_id: str
    entities: set[str]
    bg: str
    action: str = ""
    parent_node: Optional["ShotNode"] = field(default=None, repr=False)
    is_bridge: bool = False

    # Outgoing edges (children that depend on this node)
    children: list["ShotNode"] = field(default_factory=list, repr=False)

    def __repr__(self) -> str:
        kind = "Bridge" if self.is_bridge else "Shot"
        return (
            f"{kind}({self.shot_id}, entities={self.entities}, "
            f"bg='{self.bg}', parent={self.parent_node.shot_id if self.parent_node else None})"
        )


# ---------------------------------------------------------------------------
# Distance metric
# ---------------------------------------------------------------------------

def distance(s_t: ShotNode, s_k: ShotNode) -> int:
    """D(S_t, S_k) = |E_t Δ E_k| + I(B_t ≠ B_k)"""
    d_entity = len(s_t.entities.symmetric_difference(s_k.entities))
    d_bg = int(s_t.bg != s_k.bg)
    return d_entity + d_bg


# ---------------------------------------------------------------------------
# Bridge construction helpers
# ---------------------------------------------------------------------------

def _build_bridge_chain(
    source: ShotNode,
    target: ShotNode,
    bridge_counter: list[int],
) -> ShotNode:
    """
    Build exactly D-1 bridge nodes from *source* toward *target*.

    Strategy: additions first, then bg change, then subtractions last.

    By adding entities before removing them, bridge nodes pass through the
    maximal entity set (source ∪ target). This creates intermediate states
    that future shots can reuse — e.g., if the next shot after target needs
    both old and new entities, a bridge already carries their combined K/V.

    Returns the last bridge node (direct parent of target).
    """
    # Build the ordered list of intermediate (entities, bg) states
    cur_ent = set(source.entities)
    cur_bg = source.bg
    steps: list[tuple[set[str], str]] = []

    # 1. Entity additions FIRST (builds up to maximal entity set)
    for ent in sorted(target.entities - cur_ent):
        cur_ent = cur_ent | {ent}
        steps.append((set(cur_ent), cur_bg))

    # 2. Background change
    if cur_bg != target.bg:
        cur_bg = target.bg
        steps.append((set(cur_ent), cur_bg))

    # 3. Entity subtractions LAST (prunes down to target)
    for ent in sorted(cur_ent - target.entities):
        cur_ent = cur_ent - {ent}
        steps.append((set(cur_ent), cur_bg))

    # The last step should match the target state — skip it (the real shot
    # will occupy that position). Create bridges for all preceding steps.
    assert len(steps) >= 2, (
        f"Bridge injection called with D<2? steps={steps}"
    )

    prev_node = source
    # Create bridges for steps[0 .. -2] (all except the last)
    for ent_set, bg in steps[:-1]:
        bridge = ShotNode(
            shot_id=f"B{bridge_counter[0]}",
            entities=ent_set,
            bg=bg,
            is_bridge=True,
            parent_node=prev_node,
        )
        prev_node.children.append(bridge)
        bridge_counter[0] += 1
        prev_node = bridge

    return prev_node


# ---------------------------------------------------------------------------
# Routing Graph (DAG builder)
# ---------------------------------------------------------------------------

class RoutingGraph:
    """Constructs a generation DAG from a sequential shot list."""

    def __init__(self) -> None:
        self.nodes: list[ShotNode] = []          # all real shots (in order)
        self.all_nodes: list[ShotNode] = []       # real + bridge nodes
        self._bridge_counter: list[int] = [1]     # mutable counter

    # ----- core algorithm -----

    def add_shot(self, shot: ShotNode) -> None:
        """Insert a shot into the graph, routing to the optimal ancestor."""
        if not self.nodes:
            # First shot — no parent needed
            self.nodes.append(shot)
            self.all_nodes.append(shot)
            return

        # Find S_ref via explicit 3-key sort over all past nodes:
        #   1. Primary:   lowest total distance  D          (ascending)
        #   2. Secondary: lowest entity distance  D_entity   (ascending)
        #   3. Tertiary:  most recent node index             (descending)
        #
        # Rationale — in Stable Diffusion, preserving entity identity is
        # far harder than re-generating a background, so entity overlap
        # must dominate tie-breaking.  Among still-tied candidates the
        # most recent node gives best temporal coherence.
        candidates: list[tuple[int, int, int, ShotNode]] = []
        for idx, past in enumerate(self.all_nodes):
            d = distance(shot, past)
            d_ent = len(shot.entities.symmetric_difference(past.entities))
            # negative index → sort ascending = most-recent first
            candidates.append((d, d_ent, -idx, past))

        candidates.sort(key=lambda c: (c[0], c[1], c[2]))
        best_d, best_d_entity, _, best_ref = candidates[0]

        assert best_ref is not None

        if best_d == 0:
            # Rule A — optimal reuse
            shot.parent_node = best_ref
            best_ref.children.append(shot)
        elif best_d == 1:
            # Rule B — direct derivation
            shot.parent_node = best_ref
            best_ref.children.append(shot)
        else:
            # Rule C — bridge injection (D ≥ 2)
            last_bridge = _build_bridge_chain(
                best_ref, shot, self._bridge_counter
            )
            # Register all new bridge nodes
            node = last_bridge
            bridge_chain: list[ShotNode] = []
            while node is not best_ref:
                bridge_chain.append(node)
                node = node.parent_node  # type: ignore[assignment]
            bridge_chain.reverse()
            self.all_nodes.extend(bridge_chain)

            shot.parent_node = last_bridge
            last_bridge.children.append(shot)

        self.nodes.append(shot)
        self.all_nodes.append(shot)

    def build_from_shots(self, shots: list[ShotNode]) -> None:
        for s in shots:
            self.add_shot(s)

    # ----- traversal / inspection -----

    def topological_order(self) -> list[ShotNode]:
        """Return nodes in generation order (BFS from roots)."""
        from collections import deque

        in_degree: dict[str, int] = {n.shot_id: 0 for n in self.all_nodes}
        for n in self.all_nodes:
            for c in n.children:
                if c.shot_id in in_degree:
                    in_degree[c.shot_id] += 1

        queue = deque(n for n in self.all_nodes if in_degree[n.shot_id] == 0)
        order: list[ShotNode] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for c in node.children:
                if c.shot_id in in_degree:
                    in_degree[c.shot_id] -= 1
                    if in_degree[c.shot_id] == 0:
                        queue.append(c)
        return order

    def print_routing_table(self) -> None:
        """Pretty-print the routing decisions."""
        print("\n" + "=" * 70)
        print("ROUTING TABLE")
        print("=" * 70)
        for node in self.all_nodes:
            parent_id = node.parent_node.shot_id if node.parent_node else "ROOT"
            d = distance(node, node.parent_node) if node.parent_node else "-"
            kind = "BRIDGE" if node.is_bridge else "SHOT"
            print(
                f"  [{kind:6s}] {node.shot_id:5s}  "
                f"entities={node.entities!s:20s}  bg={node.bg!s:5s}  "
                f"parent={parent_id:5s}  D={d}"
            )
        print("=" * 70)

    def print_topological_path(self) -> None:
        """Print the generation-order path with arrows."""
        order = self.topological_order()
        print("\n" + "=" * 70)
        print("TOPOLOGICAL GENERATION PATH")
        print("=" * 70)
        parts = []
        for node in order:
            kind = "Bridge" if node.is_bridge else "Shot"
            parts.append(f"{kind} {node.shot_id} (E={node.entities}, BG={node.bg})")
        print(" -> ".join(parts))
        print("=" * 70)

    def print_detailed_edges(self) -> None:
        """Print every parent→child edge with distance."""
        print("\n" + "=" * 70)
        print("DETAILED EDGE LIST")
        print("=" * 70)
        for node in self.all_nodes:
            if node.parent_node is None:
                continue
            p = node.parent_node
            d = distance(node, p)
            p_kind = "Bridge" if p.is_bridge else "Shot"
            n_kind = "Bridge" if node.is_bridge else "Shot"
            print(
                f"  {p_kind} {p.shot_id} (E={p.entities}, BG={p.bg})"
                f"  --[D={d}]-->  "
                f"{n_kind} {node.shot_id} (E={node.entities}, BG={node.bg})"
            )
        print("=" * 70)
