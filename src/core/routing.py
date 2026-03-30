"""
Non-Markovian Graph Routing for Multi-Shot Video Generation.

Implements three routing strategies:
  1. RoutingGraph (Forward): S1→S2→... with bridge injection when D ≥ 2
  2. ReverseRoutingGraph (Reverse): Universal Anchor with ALL entities first,
     then derive shots by entity subtraction. Eliminates +entity transitions.
  3. EntityDecomposedRoutingGraph: Forward routing + per-entity reference
     resolution. Multi-entity nodes get one K/V source per entity (spatial
     injection), preventing identity chimera.

Core primitives:
  - ShotNode data structure
  - Distance metric D(S_t, S_k) = |E_t Δ E_k| + I(B_t ≠ B_k)
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

    Strategy (from iea.txt): subtractions first, then additions, then bg.

    We enumerate ALL D intermediate states (each one unit-change from the
    previous). The first D-1 become bridge nodes; the D-th state equals
    the target itself, so we skip it. This guarantees every edge has D ≤ 1.

    Returns the last bridge node (direct parent of target).
    """
    # Build the ordered list of intermediate (entities, bg) states
    cur_ent = set(source.entities)
    cur_bg = source.bg
    steps: list[tuple[set[str], str]] = []

    # 1. Entity subtractions
    for ent in sorted(cur_ent - target.entities):
        cur_ent = cur_ent - {ent}
        steps.append((set(cur_ent), cur_bg))

    # 2. Entity additions
    for ent in sorted(target.entities - cur_ent):
        cur_ent = cur_ent | {ent}
        steps.append((set(cur_ent), cur_bg))

    # 3. Background change
    if cur_bg != target.bg:
        cur_bg = target.bg
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


# ---------------------------------------------------------------------------
# Reverse Routing Graph (Top-Down from Universal Anchor)
# ---------------------------------------------------------------------------

class ReverseRoutingGraph:
    """Constructs a generation DAG using reverse (top-down) routing.

    Instead of building forward from S1 and injecting bridges when D≥2,
    this starts with a Universal Anchor containing ALL entities from the
    scenario, then derives each shot by entity subtraction and bg changes.

    Key property: +entity transitions NEVER occur. Every edge is either
    -entity or bg-change, both of which are much easier for the diffusion
    model to handle via K/V injection.

    Algorithm:
      1. Collect U = union of all entities across all shots
      2. Pick anchor_bg = most common background
      3. Create Universal Anchor U0 = (U, anchor_bg)
      4. For each shot, build a reduction path from U0:
         - Remove entities alphabetically (D=1 per removal)
         - Change background last (D=1)
      5. Share intermediate bridge nodes across paths
      6. Multiple shots with same state chain temporally (D=0)
    """

    def __init__(self) -> None:
        self.nodes: list[ShotNode] = []          # real shots (in order)
        self.all_nodes: list[ShotNode] = []       # real + bridge + anchor
        self._bridge_counter: list[int] = [1]
        # State registry: (frozenset(entities), bg) -> node for sharing
        self._state_map: dict[tuple[frozenset, str], ShotNode] = {}

    def build_from_shots(self, shots: list[ShotNode]) -> None:
        from collections import Counter

        # 1. Collect universal entity set
        all_entities: set[str] = set()
        for s in shots:
            all_entities |= s.entities

        # 2. Pick most common bg as anchor bg
        bg_counts = Counter(s.bg for s in shots)
        anchor_bg = bg_counts.most_common(1)[0][0]

        # 3. Create Universal Anchor
        anchor = ShotNode(
            shot_id="U0",
            entities=set(all_entities),
            bg=anchor_bg,
            is_bridge=True,
        )
        self.all_nodes.append(anchor)
        self._state_map[(frozenset(all_entities), anchor_bg)] = anchor

        # 4. Route each shot via reduction from anchor
        for shot in shots:
            self._route_shot(anchor, shot)
            self.nodes.append(shot)

    def _get_or_create_bridge(
        self, entities: frozenset, bg: str, parent: ShotNode,
    ) -> ShotNode:
        """Return existing node for this state, or create a new bridge."""
        key = (entities, bg)
        if key in self._state_map:
            return self._state_map[key]

        bridge = ShotNode(
            shot_id=f"B{self._bridge_counter[0]}",
            entities=set(entities),
            bg=bg,
            is_bridge=True,
            parent_node=parent,
        )
        parent.children.append(bridge)
        self._bridge_counter[0] += 1
        self.all_nodes.append(bridge)
        self._state_map[key] = bridge
        return bridge

    def _route_shot(self, anchor: ShotNode, target: ShotNode) -> None:
        """Build reduction path from anchor to target, then attach target."""
        target_key = (frozenset(target.entities), target.bg)

        # Fast path: state already exists (D=0 to existing node)
        if target_key in self._state_map:
            existing = self._state_map[target_key]
            target.parent_node = existing
            existing.children.append(target)
            self.all_nodes.append(target)
            # Update state_map → latest shot (better K/V for temporal coherence)
            self._state_map[target_key] = target
            return

        # Compute intermediate states (anchor → target), excluding target itself
        anchor_ent = frozenset(anchor.entities)
        anchor_bg = anchor.bg
        steps: list[tuple[frozenset, str]] = []

        cur_ent = set(anchor_ent)
        cur_bg = anchor_bg

        # Entity subtractions (alphabetical order for deterministic sharing)
        for ent in sorted(cur_ent - target.entities):
            cur_ent = cur_ent - {ent}
            state = (frozenset(cur_ent), cur_bg)
            if state != target_key:
                steps.append(state)

        # Background change (always last)
        if cur_bg != target.bg:
            cur_bg = target.bg
            state = (frozenset(cur_ent), cur_bg)
            if state != target_key:
                steps.append(state)

        # Walk intermediate steps, creating/reusing bridges
        prev_node: ShotNode = self._state_map.get(
            (anchor_ent, anchor_bg), anchor
        )
        for ent_fs, bg in steps:
            prev_node = self._get_or_create_bridge(ent_fs, bg, prev_node)

        # Attach target to last intermediate (D=1) or anchor (D=1)
        target.parent_node = prev_node
        prev_node.children.append(target)
        self.all_nodes.append(target)
        self._state_map[target_key] = target

    # ----- traversal / inspection (same interface as RoutingGraph) -----

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
        print("REVERSE ROUTING TABLE (Universal Anchor → Shots)")
        print("=" * 70)
        for node in self.all_nodes:
            parent_id = node.parent_node.shot_id if node.parent_node else "ROOT"
            d = distance(node, node.parent_node) if node.parent_node else "-"
            kind = "ANCHOR" if node.shot_id == "U0" else (
                "BRIDGE" if node.is_bridge else "SHOT"
            )
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
            kind = "Anchor" if node.shot_id == "U0" else (
                "Bridge" if node.is_bridge else "Shot"
            )
            parts.append(f"{kind} {node.shot_id} (E={node.entities}, BG={node.bg})")
        print(" -> ".join(parts))
        print("=" * 70)

    def print_detailed_edges(self) -> None:
        """Print every parent→child edge with distance and transition type."""
        print("\n" + "=" * 70)
        print("DETAILED EDGE LIST")
        print("=" * 70)
        for node in self.all_nodes:
            if node.parent_node is None:
                continue
            p = node.parent_node
            d = distance(node, p)

            # Classify transition type
            removed = p.entities - node.entities
            added = node.entities - p.entities
            bg_changed = p.bg != node.bg
            if removed:
                trans = f"-entity({removed})"
            elif bg_changed:
                trans = f"bg({p.bg}→{node.bg})"
            elif added:
                trans = f"+entity({added})"  # should never happen
            else:
                trans = "reuse(D=0)"

            p_kind = "Anchor" if p.shot_id == "U0" else (
                "Bridge" if p.is_bridge else "Shot"
            )
            n_kind = "Bridge" if node.is_bridge else "Shot"
            print(
                f"  {p_kind} {p.shot_id} (E={p.entities}, BG={p.bg})"
                f"  --[D={d}, {trans}]-->  "
                f"{n_kind} {node.shot_id} (E={node.entities}, BG={node.bg})"
            )
        print("=" * 70)


# ---------------------------------------------------------------------------
# Entity-Decomposed Routing Graph (Forward + Per-Entity Reference Resolution)
# ---------------------------------------------------------------------------

class EntityDecomposedRoutingGraph(RoutingGraph):
    """Forward routing with per-entity K/V reference resolution.

    Extends RoutingGraph (forward DAG with bridge injection) by adding
    entity_refs: for each multi-entity node, resolves the best identity
    source per entity. This enables spatial K/V injection where each
    entity region gets K/V from a separate, identity-pure ancestor.

    Key property: multi-entity nodes like S8={A,C} get:
      - ref(A) = S1({A},D)  — solo A, best identity source
      - ref(C) = S5({C},F)  — solo C, best identity source
    Each entity's K/V is injected into its spatial region, preventing
    identity chimera from mixed K/V.

    Scoring for reference selection:
      score = 10*is_solo - 2*crowd_size + 1*same_bg + 0.01*recency
    Prefers: solo > fewer-entity > same-bg > more-recent
    """

    def __init__(self) -> None:
        super().__init__()
        # shot_id -> {entity_name -> ShotNode}
        self.entity_refs: dict[str, dict[str, ShotNode]] = {}

    def build_from_shots(self, shots: list[ShotNode]) -> None:
        super().build_from_shots(shots)
        self._resolve_entity_refs()

    def _resolve_entity_refs(self) -> None:
        """For each multi-entity node, find the best identity source per entity."""
        for node in self.all_nodes:
            if len(node.entities) < 2:
                continue

            refs: dict[str, ShotNode] = {}
            for entity in sorted(node.entities):
                best_ref = self._find_best_ref(node, entity)
                if best_ref is not None:
                    refs[entity] = best_ref
            self.entity_refs[node.shot_id] = refs

    def _find_best_ref(self, target: ShotNode, entity: str) -> ShotNode | None:
        """Find the best identity source for a single entity.

        Searches all ancestors (nodes earlier in topological order) that
        contain the entity. Scores by:
          10 * is_solo — solo entity node is ideal (no identity mixing)
          -2 * len(entities) — fewer entities = less mixing
          +1 * same_bg — same background is a bonus
          +0.01 * index — more recent = better temporal coherence
        """
        best_score = -float("inf")
        best_node: ShotNode | None = None

        for idx, candidate in enumerate(self.all_nodes):
            # Must contain the target entity
            if entity not in candidate.entities:
                continue
            # Must not be the target itself
            if candidate is target:
                continue
            # Must appear before target in all_nodes (already generated)
            target_idx = self.all_nodes.index(target)
            if idx >= target_idx:
                continue

            is_solo = 1 if len(candidate.entities) == 1 else 0
            crowd = len(candidate.entities)
            same_bg = 1 if candidate.bg == target.bg else 0

            score = 10 * is_solo - 2 * crowd + 1 * same_bg + 0.01 * idx
            if score > best_score:
                best_score = score
                best_node = candidate

        return best_node

    def print_entity_refs(self) -> None:
        """Print per-entity reference resolution for multi-entity nodes."""
        print("\n" + "=" * 70)
        print("ENTITY-DECOMPOSED REFERENCES")
        print("=" * 70)
        for node in self.all_nodes:
            if node.shot_id not in self.entity_refs:
                continue
            refs = self.entity_refs[node.shot_id]
            kind = "Bridge" if node.is_bridge else "Shot"
            print(f"  [{kind}] {node.shot_id} (E={sorted(node.entities)}, BG={node.bg}):")
            for ent in sorted(refs):
                ref = refs[ent]
                ref_kind = "Bridge" if ref.is_bridge else "Shot"
                print(f"    ref({ent}) = {ref_kind} {ref.shot_id} "
                      f"(E={sorted(ref.entities)}, BG={ref.bg})")
        print("=" * 70)
