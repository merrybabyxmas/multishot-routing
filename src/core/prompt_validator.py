"""
Prompt Quality Verification Pipeline.

Validates all prompts against the design specification before generation.
Ensures NeurIPS-quality reproducibility and consistency.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.routing import ShotNode, RoutingGraph, distance
from src.core.cyberpunk_scenario import (
    ENTITY_PROMPTS, BG_PROMPTS, MOTION_PROMPTS, SHOT_IDS,
    build_scenario,
)
from src.core.generator import KeyframeGenerator


# ═══════════════════════════════════════════════════════════════════════
# Validation Checks
# ═══════════════════════════════════════════════════════════════════════

class PromptValidator:
    """Validates all prompts and routing against the design spec."""

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.passed: list[str] = []

    def _pass(self, msg: str):
        self.passed.append(msg)

    def _error(self, msg: str):
        self.errors.append(msg)

    def _warn(self, msg: str):
        self.warnings.append(msg)

    # ── 1. Prompt Length Checks ───────────────────────────────────

    def check_entity_prompt_lengths(self, min_len: int = 80):
        for sym, prompt in ENTITY_PROMPTS.items():
            if len(prompt) >= min_len:
                self._pass(f"Entity {sym}: len={len(prompt)} >= {min_len}")
            else:
                self._error(f"Entity {sym}: len={len(prompt)} < {min_len}")

    def check_bg_prompt_lengths(self, min_len: int = 60):
        for sym, prompt in BG_PROMPTS.items():
            if len(prompt) >= min_len:
                self._pass(f"BG {sym}: len={len(prompt)} >= {min_len}")
            else:
                self._error(f"BG {sym}: len={len(prompt)} < {min_len}")

    def check_motion_prompt_lengths(self, min_len: int = 150):
        for sid, prompt in MOTION_PROMPTS.items():
            if len(prompt) >= min_len:
                self._pass(f"Motion {sid}: len={len(prompt)} >= {min_len}")
            else:
                self._error(f"Motion {sid}: len={len(prompt)} < {min_len}")

    # ── 2. Consistency Checks ─────────────────────────────────────

    def check_entity_consistency(self):
        """Verify core attributes appear in entity prompts."""
        core_attrs = {
            "A": ["visor", "trenchcoat"],
            "B": ["metallic", "blue", "dog"],
            "C": ["security", "mech", "red"],
        }
        for sym, attrs in core_attrs.items():
            prompt = ENTITY_PROMPTS[sym].lower()
            for attr in attrs:
                if attr in prompt:
                    self._pass(f"Entity {sym}: contains '{attr}'")
                else:
                    self._error(f"Entity {sym}: MISSING core attribute '{attr}'")

    def check_motion_prompt_consistency(self):
        """Verify motion prompts reference correct entities."""
        scenario = build_scenario()
        entity_keywords = {
            "A": ["hacker"],
            "B": ["android dog", "dog"],
            "C": ["security mech", "mech"],
        }

        for node in scenario:
            prompt = MOTION_PROMPTS.get(node.shot_id, "").lower()
            if not prompt:
                self._error(f"Motion {node.shot_id}: MISSING motion prompt")
                continue

            # Check that all entities in the shot are mentioned
            for ent in node.entities:
                keywords = entity_keywords[ent]
                if any(kw in prompt for kw in keywords):
                    self._pass(f"Motion {node.shot_id}: mentions entity {ent}")
                else:
                    self._error(f"Motion {node.shot_id}: does NOT mention entity {ent} (keywords: {keywords})")

            # Check no extra entities are mentioned
            absent_ents = set(entity_keywords.keys()) - node.entities
            for ent in absent_ents:
                keywords = entity_keywords[ent]
                if any(kw in prompt for kw in keywords):
                    self._warn(f"Motion {node.shot_id}: mentions ABSENT entity {ent}")

    def check_motion_has_action_verb(self):
        """Verify each motion prompt has a clear action verb."""
        action_verbs = [
            "walk", "run", "stand", "sit", "crouch", "crawl", "look",
            "scan", "tap", "operate", "emerge", "face", "lean", "step",
            "trot", "sprint", "burst", "exit", "enter", "infiltrate",
            "escape", "splash", "power", "greet", "join", "stop",
        ]
        for sid, prompt in MOTION_PROMPTS.items():
            prompt_lower = prompt.lower()
            found = [v for v in action_verbs if v in prompt_lower]
            if found:
                self._pass(f"Motion {sid}: action verbs found: {found[:3]}")
            else:
                self._error(f"Motion {sid}: NO action verb detected")

    # ── 3. Spatial Anchoring Checks ───────────────────────────────

    def check_spatial_anchoring(self):
        """Verify 2-entity shots get left/right spatial anchoring in build_prompt."""
        scenario = build_scenario()
        for node in scenario:
            if len(node.entities) == 2:
                prompt = KeyframeGenerator.build_prompt(node, ENTITY_PROMPTS, BG_PROMPTS)
                if "standing on the left" in prompt and "standing on the right" in prompt:
                    self._pass(f"{node.shot_id}: spatial anchoring present (left/right)")
                else:
                    self._error(f"{node.shot_id}: 2-entity shot MISSING spatial anchoring")
            else:
                self._pass(f"{node.shot_id}: single entity, no spatial anchoring needed")

    # ── 4. Routing Validation ─────────────────────────────────────

    def check_routing_distances(self):
        """Verify routing produces expected distances."""
        # Note: routing finds OPTIMAL parents, often better than sequential analysis
        # S4: parent=S1 (D=1), not S3 (D=2) — routing skips to better ancestor
        # S6: parent=S3 (D=0) — exact match found
        # S7: parent=B2 (D=1) — bridge chain handles the gap
        # S9: parent=S2 (D=0) — ultra-long routing retrieves S2
        # S10: parent=S1 (D=0) — identity lock back to origin
        expected_d = {
            "S1": -1, "S2": 1, "S3": 1, "S4": 1, "S5": 0,
            "S6": 0, "S7": 1, "S8": 1, "S9": 0, "S10": 0,
        }

        scenario = build_scenario()
        graph = RoutingGraph()
        graph.build_from_shots(scenario)

        for node in graph.all_nodes:
            if node.is_bridge:
                continue
            d = distance(node, node.parent_node) if node.parent_node else -1
            exp = expected_d.get(node.shot_id)
            if exp is not None:
                if d == exp:
                    self._pass(f"Routing {node.shot_id}: D={d} (expected {exp})")
                else:
                    parent_id = node.parent_node.shot_id if node.parent_node else "ROOT"
                    self._warn(f"Routing {node.shot_id}: D={d} != expected {exp} (parent={parent_id})")

    def check_bridge_injection(self):
        """Verify bridges are created for D>=2 transitions."""
        scenario = build_scenario()
        graph = RoutingGraph()
        graph.build_from_shots(scenario)

        bridges = [n for n in graph.all_nodes if n.is_bridge]
        if bridges:
            self._pass(f"Bridge injection: {len(bridges)} bridges created")
            for b in bridges:
                parent_id = b.parent_node.shot_id if b.parent_node else "ROOT"
                d = distance(b, b.parent_node) if b.parent_node else -1
                if d <= 1:
                    self._pass(f"  Bridge {b.shot_id}: D={d} from {parent_id}")
                else:
                    self._error(f"  Bridge {b.shot_id}: D={d} from {parent_id} (should be <=1!)")
        else:
            self._error("Bridge injection: NO bridges created (expected for D>=2 shots)")

    def check_all_edges_d_leq_1(self):
        """Verify that after bridge injection, every edge has D<=1."""
        scenario = build_scenario()
        graph = RoutingGraph()
        graph.build_from_shots(scenario)

        for node in graph.all_nodes:
            if node.parent_node is None:
                continue
            d = distance(node, node.parent_node)
            if d <= 1:
                self._pass(f"Edge {node.parent_node.shot_id}->{node.shot_id}: D={d}")
            else:
                self._error(f"Edge {node.parent_node.shot_id}->{node.shot_id}: D={d} > 1!")

    # ── 5. Completeness Checks ────────────────────────────────────

    def check_completeness(self):
        """Verify all shot IDs have prompts."""
        for sid in SHOT_IDS:
            if sid not in MOTION_PROMPTS:
                self._error(f"Completeness: {sid} missing motion prompt")
            else:
                self._pass(f"Completeness: {sid} has motion prompt")

    # ── Run All ───────────────────────────────────────────────────

    def run_all(self) -> bool:
        """Run all validation checks. Returns True if no errors."""
        print("=" * 70)
        print("PROMPT VALIDATION PIPELINE")
        print("=" * 70)

        self.check_entity_prompt_lengths()
        self.check_bg_prompt_lengths()
        self.check_motion_prompt_lengths()
        self.check_entity_consistency()
        self.check_motion_prompt_consistency()
        self.check_motion_has_action_verb()
        self.check_spatial_anchoring()
        self.check_routing_distances()
        self.check_bridge_injection()
        self.check_all_edges_d_leq_1()
        self.check_completeness()

        print(f"\n  PASSED:   {len(self.passed)}")
        print(f"  WARNINGS: {len(self.warnings)}")
        print(f"  ERRORS:   {len(self.errors)}")

        if self.warnings:
            print("\n  WARNINGS:")
            for w in self.warnings:
                print(f"    [!] {w}")

        if self.errors:
            print("\n  ERRORS:")
            for e in self.errors:
                print(f"    [X] {e}")
            print(f"\n  VALIDATION FAILED ({len(self.errors)} errors)")
            return False

        print("\n  ALL CHECKS PASSED")
        print("=" * 70)
        return True


def main():
    validator = PromptValidator()
    success = validator.run_all()

    if success:
        # Print the routing table for reference
        from src.core.routing import RoutingGraph
        from src.core.cyberpunk_scenario import build_scenario
        scenario = build_scenario()
        graph = RoutingGraph()
        graph.build_from_shots(scenario)
        graph.print_routing_table()

        gen_order = graph.topological_order()
        print(f"\nGeneration order: {' -> '.join(n.shot_id for n in gen_order)}")

    return success


if __name__ == "__main__":
    main()
