"""
Ablation Study — 3 experiments to validate each core idea.

Exp 1: Markovian Baseline   — force linear parent (S_{t-1}) instead of routing
Exp 2: No Bridge Baseline   — skip bridge injection, direct jump for D≥2
Exp 3: Global Injection     — high blend + no spatial anchoring for +entity

Each experiment saves to outputs/ablation/exp{N}_{name}/
A combined comparison sheet is also generated.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.routing import ShotNode, RoutingGraph, distance
from src.core.generator import (
    KeyframeGenerator, AttentionControl, _stable_hash,
    ENTITY_PROMPTS, BG_PROMPTS,
)


def _fresh_scenario():
    """Return a fresh copy of the 8-shot scenario (dataclass instances can't be reused)."""
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


# ═══════════════════════════════════════════════════════════════════════
# Exp 1: Markovian Baseline — linear parent chain
# ═══════════════════════════════════════════════════════════════════════

class MarkovianGenerator(KeyframeGenerator):
    """Forces each shot to use the immediately preceding shot as parent."""

    def run(self, scenario, entity_prompts, bg_prompts, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("ABLATION EXP 1: Markovian Baseline (linear parent chain)")
        print("=" * 70)

        # Build linear chain: S1→S2→S3→S4→S5→S6→S7→S8 (no routing, no bridges)
        for i, shot in enumerate(scenario):
            if i > 0:
                shot.parent_node = scenario[i - 1]
                scenario[i - 1].children.append(shot)

        gen_order = scenario  # linear order, no bridges

        print("\nLinear chain (Markovian):")
        for s in gen_order:
            pid = s.parent_node.shot_id if s.parent_node else "ROOT"
            d = distance(s, s.parent_node) if s.parent_node else -1
            print(f"  {s.shot_id} (parent={pid}, D={d})")

        self.load_pipeline()

        print("\n[Anchors]")
        self.build_anchor_cache(entity_prompts, bg_prompts, out_dir)

        print("\n" + "=" * 70)
        print("KEYFRAME GENERATION (Markovian — linear order)")
        print("=" * 70)

        keyframes: dict[str, Image.Image] = {}

        for node in gen_order:
            prompt = self.build_prompt(node, entity_prompts, bg_prompts)
            d = distance(node, node.parent_node) if node.parent_node else -1

            ip_embeds = self._compose_ip_embeds(node)
            gen = torch.Generator(device=self.device).manual_seed(
                self.seed + _stable_hash(node.shot_id)
            )

            parent_id = node.parent_node.shot_id if node.parent_node else "ROOT"
            print(f"\n  {'─'*60}")
            print(f"  [Shot] {node.shot_id}  parent={parent_id}  D={d}")
            print(f"  Prompt: {prompt[:75]}...")

            if node.parent_node is None:
                # Root — pure T2I with store
                self.attn_ctrl.set_mode_store()
                def root_cb(pipe, step, timestep, cbk):
                    self.attn_ctrl.update_step(step)
                    return cbk
                img = self.pipe(
                    prompt=prompt,
                    negative_prompt="blurry, low quality, distorted, deformed",
                    ip_adapter_image_embeds=ip_embeds,
                    num_inference_steps=self.num_steps,
                    guidance_scale=self.guidance_scale,
                    generator=gen,
                    width=self.width, height=self.height,
                    callback_on_step_end=root_cb,
                ).images[0]
                self.attn_ctrl.save_kv_to_cache(node.shot_id)
                self.attn_ctrl.set_mode_bypass()

            else:
                # Markovian: always inject from previous shot, uniform blend
                # Use max_blend regardless of D (no special handling)
                blend = self.max_blend * 0.7
                print(f"  Mode: MARKOVIAN inject from {parent_id} (blend={blend:.0%})")
                self._generate_with_parent_kv(
                    node, prompt, ip_embeds, keyframes, gen,
                    blend_override=blend,
                )
                img = self._last_generated

            keyframes[node.shot_id] = img
            path = out_dir / f"shot_{node.shot_id}.png"
            img.save(path)
            print(f"  Saved -> {path}")

        real_shots = [n for n in gen_order if not n.is_bridge]
        self._make_contact_sheet(real_shots, keyframes, out_dir)
        print(f"\nExp 1 done -> {out_dir.resolve()}")
        return keyframes


# ═══════════════════════════════════════════════════════════════════════
# Exp 2: No Bridge — direct jump for D≥2
# ═══════════════════════════════════════════════════════════════════════

class NoBridgeGenerator(KeyframeGenerator):
    """Uses routing to find best parent, but skips bridge injection for D≥2."""

    def run(self, scenario, entity_prompts, bg_prompts, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("ABLATION EXP 2: No Bridge (direct jump for D≥2)")
        print("=" * 70)

        # Run routing but WITHOUT bridge injection
        # Find best parent for each shot, but never insert bridges
        all_nodes = []
        for i, shot in enumerate(scenario):
            if i == 0:
                all_nodes.append(shot)
                continue

            # Find best parent (same logic as RoutingGraph.add_shot)
            candidates = []
            for idx, past in enumerate(all_nodes):
                d = distance(shot, past)
                d_ent = len(shot.entities.symmetric_difference(past.entities))
                candidates.append((d, d_ent, -idx, past))
            candidates.sort(key=lambda c: (c[0], c[1], c[2]))
            _, _, _, best_ref = candidates[0]

            shot.parent_node = best_ref
            best_ref.children.append(shot)
            all_nodes.append(shot)

        gen_order = scenario  # no bridges, just real shots in order

        print("\nRouting (no bridges):")
        for s in gen_order:
            pid = s.parent_node.shot_id if s.parent_node else "ROOT"
            d = distance(s, s.parent_node) if s.parent_node else -1
            print(f"  {s.shot_id} (parent={pid}, D={d})")

        self.load_pipeline()
        self.build_anchor_cache(entity_prompts, bg_prompts, out_dir)

        print("\n" + "=" * 70)
        print("KEYFRAME GENERATION (No Bridge)")
        print("=" * 70)

        keyframes: dict[str, Image.Image] = {}

        for node in gen_order:
            prompt = self.build_prompt(node, entity_prompts, bg_prompts)
            d = distance(node, node.parent_node) if node.parent_node else -1

            ip_embeds = self._compose_ip_embeds(node)
            gen = torch.Generator(device=self.device).manual_seed(
                self.seed + _stable_hash(node.shot_id)
            )

            parent_id = node.parent_node.shot_id if node.parent_node else "ROOT"
            print(f"\n  {'─'*60}")
            print(f"  [Shot] {node.shot_id}  parent={parent_id}  D={d}")
            print(f"  Prompt: {prompt[:75]}...")

            if node.parent_node is None:
                self.attn_ctrl.set_mode_store()
                def root_cb(pipe, step, timestep, cbk):
                    self.attn_ctrl.update_step(step)
                    return cbk
                img = self.pipe(
                    prompt=prompt,
                    negative_prompt="blurry, low quality, distorted, deformed",
                    ip_adapter_image_embeds=ip_embeds,
                    num_inference_steps=self.num_steps,
                    guidance_scale=self.guidance_scale,
                    generator=gen,
                    width=self.width, height=self.height,
                    callback_on_step_end=root_cb,
                ).images[0]
                self.attn_ctrl.save_kv_to_cache(node.shot_id)
                self.attn_ctrl.set_mode_bypass()

            elif d <= 1:
                # Normal generation (same as ours)
                img = self.generate_node(node, prompt, keyframes)

            else:
                # D≥2: direct jump — force inject with moderate blend
                # This is the ablation: no bridges to smooth the transition
                blend = self.max_blend * 0.5
                print(f"  Mode: DIRECT JUMP D={d} (no bridge, blend={blend:.0%})")
                self._generate_with_parent_kv(
                    node, prompt, ip_embeds, keyframes, gen,
                    blend_override=blend,
                )
                img = self._last_generated

            keyframes[node.shot_id] = img
            path = out_dir / f"shot_{node.shot_id}.png"
            img.save(path)
            print(f"  Saved -> {path}")

        real_shots = [n for n in gen_order if not n.is_bridge]
        self._make_contact_sheet(real_shots, keyframes, out_dir)
        print(f"\nExp 2 done -> {out_dir.resolve()}")
        return keyframes


# ═══════════════════════════════════════════════════════════════════════
# Exp 3: Global Injection — no gradient, no spatial anchoring for +entity
# ═══════════════════════════════════════════════════════════════════════

class GlobalInjectGenerator(KeyframeGenerator):
    """For +entity transitions: high uniform blend, no spatial anchoring."""

    @staticmethod
    def build_prompt(node, entity_prompts, bg_prompts):
        """No spatial anchoring — just list entities without left/right."""
        parts = []
        for ent in sorted(node.entities):
            desc = entity_prompts[ent].split(",")[0]
            parts.append(desc)
        bg_desc = bg_prompts[node.bg].split(",")[0]
        return (f"{', '.join(parts)}, in {bg_desc}, {node.action}, "
                f"cinematic still frame, high quality, detailed")

    def generate_node(self, node, prompt, keyframes):
        d = distance(node, node.parent_node) if node.parent_node else -1
        parent_id = node.parent_node.shot_id if node.parent_node else "ROOT"
        kind = "Bridge" if node.is_bridge else "Shot"

        print(f"\n  {'─'*60}")
        print(f"  [{kind}] {node.shot_id}  parent={parent_id}  D={d}")
        print(f"  Entities={sorted(node.entities)}  BG={node.bg}")
        print(f"  Prompt: {prompt[:75]}...")

        ip_embeds = self._compose_ip_embeds(node)
        gen = torch.Generator(device=self.device).manual_seed(
            self.seed + _stable_hash(node.shot_id)
        )

        if node.parent_node is None:
            print(f"  Mode: T2I (root node, store K/V)")
            self.attn_ctrl.set_mode_store()
            step_log = []
            def root_callback(pipe, step, timestep, cbk):
                self.attn_ctrl.update_step(step)
                step_log.append(step)
                return cbk
            img = self.pipe(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted, deformed",
                ip_adapter_image_embeds=ip_embeds,
                num_inference_steps=self.num_steps,
                guidance_scale=self.guidance_scale,
                generator=gen,
                width=512, height=512,
                callback_on_step_end=root_callback,
            ).images[0]
            self.attn_ctrl.save_kv_to_cache(node.shot_id)
            self.attn_ctrl.set_mode_bypass()

        elif d == 0:
            self._generate_with_parent_kv(
                node, prompt, ip_embeds, keyframes, gen,
                blend_override=self.max_blend,
            )
            img = self._last_generated

        elif d == 1:
            parent = node.parent_node
            added = node.entities - parent.entities
            removed = parent.entities - node.entities

            if added:
                # ABLATION: high uniform blend, NO gradient mask
                blend = self.max_blend * 0.6
                print(f"  Mode: GLOBAL INJECT (+entity({added}), blend={blend:.0%}, NO gradient)")
                # Explicitly no gradient mask
                self.attn_ctrl.set_spatial_mask("none")
                self._generate_with_parent_kv(
                    node, prompt, ip_embeds, keyframes, gen,
                    blend_override=blend,
                )
                img = self._last_generated

            elif removed:
                blend = self.max_blend * 0.5
                self._generate_with_parent_kv(
                    node, prompt, ip_embeds, keyframes, gen,
                    blend_override=blend,
                )
                img = self._last_generated
            else:
                if len(node.entities) >= 2:
                    blend = self.max_blend * 0.4
                else:
                    blend = self.max_blend * 0.7
                self._generate_with_parent_kv(
                    node, prompt, ip_embeds, keyframes, gen,
                    blend_override=blend,
                )
                img = self._last_generated
        else:
            raise RuntimeError(f"D={d} — bridge injection failed!")

        return img


# ═══════════════════════════════════════════════════════════════════════
# Comparison Sheet Builder
# ═══════════════════════════════════════════════════════════════════════

def _load_font(size: int):
    """Try to load a TrueType font at the given size, fallback to default."""
    from PIL import ImageFont
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def make_comparison_sheet(
    results: dict[str, dict[str, Image.Image]],
    shot_ids: list[str],
    out_path: Path,
    title: str = "Ablation Comparison",
):
    """Create a grid: rows=experiments, cols=shots."""
    exp_names = list(results.keys())
    n_exps = len(exp_names)
    n_shots = len(shot_ids)

    w, h = 512, 512
    pad = 6
    label_h = 40
    header_w = 260

    font_label = _load_font(28)
    font_header = _load_font(32)

    sheet_w = header_w + n_shots * (w + pad) + pad
    sheet_h = pad + label_h + n_exps * (h + pad + label_h)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)

    # Column headers (shot IDs)
    for j, sid in enumerate(shot_ids):
        x = header_w + pad + j * (w + pad)
        draw.text((x + w // 2 - 24, pad + 4), sid, fill=(255, 255, 100), font=font_header)

    # Rows
    for i, exp_name in enumerate(exp_names):
        y = pad + label_h + i * (h + pad + label_h)
        # Row label
        draw.text((pad + 8, y + h // 2 - 14), exp_name, fill=(200, 200, 200), font=font_label)

        for j, sid in enumerate(shot_ids):
            x = header_w + pad + j * (w + pad)
            img = results[exp_name].get(sid)
            if img is not None:
                sheet.paste(img.resize((w, h)), (x, y))
            else:
                draw.rectangle([x, y, x + w, y + h], fill=(60, 60, 60))
                draw.text((x + 10, y + h // 2), "N/A", fill=(150, 150, 150), font=font_label)

    sheet.save(out_path)
    print(f"\nComparison sheet -> {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main — Run all 3 experiments
# ═══════════════════════════════════════════════════════════════════════

def main():
    base_dir = Path("outputs/ablation")
    base_dir.mkdir(parents=True, exist_ok=True)
    shot_ids = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]

    all_results: dict[str, dict[str, Image.Image]] = {}

    # ── Ours (control) ──────────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  RUNNING: Ours (Full Pipeline)")
    print("█" * 70 + "\n")
    ours_dir = base_dir / "ours"
    gen_ours = KeyframeGenerator(device="cuda:3", num_steps=25, max_blend=0.7, inject_pct=0.6, guidance_scale=7.5)
    gen_ours.run(
        scenario=_fresh_scenario(),
        entity_prompts=ENTITY_PROMPTS,
        bg_prompts=BG_PROMPTS,
        out_dir=ours_dir,
    )
    ours_kf = {}
    for sid in shot_ids:
        p = ours_dir / f"shot_{sid}.png"
        if p.exists():
            ours_kf[sid] = Image.open(p)
    all_results["Ours"] = ours_kf

    # Free GPU memory
    del gen_ours
    torch.cuda.empty_cache()

    # ── Exp 1: Markovian ────────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  RUNNING: Exp 1 — Markovian Baseline")
    print("█" * 70 + "\n")
    exp1_dir = base_dir / "exp1_markovian"
    gen1 = MarkovianGenerator(device="cuda:3", num_steps=25, max_blend=0.7, inject_pct=0.6, guidance_scale=7.5)
    gen1.run(
        scenario=_fresh_scenario(),
        entity_prompts=ENTITY_PROMPTS,
        bg_prompts=BG_PROMPTS,
        out_dir=exp1_dir,
    )
    exp1_kf = {}
    for sid in shot_ids:
        p = exp1_dir / f"shot_{sid}.png"
        if p.exists():
            exp1_kf[sid] = Image.open(p)
    all_results["Markovian"] = exp1_kf

    del gen1
    torch.cuda.empty_cache()

    # ── Exp 2: No Bridge ───────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  RUNNING: Exp 2 — No Bridge Baseline")
    print("█" * 70 + "\n")
    exp2_dir = base_dir / "exp2_no_bridge"
    gen2 = NoBridgeGenerator(device="cuda:3", num_steps=25, max_blend=0.7, inject_pct=0.6, guidance_scale=7.5)
    gen2.run(
        scenario=_fresh_scenario(),
        entity_prompts=ENTITY_PROMPTS,
        bg_prompts=BG_PROMPTS,
        out_dir=exp2_dir,
    )
    exp2_kf = {}
    for sid in shot_ids:
        p = exp2_dir / f"shot_{sid}.png"
        if p.exists():
            exp2_kf[sid] = Image.open(p)
    all_results["No Bridge"] = exp2_kf

    del gen2
    torch.cuda.empty_cache()

    # ── Exp 3: Global Injection ────────────────────────────────────
    print("\n" + "█" * 70)
    print("  RUNNING: Exp 3 — Global Injection (no gradient, no spatial)")
    print("█" * 70 + "\n")
    exp3_dir = base_dir / "exp3_global_inject"
    gen3 = GlobalInjectGenerator(device="cuda:3", num_steps=25, max_blend=0.7, inject_pct=0.6, guidance_scale=7.5)
    gen3.run(
        scenario=_fresh_scenario(),
        entity_prompts=ENTITY_PROMPTS,
        bg_prompts=BG_PROMPTS,
        out_dir=exp3_dir,
    )
    exp3_kf = {}
    for sid in shot_ids:
        p = exp3_dir / f"shot_{sid}.png"
        if p.exists():
            exp3_kf[sid] = Image.open(p)
    all_results["Global Inject"] = exp3_kf

    del gen3
    torch.cuda.empty_cache()

    # ── Comparison Sheets ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON SHEETS")
    print("=" * 70)

    # Full comparison (all 8 shots)
    make_comparison_sheet(
        all_results, shot_ids,
        base_dir / "comparison_full.png",
        title="Full Ablation Comparison",
    )

    # Focused comparisons per experiment
    # Exp 1 focus: S6, S7 (routing matters most here)
    make_comparison_sheet(
        {"Ours": ours_kf, "Markovian": exp1_kf},
        ["S5", "S6", "S7", "S8"],
        base_dir / "comparison_exp1_routing.png",
    )

    # Exp 2 focus: S4, S5 (bridge path: S3→B1→S4→B2→B3→S5)
    make_comparison_sheet(
        {"Ours": ours_kf, "No Bridge": exp2_kf},
        ["S3", "S4", "S5", "S8"],
        base_dir / "comparison_exp2_bridge.png",
    )

    # Exp 3 focus: S2, S8 (+entity shots)
    make_comparison_sheet(
        {"Ours": ours_kf, "Global Inject": exp3_kf},
        ["S1", "S2", "S5", "S8"],
        base_dir / "comparison_exp3_chimera.png",
    )

    print("\n" + "=" * 70)
    print("ALL ABLATION EXPERIMENTS COMPLETE")
    print(f"Results: {base_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
