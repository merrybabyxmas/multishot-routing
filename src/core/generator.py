"""
Phase 3: Conditioned Keyframe Generation via Non-Markovian Graph Routing.

Architecture:
  1. IP-Adapter (SDXL) — injects entity/bg identity via cross-attention
  2. AttentionStore   — hooks SDXL UNet attn1 (self-attention) layers
                        to capture & replay parent K/V tensors
  3. Graph Executor   — walks the routing DAG in topological order,
                        coordinating store/inject phases per node
"""

from __future__ import annotations

import os
import sys
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import hashlib
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def _stable_hash(s: str) -> int:
    """Deterministic hash that is consistent across Python sessions."""
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % 10000

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.core.routing import ShotNode, RoutingGraph, distance


# ═══════════════════════════════════════════════════════════════════════
# 1. AttentionStore — Custom Self-Attention Processor
# ═══════════════════════════════════════════════════════════════════════

class KVStoreAttnProcessor:
    """
    Drop-in replacement for AttnProcessor2_0 on attn1 (self-attention).

    Modes:
      - STORE:   run normal self-attention, cache K/V per timestep
      - INJECT:  blend cached parent K/V into current K/V
      - BYPASS:  normal attention, no caching (default)
    """

    def __init__(self):
        self.mode: str = "bypass"
        self.kv_bank: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.current_step: int = 0
        self.blend_ratio: float = 0.0
        self.spatial_mask_type: str | None = None

    def reset(self):
        self.kv_bank.clear()
        self.mode = "bypass"
        self.current_step = 0
        self.blend_ratio = 0.0

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # ── K/V Hook ──────────────────────────────────────────────
        if self.mode == "store":
            self.kv_bank[self.current_step] = (
                key.detach().clone(),
                value.detach().clone(),
            )

        elif self.mode in ("inject", "store_and_inject"):
            if self.blend_ratio > 0.0:
                stored = self.kv_bank.get(self.current_step)
                if stored is not None:
                    stored_k, stored_v = stored
                    if stored_k.shape[0] != key.shape[0] and key.shape[0] == 2 * stored_k.shape[0]:
                        stored_k = stored_k.repeat(2, 1, 1)
                        stored_v = stored_v.repeat(2, 1, 1)
                    if stored_k.shape == key.shape:
                        r = self.blend_ratio
                        seq_len = key.shape[1]
                        size = int(math.sqrt(seq_len))

                        if self.spatial_mask_type is not None and size * size == seq_len:
                            mask = torch.ones((1, seq_len, 1), device=key.device, dtype=key.dtype) * r
                            mask_2d = mask.view(1, size, size, 1)
                            if self.spatial_mask_type == "right":
                                mask_2d[:, :, :size // 2, :] = 0.0
                            elif self.spatial_mask_type == "left":
                                mask_2d[:, :, size // 2:, :] = 0.0
                            mask = mask_2d.view(1, seq_len, 1)
                            key = (1 - mask) * key + mask * stored_k
                            value = (1 - mask) * value + mask * stored_v
                        else:
                            key = (1 - r) * key + r * stored_k
                            value = (1 - r) * value + r * stored_v

            if self.mode == "store_and_inject":
                self.kv_bank[self.current_step] = (
                    key.detach().clone(),
                    value.detach().clone(),
                )
        # ──────────────────────────────────────────────────────────

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ═══════════════════════════════════════════════════════════════════════
# 2. AttentionControl — Manages all KV processors across the UNet
# ═══════════════════════════════════════════════════════════════════════

class AttentionControl:
    """Orchestrates store/inject across all self-attention layers."""

    def __init__(self, unet, num_steps: int, max_blend: float = 0.7,
                 inject_until_pct: float = 0.6):
        self.unet = unet
        self.num_steps = num_steps
        self.max_blend = max_blend
        self.inject_until_pct = inject_until_pct

        self.kv_processors: dict[str, KVStoreAttnProcessor] = {}
        existing = unet.attn_processors
        new_procs = {}

        for key, proc in existing.items():
            if ".attn1." in key:
                kv_proc = KVStoreAttnProcessor()
                new_procs[key] = kv_proc
                self.kv_processors[key] = kv_proc
            else:
                new_procs[key] = proc

        unet.set_attn_processor(new_procs)
        print(f"  [AttentionControl] Installed {len(self.kv_processors)} "
              f"KV-Store processors on attn1 layers")

        # Cache: shot_id -> {proc_key -> {step -> (K, V)}}
        self._kv_cache: dict[str, dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]]] = {}

    def set_spatial_mask(self, mask_type: str | None):
        """Set spatial mask for K/V injection ('left', 'right', or None)."""
        for p in self.kv_processors.values():
            p.spatial_mask_type = mask_type

    def set_mode_store_and_inject(self):
        """Store K/V while also injecting from previously loaded bank."""
        for p in self.kv_processors.values():
            p.mode = "store_and_inject"

    def set_mode_store(self):
        for p in self.kv_processors.values():
            p.mode = "store"
            p.kv_bank.clear()

    def set_mode_inject(self):
        for p in self.kv_processors.values():
            p.mode = "inject"

    def set_mode_bypass(self):
        for p in self.kv_processors.values():
            p.mode = "bypass"

    def save_kv_to_cache(self, shot_id: str):
        """Snapshot current K/V banks from all processors into cache."""
        self._kv_cache[shot_id] = {}
        for key, proc in self.kv_processors.items():
            self._kv_cache[shot_id][key] = {
                step: (k.clone(), v.clone())
                for step, (k, v) in proc.kv_bank.items()
            }

    def load_kv_from_cache(self, shot_id: str) -> bool:
        """Load cached K/V banks into processors. Returns False if not found."""
        if shot_id not in self._kv_cache:
            return False
        for key, proc in self.kv_processors.items():
            proc.kv_bank = {
                step: (k.clone(), v.clone())
                for step, (k, v) in self._kv_cache[shot_id][key].items()
            }
        return True

    def update_step(self, step: int):
        inject_steps = int(self.num_steps * self.inject_until_pct)
        if step < inject_steps:
            ratio = self.max_blend * (1.0 - step / inject_steps)
        else:
            ratio = 0.0

        for p in self.kv_processors.values():
            p.current_step = step
            p.blend_ratio = ratio


# ═══════════════════════════════════════════════════════════════════════
# 3. Graph-Driven Keyframe Generator
# ═══════════════════════════════════════════════════════════════════════

class KeyframeGenerator:

    def __init__(self, device: str = "cuda:3", num_steps: int = 8,
                 max_blend: float = 0.7, inject_pct: float = 0.6):
        self.device = device
        self.dtype = torch.float16
        self.num_steps = num_steps
        self.max_blend = max_blend
        self.inject_pct = inject_pct
        self.seed = 42

        self.pipe = None
        self.attn_ctrl: AttentionControl | None = None
        self.image_encoder = None
        self.clip_processor = None
        self.embedding_cache: dict[str, torch.Tensor] = {}

    def load_pipeline(self):
        from diffusers import AutoPipelineForText2Image

        print("[Pipeline] Loading SDXL-Turbo...")
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=self.dtype,
            variant="fp16",
        ).to(self.device)

        print("[Pipeline] Loading IP-Adapter (h94/IP-Adapter, ViT-bigG)...")
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.safetensors",
        )
        self.pipe.set_ip_adapter_scale(0.6)
        self.pipe.set_progress_bar_config(disable=True)

        self.image_encoder = self.pipe.image_encoder
        self.clip_processor = self.pipe.feature_extractor

        print("[Pipeline] Installing AttentionControl on UNet attn1...")
        self.attn_ctrl = AttentionControl(
            self.pipe.unet,
            num_steps=self.num_steps,
            max_blend=self.max_blend,
            inject_until_pct=self.inject_pct,
        )
        print("[Pipeline] Ready.\n")

    # ── Embedding Cache ────────────────────────────────────────────

    def build_anchor_cache(self, entity_prompts, bg_prompts, out_dir):
        print("[Anchors] Generating base images...")
        anchor_images: dict[str, Image.Image] = {}
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        self.attn_ctrl.set_mode_bypass()

        for symbol, prompt in {**entity_prompts, **bg_prompts}.items():
            img = self.pipe(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted",
                ip_adapter_image_embeds=[self._zero_ip_embeds()],
                num_inference_steps=4,
                guidance_scale=0.0,
                generator=gen,
                width=512, height=512,
            ).images[0]
            anchor_images[symbol] = img
            img.save(out_dir / f"anchor_{symbol}.png")
            print(f"  Anchor {symbol}: saved")

        print("[Anchors] Computing CLIP embeddings...")
        for symbol, img in anchor_images.items():
            emb = self._encode_image(img)
            self.embedding_cache[symbol] = emb
            print(f"  CLIP embed {symbol}: shape={emb.shape}")

        return anchor_images

    def _encode_image(self, img):
        inputs = self.clip_processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, self.dtype)
        with torch.no_grad():
            emb = self.image_encoder(pixel_values).image_embeds
        return emb.squeeze(0)

    def _zero_ip_embeds(self):
        return torch.zeros(1, 1, 1280, device=self.device, dtype=self.dtype)

    def _compose_ip_embeds(self, node):
        embeds = []
        for ent in sorted(node.entities):
            embeds.append(self.embedding_cache[ent].unsqueeze(0))  # (1, 1280)
        embeds.append(self.embedding_cache[node.bg].unsqueeze(0))  # (1, 1280)
        # Concatenate along sequence dim so each identity is preserved as a separate token
        combined = torch.cat(embeds, dim=0).unsqueeze(0)  # (1, N, 1280)
        return combined

    @staticmethod
    def build_prompt(node, entity_prompts, bg_prompts):
        parts = []
        for ent in sorted(node.entities):
            desc = entity_prompts[ent].split(",")[0]
            parts.append(desc)
        bg_desc = bg_prompts[node.bg].split(",")[0]
        return (f"{', '.join(parts)}, in {bg_desc}, {node.action}, "
                f"cinematic still frame, high quality, detailed")

    # ── Core Generation ────────────────────────────────────────────

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
                ip_adapter_image_embeds=[ip_embeds],
                num_inference_steps=self.num_steps,
                guidance_scale=0.0,
                generator=gen,
                width=512, height=512,
                callback_on_step_end=root_callback,
            ).images[0]
            self.attn_ctrl.save_kv_to_cache(node.shot_id)
            self.attn_ctrl.set_mode_bypass()
            print(f"  Cached K/V for {len(step_log)} steps")

        elif d == 0:
            print(f"  Mode: REUSE (D=0, blend={self.max_blend:.0%})")
            self._generate_with_parent_kv(
                node, prompt, ip_embeds, keyframes, gen,
                blend_override=self.max_blend,
            )
            img = self._last_generated

        elif d == 1:
            parent = node.parent_node
            added = node.entities - parent.entities
            removed = parent.entities - node.entities
            bg_changed = node.bg != parent.bg

            # Spatial masking: when adding entity to single-entity parent,
            # confine parent structure to one side, free the other for new entity
            spatial_mask = None
            if added and len(parent.entities) == 1 and len(node.entities) >= 2:
                spatial_mask = "right"
                print(f"  [Spatial Mask] '{spatial_mask}' — parent({parent.entities}) on right, new({added}) on left")

            if added:
                blend = self.max_blend * 0.35
                change_type = f"+entity({added})"
            elif removed:
                blend = self.max_blend * 0.5
                change_type = f"-entity({removed})"
            else:
                blend = self.max_blend * 0.6
                change_type = f"bg({parent.bg}→{node.bg})"

            print(f"  Mode: DERIVE (D=1, {change_type}, blend={blend:.0%}→0)")
            self.attn_ctrl.set_spatial_mask(spatial_mask)
            self._generate_with_parent_kv(
                node, prompt, ip_embeds, keyframes, gen,
                blend_override=blend,
            )
            self.attn_ctrl.set_spatial_mask(None)
            img = self._last_generated

        else:
            raise RuntimeError(f"D={d} — bridge injection failed!")

        return img

    def _generate_with_parent_kv(self, node, prompt, ip_embeds, keyframes,
                                  gen, blend_override=None):
        parent = node.parent_node

        # Load parent's cached K/V (from its actual generation)
        loaded = self.attn_ctrl.load_kv_from_cache(parent.shot_id)
        if loaded:
            print(f"  Loaded cached K/V from {parent.shot_id}")
        else:
            print(f"  WARNING: No cached K/V for {parent.shot_id}, using bypass")
            self.attn_ctrl.set_mode_bypass()

        if blend_override is not None:
            orig_blend = self.attn_ctrl.max_blend
            self.attn_ctrl.max_blend = blend_override

        # Generate with K/V injection + simultaneous store for this node's children
        print(f"  Generating {node.shot_id} with K/V inject + store...")
        self.attn_ctrl.set_mode_store_and_inject()

        step_log = []
        def callback(pipe, step, timestep, cbk):
            self.attn_ctrl.update_step(step)
            ratio = next(iter(self.attn_ctrl.kv_processors.values())).blend_ratio
            step_log.append((step, ratio))
            return cbk

        self._last_generated = self.pipe(
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted, deformed",
            ip_adapter_image_embeds=[ip_embeds],
            num_inference_steps=self.num_steps,
            guidance_scale=0.0,
            generator=gen,
            width=512, height=512,
            callback_on_step_end=callback,
        ).images[0]

        # Cache this node's K/V for its children
        self.attn_ctrl.save_kv_to_cache(node.shot_id)

        for step, ratio in step_log:
            bar = "█" * int(ratio * 20) + "░" * (20 - int(ratio * 20))
            print(f"    Step {step}: blend={ratio:.2f} [{bar}]")

        self.attn_ctrl.set_mode_bypass()

        if blend_override is not None:
            self.attn_ctrl.max_blend = orig_blend

    # ── Full Pipeline Execution ────────────────────────────────────

    def run(self, scenario, entity_prompts, bg_prompts, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("PHASE 3: Graph-Routed Keyframe Generation")
        print("=" * 70)

        graph = RoutingGraph()
        graph.build_from_shots(scenario)
        graph.print_routing_table()
        gen_order = graph.topological_order()
        print(f"\nGeneration order: {' -> '.join(n.shot_id for n in gen_order)}")

        self.load_pipeline()
        anchor_images = self.build_anchor_cache(entity_prompts, bg_prompts, out_dir)

        print("\n" + "=" * 70)
        print("KEYFRAME GENERATION (Topological Order)")
        print("=" * 70)

        keyframes: dict[str, Image.Image] = {}
        self._last_prompts: dict[str, str] = {}

        for node in gen_order:
            prompt = self.build_prompt(node, entity_prompts, bg_prompts)
            self._last_prompts[node.shot_id] = prompt

            img = self.generate_node(node, prompt, keyframes)
            keyframes[node.shot_id] = img

            tag = "bridge" if node.is_bridge else "shot"
            path = out_dir / f"{tag}_{node.shot_id}.png"
            img.save(path)
            print(f"  Saved -> {path}")

        real_shots = [n for n in gen_order if not n.is_bridge]
        self._make_contact_sheet(real_shots, keyframes, out_dir)

        print("\n" + "=" * 70)
        print(f"DONE — {len(keyframes)} keyframes generated")
        print(f"Output: {out_dir.resolve()}")
        print("=" * 70)

    @staticmethod
    def _make_contact_sheet(shots, keyframes, out_dir):
        imgs = [keyframes[s.shot_id] for s in shots]
        n = len(imgs)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        w, h = imgs[0].size
        pad = 4
        sheet = Image.new(
            "RGB",
            (cols * (w + pad) + pad, rows * (h + pad + 24) + pad),
            (30, 30, 30),
        )
        draw = ImageDraw.Draw(sheet)

        for i, (shot, img) in enumerate(zip(shots, imgs)):
            r, c = divmod(i, cols)
            x = pad + c * (w + pad)
            y = pad + r * (h + pad + 24)
            sheet.paste(img, (x, y))
            label = f"{shot.shot_id}: E={sorted(shot.entities)} BG={shot.bg}"
            draw.text((x + 2, y + h + 4), label, fill=(200, 200, 200))

        path = out_dir / "contact_sheet.png"
        sheet.save(path)
        print(f"\n  Contact sheet -> {path}")


# ═══════════════════════════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════════════════════════

ENTITY_PROMPTS = {
    "A": "a young man with short black hair, hero, full body, standing, white background",
    "B": "a young woman with red hair, sidekick, full body, standing, white background",
    "C": "a menacing villain in dark cloak, full body, standing, white background",
}

BG_PROMPTS = {
    "D": "a futuristic neon-lit city street at night, cinematic, no people",
    "E": "a dense enchanted forest with sunlight filtering through, cinematic, no people",
    "F": "a dark underground cave with glowing crystals, cinematic, no people",
}

SCENARIO = [
    ShotNode(shot_id="S1", entities={"A"},      bg="D", action="A walks through city"),
    ShotNode(shot_id="S2", entities={"A", "B"}, bg="D", action="B joins A in city"),
    ShotNode(shot_id="S3", entities={"A", "B"}, bg="D", action="A and B have conversation"),
    ShotNode(shot_id="S4", entities={"B"},      bg="E", action="B alone in forest"),
    ShotNode(shot_id="S5", entities={"C"},      bg="F", action="C lurks in cave"),
    ShotNode(shot_id="S6", entities={"A", "B"}, bg="E", action="A and B regroup in forest"),
    ShotNode(shot_id="S7", entities={"A", "B"}, bg="D", action="A and B return to city"),
    ShotNode(shot_id="S8", entities={"A", "C"}, bg="F", action="A confronts C in cave"),
]


def main():
    gen = KeyframeGenerator(
        device="cuda:3",
        num_steps=8,
        max_blend=0.7,
        inject_pct=0.6,
    )
    gen.run(
        scenario=SCENARIO,
        entity_prompts=ENTITY_PROMPTS,
        bg_prompts=BG_PROMPTS,
        out_dir=Path("outputs/phase3_keyframes"),
    )


if __name__ == "__main__":
    main()
