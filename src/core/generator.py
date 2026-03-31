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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def _stable_hash(s: str) -> int:
    """Deterministic hash that is consistent across Python sessions."""
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % 10000

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.core.routing import (
    ShotNode, RoutingGraph, ReverseRoutingGraph,
    EntityDecomposedRoutingGraph, distance,
)


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
        self.spatial_mask_type: str = "none"  # "none" | "gradient"
        # Spatial multi-source injection: list of (kv_bank, slot_index)
        # slot_index determines spatial region (0=left, 1=right, etc.)
        self.spatial_sources: list[tuple[dict, int]] = []
        self.num_spatial_slots: int = 2

    def reset(self):
        self.kv_bank.clear()
        self.mode = "bypass"
        self.current_step = 0
        self.blend_ratio = 0.0
        self.spatial_mask_type = "none"
        self.spatial_sources = []
        self.num_spatial_slots = 2

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

                        if self.spatial_mask_type == "gradient":
                            # Gradient mask: left=0 (free for new entity), right=1 (parent K/V)
                            seq_len = key.shape[1]
                            size = int(seq_len ** 0.5)
                            grad_1d = torch.linspace(0.0, 1.0, size, device=key.device, dtype=key.dtype)
                            grad_2d = grad_1d.unsqueeze(0).expand(size, -1)  # (H, W)
                            spatial_r = (grad_2d.reshape(1, seq_len, 1)) * r
                            key = (1 - spatial_r) * key + spatial_r * stored_k
                            value = (1 - spatial_r) * value + spatial_r * stored_v
                        else:
                            key = (1 - r) * key + r * stored_k
                            value = (1 - r) * value + r * stored_v

            if self.mode == "store_and_inject":
                self.kv_bank[self.current_step] = (
                    key.detach().clone(),
                    value.detach().clone(),
                )

        elif self.mode in ("spatial_inject", "spatial_store_and_inject"):
            # Multi-source K/V injection: each entity source contributes
            # globally (not spatially masked) with equal weight.
            # The prompt + IP-Adapter handle spatial positioning;
            # K/V injection handles identity preservation.
            if self.blend_ratio > 0.0 and self.spatial_sources:
                r = self.blend_ratio
                n_sources = len(self.spatial_sources)
                per_source_r = r / n_sources  # split blend across sources
                orig_dtype = key.dtype

                for src_bank, slot_idx in self.spatial_sources:
                    stored = src_bank.get(self.current_step)
                    if stored is None:
                        continue
                    src_k, src_v = stored
                    if src_k.shape[0] != key.shape[0] and key.shape[0] == 2 * src_k.shape[0]:
                        src_k = src_k.repeat(2, 1, 1)
                        src_v = src_v.repeat(2, 1, 1)
                    if src_k.shape != key.shape:
                        continue

                    key = ((1 - per_source_r) * key.float() + per_source_r * src_k.float()).to(orig_dtype)
                    value = ((1 - per_source_r) * value.float() + per_source_r * src_v.float()).to(orig_dtype)

            if self.mode == "spatial_store_and_inject":
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

    def clear_all_caches(self):
        """Clear all K/V caches and banks. Call between scenarios."""
        self._kv_cache.clear()
        for p in self.kv_processors.values():
            p.kv_bank.clear()

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
        """Snapshot current K/V banks from all processors into CPU cache."""
        self._kv_cache[shot_id] = {}
        for key, proc in self.kv_processors.items():
            self._kv_cache[shot_id][key] = {
                step: (k.cpu().clone(), v.cpu().clone())
                for step, (k, v) in proc.kv_bank.items()
            }

    def load_kv_from_cache(self, shot_id: str) -> bool:
        """Load cached K/V banks from CPU cache into GPU processors."""
        if shot_id not in self._kv_cache:
            return False
        device = next(self.unet.parameters()).device
        for key, proc in self.kv_processors.items():
            proc.kv_bank = {
                step: (k.clone().to(device), v.clone().to(device))
                for step, (k, v) in self._kv_cache[shot_id][key].items()
            }
        return True

    def set_spatial_mask(self, mask_type: str = "none"):
        """Set spatial mask type on all KV processors. 'gradient' or 'none'."""
        for p in self.kv_processors.values():
            p.spatial_mask_type = mask_type

    def load_spatial_kv(self, entity_sources: list[tuple[str, int]]) -> bool:
        """Load K/V from multiple entity sources with spatial slot assignments.

        Args:
            entity_sources: list of (shot_id, slot_index) pairs.
                slot_index: 0=left, 1=right (for 2 entities), etc.

        Returns True if at least one source was loaded.
        """
        device = next(self.unet.parameters()).device
        n_slots = max(s[1] for s in entity_sources) + 1 if entity_sources else 2
        any_loaded = False

        for key, proc in self.kv_processors.items():
            proc.spatial_sources = []
            proc.num_spatial_slots = n_slots

            for shot_id, slot_idx in entity_sources:
                if shot_id not in self._kv_cache:
                    continue
                if key not in self._kv_cache[shot_id]:
                    continue
                # Build a GPU kv_bank for this source
                src_bank = {
                    step: (k.clone().to(device), v.clone().to(device))
                    for step, (k, v) in self._kv_cache[shot_id][key].items()
                }
                proc.spatial_sources.append((src_bank, slot_idx))
                any_loaded = True

        return any_loaded

    def set_mode_spatial_inject(self):
        """Set all processors to spatial multi-source injection mode."""
        for p in self.kv_processors.values():
            p.mode = "spatial_inject"

    def set_mode_spatial_store_and_inject(self):
        """Set all processors to spatial injection + store mode."""
        for p in self.kv_processors.values():
            p.mode = "spatial_store_and_inject"

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
                 max_blend: float = 0.7, inject_pct: float = 0.6,
                 collage_steps: int = 20, collage_strength: float = 0.45,
                 width: int = 768, height: int = 768):
        self.device = device
        self.dtype = torch.float16
        self.num_steps = num_steps
        self.max_blend = max_blend
        self.inject_pct = inject_pct
        self.collage_steps = collage_steps
        self.collage_strength = collage_strength
        self.width = width
        self.height = height
        self.seed = 42

        self.pipe = None
        self.attn_ctrl: AttentionControl | None = None
        self.image_encoder = None
        self.clip_processor = None
        self.embedding_cache: dict[str, torch.Tensor] = {}
        self.anchor_images: dict[str, Image.Image] = {}

    def load_pipeline(self):
        if hasattr(self, "pipe") and self.pipe is not None:
            print("[Pipeline] SDXL-Turbo already loaded, reusing.")
            return
        from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

        print("[Pipeline] Loading SDXL-Turbo...")
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=self.dtype,
            variant="fp16",
        ).to(self.device)

        print("[Pipeline] Loading IP-Adapter...")
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.safetensors",
        )
        self.pipe.set_ip_adapter_scale(0.6)
        self.pipe.set_progress_bar_config(disable=True)

        print("[Pipeline] Creating Img2Img pipeline (shared components)...")
        self.pipe_i2i = AutoPipelineForImage2Image.from_pipe(self.pipe)
        self.pipe_i2i.set_progress_bar_config(disable=True)

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

    def _validate_single_entity(self, img: Image.Image) -> float:
        """Return CLIP similarity to 'one person' minus 'two people'.
        Positive = likely single entity. Negative = likely multiple."""
        import open_clip
        if not hasattr(self, '_val_clip_model'):
            self._val_clip_model, _, self._val_clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device,
            )
            self._val_clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

        img_tensor = self._val_clip_preprocess(img).unsqueeze(0).to(self.device)
        texts = self._val_clip_tokenizer(["one person, solo", "two people, duo, multiple characters"]).to(self.device)

        with torch.no_grad():
            img_feat = self._val_clip_model.encode_image(img_tensor)
            txt_feat = self._val_clip_model.encode_text(texts)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sims = (img_feat @ txt_feat.T).squeeze(0)

        # positive means "one person" is more likely
        return (sims[0] - sims[1]).item()

    def build_anchor_cache(self, entity_prompts, bg_prompts, out_dir):
        print("[Anchors] Generating base images...")
        anchor_images: dict[str, Image.Image] = {}
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        self.attn_ctrl.set_mode_bypass()

        for symbol, prompt in {**entity_prompts, **bg_prompts}.items():
            is_entity = symbol in entity_prompts
            if is_entity:
                import re
                # Detect entity type for appropriate framing
                human_kw = r'person|man|woman|girl|boy|warrior|marine|soldier|knight|wizard|witch|king|queen|prince|princess|assassin|thief|pilot|captain|commander|agent|detective|samurai|ninja|monk|priest'
                is_human = bool(re.search(human_kw, prompt, re.IGNORECASE))

                clean_prompt = self._clean_entity_desc(prompt)
                if is_human:
                    anchor_prompt = (
                        f"solo, one single character, full body, standing, centered, "
                        f"white background, {clean_prompt}"
                    )
                else:
                    # Non-human: drone, robot, creature, vehicle — keep full description
                    anchor_prompt = (
                        f"solo, single subject, full body, centered, "
                        f"white background, {prompt}"
                    )
                anchor_neg = (
                    "two people, two characters, duo, pair, couple, group, crowd, "
                    "multiple people, multiple figures, split image, side by side, "
                    "duplicate, second character, blurry, low quality, distorted"
                )
                anchor_steps = 8
                anchor_guidance = 2.0
                max_retries = 3
            else:
                anchor_prompt = prompt
                anchor_neg = "people, person, blurry, low quality, distorted"
                anchor_steps = 4
                anchor_guidance = 0.0
                max_retries = 1

            best_img = None
            best_score = -999.0
            for attempt in range(max_retries):
                attempt_gen = torch.Generator(device=self.device).manual_seed(self.seed + attempt)
                do_cfg = anchor_guidance > 0.0
                img = self.pipe(
                    prompt=anchor_prompt,
                    negative_prompt=anchor_neg if do_cfg else None,
                    ip_adapter_image_embeds=self._zero_ip_embeds(do_cfg=do_cfg),
                    num_inference_steps=anchor_steps,
                    guidance_scale=anchor_guidance,
                    generator=attempt_gen,
                    width=self.width, height=self.height,
                ).images[0]

                if not is_entity:
                    best_img = img
                    break

                # Validate single entity via CLIP
                score = self._validate_single_entity(img)
                print(f"  Anchor {symbol} attempt {attempt+1}: single-entity score = {score:.3f}")
                if score > best_score:
                    best_score = score
                    best_img = img
                if score > 0.02:  # confident single entity
                    break

            anchor_images[symbol] = best_img
            self.anchor_images[symbol] = best_img.copy()
            best_img.save(out_dir / f"anchor_{symbol}.png")
            if is_entity:
                print(f"  Anchor {symbol}: saved (single-entity score={best_score:.3f})")
            else:
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

    def _zero_ip_embeds(self, do_cfg=False):
        """Return zero embedding for IP-Adapter.
        When do_cfg=True, return [neg, pos] concatenated for classifier-free guidance."""
        zero = torch.zeros(1, 1, 1280, device=self.device, dtype=self.dtype)
        if do_cfg:
            return [torch.cat([zero, zero], dim=0)]  # [2, 1, 1280]
        return [zero]

    def _compose_ip_embeds(self, node):
        """Concatenate all entity + bg embeddings into one token sequence.
        Returns list with single (1, N, 1280) tensor where N = num_entities + 1."""
        tokens = []
        for ent in sorted(node.entities):
            tokens.append(self.embedding_cache[ent].unsqueeze(0).unsqueeze(0))  # (1,1,1280)
        tokens.append(self.embedding_cache[node.bg].unsqueeze(0).unsqueeze(0))  # (1,1,1280)
        concat = torch.cat(tokens, dim=1)  # (1, N, 1280)
        return [concat]

    @staticmethod
    def _clean_entity_desc(raw_desc: str) -> str:
        """Strip weapon/equipment/accessory phrases that cause SDXL to
        render extra figures (mech suits, large weapons as separate entities).
        Works at both clause-level (comma-separated) and phrase-level."""
        import re
        # Phrases to remove entirely (multi-word patterns)
        phrase_removals = [
            r'powered armor', r'power armor', r'power suit', r'battle suit',
            r'exo.?suit', r'mech suit', r'plasma rifle', r'assault rifle',
            r'sniper rifle', r'laser rifle', r'energy weapon',
            r'across (her|his|their) chest', r'on (her|his|their) back',
            r'in (her|his|their) hand(s)?', r'wielding \w+',
        ]
        result = raw_desc
        for pat in phrase_removals:
            result = re.sub(pat, '', result, flags=re.IGNORECASE)
        # Clause-level removal for remaining weapon keywords
        weapon_kw = (
            r'\brifle\b|\bgun\b|\bsword\b|\bweapon\b|\bblade\b|\bstaff\b|'
            r'\bshield\b|\bbow\b|\bspear\b|\bcannon\b|\bpistol\b|\baxe\b|'
            r'\bmech\b|\brobot\b|\bdrone\b|\bvehicle\b|\bmount\b|\bholster\b'
        )
        clauses = [c.strip() for c in result.split(',')]
        clean = [c for c in clauses if c and not re.search(weapon_kw, c, re.IGNORECASE)]
        result = ', '.join(clean) if clean else clauses[0]
        # Clean up double spaces and trailing commas
        result = re.sub(r'\s{2,}', ' ', result).strip(' ,')
        return result

    @staticmethod
    def build_prompt(node, entity_prompts, bg_prompts):
        entities_sorted = sorted(node.entities)
        parts = []
        if len(entities_sorted) >= 3:
            positions = ["on the left", "in the center", "on the right"]
            for i, ent in enumerate(entities_sorted):
                desc = KeyframeGenerator._clean_entity_desc(
                    entity_prompts[ent].split(",")[0])
                pos = positions[i] if i < len(positions) else ""
                parts.append(f"{desc} {pos}".strip())
        elif len(entities_sorted) == 2:
            desc0 = KeyframeGenerator._clean_entity_desc(
                entity_prompts[entities_sorted[0]].split(",")[0])
            desc1 = KeyframeGenerator._clean_entity_desc(
                entity_prompts[entities_sorted[1]].split(",")[0])
            parts.append(f"{desc0} standing on the left")
            parts.append(f"{desc1} standing on the right")
        else:
            for ent in entities_sorted:
                desc = KeyframeGenerator._clean_entity_desc(
                    entity_prompts[ent].split(",")[0])
                parts.append(f"solo, {desc}")  # enforce single entity
        bg_desc = bg_prompts[node.bg].split(",")[0]
        action = node.action if node.action else "group scene"
        return (f"{', '.join(parts)}, in {bg_desc}, {action}, "
                f"cinematic still frame, high quality, detailed")

    # ── Collage Composition ───────────────────────────────────────

    def _compose_collage(self, node, source_images=None):
        """Compose collage: entity images on background.

        Args:
            node: target node with entities and bg
            source_images: optional dict {entity -> PIL.Image} to use instead
                          of anchor images. Used for entity-decomposed routing
                          where each entity has a specific reference shot.

        Pastes full-size entity images (with white-bg removed) at horizontal
        offsets over the background anchor, so characters are spread across
        the canvas without being cropped.
        """
        entities = sorted(node.entities)
        n = len(entities)
        w, h = self.width, self.height

        # Base canvas: background image
        bg_img = self.anchor_images.get(node.bg)
        canvas = bg_img.copy().resize((w, h)) if bg_img else Image.new("RGB", (w, h), (30, 30, 30))

        # Target character centers: evenly spaced across canvas
        if n == 1:
            offsets = [0]
        else:
            margin = w // (n + 1)
            targets = [margin * (i + 1) for i in range(n)]
            center = w // 2
            offsets = [t - center for t in targets]

        for i, ent in enumerate(entities):
            # Use source_images if provided, otherwise fall back to anchors
            if source_images and ent in source_images:
                ent_img = source_images[ent].convert("RGB").resize((w, h))
            else:
                ent_img = self.anchor_images[ent].convert("RGB").resize((w, h))
            ent_arr = np.array(ent_img)

            # Mask: non-white pixels = entity foreground
            is_white = np.all(ent_arr > 220, axis=2)
            mask_arr = (~is_white).astype(np.uint8) * 255
            mask = Image.fromarray(mask_arr, "L")

            # Paste full entity image at horizontal offset (white bg masked out)
            canvas.paste(ent_img, (offsets[i], 0), mask)

        return canvas

    # ── Multi-Stream Latent Fusion ────────────────────────────────

    def _generate_multi_stream(self, node, entity_prompts, bg_prompts,
                                entity_sources, keyframes, gen):
        """Multi-Stream Latent Fusion for multi-entity generation.

        Instead of collage + img2img, runs separate U-Net forward passes
        per entity stream (each with its own prompt, IP-Adapter, K/V),
        then fuses predicted noise in latent space via sigmoid masks.

        Streams:
          - Stream 0 (background): bg prompt only, no IP-Adapter identity
          - Stream 1..N (entities): per-entity prompt + IP-Adapter + K/V
        """
        pipe = self.pipe
        unet = pipe.unet
        scheduler = pipe.scheduler
        vae = pipe.vae
        entities_sorted = sorted(node.entities)
        n_ents = len(entities_sorted)
        device = self.device
        dtype = self.dtype

        # ── Build per-stream conditions ──
        # Background stream: use full bg description for better consistency
        bg_full = bg_prompts[node.bg]
        bg_desc = bg_full.split(",")[0]
        bg_prompt = f"{bg_full}, cinematic, detailed, no people"

        # Entity streams: use cleaned description (no weapons/mech that cause duplicates)
        ent_prompts = []
        ent_ip_embeds = []
        for ent in entities_sorted:
            ent_parts = entity_prompts[ent].split(",")
            raw_desc = ",".join(ent_parts[:min(3, len(ent_parts))])
            desc = self._clean_entity_desc(raw_desc)
            ent_prompts.append(
                f"solo, {desc}, in {bg_desc}, cinematic still frame, high quality"
            )
            ent_ip_embeds.append(
                self.embedding_cache[ent].unsqueeze(0).unsqueeze(0)  # (1,1,1280)
            )

        # Encode all prompts (SDXL returns prompt_embeds + pooled_prompt_embeds)
        all_prompts = [bg_prompt] + ent_prompts
        prompt_embeds_list = []
        add_text_embeds_list = []
        for p in all_prompts:
            (prompt_embeds, negative_prompt_embeds,
             pooled_prompt_embeds, negative_pooled) = pipe.encode_prompt(
                prompt=p,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            prompt_embeds_list.append(prompt_embeds)
            add_text_embeds_list.append(pooled_prompt_embeds)

        # ── Build sigmoid spatial masks (latent space) ──
        latent_h = self.height // 8  # VAE downscale factor
        latent_w = self.width // 8
        masks = self._build_stream_masks(n_ents, latent_h, latent_w, device, dtype)

        # ── Prepare latents ──
        num_steps = self.num_steps
        scheduler.set_timesteps(num_steps, device=device)
        timesteps = scheduler.timesteps

        latent_shape = (1, unet.config.in_channels, latent_h, latent_w)
        latents = torch.randn(latent_shape, generator=gen, device=device, dtype=dtype)
        latents = latents * scheduler.init_noise_sigma

        # ── Prepare added time IDs (SDXL requirement) ──
        # Manually construct: [orig_h, orig_w, crop_y, crop_x, target_h, target_w]
        add_time_ids = torch.tensor(
            [[self.height, self.width, 0, 0, self.height, self.width]],
            dtype=dtype, device=device,
        )

        # ── Per-entity K/V: load into separate banks ──
        # entity_sources: [(shot_id, slot_idx), ...]
        entity_kv_caches = {}  # slot_idx -> {proc_key -> {step -> (K, V)}}
        for shot_id, slot_idx in entity_sources:
            if shot_id in self.attn_ctrl._kv_cache:
                entity_kv_caches[slot_idx] = self.attn_ctrl._kv_cache[shot_id]

        # ── Prepare IP-Adapter image embeds for each stream ──
        zero_ip = self._zero_ip_embeds()  # for bg stream
        stream_ip_embeds = [zero_ip]  # stream 0 = bg
        for emb in ent_ip_embeds:
            stream_ip_embeds.append([emb])  # stream 1..N = entity

        print(f"  Streams: 1 bg + {n_ents} entity, {num_steps} steps")
        print(f"  Latent shape: {latent_shape}, masks: {[m.shape for m in masks]}")

        # ── Custom denoising loop ──
        for step_idx, t in enumerate(timesteps):
            # Scale input for scheduler
            latent_input = scheduler.scale_model_input(latents, t)

            # Run U-Net for each stream
            noise_preds = []
            for stream_idx in range(n_ents + 1):
                # FIX: Clear K/V banks BEFORE each stream to prevent cross-contamination
                for proc in self.attn_ctrl.kv_processors.values():
                    proc.kv_bank.clear()

                # Set K/V injection for entity streams
                if stream_idx > 0:
                    slot_idx = stream_idx - 1
                    if slot_idx in entity_kv_caches:
                        # Load this entity's reference K/V
                        kv_cache = entity_kv_caches[slot_idx]
                        has_any_kv = False
                        for key, proc in self.attn_ctrl.kv_processors.items():
                            if key in kv_cache:
                                step_kv = kv_cache[key].get(step_idx)
                                if step_kv is not None:
                                    k, v = step_kv
                                    proc.kv_bank[step_idx] = (
                                        k.clone().to(device),
                                        v.clone().to(device),
                                    )
                                    has_any_kv = True
                            proc.current_step = step_idx
                            # Decay blend over steps
                            inject_steps = int(num_steps * self.inject_pct)
                            # Higher blend for multi-stream (identity preservation)
                            ms_blend = min(self.max_blend + 0.15, 0.9)
                            if step_idx < inject_steps and has_any_kv:
                                proc.blend_ratio = ms_blend * (1.0 - step_idx / inject_steps)
                            else:
                                proc.blend_ratio = 0.0
                            proc.mode = "inject"
                    else:
                        self.attn_ctrl.set_mode_bypass()
                else:
                    # Background stream: no K/V injection
                    self.attn_ctrl.set_mode_bypass()

                # Set IP-Adapter embeds for this stream (higher scale for identity)
                pipe.set_ip_adapter_scale(0.8 if stream_idx > 0 else 0.0)

                # Prepare added conditions
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds_list[stream_idx],
                    "time_ids": add_time_ids,
                }

                # Handle IP-Adapter embeds
                ip_emb = stream_ip_embeds[stream_idx]
                if hasattr(unet, 'encoder_hid_proj') and unet.encoder_hid_proj is not None:
                    added_cond_kwargs["image_embeds"] = ip_emb

                with torch.no_grad():
                    noise_pred = unet(
                        latent_input,
                        t,
                        encoder_hidden_states=prompt_embeds_list[stream_idx],
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                # FIX: Guard against NaN — replace with zero (bg stream will fill)
                if torch.isnan(noise_pred).any():
                    nan_pct = torch.isnan(noise_pred).float().mean().item() * 100
                    print(f"  WARNING: NaN in stream {stream_idx} ({nan_pct:.1f}%), replacing with zeros")
                    noise_pred = torch.nan_to_num(noise_pred, nan=0.0)

                noise_preds.append(noise_pred)

            # ── Fuse noise predictions via spatial masks ──
            fused_noise = torch.zeros_like(noise_preds[0])
            for stream_idx, (noise, mask) in enumerate(zip(noise_preds, masks)):
                fused_noise = fused_noise + noise * mask

            # FIX: Final NaN guard on fused noise
            if torch.isnan(fused_noise).any():
                print(f"  WARNING: NaN in fused noise at step {step_idx}, clamping")
                fused_noise = torch.nan_to_num(fused_noise, nan=0.0)

            # ── Scheduler step ──
            latents = scheduler.step(fused_noise, t, latents, return_dict=False)[0]

            # Clear K/V banks after each step
            for proc in self.attn_ctrl.kv_processors.values():
                proc.kv_bank.clear()

        # ── Decode latents to image ──
        self.attn_ctrl.set_mode_bypass()
        pipe.set_ip_adapter_scale(0.6)

        latents_scaled = latents / vae.config.scaling_factor
        # FIX: Upcast VAE to float32 for decode (matches standard SDXL pipeline behavior)
        # fp16 VAE produces NaN on large latent values (std≈8 after scaling)
        needs_upcast = vae.dtype == torch.float16 and vae.config.force_upcast
        if needs_upcast:
            vae.to(dtype=torch.float32)
            latents_for_decode = latents_scaled.to(torch.float32)
        else:
            latents_for_decode = latents_scaled.to(torch.float32)  # always use fp32 for safety

        with torch.no_grad():
            decoded = vae.decode(latents_for_decode, return_dict=False)[0]

        if needs_upcast:
            vae.to(dtype=torch.float16)

        image = pipe.image_processor.postprocess(decoded.float(), output_type="pil")[0]

        # Store K/V cache for this node (re-run with store mode)
        # We'll do a lightweight store pass using the fused result
        self._store_kv_for_node(node, image, entity_prompts, bg_prompts, gen)

        return image

    def _store_kv_for_node(self, node, image, entity_prompts, bg_prompts, gen):
        """Run a quick img2img pass on the generated image to capture K/V."""
        prompt = self.build_prompt(node, entity_prompts, bg_prompts)
        ip_embeds = self._compose_ip_embeds(node)

        self.attn_ctrl.set_mode_store()
        step_log = []
        def cb(pipe, step, timestep, cbk):
            self.attn_ctrl.update_step(step)
            step_log.append(step)
            return cbk

        # Light img2img pass just to capture K/V
        # strength=0.75 ensures enough steps (ceil(8*0.75)=6) for K/V to cover
        # the inject_pct=0.6 window (ceil(8*0.6)=5 steps needed)
        _ = self.pipe_i2i(
            prompt=prompt,
            image=image,
            ip_adapter_image_embeds=ip_embeds,
            strength=0.75,
            num_inference_steps=self.num_steps,
            guidance_scale=0.0,
            generator=gen,
            callback_on_step_end=cb,
        ).images[0]

        self.attn_ctrl.save_kv_to_cache(node.shot_id)
        for p in self.attn_ctrl.kv_processors.values():
            p.kv_bank.clear()
        self.attn_ctrl.set_mode_bypass()
        print(f"  Stored K/V via img2img ({len(step_log)} steps)")

    def _build_stream_masks(self, n_ents, h, w, device, dtype):
        """Build sigmoid spatial masks for bg + N entity streams.

        Returns list of (1, 1, h, w) masks that sum to 1.0 at every position.
        Entity masks are sigmoid-based soft regions; bg mask fills the remainder.
        """
        x = torch.linspace(0, 1, w, device=device, dtype=torch.float32)

        entity_masks = []
        sharpness = 12.0  # controls softness of boundaries

        for i in range(n_ents):
            center = (i + 0.5) / n_ents
            half_width = 0.5 / n_ents
            # Sigmoid: rises at left edge, falls at right edge
            left_edge = torch.sigmoid(sharpness * (x - (center - half_width)))
            right_edge = torch.sigmoid(sharpness * ((center + half_width) - x))
            mask_1d = left_edge * right_edge  # bell-shaped
            mask_2d = mask_1d.unsqueeze(0).expand(h, -1)  # (h, w)
            entity_masks.append(mask_2d)

        # Stack and normalize so entity masks + bg = 1.0
        if entity_masks:
            ent_stack = torch.stack(entity_masks, dim=0)  # (N, h, w)
            ent_sum = ent_stack.sum(dim=0, keepdim=True)  # (1, h, w)
            # Cap entity contribution at 0.85 to leave room for bg
            max_ent = 0.85
            scale = torch.where(ent_sum > max_ent,
                                max_ent / ent_sum.clamp(min=1e-6),
                                torch.ones_like(ent_sum))
            ent_stack = ent_stack * scale
            bg_mask = 1.0 - ent_stack.sum(dim=0)  # (h, w)
        else:
            bg_mask = torch.ones(h, w, device=device, dtype=torch.float32)
            ent_stack = torch.zeros(0, h, w, device=device, dtype=torch.float32)

        # Assemble: [bg_mask, ent_mask_0, ent_mask_1, ...]
        masks = [bg_mask.unsqueeze(0).unsqueeze(0).to(dtype)]  # (1,1,h,w)
        for i in range(n_ents):
            masks.append(ent_stack[i].unsqueeze(0).unsqueeze(0).to(dtype))

        return masks

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
            # Universal Anchor: compose collage from individual anchors,
            # then harmonize via img2img to create a coherent multi-entity image
            is_universal = node.is_bridge and len(node.entities) >= 2

            if is_universal:
                collage = self._compose_collage(node)
                self._last_collage = collage  # save for debug output
                print(f"  Mode: COLLAGE + HARMONIZE (Universal Anchor, store K/V)")
                self.attn_ctrl.set_mode_store()

                step_log = []
                def root_callback(pipe, step, timestep, cbk):
                    self.attn_ctrl.update_step(step)
                    step_log.append(step)
                    return cbk

                img = self.pipe_i2i(
                    prompt=prompt,
                    negative_prompt="blurry, low quality, distorted, deformed, chimera",
                    image=collage,
                    ip_adapter_image_embeds=ip_embeds,
                    strength=self.collage_strength,
                    num_inference_steps=self.collage_steps,
                    guidance_scale=0.0,
                    generator=gen,
                    callback_on_step_end=root_callback,
                ).images[0]
                self.attn_ctrl.save_kv_to_cache(node.shot_id)
                for p in self.attn_ctrl.kv_processors.values():
                    p.kv_bank.clear()
                self.attn_ctrl.set_mode_bypass()
                print(f"  Cached K/V for {len(step_log)} steps")

            else:
                is_single = len(node.entities) <= 1
                cfg_scale = 3.0 if is_single else 0.0
                neg_prompt = (
                    "two people, two characters, duplicate, duo, pair, "
                    "multiple figures, robot behind, mech behind, "
                    "split image, side by side, "
                    "blurry, low quality, distorted, deformed"
                ) if is_single else "blurry, low quality, distorted, deformed"
                print(f"  Mode: T2I (root node, store K/V, cfg={cfg_scale})")
                self.attn_ctrl.set_mode_store()

                step_log = []
                def root_callback(pipe, step, timestep, cbk):
                    self.attn_ctrl.update_step(step)
                    step_log.append(step)
                    return cbk

                # For CFG > 0, IP-Adapter needs doubled embeds [neg, pos]
                use_cfg = cfg_scale > 0.0
                if use_cfg:
                    pos_emb = ip_embeds[0]  # (1, N, 1280)
                    neg_emb = torch.zeros_like(pos_emb)
                    ip_for_gen = [torch.cat([neg_emb, pos_emb], dim=0)]
                else:
                    ip_for_gen = ip_embeds

                best_img = None
                best_score = -999.0
                max_tries = 3 if (is_single and use_cfg) else 1
                for attempt in range(max_tries):
                    attempt_gen = torch.Generator(device=self.device).manual_seed(
                        self.seed + _stable_hash(node.shot_id) + attempt
                    )
                    self.attn_ctrl.set_mode_store()
                    step_log.clear()

                    img = self.pipe(
                        prompt=prompt,
                        negative_prompt=neg_prompt if use_cfg else None,
                        ip_adapter_image_embeds=ip_for_gen,
                        num_inference_steps=8,
                        guidance_scale=cfg_scale,
                        generator=attempt_gen,
                        width=self.width, height=self.height,
                        callback_on_step_end=root_callback,
                    ).images[0]

                    if not is_single or max_tries == 1:
                        best_img = img
                        break

                    # Validate single entity
                    score = self._validate_single_entity(img)
                    print(f"    Attempt {attempt+1}: single-entity score={score:.3f}")
                    if score > best_score:
                        best_score = score
                        best_img = img
                        # Save this attempt's K/V
                        self.attn_ctrl.save_kv_to_cache(f"_best_{node.shot_id}")
                    if score > 0.03:
                        break

                img = best_img
                # Restore best K/V if we did multiple attempts
                if max_tries > 1 and f"_best_{node.shot_id}" in self.attn_ctrl._kv_cache:
                    self.attn_ctrl._kv_cache[node.shot_id] = self.attn_ctrl._kv_cache.pop(f"_best_{node.shot_id}")
                self.attn_ctrl.save_kv_to_cache(node.shot_id)
                for p in self.attn_ctrl.kv_processors.values():
                    p.kv_bank.clear()
                self.attn_ctrl.set_mode_bypass()
                print(f"  Cached K/V for {len(step_log)} steps")

        elif d == 0:
            print(f"  Mode: REUSE (D=0, blend={self.max_blend:.0%})")
            self._generate_with_parent_kv(
                node, prompt, ip_embeds, keyframes, gen,
                blend_override=self.max_blend,
            )
            img = self._last_generated

        elif len(node.entities) >= 2 and hasattr(self, '_entity_graph') and self._entity_graph is not None:
            # ── Multi-entity with D≥1: MULTI-STREAM LATENT FUSION ──
            # Each entity gets its own U-Net forward pass with dedicated
            # prompt, IP-Adapter, and K/V injection. Noise predictions
            # are fused in latent space via sigmoid spatial masks.
            refs = self._entity_graph.entity_refs.get(node.shot_id, {})
            entities_sorted = sorted(node.entities)
            n_ents = len(entities_sorted)

            entity_sources = []
            for slot_idx, ent in enumerate(entities_sorted):
                if ent in refs:
                    ref_node = refs[ent]
                    entity_sources.append((ref_node.shot_id, slot_idx))
                    print(f"  ref({ent}) = {ref_node.shot_id} "
                          f"(E={sorted(ref_node.entities)}, BG={ref_node.bg}) [K/V source]")
                else:
                    print(f"  ref({ent}) = NONE (first appearance, no K/V)")

            print(f"  Mode: MULTI-STREAM LATENT FUSION ({n_ents} entities)")
            img = self._generate_multi_stream(
                node, self._entity_prompts, self._bg_prompts,
                entity_sources, keyframes, gen,
            )

        elif len(node.entities) >= 2:
            # ── Fallback: Multi-entity with D≥1 (no entity graph) ──
            # Uses collage + harmonize approach
            collage = self._compose_collage(node)
            self._last_collage = collage

            parent = node.parent_node
            style_blend = 0.15
            loaded = self.attn_ctrl.load_kv_from_cache(parent.shot_id)
            if loaded:
                print(f"  Mode: COLLAGE + HARMONIZE (multi-entity, style K/V from {parent_id}, blend={style_blend:.0%})")
                orig_blend = self.attn_ctrl.max_blend
                self.attn_ctrl.max_blend = style_blend
                self.attn_ctrl.set_mode_store_and_inject()
            else:
                print(f"  Mode: COLLAGE + HARMONIZE (multi-entity, no parent K/V)")
                self.attn_ctrl.set_mode_store()

            step_log = []
            def multi_callback(pipe, step, timestep, cbk):
                self.attn_ctrl.update_step(step)
                step_log.append(step)
                return cbk

            img = self.pipe_i2i(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted, deformed, chimera, merged faces",
                image=collage,
                ip_adapter_image_embeds=ip_embeds,
                strength=self.collage_strength,
                num_inference_steps=self.collage_steps,
                guidance_scale=0.0,
                generator=gen,
                callback_on_step_end=multi_callback,
            ).images[0]

            self.attn_ctrl.save_kv_to_cache(node.shot_id)
            for p in self.attn_ctrl.kv_processors.values():
                p.kv_bank.clear()
            self.attn_ctrl.set_mode_bypass()
            if loaded:
                self.attn_ctrl.max_blend = orig_blend
            print(f"  Cached K/V for {len(step_log)} steps")

        elif d == 1:
            # ── Single entity D=1 ──
            parent = node.parent_node
            removed = parent.entities - node.entities

            if removed:
                blend = self.max_blend * 0.5
                print(f"  Mode: DERIVE (D=1, -entity({removed}), blend={blend:.0%}→0)")
                self._generate_with_parent_kv(
                    node, prompt, ip_embeds, keyframes, gen,
                    blend_override=blend,
                )
                img = self._last_generated
            else:
                blend = self.max_blend * 0.7
                print(f"  Mode: DERIVE (D=1, bg({parent.bg}→{node.bg}), blend={blend:.0%}→0)")
                self._generate_with_parent_kv(
                    node, prompt, ip_embeds, keyframes, gen,
                    blend_override=blend,
                )
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
        is_single = len(node.entities) <= 1
        cfg_scale = 3.0 if is_single else 0.0
        neg_prompt = (
            "two people, two characters, duplicate, duo, pair, "
            "multiple figures, robot behind, mech behind, "
            "split image, side by side, "
            "blurry, low quality, distorted, deformed"
        ) if is_single else "blurry, low quality, distorted, deformed"
        print(f"  Generating {node.shot_id} with K/V inject + store (cfg={cfg_scale})...")
        self.attn_ctrl.set_mode_store_and_inject()

        step_log = []
        def callback(pipe, step, timestep, cbk):
            self.attn_ctrl.update_step(step)
            ratio = next(iter(self.attn_ctrl.kv_processors.values())).blend_ratio
            step_log.append((step, ratio))
            return cbk

        use_cfg = cfg_scale > 0.0
        if use_cfg:
            pos_emb = ip_embeds[0]
            neg_emb = torch.zeros_like(pos_emb)
            ip_for_gen = [torch.cat([neg_emb, pos_emb], dim=0)]
        else:
            ip_for_gen = ip_embeds

        best_img = None
        best_score = -999.0
        max_tries = 3 if is_single else 1
        for attempt in range(max_tries):
            attempt_gen = torch.Generator(device=self.device).manual_seed(
                self.seed + _stable_hash(node.shot_id) + attempt
            )
            self.attn_ctrl.set_mode_store_and_inject()
            step_log.clear()

            img = self.pipe(
                prompt=prompt,
                negative_prompt=neg_prompt if use_cfg else None,
                ip_adapter_image_embeds=ip_for_gen,
                num_inference_steps=8,
                guidance_scale=cfg_scale,
                generator=attempt_gen,
                width=self.width, height=self.height,
                callback_on_step_end=callback,
            ).images[0]

            if not is_single or max_tries == 1:
                best_img = img
                break

            score = self._validate_single_entity(img)
            print(f"    Attempt {attempt+1}: single-entity score={score:.3f}")
            if score > best_score:
                best_score = score
                best_img = img
                self.attn_ctrl.save_kv_to_cache(f"_best_{node.shot_id}")
            if score > 0.03:
                break

        self._last_generated = best_img
        # Restore best K/V
        if max_tries > 1 and f"_best_{node.shot_id}" in self.attn_ctrl._kv_cache:
            self.attn_ctrl._kv_cache[node.shot_id] = self.attn_ctrl._kv_cache.pop(f"_best_{node.shot_id}")

        # Cache this node's K/V to CPU for its children, then clear GPU banks
        self.attn_ctrl.save_kv_to_cache(node.shot_id)
        for p in self.attn_ctrl.kv_processors.values():
            p.kv_bank.clear()

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
        print("PHASE 3: Entity-Decomposed Graph-Routed Keyframe Generation")
        print("=" * 70)

        graph = EntityDecomposedRoutingGraph()
        graph.build_from_shots(scenario)
        graph.print_routing_table()
        graph.print_detailed_edges()
        graph.print_entity_refs()
        gen_order = graph.topological_order()
        print(f"\nGeneration order: {' -> '.join(n.shot_id for n in gen_order)}")

        # Store reference to entity graph for spatial injection in generate_node
        self._entity_graph = graph

        self.load_pipeline()
        # Clear K/V caches from any previous scenario
        if hasattr(self, "attn_ctrl"):
            self.attn_ctrl.clear_all_caches()
        self._entity_prompts = entity_prompts
        self._bg_prompts = bg_prompts
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

            # Save raw collage for Universal Anchor (debug)
            if hasattr(self, "_last_collage") and self._last_collage is not None:
                collage_path = out_dir / f"collage_raw_{node.shot_id}.png"
                self._last_collage.save(collage_path)
                print(f"  Collage -> {collage_path}")
                self._last_collage = None

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
