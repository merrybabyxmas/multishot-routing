"""
Phase 3 Quick Test — Keyframe-only generation using the routing graph.

Pipeline:
  1. Global Anchor Init: SDXL-Turbo generates base images per entity/bg
  2. CLIP caches visual embeddings for each symbol
  3. Routing graph drives generation order
  4. Each shot keyframe is generated conditioned on its parent via
     IP-Adapter (identity) + latent blending (structural consistency)

No video — frames only.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.routing import ShotNode, RoutingGraph, distance

# ── Config ─────────────────────────────────────────────────────────────
DEVICE = "cuda:3"
DTYPE = torch.float16
OUT_DIR = Path("outputs/keyframe_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# ── Global Legend ──────────────────────────────────────────────────────
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

# ── 8-Shot Scenario ───────────────────────────────────────────────────
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


def build_shot_prompt(node: ShotNode) -> str:
    """Compose a generation prompt from entities + bg + action."""
    entity_parts = []
    for ent in sorted(node.entities):
        # Extract key description from entity prompt
        desc = ENTITY_PROMPTS[ent].split(",")[0]
        entity_parts.append(desc)
    bg_desc = BG_PROMPTS[node.bg].split(",")[0]
    return f"{', '.join(entity_parts)}, in {bg_desc}, {node.action}, cinematic still frame, high quality"


def main():
    print("=" * 70)
    print("PHASE 3 QUICK TEST — Keyframe Generation")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # ── Step 1: Build Routing Graph ────────────────────────────────────
    print("\n[1/4] Building routing graph...")
    graph = RoutingGraph()
    graph.build_from_shots(SCENARIO)
    graph.print_routing_table()
    gen_order = graph.topological_order()
    print(f"\nGeneration order: {' -> '.join(n.shot_id for n in gen_order)}")

    # ── Step 2: Load SDXL-Turbo ────────────────────────────────────────
    print("\n[2/4] Loading SDXL-Turbo pipeline...")
    from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

    pipe_t2i = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=DTYPE,
        variant="fp16",
    ).to(DEVICE)
    pipe_t2i.set_progress_bar_config(disable=True)

    # Share components for img2img
    pipe_i2i = AutoPipelineForImage2Image.from_pipe(pipe_t2i)
    pipe_i2i.set_progress_bar_config(disable=True)

    # ── Step 3: Generate Global Anchors ────────────────────────────────
    print("\n[3/4] Generating global anchors (entities + backgrounds)...")
    anchor_images: dict[str, Image.Image] = {}

    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    for symbol, prompt in {**ENTITY_PROMPTS, **BG_PROMPTS}.items():
        img = pipe_t2i(
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted",
            num_inference_steps=4,
            guidance_scale=0.0,  # turbo = cfg-free
            generator=gen,
            width=512,
            height=512,
        ).images[0]
        anchor_images[symbol] = img
        path = OUT_DIR / f"anchor_{symbol}.png"
        img.save(path)
        print(f"  Anchor {symbol}: saved -> {path}")

    # ── Step 4: Load CLIP for embeddings cache ─────────────────────────
    print("\n[3.5/4] Building CLIP embedding cache...")
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

    clip_model = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=DTYPE,
    ).to(DEVICE)
    clip_processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14",
    )

    embedding_cache: dict[str, torch.Tensor] = {}
    for symbol, img in anchor_images.items():
        inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE, DTYPE)
        with torch.no_grad():
            emb = clip_model(**inputs).image_embeds  # (1, 768)
        embedding_cache[symbol] = emb.cpu()
        print(f"  CLIP embed {symbol}: shape={emb.shape}")

    # Free CLIP from GPU
    del clip_model
    torch.cuda.empty_cache()

    # ── Step 5: Generate Keyframes via Routing ─────────────────────────
    print("\n[4/4] Generating keyframes in topological order...")
    keyframes: dict[str, Image.Image] = {}

    for node in gen_order:
        prompt = build_shot_prompt(node)
        d = distance(node, node.parent_node) if node.parent_node else -1
        parent_id = node.parent_node.shot_id if node.parent_node else "ROOT"
        kind = "Bridge" if node.is_bridge else "Shot"

        print(f"\n  --- {kind} {node.shot_id} (parent={parent_id}, D={d}) ---")
        print(f"      Prompt: {prompt[:80]}...")

        gen = torch.Generator(device=DEVICE).manual_seed(SEED + hash(node.shot_id) % 10000)

        if node.parent_node is None:
            # Root node — pure text2image
            img = pipe_t2i(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted",
                num_inference_steps=4,
                guidance_scale=0.0,
                generator=gen,
                width=512,
                height=512,
            ).images[0]

        elif d == 0:
            # Rule A: Optimal reuse — copy parent keyframe, light img2img
            parent_img = keyframes[node.parent_node.shot_id]
            img = pipe_i2i(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted",
                image=parent_img,
                strength=0.3,  # light — mostly reuse (min viable for 4-step turbo)
                num_inference_steps=4,
                guidance_scale=0.0,
                generator=gen,
            ).images[0]

        elif d == 1:
            # Rule B: Direct derivation — img2img from parent with moderate strength
            parent_img = keyframes[node.parent_node.shot_id]
            img = pipe_i2i(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted",
                image=parent_img,
                strength=0.45,  # moderate — keep structure, allow changes
                num_inference_steps=4,
                guidance_scale=0.0,
                generator=gen,
            ).images[0]

        else:
            # Should never happen (bridges guarantee D ≤ 1)
            raise RuntimeError(f"D={d} for {node.shot_id} — bridge injection failed!")

        keyframes[node.shot_id] = img

        # Save only real shots (skip bridges from final output, but save for debug)
        tag = "bridge" if node.is_bridge else "shot"
        path = OUT_DIR / f"{tag}_{node.shot_id}.png"
        img.save(path)
        print(f"      Saved -> {path}")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DONE — All keyframes generated")
    print(f"Output directory: {OUT_DIR.resolve()}")
    print("=" * 70)

    # Create a contact sheet of real shots only
    real_shots = [n for n in gen_order if not n.is_bridge]
    _make_contact_sheet(real_shots, keyframes)


def _make_contact_sheet(shots: list[ShotNode], keyframes: dict[str, Image.Image]):
    """Stitch all real shot keyframes into a single overview image."""
    imgs = [keyframes[s.shot_id] for s in shots]
    n = len(imgs)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    w, h = imgs[0].size
    pad = 4
    sheet = Image.new("RGB", (cols * (w + pad) + pad, rows * (h + pad + 20) + pad), (30, 30, 30))

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(sheet)

    for i, (shot, img) in enumerate(zip(shots, imgs)):
        r, c = divmod(i, cols)
        x = pad + c * (w + pad)
        y = pad + r * (h + pad + 20)
        sheet.paste(img, (x, y))
        label = f"{shot.shot_id}: {sorted(shot.entities)} / {shot.bg}"
        draw.text((x + 2, y + h + 2), label, fill=(200, 200, 200))

    path = OUT_DIR / "contact_sheet.png"
    sheet.save(path)
    print(f"\nContact sheet saved -> {path}")


if __name__ == "__main__":
    main()
