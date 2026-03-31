# Multishot Routing — Non-Markovian Graph Routing for Multi-Shot Generation

## 1. Project Overview

**Goal:** Solve **Long-term Consistency Collapse** and **Identity Chimera** in training-free, multi-shot T2I generation.

**Core thesis:** By replacing Markovian (sequential) K/V injection with **Non-Markovian Graph Routing** (state-distance $D$) and **Multi-Stream Latent Fusion**, background and identity consistency can be maintained across 10+ shots without any finetuning.

**Benchmark:** MSR-50 — 50 scenarios × 10 shots × 5 domains (scifi, fantasy, modern, nature, stylized).

---

## 2. Core Methodology

### A. Non-Markovian Graph Routing (The Routing Engine)
* **State Distance:** $D(S_t, S_k) = |E_t \triangle E_k| + \mathbb{1}(B_t \neq B_k)$
* **Optimal Parent:** Scan all past shots, pick the one with minimum $D$ as K/V parent (not just $t-1$).
* **Bridge Nodes:** When $D \ge 2$, inject synthetic bridge nodes to ensure every edge has $D \le 1$.
  * Order: Add Entity → Change Background → Remove Entity.

### B. Entity-Decomposed Routing (Current Active Strategy)
* Extends forward routing with **per-entity reference resolution**.
* For multi-entity shots (e.g., S8={A,C}), find the best **solo** source node per entity:
  * `ref(A) = S1` (solo A), `ref(C) = S5` (solo C).
* Scoring: `10×is_solo - 2×crowd + 1×same_bg + 0.01×recency`.
* Stored in `EntityDecomposedRoutingGraph.entity_refs: dict[shot_id, dict[entity, ShotNode]]`.

### C. Multi-Stream Latent Fusion (The Rendering Engine — CORE)
* **Pixel-level collage is STRICTLY FORBIDDEN.** All composition happens in latent space.
* **Stream Decoupling:** Separate U-Net forward pass per entity + 1 background stream.
* **Per-stream conditioning:** Each entity stream gets its own prompt, IP-Adapter identity embedding, and routed K/V cache from its reference node.
* **Steep Sigmoid Spatial Masks:** `torch.sigmoid(x * steepness)` (steepness=12.0) partitions the latent space. Entity contribution capped at 0.85, background gets remainder. NO linear masking (`torch.linspace`) allowed.
* **Fusion:** $\epsilon_{final} = \sum_i \epsilon_i \times mask_i$ (noise-level blending per denoising step).
* **K/V Storage:** After multi-stream generation, a lightweight img2img pass (strength=0.2) captures unified K/V cache for downstream child nodes.

---

## 3. Evaluation Metrics

### A. Background Consistency — CLIP-I (Image Similarity)
* Cosine similarity of CLIP image embeddings between shots sharing the same background symbol.
* Target: Flat consistency curve over 10 shots (Markovian degrades).

### B. Identity Preservation — DINOv2 Similarity
* DINOv2 embedding similarity of the same entity across different shots.
* Target: Multi-Stream Fusion prevents chimera/morphing at entity transitions.
* Future: Add FaceID (InsightFace) for human-face scenarios and VLM (GPT-4o) binary entity presence check.

### C. Prompt Alignment — CLIP-T (Text-Image)
* CLIP text-image cosine similarity between each frame and its keyframe prompt (truncated to 300 chars for CLIP token limit).

---

## 4. Pipelines (5 variants for ablation)

| Label | Description | Implementation |
|-------|-------------|----------------|
| **Ours** | Entity-Decomposed Routing + Multi-Stream Latent Fusion + Bridge Nodes | `generator.py` → `KeyframeGenerator` |
| **StoryDiffusion** | External baseline (consistent self-attention, 25 steps, guidance=5.0) | `storydiff_baseline.py` |
| **Markovian** | Sequential $t-1$ parent only (no graph routing) | `ablation.py` → `MarkovianGenerator` |
| **NoBridge** | Our routing but skip bridge nodes for $D \ge 2$ | `ablation.py` → `NoBridgeGenerator` |
| **GlobalInject** | Global K/V blend (no entity decomposition or spatial routing) | `ablation.py` → `GlobalInjectGenerator` |

---

## 5. Project Structure

```text
src/
 ├── core/
 │   ├── routing.py              # DAG construction, distance metric, 3 routing strategies
 │   │                           #   (RoutingGraph, ReverseRoutingGraph, EntityDecomposedRoutingGraph)
 │   ├── generator.py            # Multi-Stream Latent Fusion, K/V injection, SDXL pipeline
 │   │                           #   (KVStoreAttnProcessor, AttentionControl, KeyframeGenerator)
 │   ├── evaluate_benchmark.py   # Two-phase harness: Phase1=Generate, Phase2=Evaluate+Report
 │   │                           #   (CLIPScorer, DINOScorer, compute_metrics, CSV/Markdown output)
 │   ├── ablation.py             # Ablation generators (Markovian, NoBridge, GlobalInject)
 │   ├── storydiff_baseline.py   # StoryDiffusion baseline implementation
 │   ├── dataset_builder.py      # Procedural MSR-50 template generator
 │   └── prompt_validator.py     # Prompt validation & formatting
tests/
 └── test_router.py              # Unit tests: Forward, Reverse, Entity-Decomposed routing
datasets/
 └── MSR-50/
     ├── MSR-50.json             # Master index (50 scenarios, 500 total shots)
     └── {domain}_{01-10}.json   # Individual scenario files (5 domains × 10 each)
outputs/
 ├── phase3_keyframes/           # Latest keyframe generation results
 └── msr50_eval/                 # Benchmark results (evaluation_results.csv, evaluation_summary.json)
```

---

## 6. Tech Stack

* **Backbone:** SDXL-Turbo (`stabilityai/sdxl-turbo`), fp16, 768×768
* **Identity Injection:** IP-Adapter (`h94/IP-Adapter`, `ip-adapter_sdxl.bin`)
* **K/V Cache:** Custom `KVStoreAttnProcessor` on UNet self-attention (`attn1`) layers
* **Evaluation:** OpenCLIP (`ViT-B-32`), DINOv2 (`dinov2_vitb14`)
* **Video (future):** I2VGen-XL (`ali-vilab/i2vgen-xl`)
* **Optimization:** `torch.float16`, `F.scaled_dot_product_attention`
* **GPU:** `cuda:3` (default)

---

## 7. Execution

```bash
# Quick sanity check (2 scenarios, our pipeline only)
python -m src.core.evaluate_benchmark --max-scenarios 2 --pipelines Ours

# Full MSR-50 benchmark (all 50 scenarios × 5 pipelines)
python -m src.core.evaluate_benchmark

# Evaluate only (skip generation, reuse existing frames)
python -m src.core.evaluate_benchmark --eval-only

# Run routing unit tests
python tests/test_router.py
```

---

## 8. Strict Constraints

### DO NOT:
1. **NO PIXEL COLLAGE:** Do NOT use `PIL.Image.paste` or `ImageChops` to stitch images. Multi-entity synthesis MUST use Latent Noise Fusion only.
2. **NO LINEAR MASKING:** Do NOT use `torch.linspace(0, 1)` for spatial division. MUST use steep sigmoid (`torch.sigmoid(x * steepness)`).
3. **NO HOMOGENEOUS IP-ADAPTER PACKING:** Do NOT pack human faces and abstract objects into the same IP-Adapter. Humans/Animals → IP-Adapter, abstract objects → Text Prompt only.
4. **NO PROMPT OVERLOADING (>77 tokens):** Delegate visual details to IP-Adapter and K/V Cache. Text prompt = "Who, Where, What action" only.
5. **RESPECT MSR-50 TEMPLATE:** Do NOT alter the distance array `[D=-, 1, 1, 2, 0, 1, 3, 1, 1, 0]`. This is a controlled template designed to trigger specific ablation vulnerabilities.

### MUST DO:
1. **VRAM Management:** Include `torch.cuda.empty_cache()` and memory cleanup after every pipeline unload to prevent OOM across 2,500 generations.
2. **Reproducibility:** Use deterministic seeds (`_stable_hash(shot_id)`) for all generation.
3. **Two-phase evaluation:** Generate all frames first (GPU-heavy), then evaluate (CLIP/DINO only). Never run generator and scorer simultaneously.
4. **Resume support:** Skip regeneration if frames already exist on disk.
5. **Pre-flight check:** Before fixing errors, verify the proposed solution does not violate constraints above.
