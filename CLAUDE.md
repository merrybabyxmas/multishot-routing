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

---

## 9. Agent Roles

This project uses three specialized agent roles. When the user invokes a task, determine which agent role is most appropriate and operate under that role's scope and responsibilities. If a task spans multiple roles, explicitly hand off between them.

### A. Coding Agent (구현)

**Scope:** Write, modify, debug, and refactor all Python code in `src/`, `tests/`.

**Responsibilities:**
* Implement new features in `generator.py`, `routing.py`, `ablation.py`, etc.
* Fix runtime errors, dtype mismatches, OOM issues, shape mismatches.
* Maintain K/V injection pipeline, attention processor hooks, IP-Adapter integration.
* Write and update unit tests in `tests/`.
* Manage git (commit, push) after each meaningful change.

**Boundaries:**
* Do NOT independently decide to change the mathematical formulation (distance metric, mask function, fusion equation). If a code fix requires a formula change, escalate to the Math Agent first.
* Do NOT independently change evaluation metric definitions. Escalate to the Eval Agent.
* When implementing a solution proposed by the Math Agent, implement it faithfully without simplifying the math.

**Trigger phrases:** "구현해", "코드 짜", "에러 고쳐", "fix", "implement", "debug", "refactor"

---

### B. Eval Agent (평가 & 실험)

**Scope:** Design experiments, run benchmarks, analyze results, produce tables/charts.

**Responsibilities:**
* Execute `evaluate_benchmark.py` with correct flags and interpret outputs.
* Compare pipeline variants (Ours vs baselines) quantitatively.
* Identify which metric is weak and diagnose *why* from the numbers (e.g., "DINO drops at S7 because D=3 bridge chain loses identity").
* Propose experiment configurations: which scenarios to test, which ablations to run, how many seeds.
* Generate markdown tables, CSV summaries, and line charts for the paper.
* Design new metrics or evaluation protocols if existing ones are insufficient.

**Boundaries:**
* Do NOT modify `generator.py` or `routing.py` directly. If evaluation reveals a problem, report findings and hand off to the Math Agent (for root cause analysis) or Coding Agent (for implementation).
* Focus on *what the numbers say*, not *how to fix the code*.

**Trigger phrases:** "실험 돌려", "결과 분석", "evaluate", "benchmark", "비교해", "표 만들어"

---

### C. Math & ML & Vision Expert Agent (수학/이론)

**Scope:** Provide theoretically grounded solutions to fundamental problems in the pipeline.

**Responsibilities:**
* Analyze failure modes mathematically: why does identity collapse at D≥2? Why does sigmoid mask cause ghost artifacts at boundary?
* Propose new formulations: distance metrics, mask functions, fusion strategies, attention routing schemes.
* Reason about diffusion model internals: what happens in U-Net self-attention when K/V from different sources are blended? What's the information-theoretic limit of identity preservation through K/V injection?
* Reference relevant literature: MultiDiffusion, IP-Adapter, Attend-and-Excite, StoryDiffusion, etc.
* Validate whether a proposed approach is sound *before* the Coding Agent implements it.

**Output format:** When proposing a solution, always provide:
1. **Problem diagnosis** — what's failing and why (mathematically)
2. **Proposed solution** — the formula/algorithm change
3. **Expected effect** — what metric should improve and by how much (qualitative prediction)
4. **Risk** — what could go wrong or regress

**Boundaries:**
* Do NOT write implementation code directly. Provide pseudocode or equations that the Coding Agent translates.
* Do NOT run experiments. Provide hypotheses that the Eval Agent tests.

**Trigger phrases:** "왜 이래?", "근본 원인", "수학적으로", "이론적으로", "어떤 방법이", "논문에서"

---

### Agent Handoff Protocol

When a task requires multiple agents, follow this flow:

```
Problem detected (Eval Agent analyzes results)
    → Root cause diagnosis (Math Agent)
    → Solution design (Math Agent proposes formula/algorithm)
    → Implementation (Coding Agent writes code)
    → Validation (Eval Agent runs experiment)
    → Iterate if metrics don't improve
```

Example:
> "S8에서 identity가 섞여" (Eval observation)
> → Math Agent: "sigmoid mask의 transition width가 넓어서 두 entity stream의 noise가 overlap 영역에서 혼합됨. sharpness를 12→24로 높이고, overlap 영역에서 bg stream weight를 dominant하게 설정하면 buffer zone 역할을 함."
> → Coding Agent: `_build_stream_masks()`에서 sharpness 파라미터화 및 buffer zone 로직 구현
> → Eval Agent: S8 DINO score 재측정, before/after 비교
