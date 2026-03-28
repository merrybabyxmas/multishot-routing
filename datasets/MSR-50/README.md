# MSR-50: Multi-Shot Routing Benchmark

## Overview
- **50 scenarios** across 5 domains
- **500 total shots** (50 × 10)
- Standardized D-score transition template

## Domains
- **Sci-Fi / Cyberpunk** (10 scenarios)
- **High Fantasy** (10 scenarios)
- **Modern Realistic** (10 scenarios)
- **Nature / Animals** (10 scenarios)
- **Stylized / Animation** (10 scenarios)

## Transition Template
| Shot | D | Test Point |
|------|---|---|
| S1 | - | Init: single entity |
| S2 | 1 | +Entity: chimera prevention |
| S3 | 1 | BG change: identity preservation |
| S4 | 2 | Bridge needed: -entity + BG change |
| S5 | 0 | Identity lock: action change only |
| S6 | 1 | Long-range routing: non-Markovian parent |
| S7 | 3 | Extreme swap: complete entity change → bridge chain |
| S8 | 1 | Heterogeneous pair: dissimilar entities |
| S9 | 1 | Ultra-long routing: 7-shot parent retrieval |
| S10 | 0 | Long-term consistency: round-trip identity |
