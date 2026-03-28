# Prompt Design Specification — 10-Shot Cyberpunk Scenario

## 1. Design Principles

Every prompt in this scenario must satisfy **all** of the following:

### 1.1 Structural Requirements
- **Minimum length**: 80+ characters for entity prompts, 60+ for BG prompts, 120+ for shot motion prompts
- **Entity prompts**: Include physical appearance (2+ attributes), posture, attire, and "white background" for anchor generation
- **BG prompts**: Include lighting, atmosphere, spatial features, "cinematic, no people"
- **Shot prompts (keyframe)**: Entity descriptions + spatial anchoring (left/right for 2-entity) + background + action + quality tags
- **Motion prompts (I2V)**: Clear action verb + direction of movement + camera instruction + atmospheric detail

### 1.2 Consistency Rules
- Entity A must ALWAYS be described with the same core attributes (cyberpunk hacker, neon visor, dark trenchcoat)
- Entity B must ALWAYS be described with the same core attributes (combat android dog, metallic, glowing blue eyes)
- Entity C must ALWAYS be described with the same core attributes (massive armored security mech, red optical sensor)
- Background keywords must match the BG symbol exactly

### 1.3 Quality Tags
- All prompts end with: "cinematic, high quality, detailed, 8k"
- Negative prompt: "blurry, low quality, distorted, deformed, watermark, text, static"

## 2. Entity Definitions

| Symbol | Name | Core Prompt |
|--------|------|-------------|
| A | Hacker | "a cyberpunk hacker with a neon-blue visor and dark trenchcoat, lean build, standing, full body, white background" |
| B | Android Dog | "a combat android dog with sleek metallic silver body and glowing blue eyes, quadruped stance, full body, white background" |
| C | Security Mech | "a massive armored security mech with a red optical sensor and heavy cannon arms, imposing, full body, white background" |

## 3. Background Definitions

| Symbol | Name | Core Prompt |
|--------|------|-------------|
| D | Neon Alley | "a rain-soaked neon-lit back alley at night with holographic signs and puddle reflections, cyberpunk, cinematic, no people" |
| E | Core Room | "a vast high-tech server core room with towering glowing data columns and holographic displays, cyberpunk, cinematic, no people" |
| F | Vent Shaft | "a narrow dark ventilation shaft with exposed cables and dim red emergency lights, claustrophobic, cinematic, no people" |

## 4. Shot Sequence & Routing Validation

| Shot | Entities | BG | Expected D | Parent (Ours) | Validation Point |
|------|----------|----|-----------|----------------|-----------------|
| S1 | {A} | D | - | ROOT | Init: hacker alone in alley |
| S2 | {A,B} | D | 1 | S1 | +entity: chimera prevention (human + dog) |
| S3 | {A,B} | E | 1 | S2 | BG change: both entities preserved |
| S4 | {A} | F | 2 | S3→bridge | Bridge: remove B + change BG simultaneously |
| S5 | {A} | F | 0 | S4 | D=0 reuse: action change, identity lock |
| S6 | {A,B} | E | 1 | S3 | Routing: long-range parent retrieval to S3 |
| S7 | {C} | E | 3 | S6→bridges | Extreme swap: remove A,B + add C via bridge chain |
| S8 | {A,C} | E | 1 | S7 | +entity: human + mech chimera prevention |
| S9 | {A,B} | D | 1 | S2 | Ultra-long routing: retrieve S2 across 7 shots |
| S10 | {A} | D | 1 | S9 or S1 | -entity: S1 identity preservation after 10 shots |

## 5. Motion Prompt Design (I2V)

Motion prompts must contain:
1. **Subject**: who is doing the action
2. **Action verb**: specific, directional (walking forward, looking around, descending)
3. **Manner**: speed/intensity (slowly, cautiously, confidently)
4. **Environment interaction**: rain drops, reflections, sparks, glow
5. **Camera hint**: optional (static wide shot, slow pan, tracking shot)

## 6. Verification Checklist

- [ ] All entity prompts >= 80 chars
- [ ] All BG prompts >= 60 chars
- [ ] All shot prompts >= 120 chars
- [ ] All motion prompts >= 150 chars
- [ ] 2-entity shots include "standing on the left/right" spatial anchoring
- [ ] Entity core attributes are consistent across all prompts
- [ ] No prompt references entities not in that shot's entity set
- [ ] Routing distance D matches expected values
- [ ] Bridge injection triggers correctly for D>=2
- [ ] Motion prompts contain action verb + direction + manner
