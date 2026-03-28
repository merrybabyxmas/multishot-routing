"""
10-Shot Cyberpunk Scenario — "Neon Heist"

Full prompt set with entity/BG definitions, keyframe prompts, motion prompts,
and routing validation. Designed for NeurIPS-quality ablation + I2V pipeline.
"""

from __future__ import annotations
from src.core.routing import ShotNode


# ═══════════════════════════════════════════════════════════════════════
# Entity & Background Definitions
# ═══════════════════════════════════════════════════════════════════════

ENTITY_PROMPTS = {
    "A": (
        "a cyberpunk hacker wearing a neon-blue holographic visor and a long dark trenchcoat, "
        "lean athletic build, short messy silver hair, fingerless gloves with circuit patterns, "
        "full body, standing, white background"
    ),
    "B": (
        "a combat android dog with a sleek metallic silver body and glowing blue optical eyes, "
        "armored plating on legs and back, antenna ears, quadruped aggressive stance, "
        "full body, white background"
    ),
    "C": (
        "a massive armored security mech with a single red optical sensor visor, "
        "heavy cannon arms, bulky dark-grey titanium plating, imposing bipedal stance, "
        "full body, white background"
    ),
}

BG_PROMPTS = {
    "D": (
        "a rain-soaked neon-lit cyberpunk back alley at night with flickering holographic signs, "
        "puddle reflections of pink and blue neon, steam rising from grates, dark moody atmosphere, "
        "cinematic, no people"
    ),
    "E": (
        "a vast high-tech server core room with towering rows of glowing cyan data columns, "
        "holographic displays floating in mid-air, polished reflective floor, cold blue ambient lighting, "
        "cinematic, no people"
    ),
    "F": (
        "a narrow dark ventilation shaft with exposed cables and conduits, dim red emergency lights, "
        "metal grate flooring, claustrophobic tight space with industrial pipes, "
        "cinematic, no people"
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# 10-Shot Scenario
# ═══════════════════════════════════════════════════════════════════════

def build_scenario() -> list[ShotNode]:
    return [
        ShotNode(
            shot_id="S1", entities={"A"}, bg="D",
            action="hacker stands still in the rain-soaked neon alley, scanning surroundings",
        ),
        ShotNode(
            shot_id="S2", entities={"A", "B"}, bg="D",
            action="android dog trots up and joins the hacker in the neon alley",
        ),
        ShotNode(
            shot_id="S3", entities={"A", "B"}, bg="E",
            action="hacker and android dog infiltrate the server core room together",
        ),
        ShotNode(
            shot_id="S4", entities={"A"}, bg="F",
            action="hacker crawls alone into the narrow ventilation shaft",
        ),
        ShotNode(
            shot_id="S5", entities={"A"}, bg="F",
            action="hacker operates a holographic hacking pad at the end of the vent shaft",
        ),
        ShotNode(
            shot_id="S6", entities={"A", "B"}, bg="E",
            action="hacker exits back into the core room and reunites with the android dog",
        ),
        ShotNode(
            shot_id="S7", entities={"C"}, bg="E",
            action="massive security mech emerges from the shadows in the core room",
        ),
        ShotNode(
            shot_id="S8", entities={"A", "C"}, bg="E",
            action="hacker faces the security mech in a tense standoff in the core room",
        ),
        ShotNode(
            shot_id="S9", entities={"A", "B"}, bg="D",
            action="hacker and android dog escape back to the neon alley in the rain",
        ),
        ShotNode(
            shot_id="S10", entities={"A"}, bg="D",
            action="hacker leans against a wall alone in the neon alley, relieved",
        ),
    ]


SHOT_IDS = [f"S{i}" for i in range(1, 11)]


# ═══════════════════════════════════════════════════════════════════════
# Motion Prompts (I2V — action-enhanced for video generation)
# ═══════════════════════════════════════════════════════════════════════

MOTION_PROMPTS = {
    "S1": (
        "A cyberpunk hacker with a neon-blue visor and dark trenchcoat stands in a rain-soaked "
        "neon back alley at night, slowly looking around and scanning the environment, rain drops "
        "falling on his shoulders, neon reflections shimmering in puddles, atmospheric steam rising, "
        "static wide shot, cinematic, high quality, detailed"
    ),
    "S2": (
        "A sleek metallic combat android dog with glowing blue eyes trots forward through the rain "
        "and stops next to a cyberpunk hacker in a neon-lit back alley, the dog's metal paws "
        "splashing in neon-colored puddles, both figures illuminated by flickering holographic signs, "
        "tracking shot, cinematic, high quality, detailed"
    ),
    "S3": (
        "A cyberpunk hacker and a combat android dog cautiously walk forward together into a vast "
        "high-tech server core room, holographic displays floating around them, their reflections "
        "visible on the polished floor, cold blue lighting casting long shadows, "
        "slow dolly forward, cinematic, high quality, detailed"
    ),
    "S4": (
        "A cyberpunk hacker in a dark trenchcoat crawls forward on hands and knees through a narrow "
        "dark ventilation shaft, red emergency lights flickering, exposed cables brushing against "
        "his coat, the tight claustrophobic space barely fitting his body, "
        "POV tracking shot, cinematic, high quality, detailed"
    ),
    "S5": (
        "A cyberpunk hacker crouches at the end of a ventilation shaft, fingers rapidly tapping on "
        "a holographic hacking pad that projects blue light onto his face, data streams scrolling "
        "across the holographic display, dim red emergency lights pulsing in the background, "
        "close-up static shot, cinematic, high quality, detailed"
    ),
    "S6": (
        "A cyberpunk hacker steps out from a dark opening back into a vast server core room, "
        "a metallic combat android dog runs up to greet him with its tail antenna wagging, "
        "holographic displays illuminate both figures, blue ambient light reflecting off metal surfaces, "
        "medium wide shot, cinematic, high quality, detailed"
    ),
    "S7": (
        "A massive armored security mech with a glowing red optical sensor slowly emerges from the "
        "shadows in a high-tech server core room, heavy footsteps causing vibrations, its cannon arms "
        "powering up with orange energy, holographic alarms activating around it, "
        "dramatic low-angle shot, cinematic, high quality, detailed"
    ),
    "S8": (
        "A cyberpunk hacker stands defiantly facing a massive armored security mech in a server core "
        "room, tension visible in their stances, the mech's red sensor locked onto the hacker, "
        "holographic displays flickering between them, dramatic blue and red lighting contrast, "
        "wide static shot, cinematic, high quality, detailed"
    ),
    "S9": (
        "A cyberpunk hacker and a combat android dog burst through a door and run together through "
        "a rain-soaked neon back alley at night, splashing through puddles, neon signs blurring past, "
        "the dog sprinting alongside the hacker, rain streaking through the air, "
        "dynamic tracking shot, cinematic, high quality, detailed"
    ),
    "S10": (
        "A cyberpunk hacker leans back against a wet brick wall in a neon-lit back alley, exhaling "
        "with visible breath in the cold air, rain gently falling, neon reflections playing across "
        "his visor, a moment of quiet relief after the mission, puddles at his feet, "
        "slow push-in shot, cinematic, high quality, detailed"
    ),
}

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, watermark, text, static, "
    "ugly, bad anatomy, extra limbs, disfigured"
)
