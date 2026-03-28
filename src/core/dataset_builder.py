"""
MSR-50 Benchmark Dataset Builder.

Generates 50 standardized 10-shot scenarios across 5 domains for
Multi-Shot Routing evaluation. Each scenario follows the same
D-score transition template to enable cross-scenario analysis.

Domains (10 scenarios each):
  1. Sci-Fi / Cyberpunk
  2. High Fantasy
  3. Modern Realistic
  4. Nature / Animals
  5. Stylized / Animation
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict


# ═══════════════════════════════════════════════════════════════════════
# Standard 10-Shot Transition Template
# ═══════════════════════════════════════════════════════════════════════
#
# S1  (D=-)  : Init — 1 entity, BG_1
# S2  (D=1)  : +entity — 2 entities, BG_1     → gradient chimera test
# S3  (D=1)  : BG change — 2 entities, BG_2
# S4  (D=2)  : -entity + BG change — 1 entity, BG_3  → bridge needed
# S5  (D=0)  : Action change — 1 entity, BG_3        → identity lock
# S6  (D=1)  : Long-range routing — 2 entities, BG_2  → S3 retrieval
# S7  (D=3)  : Extreme swap — new entity, BG_2  → bridge chain
# S8  (D=1)  : +entity — 2 entities (A+C), BG_2  → heterogeneous pair
# S9  (D=1)  : Ultra-long routing — 2 entities (A+B), BG_1  → S2 retrieval
# S10 (D=0)  : Identity lock — 1 entity (A), BG_1  → long-term consistency
#
# Template: entities per shot = [A, AB, AB, A, A, AB, C, AC, AB, A]
# Template: bg per shot       = [D, D,  E,  F, F, E,  E, E,  D,  D]

SHOT_TEMPLATE = [
    {"entities": ["A"],      "bg": "D", "expected_d": -1},
    {"entities": ["A", "B"], "bg": "D", "expected_d": 1},
    {"entities": ["A", "B"], "bg": "E", "expected_d": 1},
    {"entities": ["A"],      "bg": "F", "expected_d": 2},
    {"entities": ["A"],      "bg": "F", "expected_d": 0},
    {"entities": ["A", "B"], "bg": "E", "expected_d": 1},
    {"entities": ["C"],      "bg": "E", "expected_d": 3},
    {"entities": ["A", "C"], "bg": "E", "expected_d": 1},
    {"entities": ["A", "B"], "bg": "D", "expected_d": 1},
    {"entities": ["A"],      "bg": "D", "expected_d": 0},
]


# ═══════════════════════════════════════════════════════════════════════
# Domain Definitions (5 domains × 10 variants each)
# ═══════════════════════════════════════════════════════════════════════

DOMAINS = {
    "scifi": {
        "name": "Sci-Fi / Cyberpunk",
        "entities": [
            # Each tuple: (A, B, C) — three entity definitions per scenario
            (
                "a cyberpunk hacker wearing a neon-blue holographic visor and a long dark trenchcoat, lean athletic build, short messy silver hair, fingerless gloves with circuit patterns, full body, standing, white background",
                "a combat android dog with a sleek metallic silver body and glowing blue optical eyes, armored plating on legs and back, antenna ears, quadruped aggressive stance, full body, white background",
                "a massive armored security mech with a single red optical sensor visor, heavy cannon arms, bulky dark-grey titanium plating, imposing bipedal stance, full body, white background",
            ),
            (
                "a female space marine in white and gold powered armor, short red mohawk hair, plasma rifle across her chest, battle scars on the suit, athletic build, full body, standing, white background",
                "a hovering scout drone with four rotors and a blue scanning beam, chrome shell with orange warning stripes, compact cylindrical body, full body, white background",
                "a towering alien warlord with dark purple skin and four arms, ornate bone-like crown, wearing scale armor made of obsidian plates, full body, standing, white background",
            ),
            (
                "a young cyborg mechanic with a prosthetic left arm made of exposed chrome gears, oil-stained orange jumpsuit, goggles on forehead, messy brown hair, full body, standing, white background",
                "a quadruped cargo bot with six stubby legs, worn green paint with yellow hazard stripes, a large flat cargo platform on top, full body, white background",
                "a sleek assassin android with pure black matte skin, glowing violet eyes, retractable blade arms, featureless face, humanoid athletic build, full body, standing, white background",
            ),
            (
                "a grizzled starship captain in a weathered grey naval uniform, salt-and-pepper beard, cybernetic right eye glowing amber, tall imposing build, full body, standing, white background",
                "a small floating AI companion sphere with a glowing cyan core, translucent blue shell, holographic antenna ring, hovering at chest height, full body, white background",
                "a biomechanical alien creature with elongated limbs, bioluminescent green veins under translucent grey skin, eyeless head with mandibles, full body, standing, white background",
            ),
            (
                "a rebel smuggler girl with twin holstered laser pistols, leather jacket over neon-green tank top, braided black hair with gold beads, confident stance, full body, standing, white background",
                "a modified racing drone bike hovering off the ground, red and black carbon fiber hull, exposed thruster engines, compact sleek design, full body, white background",
                "a corrupt corporate enforcer in a pristine white suit with a chrome mask covering upper face, concealed arm-blade, imposing tall build, full body, standing, white background",
            ),
            (
                "a child prodigy inventor wearing oversized lab goggles, messy auburn hair, a utility belt full of gadgets, patchwork vest over a striped shirt, full body, standing, white background",
                "a loyal robot companion shaped like a large cat with brass plating, green LED eyes, articulated tail, steam-powered joints, full body, white background",
                "a rogue military AI housed in a humanoid frame, gunmetal grey exterior, cracked red visor, exposed wiring on one arm, menacing stance, full body, standing, white background",
            ),
            (
                "a bounty hunter in desert-worn power armor, sand-blasted orange and brown coloring, helmet with T-shaped visor, shoulder-mounted scanner, full body, standing, white background",
                "a trained cyber-falcon with metallic feathers and glowing red eye implants, razor-sharp talons, wingspan visible, perched on handler's arm, full body, white background",
                "a massive sand wurm creature partially emerged from ground, segmented armored body, concentric ring of teeth, dust cloud around it, full body, white background",
            ),
            (
                "a neon-punk street samurai with katana on back, face tattoos glowing cyan, black leather armor with neon piping, muscular build, shaved head, full body, standing, white background",
                "a compact wheeled security turret with twin barrels, matte black with red LED strip, treaded base, sensor dome on top, full body, white background",
                "a genetically modified hulking brute with bulging muscles and grey-green skin, metal jaw implant, chains wrapped around fists, torn pants only, full body, standing, white background",
            ),
            (
                "a diplomatic android with porcelain-white skin and perfectly symmetrical features, wearing a formal navy kimono, silver hair in a bun, serene expression, full body, standing, white background",
                "a small maintenance spider-bot with eight articulated chrome legs, red tool appendages, dome-shaped body with a single camera eye, full body, white background",
                "a void entity manifesting as a shifting humanoid silhouette of deep purple and black, star-like points of light within its form, no distinct features, full body, standing, white background",
            ),
            (
                "a veteran pilot in a worn flight suit with mission patches, aviator shades, salt-and-pepper crew cut, prosthetic right leg, carrying helmet under arm, full body, standing, white background",
                "a bipedal combat mech suit standing at half human height, angular olive-drab armor, twin arm-mounted gatlings, visor slit glowing yellow, full body, white background",
                "a crystalline alien diplomat with transparent body showing internal light structures, geometric head formation, floating crystal fragments orbiting it, full body, standing, white background",
            ),
        ],
        "backgrounds": [
            # Each tuple: (D, E, F) — three background definitions per scenario
            (
                "a rain-soaked neon-lit cyberpunk back alley at night with flickering holographic signs, puddle reflections of pink and blue neon, steam rising from grates, dark moody atmosphere, cinematic, no people",
                "a vast high-tech server core room with towering rows of glowing cyan data columns, holographic displays floating in mid-air, polished reflective floor, cold blue ambient lighting, cinematic, no people",
                "a narrow dark ventilation shaft with exposed cables and conduits, dim red emergency lights, metal grate flooring, claustrophobic tight space with industrial pipes, cinematic, no people",
            ),
            (
                "the bridge of a damaged starship with sparking consoles and flickering emergency lights, large viewport showing a nebula, debris floating in zero-gravity sections, cinematic, no people",
                "a derelict alien temple on a barren moon, massive carved pillars with unknown symbols, bioluminescent moss, shattered dome revealing stars above, cinematic, no people",
                "a cramped escape pod interior with a single viewport, emergency lighting strips, life support readouts, strapped-in seats, tight metallic walls, cinematic, no people",
            ),
            (
                "an underground junkyard workshop lit by hanging industrial lights, mountains of scrap metal and old robot parts, oil puddles on concrete floor, warm amber lighting, cinematic, no people",
                "a sterile white corporate testing facility with observation windows, robotic arms along assembly line, clean fluorescent lighting, warning markers on floor, cinematic, no people",
                "a narrow maintenance tunnel between building walls, exposed pipes dripping water, dim yellow utility lights, graffiti on concrete surfaces, claustrophobic, cinematic, no people",
            ),
            (
                "the vast hangar bay of a space station with docked ships and fuel lines, orange work lights illuminating metal catwalks, viewport showing planet below, cinematic, no people",
                "an alien jungle planet with bioluminescent purple vegetation, giant fern-like trees with glowing sap, misty humid atmosphere, two moons visible, cinematic, no people",
                "inside a decompressing airlock chamber, red warning strobes flashing, frost forming on metal walls, emergency pressure gauge, small window showing space, cinematic, no people",
            ),
            (
                "a bustling neon marketplace in a space station, holographic vendor signs in multiple alien languages, crowded narrow walkways with steam vents, colorful atmospheric lighting, cinematic, no people",
                "a hidden rebel base inside a hollowed-out asteroid, rough stone walls with bolted metal panels, tactical holographic map in center, dim blue and green lighting, cinematic, no people",
                "a cramped smuggler ship cargo hold stacked with crates and contraband, low ceiling, single swinging overhead light, hammock in corner, cinematic, no people",
            ),
            (
                "a Victorian-steampunk rooftop at sunset with brass chimneys and gear-covered towers, airships visible in golden sky, pigeons and steam vents, warm copper lighting, cinematic, no people",
                "a massive clockwork factory interior with enormous rotating gears and conveyor belts, steam pipes overhead, rhythmic mechanical sounds implied by motion blur, cinematic, no people",
                "a narrow cobblestone alley between tall brick buildings, gas lamps flickering, puddles reflecting amber light, distant silhouette of airship, cinematic, no people",
            ),
            (
                "a scorched desert canyon at high noon with heat haze and red rock formations, distant mesa plateaus, cracked earth with sparse dead vegetation, harsh sunlight, cinematic, no people",
                "a fortified desert outpost with sandbag walls and antenna arrays, solar panels on roof, dust devils in background, stark shadows under blazing sun, cinematic, no people",
                "inside a dark cave system with stalactites and glowing mineral veins, underground stream reflecting blue light, narrow passage opening to larger cavern, cinematic, no people",
            ),
            (
                "a rain-drenched mega-city street at night with towering skyscrapers covered in holographic ads, cars with light trails, crowded elevated walkways, vivid neon reflections, cinematic, no people",
                "an illegal underground fighting arena with chain-link cage, harsh spotlights from above, concrete walls covered in graffiti, smoke-filled atmosphere, cinematic, no people",
                "a flooded subway tunnel with emergency lights half-submerged, water reflections on ceiling, abandoned train cars, eerie echo atmosphere, cinematic, no people",
            ),
            (
                "a tranquil Japanese cyber-garden with holographic cherry blossoms, a still koi pond with digital fish, stone lanterns with LED glow, misty dawn atmosphere, cinematic, no people",
                "a high-security vault room with laser grid visible in dim light, circular titanium door, biometric scanners on walls, cold sterile atmosphere, cinematic, no people",
                "a narrow rooftop bridge between two buildings at night, wind turbines spinning, antenna forest silhouetted against city glow, gusty atmosphere, cinematic, no people",
            ),
            (
                "a military aircraft carrier deck at dusk, fighter jets parked with maintenance platforms, orange deck lights, ocean horizon with dramatic clouds, strong wind atmosphere, cinematic, no people",
                "an orbital space station observation deck with panoramic viewport, Earth visible below with city lights, minimalist white interior with floating holographic panels, cinematic, no people",
                "a claustrophobic submarine control room with sonar screens and analog gauges, green ambient lighting, periscope in center, pressurized atmosphere, cinematic, no people",
            ),
        ],
        "actions": [
            ["{A} stands still in the rain, slowly scanning the neon-lit surroundings",
             "{B} trots through the puddles and stops beside {A}",
             "{A} and {B} cautiously advance into the glowing room together",
             "{A} drops to hands and knees and crawls into the shaft alone",
             "{A} crouches and taps rapidly on a holographic hacking interface",
             "{A} steps out and reunites with {B} who runs up to greet",
             "{C} powers up and emerges menacingly from the deep shadows",
             "{A} stands defiantly as {C} locks its sensor onto the hacker",
             "{A} and {B} burst through the door and sprint away together",
             "{A} leans against the wet wall, exhaling with visible relief"],
            ["{A} emerges from the airlock, scanning the damaged bridge warily",
             "{B} hovers through floating debris and circles around {A}",
             "{A} and {B} move cautiously through the alien temple ruins",
             "{A} squeezes alone into the cramped escape pod interior",
             "{A} checks life-support readouts with focused intensity",
             "{A} exits the pod and finds {B} scanning the area nearby",
             "{C} rises from behind the crumbling stone pillars",
             "{A} draws a weapon and faces {C} across the broken floor",
             "{A} and {B} race back toward the damaged starship",
             "{A} collapses into a seat, staring out at the nebula"],
            ["{A} wipes oil from goggles and surveys the cluttered workshop",
             "{B} trundles in on stubby legs, cargo platform rattling",
             "{A} and {B} infiltrate the sterile testing facility together",
             "{A} slips alone into the narrow maintenance tunnel",
             "{A} examines exposed wiring with prosthetic arm clicking softly",
             "{A} climbs back out and finds {B} loaded with salvaged parts",
             "{C} drops silently from the ceiling with blades extended",
             "{A} raises the prosthetic arm defensively against {C}",
             "{A} and {B} crash through the tunnel back toward safety",
             "{A} sits among scrap metal, catching breath under warm lights"],
            ["{A} stands at the railing, gazing down at the planet below",
             "{B} drifts in glowing softly and hovers beside {A}",
             "{A} and {B} push through bioluminescent jungle vegetation",
             "{A} seals the airlock door and waits alone inside",
             "{A} monitors the pressure gauge with jaw clenched",
             "{A} steps out of the airlock to find {B} orbiting nearby",
             "{C} crawls out of the vegetation, bioluminescent veins pulsing",
             "{A} pulls a sidearm and faces {C} across the clearing",
             "{A} and {B} sprint together back across the hangar catwalks",
             "{A} slumps against a fuel line, finally safe"],
            ["{A} walks through the neon marketplace, hand resting on holster",
             "{B} roars up on thrusters and parks hovering beside {A}",
             "{A} and {B} sneak through the rebel base entrance together",
             "{A} ducks alone into the cramped smuggler ship cargo hold",
             "{A} rummages through crates, searching for the right device",
             "{A} exits the hold and meets {B} hovering at the doorway",
             "{C} steps out from behind the tactical map, chrome mask gleaming",
             "{A} quick-draws both pistols and faces {C} across the room",
             "{A} and {B} blast out of the base into the busy marketplace",
             "{A} holsters the pistols and breathes out under the neon signs"],
            ["{A} perches on a brass chimney, adjusting goggles to scan the rooftops",
             "{B} leaps across brass pipes and lands purring beside {A}",
             "{A} and {B} descend into the massive clockwork factory interior",
             "{A} darts alone into the narrow cobblestone alley",
             "{A} tinkers with a gadget pulled from the utility belt",
             "{A} runs back and finds {B} waiting at the factory entrance",
             "{C} bursts through a wall, red visor cracked and wiring sparking",
             "{A} launches a gadget and squares off bravely against {C}",
             "{A} and {B} flee together across the rooftops at sunset",
             "{A} sits on a chimney ledge, dangling feet, mission complete"],
            ["{A} crouches behind a red rock formation, scanning the canyon",
             "{B} swoops in on metallic wings and perches on {A}'s shoulder",
             "{A} and {B} advance together toward the fortified desert outpost",
             "{A} enters the dark cave system alone, stepping carefully",
             "{A} examines glowing mineral veins along the cave wall",
             "{A} emerges from the cave to find {B} circling overhead",
             "{C} erupts from the sand in a massive cloud of dust",
             "{A} raises the scanner and braces against {C}'s thundering advance",
             "{A} and {B} retreat rapidly across the scorched canyon floor",
             "{A} takes shelter under a mesa overhang as dust settles"],
            ["{A} stands under a holographic ad, rain streaming off the katana",
             "{B} rolls in on treaded base, twin barrels tracking for threats",
             "{A} and {B} descend the stairs into the underground arena",
             "{A} wades alone through the flooded subway tunnel",
             "{A} checks reflections in the dark water for any movement",
             "{A} emerges from the tunnel and finds {B} standing guard",
             "{C} smashes through the cage wall, chains dragging behind",
             "{A} draws the katana and faces {C} under the harsh spotlights",
             "{A} and {B} fight their way back to the rain-soaked street",
             "{A} sheathes the katana and stands alone in the neon rain"],
            ["{A} kneels beside the koi pond, observing the digital fish serenely",
             "{B} scuttles across the stone path on eight precise chrome legs",
             "{A} and {B} slip past laser grids into the vault room",
             "{A} climbs alone onto the narrow rooftop bridge between buildings",
             "{A} balances on the bridge, wind tugging at the formal kimono",
             "{A} returns through the vault door to find {B} disabling sensors",
             "{C} materializes from shifting purple-black void energy",
             "{A} assumes a defensive stance as {C} pulses with starlight",
             "{A} and {B} dash together through the tranquil cyber-garden",
             "{A} sits peacefully by the pond, composure fully restored"],
            ["{A} walks across the carrier deck, flight suit whipping in the wind",
             "{B} marches out of the hangar bay in precise lockstep formation",
             "{A} and {B} enter the orbital observation deck together",
             "{A} descends alone into the claustrophobic submarine control room",
             "{A} studies the sonar display intently, tracking an anomaly",
             "{A} climbs out and reunites with {B} on the observation deck",
             "{C} phases through the viewport, crystal fragments refracting light",
             "{A} stands firm and faces {C} with Earth glowing in the background",
             "{A} and {B} evacuate through the station corridor together",
             "{A} stands alone on the carrier deck, watching the sunset horizon"],
        ],
    },

    "fantasy": {
        "name": "High Fantasy",
        "entities": [
            (
                "a brave elven archer with long platinum blonde hair, green leather tunic with leaf patterns, carrying a glowing wooden longbow, lean tall build, pointed ears, full body, standing, white background",
                "a massive stone golem with moss-covered boulder body, glowing blue runic eyes carved into face, tree-trunk-thick arms, towering eight feet tall, full body, standing, white background",
                "a dark necromancer in tattered black robes with skull-topped iron staff, pale gaunt face, sunken glowing purple eyes, skeletal hands, full body, standing, white background",
            ),
            (
                "a dwarven warrior king with braided red beard decorated with gold rings, heavy mithril plate armor, wielding a rune-inscribed war hammer, stocky powerful build, full body, standing, white background",
                "a celestial white stag with crystalline antlers that glow with inner light, ethereal mist surrounding hooves, sapphire blue eyes, majestic tall stance, full body, white background",
                "a fire demon with cracked obsidian skin revealing magma underneath, curling ram horns wreathed in flame, massive bat-like wings folded, towering menacing stance, full body, standing, white background",
            ),
            (
                "a young witch apprentice with wild curly red hair, patched purple robe too large for her, a fat toad sitting on her shoulder, freckled face with crooked hat, full body, standing, white background",
                "a miniature fairy dragon with iridescent butterfly wings, shimmering scales of teal and gold, playful hovering pose, no larger than a cat, full body, white background",
                "a cursed black knight in corroded armor fused to flesh, hollow eye sockets with ghostly green fire, a jagged greatsword covered in rust, imposing heavy stance, full body, standing, white background",
            ),
            (
                "a wandering bard with a lute on his back, feathered wide-brim hat, colorful patchwork cloak, charming smile, lean build with knee-high boots, full body, standing, white background",
                "a loyal timber wolf companion with grey and white fur, intelligent amber eyes, leather harness with small saddlebags, alert standing pose, full body, white background",
                "a troll chieftain with mossy green skin and tusks, bone necklace and fur loincloth, crude wooden club, massively muscular build, full body, standing, white background",
            ),
            (
                "a paladin priestess in gleaming silver armor with golden sun emblems, flowing white cape, short-cropped blonde hair, carrying a radiant mace, full body, standing, white background",
                "a clockwork automaton butler in a tuxedo-styled brass body, monocle-like lens eye, top hat welded on head, jerky articulated movements, full body, standing, white background",
                "a lich queen with a jeweled crown fused to exposed skull, flowing spectral robes of deep violet, chains of soul gems around neck, hovering slightly, full body, white background",
            ),
            (
                "a half-orc ranger with green-tinged skin and small tusks, braided dark hair, leather armor with bone toggles, compound bow on back, scarred muscular build, full body, standing, white background",
                "a trained battle griffon with eagle head and lion body, golden feathers and tawny fur, leather saddle and reins, fierce proud stance, full body, white background",
                "a giant frost spider with ice-blue translucent body, eight crystalline legs, mandibles dripping with frost, web patterns visible, full body, white background",
            ),
            (
                "a gnome alchemist with oversized round spectacles, wild white Einstein hair, brown leather apron covered in potion stains, belt of clinking vials, short stature, full body, standing, white background",
                "a baby dragon hatchling with emerald green scales, stubby wings too small to fly, big curious orange eyes, smoke puffing from nostrils, full body, white background",
                "a shadow assassin wrapped in living darkness, only glowing white eyes visible, twin curved daggers that absorb light, ephemeral wispy form, full body, standing, white background",
            ),
            (
                "a centaur warrior with human torso of a woman with braided auburn hair, horse body of chestnut color, wielding a long spear and round shield, proud stance, full body, standing, white background",
                "a pixie swarm appearing as a cluster of tiny humanoids with dragonfly wings, each emitting soft pastel glow, moving as a collective cloud, full body, white background",
                "an ancient treant with bark-skin face, branch antlers, root-like legs, leaves as hair, moss beard, towering ancient presence, glowing amber sap eyes, full body, standing, white background",
            ),
            (
                "a sea-elf mermaid warrior with turquoise scales on lower body, coral armor on torso, trident in hand, flowing teal hair with shell ornaments, fierce expression, full body, standing, white background",
                "a hermit crab golem with a crystal shell, stone crab body with gem-encrusted pincers, barnacles and coral growing on legs, slow heavy stance, full body, white background",
                "a kraken priest with octopus head and human body in dark ceremonial robes, tentacle beard, staff topped with a glowing pearl, full body, standing, white background",
            ),
            (
                "a tiefling rogue with crimson skin, small curved horns, long black tail, wearing a dark hooded cloak with daggers strapped to thighs, yellow cat-like eyes, full body, standing, white background",
                "a spectral wolf made of blue ethereal energy, translucent body with visible skeleton, glowing blue eyes, chain collar with a broken leash, full body, white background",
                "a vampire lord in an ornate black and red coat with high collar, pale aristocratic features, slicked-back dark hair, bat-wing cape clasp, regal pose, full body, standing, white background",
            ),
        ],
        "backgrounds": [
            (
                "an enchanted glowing forest with giant bioluminescent mushrooms, misty atmosphere, fireflies drifting between ancient oaks, dappled moonlight, cinematic, no people",
                "a crumbling ancient ruined temple with cracked stone pillars, moonlight streaming through collapsed ceiling, overgrown with vines, cinematic, no people",
                "a dark underground crypt lit by green torches, skull carvings on stone walls, cobwebs across sarcophagi, damp musty atmosphere, cinematic, no people",
            ),
            (
                "a grand dwarven throne hall carved inside a mountain, massive stone pillars with gold inlay, lava channels providing warm light, echoing vast space, cinematic, no people",
                "a frozen tundra battlefield with broken weapons in snow, aurora borealis overhead, distant ice mountains, howling wind implied by snow trails, cinematic, no people",
                "inside a volcanic forge with rivers of lava, anvils and chains hanging from ceiling, extreme heat distortion, orange and red glow, cinematic, no people",
            ),
            (
                "a cozy cottage interior with bubbling cauldrons, shelves overflowing with jars of strange ingredients, cats sleeping by fireplace, warm amber firelight, cinematic, no people",
                "a dark thorny bramble labyrinth under a blood-red moon, twisted black branches forming walls, scattered bone fragments, eerie silence, cinematic, no people",
                "a narrow dungeon corridor with iron-barred cells, dripping water from stone ceiling, single torch providing flickering light, rat shadows, cinematic, no people",
            ),
            (
                "a vibrant medieval market square with colorful tent stalls, cobblestone ground, a fountain in the center, hanging lanterns at dusk, warm festive atmosphere, cinematic, no people",
                "the interior of a grand wizard tower library, spiraling shelves of ancient books, floating candles, astrolabe and crystal ball on desk, warm scholarly light, cinematic, no people",
                "a murky swamp with twisted dead trees, bubbling pools of green water, hanging Spanish moss, occasional will-o-wisps, fog-covered, cinematic, no people",
            ),
            (
                "a sunlit cathedral with stained glass windows casting rainbow light, marble pillars, golden altar with divine radiance, soaring vaulted ceiling, cinematic, no people",
                "a necropolis city of the dead stretching to the horizon, crumbling mausoleums and obelisks, perpetual grey overcast sky, ghostly mist between graves, cinematic, no people",
                "a narrow mine shaft with wooden support beams, glittering gem deposits in walls, mine cart tracks, lantern light revealing depth, cinematic, no people",
            ),
            (
                "a dense jungle canopy with massive ancient trees, rope bridges between platforms, colorful birds in branches, shafts of golden sunlight, humid atmosphere, cinematic, no people",
                "a crumbling elven palace overtaken by nature, marble floors cracked with tree roots, shattered crystal chandeliers, bioluminescent ivy on walls, cinematic, no people",
                "a deep ice cavern with frozen waterfalls, translucent blue ice walls, icicle formations reflecting light, cold mist at floor level, cinematic, no people",
            ),
            (
                "an alchemist laboratory with bubbling flasks and copper tubes, smoke and colored vapors, ingredient cabinets, books piled on every surface, warm candlelight, cinematic, no people",
                "a dragon's treasure hoard inside a vast cavern, mountains of gold coins and gems, dragon skulls on walls, warm golden light from the treasure, cinematic, no people",
                "a shadow realm with floating rock platforms, void between islands, purple lightning in a starless sky, gravity-defying waterfalls, cinematic, no people",
            ),
            (
                "a sun-dappled meadow with wildflowers and a babbling brook, distant rolling green hills, white clouds in blue sky, peaceful pastoral atmosphere, cinematic, no people",
                "a mystical standing stone circle on a hilltop at twilight, ancient runes glowing faintly, mist rolling in from surrounding moors, dramatic sky, cinematic, no people",
                "inside a massive hollow tree trunk turned dwelling, spiral staircase carved in wood, mushroom lights, cozy earthy interior, cinematic, no people",
            ),
            (
                "a coastal cliff fortress overlooking stormy seas, waves crashing on rocks below, lightning flashing in dark clouds, sea spray and wind, cinematic, no people",
                "an underwater coral palace with bioluminescent coral architecture, shafts of light from surface above, schools of tropical fish, serene blue-green atmosphere, cinematic, no people",
                "a narrow sea cave with tide pools and barnacle-covered walls, single opening letting in ocean light, echoing water sounds, damp salty atmosphere, cinematic, no people",
            ),
            (
                "a gothic vampire castle courtyard at midnight, stone gargoyles, dead rose garden, full moon casting silver light, iron gate entrance, cinematic, no people",
                "a decaying ballroom with dusty chandelier, cracked mirrors, cobweb-covered furniture, faded velvet curtains, ghostly moonlight through windows, cinematic, no people",
                "a narrow secret passage behind castle walls, torch sconces on stone walls, spider webs, hidden door mechanisms visible, claustrophobic, cinematic, no people",
            ),
        ],
        "actions": [
            ["{A} stands alert at the forest edge, bow drawn and scanning for danger",
             "{B} lumbers through the undergrowth and stops beside {A}",
             "{A} and {B} venture cautiously into the ancient ruins together",
             "{A} squeezes alone into the dark underground crypt passage",
             "{A} kneels to examine glowing runes carved into the stone floor",
             "{A} returns to the temple and finds {B} standing guard faithfully",
             "{C} materializes from swirling dark energy with staff raised",
             "{A} draws the bow and faces {C} across the crumbling hall",
             "{A} and {B} flee together through the moonlit forest",
             "{A} leans against an ancient oak, catching breath in safety"],
            ["{A} surveys the grand throne hall from atop the stone staircase",
             "{B} trots across the frozen ground, antlers glowing softly",
             "{A} and {B} advance together across the wind-swept tundra battlefield",
             "{A} descends alone into the volcanic forge, heat distorting the air",
             "{A} hammers a rune into the anvil with powerful focused strikes",
             "{A} climbs out of the forge and finds {B} waiting in the snow",
             "{C} erupts from the lava in a column of flame and smoke",
             "{A} raises the war hammer and faces {C} with unwavering resolve",
             "{A} and {B} charge together back toward the throne hall",
             "{A} sits heavily on the throne, victorious but exhausted"],
            ["{A} stirs a bubbling cauldron in the cozy cottage, reading a spell",
             "{B} flutters in through the window and perches on {A}'s hat",
             "{A} and {B} tiptoe together into the dark thorny labyrinth",
             "{A} creeps alone down the narrow dungeon corridor",
             "{A} peers through iron bars, whispering an incantation",
             "{A} hurries back to the labyrinth and finds {B} hovering anxiously",
             "{C} strides out of the shadows, ghostly green fire in empty eye sockets",
             "{A} raises the wand and stands bravely before {C}",
             "{A} and {B} dash together back toward the cottage",
             "{A} collapses into the armchair by the fire, toad croaking approval"],
            ["{A} strums the lute by the fountain in the bustling market square",
             "{B} pads through the crowd and sits loyally at {A}'s feet",
             "{A} and {B} climb the wizard tower staircase together",
             "{A} ventures alone into the murky swamp, boots squelching",
             "{A} examines strange glowing will-o-wisps from a safe distance",
             "{A} returns to the tower and finds {B} by the library door",
             "{C} crashes through the undergrowth, tusks gleaming with fury",
             "{A} plants feet firmly and raises a hand to face {C}",
             "{A} and {B} race together through the market at full speed",
             "{A} sits by the fountain, playing a relieved tune on the lute"],
            ["{A} stands in the cathedral, golden sunlight streaming through stained glass",
             "{B} clanks forward on brass joints and bows beside {A}",
             "{A} and {B} descend together into the necropolis city of the dead",
             "{A} enters the narrow mine shaft alone, lantern held high",
             "{A} discovers a vein of rare crystals and examines them carefully",
             "{A} emerges and finds {B} polishing its monocle at the entrance",
             "{C} floats out of a mausoleum, soul gems rattling on spectral chains",
             "{A} raises the radiant mace which blazes with divine light against {C}",
             "{A} and {B} retreat together up the cathedral steps",
             "{A} kneels alone at the golden altar, whispering a prayer of thanks"],
            ["{A} crouches on a jungle platform, scanning the canopy through leaves",
             "{B} glides in and lands beside {A} with a proud screech",
             "{A} and {B} explore the crumbling elven palace interior together",
             "{A} enters the deep ice cavern alone, breath visible in the cold",
             "{A} touches a frozen waterfall, studying the ice formations",
             "{A} returns and finds {B} preening feathers at the palace entrance",
             "{C} descends from a web overhead on crystalline ice-blue legs",
             "{A} nocks an arrow and faces {C} across the frozen chamber",
             "{A} and {B} soar together away through the jungle canopy",
             "{A} rests on a rope bridge platform, watching the sunset"],
            ["{A} adjusts oversized spectacles and examines a bubbling flask carefully",
             "{B} waddles in, puffing tiny smoke rings and chirping excitedly",
             "{A} and {B} venture into the dragon's treasure cavern together",
             "{A} steps alone through a portal into the shadow realm",
             "{A} studies floating rock platforms, calculating distances to jump",
             "{A} leaps back through the portal and finds {B} chewing on gold coins",
             "{C} melts out of the living darkness, twin daggers absorbing all light",
             "{A} hurls a potion and faces {C} amid exploding alchemical sparks",
             "{A} and {B} scramble together back to the safety of the laboratory",
             "{A} slumps at the workbench, {B} curled up asleep on a pile of notes"],
            ["{A} canters through the sun-dappled meadow, spear gleaming",
             "{B} swarm in glowing pastel colors and flutter around {A}'s head",
             "{A} and {B} approach the mystical standing stones at twilight together",
             "{A} squeezes alone into the hollow tree dwelling interior",
             "{A} examines carved wooden artifacts along the spiral staircase",
             "{A} trots back out and finds {B} dancing among the standing stones",
             "{C} awakens with a deep groan, bark-face eyes glowing amber",
             "{A} lowers the spear and faces {C} across the moonlit hilltop",
             "{A} and {B} gallop together across the meadow at full speed",
             "{A} stands alone by the babbling brook, spear planted in the earth"],
            ["{A} stands on the fortress cliff edge, wind whipping teal hair",
             "{B} scuttles sideways along the rocks, crystal shell sparkling",
             "{A} and {B} descend together into the underwater coral palace",
             "{A} swims alone into the narrow sea cave, light fading quickly",
             "{A} examines strange bioluminescent patterns on the cave walls",
             "{A} emerges and finds {B} guarding the palace entrance loyally",
             "{C} rises from the deep water, tentacle beard writhing and staff glowing",
             "{A} grips the trident tightly and faces {C} in the underwater hall",
             "{A} and {B} swim together up toward the stormy surface",
             "{A} stands alone on the cliff, watching waves crash below"],
            ["{A} lurks in the castle courtyard, scanning gargoyles with yellow eyes",
             "{B} materializes from blue ethereal mist and sits beside {A}",
             "{A} and {B} creep together into the dusty decaying ballroom",
             "{A} slips alone into the narrow secret passage behind the walls",
             "{A} examines hidden door mechanisms, tail twitching with focus",
             "{A} emerges and finds {B} pacing the ballroom, chain rattling",
             "{C} descends the grand staircase, bat-wing cape billowing",
             "{A} draws twin daggers and faces {C} under the dusty chandelier",
             "{A} and {B} sprint together through the courtyard under moonlight",
             "{A} perches alone on a gargoyle, watching dawn break over the castle"],
        ],
    },

    "modern": {
        "name": "Modern Realistic",
        "entities": [
            (
                "a young professional woman in a tailored navy blazer and white blouse, shoulder-length dark hair, holding a tablet device, confident posture, full body, standing, white background",
                "a golden retriever therapy dog wearing a blue service vest, friendly alert expression, sitting obediently, well-groomed fluffy coat, full body, white background",
                "a stern security guard in a black uniform with radio earpiece, buzz-cut grey hair, crossing arms, tall imposing build, aviator sunglasses, full body, standing, white background",
            ),
            (
                "a bearded hipster barista with man-bun, flannel shirt rolled to elbows revealing tattoo sleeves, leather apron, carrying a latte art cup, full body, standing, white background",
                "a calico cat with distinctive orange-black-white patches, green eyes, fluffy tail curled upward, sitting with one paw raised, full body, white background",
                "a food inspector in a white lab coat with clipboard, wire-rim glasses, stern expression, latex gloves, official ID badge on lanyard, full body, standing, white background",
            ),
            (
                "a female firefighter in full turnout gear with yellow helmet under arm, soot marks on face, short blonde ponytail, heavy boots, determined expression, full body, standing, white background",
                "a dalmatian firehouse dog with classic black spots on white coat, red collar with brass tag, alert standing pose, well-muscled, full body, white background",
                "a panicked civilian man in business casual holding a baby, loosened tie, disheveled brown hair, wide eyes, protective posture over infant, full body, standing, white background",
            ),
            (
                "a street musician young man with an acoustic guitar strapped on, faded jeans and vintage band t-shirt, curly brown hair, sneakers, relaxed easy smile, full body, standing, white background",
                "a border collie with black and white markings, intelligent brown eyes, red bandana around neck, alert herding stance, full body, white background",
                "a police officer in standard blue uniform with badge and utility belt, short dark hair, professional stern demeanor, hand on radio, full body, standing, white background",
            ),
            (
                "a elderly Japanese grandmother in a floral kimono-style top and dark pants, silver hair in a neat bun, kind wrinkled face, carrying a cloth shopping bag, full body, standing, white background",
                "a small white Shiba Inu dog with curled tail, fox-like face, orange leash, alert curious expression, compact fluffy body, full body, white background",
                "a delivery driver in brown uniform with cap, carrying a large cardboard box, tired but friendly expression, sneakers, full body, standing, white background",
            ),
            (
                "a male surgeon in green scrubs with surgical cap and mask pulled down, stethoscope around neck, kind tired eyes, latex gloves, full body, standing, white background",
                "a grey British Shorthair cat with round orange eyes, plush dense coat, sitting with paws tucked, dignified expression, full body, white background",
                "a hospital administrator woman in power suit with clipboard, reading glasses perched on nose, stern efficient demeanor, heels, full body, standing, white background",
            ),
            (
                "a college student with oversized hoodie and backpack, headphones around neck, messy black hair, holding a coffee cup, casual slouched posture, full body, standing, white background",
                "a corgi puppy with short legs and big ears, tri-color coat, tongue out, playful bouncing pose, tiny but energetic, full body, white background",
                "a university professor in tweed jacket with elbow patches, wire glasses, grey beard, holding a thick book, dignified academic posture, full body, standing, white background",
            ),
            (
                "a female yoga instructor in athletic wear, toned build, dark skin, braided hair, carrying a rolled yoga mat, serene balanced posture, full body, standing, white background",
                "a Maine Coon cat with magnificent flowing tabby fur, lynx-like ear tufts, large plumed tail, regal sitting pose, full body, white background",
                "a construction foreman in orange safety vest and hard hat, holding rolled blueprints, weathered tan face, steel-toed boots, full body, standing, white background",
            ),
            (
                "a teenage skateboarder with baggy jeans and graphic tee, colorful skateboard under arm, beanie hat, band-aid on knee, carefree grin, full body, standing, white background",
                "a German Shepherd police dog in a tactical vest, alert ears, intelligent dark eyes, powerful muscular build, on-duty stance, full body, white background",
                "a park ranger in khaki uniform with wide-brimmed hat, binoculars around neck, sun-weathered friendly face, hiking boots, full body, standing, white background",
            ),
            (
                "a female chef in white double-breasted jacket and tall toque, flour dusted on apron, holding a whisk, determined focused expression, full body, standing, white background",
                "a miniature poodle with an elegant groomed white coat, red bow on head, prancing confident stance, alert dark eyes, full body, white background",
                "a health inspector in business attire with official badge, carrying a tablet for notes, serious scrutinizing expression, full body, standing, white background",
            ),
        ],
        "backgrounds": [
            (
                "a cozy modern coffee shop interior with exposed brick walls, warm pendant lighting, wooden tables, steaming espresso machine, rain visible through large windows, cinematic, no people",
                "a sleek open-plan corporate office with glass partitions, computer monitors on desks, city skyline view from floor-to-ceiling windows, cool white lighting, cinematic, no people",
                "a narrow back-of-house restaurant kitchen with stainless steel counters, hanging pots, steam from cooking, tile floor, cramped busy atmosphere, cinematic, no people",
            ),
            (
                "a trendy artisan bakery with display cases of pastries, chalkboard menu, warm wood and white tile decor, morning sunlight through front window, cinematic, no people",
                "a commercial kitchen during health inspection with bright fluorescent lights, stainless steel everywhere, organized ingredient stations, clipboard on counter, cinematic, no people",
                "a cluttered storage room behind a bakery with flour sacks, stacked boxes, narrow shelves, single bare lightbulb, dusty atmosphere, cinematic, no people",
            ),
            (
                "a city fire station garage with red fire trucks parked, equipment hanging on walls, polished concrete floor, large open bay doors, institutional lighting, cinematic, no people",
                "a smoke-filled apartment building hallway with emergency exit signs, peeling wallpaper, flickering overhead light, fire hose on wall, urgent atmosphere, cinematic, no people",
                "a narrow fire escape stairwell with metal grating steps, brick walls, looking down multiple stories, emergency lighting, cinematic, no people",
            ),
            (
                "a vibrant city park in autumn with golden-leaved trees, paved paths, benches along a pond, joggers in background blur, warm late afternoon sun, cinematic, no people",
                "a busy subway platform with tiled walls and arriving train lights, electronic departure board, yellow safety line, underground fluorescent lighting, cinematic, no people",
                "a dimly lit underground parking garage with concrete pillars, numbered parking spaces, security cameras visible, distant car headlights, cinematic, no people",
            ),
            (
                "a traditional Japanese shopping street with paper lanterns, small storefronts with noren curtains, stone-paved walkway, soft evening light, cinematic, no people",
                "a modern minimalist apartment interior with tatami-style flooring, sliding paper screens, ikebana flower arrangement, natural soft lighting through shoji, cinematic, no people",
                "a narrow alley between traditional buildings with potted plants, hanging laundry above, bicycle parked against wall, quiet afternoon sunlight, cinematic, no people",
            ),
            (
                "a modern hospital lobby with white walls and green potted plants, reception desk with monitors, sanitizer stations, bright clinical lighting, cinematic, no people",
                "a hospital operating room with surgical lights, monitors displaying vitals, sterile blue draping, stainless steel instrument trays, clinical atmosphere, cinematic, no people",
                "a hospital utility corridor with gurney against wall, supply carts, pipe-exposed ceiling, fluorescent tube lighting, institutional feel, cinematic, no people",
            ),
            (
                "a university campus quad with old brick buildings and ivy, students walking in background blur, autumn trees, bronze statue on pedestal, cinematic, no people",
                "a large lecture hall with tiered seating, projector screen at front, wooden desks, institutional fluorescent lighting, academic atmosphere, cinematic, no people",
                "a cramped student dorm room with bunk beds, posters on walls, textbooks and laptop on small desk, warm lamp light, cinematic, no people",
            ),
            (
                "a bright modern yoga studio with bamboo floors and mirror walls, natural light from skylights, plants in corners, meditation cushions, serene atmosphere, cinematic, no people",
                "a bustling urban construction site with steel framework, crane in background, gravel and equipment, orange safety barriers, morning light, cinematic, no people",
                "a narrow building maintenance corridor with exposed pipes and junction boxes, concrete floor, dim industrial lighting, utility access doors, cinematic, no people",
            ),
            (
                "a concrete skatepark with ramps and rails, graffiti-covered walls, late afternoon golden hour light, city skyline in background, urban energy, cinematic, no people",
                "a wooded nature trail in a city park with dappled sunlight, wooden boardwalk through wetlands, bird feeders along path, peaceful green atmosphere, cinematic, no people",
                "a dark underpass tunnel with concrete walls, puddles on ground, distant streetlight at end, echo-chamber atmosphere, urban grit, cinematic, no people",
            ),
            (
                "a high-end restaurant kitchen with copper pots hanging, marble countertops, herb garden windowsill, warm professional lighting, organized mise en place, cinematic, no people",
                "a bustling farmer's market with colorful produce stalls, canvas canopies, morning dew on vegetables, cheerful outdoor atmosphere, cinematic, no people",
                "a restaurant walk-in refrigerator with metal shelving, labeled containers, cold mist visible, industrial stainless steel walls, harsh overhead light, cinematic, no people",
            ),
        ],
        "actions": [
            ["{A} sits at a window table, tapping on the tablet while sipping coffee",
             "{B} trots in wearing the service vest and sits beside {A}",
             "{A} and {B} walk through the glass partitions into the corporate office",
             "{A} pushes through the swinging door into the restaurant kitchen alone",
             "{A} inspects stainless steel counters, making notes on the tablet",
             "{A} exits the kitchen and finds {B} waiting by the door, tail wagging",
             "{C} rounds the corner with arms crossed and a stern expression",
             "{A} straightens up and addresses {C} with professional confidence",
             "{A} and {B} walk briskly together back toward the coffee shop",
             "{A} sits alone at the window table, watching rain streak the glass"],
            ["{A} arranges pastries in the display case with careful precision",
             "{B} leaps onto the counter and watches {A} with bright green eyes",
             "{A} and {B} enter the fluorescent-lit commercial kitchen together",
             "{A} opens the storage room door and steps inside alone",
             "{A} checks expiration dates on flour sacks, brow furrowed",
             "{A} emerges carrying supplies and finds {B} napping on the counter",
             "{C} enters clipboard-first, scanning the premises with scrutiny",
             "{A} turns to face {C} and gestures to the organized workspace",
             "{A} and {B} slip together out the bakery's back entrance",
             "{A} leans against the flour-dusted counter, relieved the inspection passed"],
            ["{A} checks gear by the fire truck, adjusting the helmet strap",
             "{B} stands alert beside the truck, ears pricked forward",
             "{A} and {B} rush together into the smoke-filled hallway",
             "{A} climbs alone down the narrow fire escape stairwell",
             "{A} tests the metal grating for stability, looking down the flights",
             "{A} descends and finds {B} waiting at the bottom, bark echoing",
             "{C} stumbles out of the smoke, clutching the baby protectively",
             "{A} reaches out calmly and guides {C} toward the exit",
             "{A} and {B} escort everyone together down to the station garage",
             "{A} sits on the bumper of the fire truck, wiping soot from the face"],
            ["{A} sits on a park bench, strumming guitar in the golden autumn light",
             "{B} bounds over with a red bandana and drops a stick at {A}'s feet",
             "{A} and {B} descend the subway stairs into the underground platform",
             "{A} walks alone into the dim parking garage, footsteps echoing",
             "{A} checks behind concrete pillars, looking for a parked car",
             "{A} jogs back up and finds {B} sitting patiently by the subway entrance",
             "{C} steps out from behind a pillar in full uniform, hand on radio",
             "{A} pauses and faces {C}, guitar still slung across shoulders",
             "{A} and {B} walk quickly together back through the autumn park",
             "{A} sits alone on the bench, watching leaves drift onto the pond"],
            ["{A} walks slowly down the lantern-lit shopping street, carrying the cloth bag",
             "{B} trots alongside on the orange leash, curly tail wagging",
             "{A} and {B} enter the minimalist apartment through the sliding screen",
             "{A} turns into the narrow alley between buildings alone",
             "{A} pauses to admire potted plants along the quiet alley wall",
             "{A} returns and finds {B} sitting neatly outside the apartment door",
             "{C} arrives with the large cardboard box, slightly out of breath",
             "{A} bows politely and accepts the delivery from {C}",
             "{A} and {B} walk together back down the lamplit shopping street",
             "{A} sits on a stone bench alone, watching the evening sky darken"],
            ["{A} walks through the hospital lobby checking charts on the clipboard",
             "{B} sits regally on a chair in the waiting area, orange eyes blinking",
             "{A} and {B} enter the operating room through the double doors",
             "{A} walks alone down the utility corridor, scrubs swishing quietly",
             "{A} leans against the wall, reviewing patient vitals on a tablet",
             "{A} returns to the OR and finds {B} curled on a supply cart nearby",
             "{C} strides in with heels clicking, adjusting reading glasses",
             "{A} turns to face {C} and presents the surgical report calmly",
             "{A} and {B} walk together back through the hospital lobby",
             "{A} sits alone in the break room, finally taking a long breath"],
            ["{A} walks across the university quad, hoodie pulled up against the wind",
             "{B} bounces alongside on short legs, ears flapping with each stride",
             "{A} and {B} enter the large lecture hall together through the side door",
             "{A} climbs the stairs to the cramped dorm room alone",
             "{A} opens the laptop and starts studying at the small desk",
             "{A} comes back down and finds {B} waiting at the lecture hall entrance",
             "{C} enters the lecture hall adjusting wire glasses, holding a thick book",
             "{A} looks up and meets {C}'s eyes from across the tiered seating",
             "{A} and {B} walk quickly together back across the campus quad",
             "{A} sits alone under the old oak tree, headphones back on"],
            ["{A} flows through a sun salutation pose in the bright yoga studio",
             "{B} lounges on a meditation cushion, watching with half-closed eyes",
             "{A} and {B} walk together into the bustling construction site",
             "{A} enters the building maintenance corridor alone, mat under arm",
             "{A} pauses to stretch in the narrow space, breathing deeply",
             "{A} returns and finds {B} batting at a hard hat left on the ground",
             "{C} walks over in the orange safety vest, rolling up the blueprints",
             "{A} turns and discusses something with {C}, gesturing at the site",
             "{A} and {B} walk together back to the peaceful yoga studio",
             "{A} sits alone in lotus pose, eyes closed, completely centered"],
            ["{A} grinds down a rail, skateboard sparking, in the golden hour light",
             "{B} runs alongside in the tactical vest, matching pace precisely",
             "{A} and {B} walk together onto the wooded nature trail",
             "{A} ducks alone into the dark underpass tunnel, skateboard tucked",
             "{A} examines graffiti on the concrete walls, looking for a tag",
             "{A} emerges from the tunnel and finds {B} waiting at the entrance",
             "{C} appears on the trail in khaki uniform, binoculars around neck",
             "{A} stops and faces {C}, skateboard resting against one hip",
             "{A} and {B} jog together back through the skatepark",
             "{A} sits alone on a ramp edge, watching the sunset over the city"],
            ["{A} plates a dish with precise tweezers under the warm kitchen lights",
             "{B} prances through the kitchen in perfectly groomed fashion",
             "{A} and {B} walk together through the colorful farmer's market",
             "{A} enters the walk-in refrigerator alone, pulling the heavy door shut",
             "{A} checks labeled containers, breath misting in the cold air",
             "{A} exits the fridge and finds {B} sitting daintily by the herb garden",
             "{C} enters with the official badge, already scanning the premises",
             "{A} turns to face {C} and presents the sanitation records confidently",
             "{A} and {B} walk together back to the warm restaurant kitchen",
             "{A} sits alone at the counter, savoring a quiet moment with espresso"],
        ],
    },

    "nature": {
        "name": "Nature / Animals",
        "entities": [
            (
                "a male African lion with a full dark mane, golden tawny fur, muscular build, proud regal stance, intense amber eyes, full body, standing, white background",
                "a spotted cheetah with sleek build and characteristic tear-mark face lines, golden coat with black spots, alert crouching stance, full body, white background",
                "a massive African elephant bull with large tusks, wrinkled grey skin, flared ears, powerful towering stance, dust on hide, full body, standing, white background",
            ),
            (
                "a grey timber wolf with thick winter coat, silver-grey fur with white chest, keen yellow eyes, alert standing pose on all fours, full body, white background",
                "a large brown bear with thick cinnamon-colored fur, powerful build, standing on all fours, small rounded ears, heavy paws, full body, white background",
                "a bald eagle with brilliant white head, dark brown body feathers, sharp yellow beak and talons, wingspan partially open, full body, white background",
            ),
            (
                "a red fox with vivid orange-red fur and white-tipped bushy tail, black leg stockings, pointed ears, intelligent golden eyes, alert standing pose, full body, white background",
                "a snowy owl with pure white plumage and scattered dark barring, large round yellow eyes, soft fluffy feathers, perched alert stance, full body, white background",
                "a large bull moose with massive palmate antlers, dark brown coat, long legs, distinctive bell hanging from throat, imposing tall stance, full body, standing, white background",
            ),
            (
                "a Bengal tiger with vivid orange coat and black stripes, white chest, powerful muscular build, intense green eyes, stalking low stance, full body, white background",
                "a red panda with russet fur and ringed bushy tail, black belly, white face markings, round ears, cute small build perched on hind legs, full body, white background",
                "an Indian rhinoceros with thick grey armored skin, single horn on snout, massive barrel body, folds resembling plate armor, full body, standing, white background",
            ),
            (
                "a snow leopard with thick pale grey spotted fur, long fluffy tail, wide paws, ice-blue eyes, stealthy crouching stance, full body, white background",
                "a Himalayan monal pheasant with iridescent rainbow plumage of green blue and copper, red face, elaborate tail feathers displayed, full body, white background",
                "a yak with long shaggy dark brown coat, curved horns, massive stocky build, frost on fur, heavy breathing stance, full body, standing, white background",
            ),
            (
                "a green sea turtle with patterned brown and green shell, smooth skin with white and green coloring, flippers extended, gentle swimming pose, full body, white background",
                "a bottlenose dolphin with sleek grey body, permanent smile expression, dorsal fin prominent, playful jumping pose, full body, white background",
                "a great white shark with powerful torpedo-shaped grey body, white underbelly, rows of visible teeth, cold dark eyes, swimming pose, full body, white background",
            ),
            (
                "a scarlet macaw with vivid red, blue, and yellow plumage, long tail feathers, curved grey beak, perched with wings slightly spread, full body, white background",
                "a capuchin monkey with brown and cream fur, intelligent dark eyes, long prehensile tail, small hands gripping a branch, curious pose, full body, white background",
                "a black jaguar with barely visible rosette pattern in dark fur, muscular build, green-gold eyes, powerful stalking stance, full body, white background",
            ),
            (
                "an Arctic fox with pure white winter coat, small rounded ears, bushy tail wrapped around body, black nose, compact body, alert standing pose, full body, white background",
                "a walrus with brown wrinkled skin, long ivory tusks, thick whiskers, massive blubbery body, flippers visible, resting pose, full body, white background",
                "a polar bear with thick white fur, black nose and eyes, massive paws, powerful muscular build, walking stance, full body, standing, white background",
            ),
            (
                "a mandrill with vivid blue and red face markings, olive-brown fur, prominent muzzle, strong arms, walking on all fours with tail up, full body, white background",
                "a grey crowned crane with grey body, striking golden crown feathers, red throat patch, long elegant legs, dancing pose, full body, white background",
                "a silverback gorilla with dark fur and silver saddle marking on back, massive muscular build, intelligent brown eyes, knuckle-walking stance, full body, white background",
            ),
            (
                "a red deer stag with magnificent multi-pointed antlers, rich brown coat, white rump patch, proud head-up stance, misty breath, full body, standing, white background",
                "a European badger with black and white striped face, grey body fur, sturdy low build, small eyes, digging-ready stance, full body, white background",
                "a wild boar with coarse dark brown bristly fur, prominent curved tusks, muscular compact body, scarred snout, aggressive low stance, full body, standing, white background",
            ),
        ],
        "backgrounds": [
            (
                "the African savanna at golden hour with acacia trees silhouetted, tall golden grass swaying, distant mountains, warm amber light, cinematic, no people",
                "a rocky kopje outcrop on the savanna with flat warm stones, scattered bushes, panoramic view of plains below, harsh midday sun, cinematic, no people",
                "a dense riverine thicket along a muddy African river, tangled roots and overhanging branches, dappled shade, crocodile-still water nearby, cinematic, no people",
            ),
            (
                "a misty pine forest in early morning with shafts of golden light, moss-covered ground, fern undergrowth, distant mountain visible, cinematic, no people",
                "a rocky mountain stream with clear rushing water over boulders, autumn-colored trees along banks, salmon visible in shallows, cinematic, no people",
                "inside a hollow fallen tree trunk, soft decomposing wood interior, mushrooms growing, filtered green light from outside, cozy den atmosphere, cinematic, no people",
            ),
            (
                "a snow-covered winter forest with bare birch trees, fresh fox tracks in powder snow, soft grey overcast sky, quiet serene atmosphere, cinematic, no people",
                "an open frozen lake under northern lights with green and purple aurora, smooth ice surface reflecting sky, distant tree line, cinematic, no people",
                "a narrow snow-covered animal den entrance among rocks, icicles hanging, warm light visible inside, frost patterns on stone, cinematic, no people",
            ),
            (
                "a lush bamboo forest in Asia with tall green stalks creating natural corridor, filtered sunlight, carpet of fallen leaves, misty humid atmosphere, cinematic, no people",
                "an ancient stone temple courtyard overgrown with jungle vegetation, moss on carved stone, golden light through canopy, exotic flowers, cinematic, no people",
                "a narrow river gorge with steep rock walls, rushing white water below, tropical ferns clinging to rock face, spray mist, cinematic, no people",
            ),
            (
                "a high-altitude Himalayan mountain meadow with wildflowers, distant snow-capped peaks, prayer flags fluttering, thin blue sky, cinematic, no people",
                "a rocky mountain pass with dramatic cliff edges, clouds below the viewpoint, sparse alpine vegetation, harsh wind-sculpted rocks, cinematic, no people",
                "a sheltered mountain cave with view of valley below, smooth wind-worn walls, lichen patterns, snow at entrance, cinematic, no people",
            ),
            (
                "a vibrant coral reef with colorful coral formations and tropical fish, clear turquoise water, shafts of sunlight from surface, marine diversity, cinematic, no people",
                "a dramatic rocky coastline with crashing waves and tide pools, sea spray, barnacle-covered rocks, overcast sky, cinematic, no people",
                "a dark deep-ocean environment with bioluminescent organisms, distant hydrothermal vent glow, particle-filled water, pressure-darkness, cinematic, no people",
            ),
            (
                "a dense Amazon rainforest canopy with enormous trees, hanging vines and bromeliads, filtered green light, cacophony of color, humid misty, cinematic, no people",
                "a jungle river clearing with still brown water reflecting trees, lily pads, colorful butterflies, golden afternoon light, cinematic, no people",
                "a narrow jungle floor trail between massive buttress roots, leaf litter, fungus on fallen logs, dim dappled light, earthy atmosphere, cinematic, no people",
            ),
            (
                "an Arctic tundra landscape with flat snowy terrain stretching to horizon, sparse low vegetation poking through snow, pale winter sun, cinematic, no people",
                "a rocky Arctic coastline with ice floes in dark cold water, snow-covered cliffs, distant glaciers, grey overcast sky, cinematic, no people",
                "inside an ice cave with translucent blue walls, frozen formations, light filtering through ice ceiling, subzero stillness, cinematic, no people",
            ),
            (
                "an African tropical forest clearing with massive fig trees, filtered sunlight creating spotlight patches, lush green undergrowth, humid atmosphere, cinematic, no people",
                "a volcanic landscape with black lava rock and sparse green vegetation, steam vents, dramatic cloudy sky, harsh terrain contrasts, cinematic, no people",
                "a narrow forest stream bed with smooth round stones, overhanging ferns, crystal clear shallow water, cool shade, cinematic, no people",
            ),
            (
                "a Scottish highland moor at dawn with heather-covered rolling hills, morning mist in valleys, distant loch visible, soft golden light, cinematic, no people",
                "a European ancient oak forest with massive gnarled trees, carpet of autumn leaves, soft golden afternoon light, fairy-tale atmosphere, cinematic, no people",
                "a dark badger sett entrance among tree roots, packed earth, paw prints in mud, dim forest light, underground den opening, cinematic, no people",
            ),
        ],
        "actions": [
            ["{A} stands on a rocky outcrop, mane blowing in the warm wind, surveying the golden savanna",
             "{B} slinks through the tall grass and crouches low beside {A}",
             "{A} and {B} climb together onto the sun-warmed kopje rocks",
             "{A} pushes alone through the dense riverine thicket",
             "{A} drinks cautiously from the muddy river, ears pivoting",
             "{A} emerges from the thicket and finds {B} resting in the shade",
             "{C} appears on the horizon, massive silhouette against the setting sun",
             "{A} stands ground and faces {C} across the open savanna",
             "{A} and {B} bound together through the tall golden grass",
             "{A} lies alone under an acacia tree, breathing deeply in the dusk"],
            ["{A} lifts nose to the misty morning air, testing the forest scents",
             "{B} ambles out from behind a pine and sits heavily nearby",
             "{A} and {B} follow the rocky stream together, stepping on boulders",
             "{A} pads alone into the hollow fallen tree trunk",
             "{A} curls up inside the decomposing wood, listening to the forest",
             "{A} emerges and finds {B} fishing in the stream nearby",
             "{C} soars overhead, casting a massive shadow across the clearing",
             "{A} crouches low and watches {C} circling in the mountain sky",
             "{A} and {B} retreat together into the dense pine forest",
             "{A} sits alone on a mossy log, morning mist swirling around"],
            ["{A} trots across fresh powder snow, leaving delicate tracks behind",
             "{B} glides silently down from a birch branch and perches near {A}",
             "{A} and {B} cross the frozen lake together under the aurora borealis",
             "{A} squeezes alone into the snow-covered den among the rocks",
             "{A} rests in the den, warm breath melting icicles at the entrance",
             "{A} emerges and finds {B} watching from a branch, big yellow eyes unblinking",
             "{C} crashes through a snowdrift, antlers towering above the treeline",
             "{A} stands alert and faces {C} across the frozen landscape",
             "{A} and {B} dash together through the birch forest, kicking up snow",
             "{A} curls up alone at the den entrance, snow falling softly"],
            ["{A} stalks silently through the bamboo forest, stripes melting into shadows",
             "{B} scampers along a branch and drops down beside {A}",
             "{A} and {B} explore the ancient temple courtyard together",
             "{A} squeezes alone into the narrow river gorge, water rushing below",
             "{A} balances on the rocks, watching the white water intently",
             "{A} leaps back and finds {B} nibbling bamboo at the temple steps",
             "{C} charges through the vegetation, armored body crashing through ferns",
             "{A} lowers into a crouch and faces {C} across the mossy stones",
             "{A} and {B} retreat together into the thick bamboo forest",
             "{A} lies alone on a sun-warmed temple stone, tail flicking lazily"],
            ["{A} prowls across the alpine meadow, fur blending with the rocky terrain",
             "{B} struts through the wildflowers, iridescent plumage catching the light",
             "{A} and {B} traverse the rocky mountain pass together, clouds below",
             "{A} slinks alone into the sheltered mountain cave",
             "{A} rests in the cave, gazing out at the valley far below",
             "{A} pads back out and finds {B} displaying feathers at the cave mouth",
             "{C} appears at the ridge, shaggy coat frosted with ice, horns lowered",
             "{A} crouches on a rock ledge and faces {C} across the narrow pass",
             "{A} and {B} descend together through the meadow at speed",
             "{A} lies alone on a high ledge, snow-capped peaks glowing at sunset"],
            ["{A} glides gracefully through the coral reef, patterned shell catching light",
             "{B} leaps playfully from the water and splashes near {A}",
             "{A} and {B} swim together along the dramatic rocky coastline",
             "{A} dives alone into the dark deep-ocean environment",
             "{A} drifts near a hydrothermal vent, observing bioluminescent life",
             "{A} surfaces and finds {B} clicking nearby in the shallows",
             "{C} surges up from the deep, rows of teeth gleaming white",
             "{A} turns and faces {C} in the open water, holding position",
             "{A} and {B} swim rapidly together back toward the reef",
             "{A} rests alone in a warm patch of sunlight on the sandy floor"],
            ["{A} spreads vivid red wings on a high branch, calling out across the canopy",
             "{B} swings through the vines and lands on a nearby branch",
             "{A} and {B} descend together to the jungle river clearing",
             "{A} hops alone down the narrow jungle floor trail between roots",
             "{A} pecks at colorful fungi growing on a fallen log",
             "{A} flutters back and finds {B} grooming by the river edge",
             "{C} pads out of the shadows, dark coat barely visible, green-gold eyes glowing",
             "{A} takes flight and circles warily above as {C} prowls below",
             "{A} and {B} flee together through the canopy, branches swaying",
             "{A} perches alone on the highest branch, feathers ruffled but safe"],
            ["{A} stands on the frozen tundra, white coat perfectly camouflaged against the snow",
             "{B} hauls out of the dark water onto an ice floe, tusks gleaming",
             "{A} and {B} move together along the rocky Arctic coastline",
             "{A} enters the ice cave alone, blue translucent walls all around",
             "{A} explores the frozen formations, nose twitching at the stillness",
             "{A} trots back out and finds {B} basking on the sun-warmed rocks",
             "{C} rises on hind legs, massive silhouette against the grey sky",
             "{A} lowers body and faces {C} across the snow-covered ice",
             "{A} and {B} lumber together back across the barren tundra",
             "{A} curls up alone in a snow hollow, bushy tail wrapped tight"],
            ["{A} sits on a sun-dappled branch, vivid face markings catching the light",
             "{B} lands gracefully nearby, golden crown feathers fanning out",
             "{A} and {B} clamber together across the volcanic black lava rock",
             "{A} descends alone into the narrow stream bed among smooth stones",
             "{A} washes hands in the crystal clear water, watching for fish",
             "{A} climbs back up and finds {B} performing an elegant dance display",
             "{C} emerges from the dense vegetation, knuckle-walking with slow power",
             "{A} stands upright and displays teeth, facing {C} across the clearing",
             "{A} and {B} flee together through the tropical forest canopy",
             "{A} sits alone on a high rock, grooming in the golden afternoon light"],
            ["{A} stands on a misty highland moor, antlers silhouetted against the dawn sky",
             "{B} waddles out from behind a heather bush, striped face alert",
             "{A} and {B} walk together through the ancient oak forest",
             "{A} pushes alone into the dark sett entrance among tree roots",
             "{A} pauses in the packed-earth tunnel, listening carefully",
             "{A} backs out and finds {B} sniffing the autumn leaves nearby",
             "{C} charges out of the undergrowth, tusks gleaming, bristles raised",
             "{A} lowers the antlers and faces {C} across the forest clearing",
             "{A} and {B} bolt together across the rolling heather-covered hills",
             "{A} stands alone on the hilltop, mist swirling around, loch visible below"],
        ],
    },

    "stylized": {
        "name": "Stylized / Animation",
        "entities": [
            (
                "a cheerful cartoon robot with a round blue body, antenna with a lightbulb tip, wide friendly LED eyes, roller-wheel feet, waving one claw hand, 3D Pixar style, full body, standing, white background",
                "a fluffy orange cartoon cat with enormous green eyes, striped tabby markings, oversized head on small body, curly tail, mischievous grin, 3D Pixar style, full body, white background",
                "a grumpy cartoon villain mad scientist with wild white hair, oversized purple goggles, lab coat with burn marks, tiny legs under big torso, 3D Pixar style, full body, standing, white background",
            ),
            (
                "a brave cartoon mouse knight in tiny silver armor, holding a needle as sword, determined expression, red plume on helmet, standing on hind legs, 3D Pixar style, full body, white background",
                "a loyal cartoon frog squire with a leaf hat, vest made of woven grass, big bulging kind eyes, carrying a tiny backpack, green skin with spots, 3D Pixar style, full body, white background",
                "a menacing cartoon snake sorcerer with purple scales, tiny wizard hat between eyes, coiled body with floating spell rings, forked tongue, 3D Pixar style, full body, white background",
            ),
            (
                "a tiny cartoon fairy with sparkly pink wings, blue dress made of flower petals, messy blonde hair with daisy crown, round face with freckles, hovering pose, 3D Pixar style, full body, white background",
                "a cartoon ladybug with oversized cute eyes, red shell with black spots, tiny blue shoes on each of six legs, cheerful expression, 3D Pixar style, full body, white background",
                "a cartoon garden gnome come to life with bushy white beard, red pointed hat, chubby cheeks, holding a tiny lantern, waddle-ready stance, 3D Pixar style, full body, standing, white background",
            ),
            (
                "a cartoon space explorer puppy in a tiny astronaut suit, bubble helmet, fluffy brown ears poking out, big excited eyes, wagging tail visible through suit, 3D Pixar style, full body, standing, white background",
                "a cartoon alien blob with translucent green jelly body, three eyes on stalks, tiny tentacle arms, floating slightly off ground, cute confused expression, 3D Pixar style, full body, white background",
                "a cartoon pirate octopus with an eyepatch, tiny tricorn hat, four tentacles holding different weapons, scarred purple body, menacing grin, 3D Pixar style, full body, standing, white background",
            ),
            (
                "a cartoon bear cub wearing a yellow raincoat and red boots, round tummy, small round ears, holding a tiny umbrella, wide innocent eyes, 3D Pixar style, full body, standing, white background",
                "a cartoon baby penguin with fluffy grey down, oversized feet, tiny flippers, curious head tilt, round body, 3D Pixar style, full body, white background",
                "a cartoon storm cloud with an angry face, dark grey puffy body, tiny lightning bolts for arms, rain drops as tears, floating menacingly, 3D Pixar style, full body, white background",
            ),
            (
                "a cartoon fox detective in a brown trenchcoat and fedora, magnifying glass in paw, sharp amber eyes, bushy red tail, sly knowing expression, 3D Pixar style, full body, standing, white background",
                "a cartoon owl assistant with enormous round spectacles, scholarly bowtie, grey feathers, sitting on a stack of tiny books, wise expression, 3D Pixar style, full body, white background",
                "a cartoon rat crime boss in a pinstripe suit, slicked-back fur, gold tooth, cigar in mouth, tiny but menacing, 3D Pixar style, full body, standing, white background",
            ),
            (
                "a cartoon dinosaur kid with green scaly skin and tiny T-Rex arms, oversized head with big toothy grin, wearing a red baseball cap backwards, stubby tail, 3D Pixar style, full body, standing, white background",
                "a cartoon butterfly with rainbow gradient wings, tiny human-like face with button nose, delicate antenna, sparkling trail behind, 3D Pixar style, full body, white background",
                "a cartoon volcano monster with rocky body, lava cracks glowing orange, angry crater head with smoke coming out, stubby stone feet, 3D Pixar style, full body, standing, white background",
            ),
            (
                "a cartoon witch girl with purple hair in twin buns, oversized witch hat, striped stockings, riding a tiny broom, mischievous wink, 3D Pixar style, full body, standing, white background",
                "a cartoon black cat familiar with crescent moon marking on forehead, big yellow eyes, sleek body with curled tail, sitting primly, 3D Pixar style, full body, white background",
                "a cartoon pumpkin-headed scarecrow with stitched smile, straw poking from patched clothes, crow on shoulder, lanky awkward stance, 3D Pixar style, full body, standing, white background",
            ),
            (
                "a cartoon toy soldier with red uniform and tall bearskin hat, stiff wooden limbs, painted rosy cheeks, drum strapped to front, marching pose, 3D Pixar style, full body, standing, white background",
                "a cartoon wind-up ballerina with a key in her back, pink tutu, porcelain-like skin, pointe shoes, graceful pose, 3D Pixar style, full body, white background",
                "a cartoon jack-in-the-box with spring body, jester hat with bells, maniacal grin, popping out of a decorative box, 3D Pixar style, full body, standing, white background",
            ),
            (
                "a cartoon mushroom adventurer with a red-and-white spotted cap as hat, tiny legs in green boots, carrying a twig walking stick, cheerful round face, 3D Pixar style, full body, standing, white background",
                "a cartoon snail companion with a spiral rainbow shell, slimy translucent body, eyestalks with cute eyes, slow but determined expression, 3D Pixar style, full body, white background",
                "a cartoon thorny bramble monster with twisted vine body, glowing red berry eyes, leaf-covered arms reaching outward, rooted feet, menacing, 3D Pixar style, full body, standing, white background",
            ),
        ],
        "backgrounds": [
            (
                "a colorful cartoon laboratory with bubbling beakers, spinning gears, and blinking control panels, polished tiled floor, cheerful warm lighting, 3D Pixar style, cinematic, no people",
                "a cartoon candy factory interior with rivers of chocolate, gumdrop machines, lollipop columns, rainbow conveyor belts, sweet warm lighting, 3D Pixar style, cinematic, no people",
                "a narrow cartoon ventilation duct with visible rivets, tiny cobwebs, a distant light at the end, metallic walls, 3D Pixar style, cinematic, no people",
            ),
            (
                "a cartoon medieval village square with tiny cobblestone paths, mushroom-shaped houses, hanging flower baskets, warm sunset glow, 3D Pixar style, cinematic, no people",
                "a dark cartoon dungeon with stone walls, flickering cartoon torches, puddles on floor, chains hanging from ceiling, spooky but charming, 3D Pixar style, cinematic, no people",
                "a narrow cartoon castle corridor with suits of armor, tapestries, red carpet, candle sconces, slightly spooky but cozy, 3D Pixar style, cinematic, no people",
            ),
            (
                "a magical cartoon flower garden with oversized daisies and tulips, dewdrops sparkling, butterfly-dotted air, small stone path, warm golden light, 3D Pixar style, cinematic, no people",
                "a cartoon mushroom village nestled in tree roots, tiny doors and windows in mushrooms, fairy lights strung between caps, soft green ambient glow, 3D Pixar style, cinematic, no people",
                "a narrow cartoon underground tunnel with root tendrils as ceiling, glowing worm lights, smooth earthen walls, cozy but tight, 3D Pixar style, cinematic, no people",
            ),
            (
                "a cartoon outer space scene with colorful planets, sparkly stars, a ringed planet in background, asteroid belt, vibrant cosmic colors, 3D Pixar style, cinematic, no people",
                "a cartoon alien planet surface with purple sand dunes, two suns in pink sky, crystal rock formations, floating jellyfish creatures in air, 3D Pixar style, cinematic, no people",
                "inside a cartoon spaceship corridor with blinking lights, round porthole windows showing space, curved white walls, futuristic sliding doors, 3D Pixar style, cinematic, no people",
            ),
            (
                "a cartoon rainy city street with puddles reflecting streetlights, colorful umbrellas in distance, old-fashioned lampposts, cobblestone wet sheen, moody but charming, 3D Pixar style, cinematic, no people",
                "a cartoon sunny meadow with rolling green hills, white fluffy clouds, rainbow in distance, wildflowers everywhere, cheerful bright, 3D Pixar style, cinematic, no people",
                "a cartoon dark hollow inside a tree with cozy furniture, tiny windows, acorn cups on table, warm fireplace glow, 3D Pixar style, cinematic, no people",
            ),
            (
                "a cartoon noir city street at night with dramatic shadows, old brick buildings, fire escape ladders, single streetlight cone of light, moody atmosphere, 3D Pixar style, cinematic, no people",
                "a cartoon police evidence room with shelves of labeled boxes, bulletin board with string connections, desk with lamp, cluttered detective atmosphere, 3D Pixar style, cinematic, no people",
                "a narrow cartoon sewer tunnel with arched brick ceiling, shallow water stream, distant manhole light above, echo-chamber feel, 3D Pixar style, cinematic, no people",
            ),
            (
                "a cartoon prehistoric landscape with giant ferns and palm trees, erupting volcano in distance, lava rivers, pterodactyls in sky, vibrant primordial, 3D Pixar style, cinematic, no people",
                "a cartoon crystal cave interior with multicolored gemstone formations, sparkling reflections, underground lake, magical ambient glow, 3D Pixar style, cinematic, no people",
                "a narrow cartoon lava tube tunnel with glowing magma walls, obsidian floor, heat shimmer, tiny stalactites, 3D Pixar style, cinematic, no people",
            ),
            (
                "a cartoon spooky forest at twilight with twisted trees, jack-o-lantern path lights, bats in purple sky, leaf-covered ground, Halloween atmosphere, 3D Pixar style, cinematic, no people",
                "a cartoon witch's cottage interior with bubbling cauldron, spell books, potion bottles on shelves, black cat basket by fire, cozy spooky, 3D Pixar style, cinematic, no people",
                "a narrow cartoon secret passage behind bookshelf, stone steps going down, cobwebs, faint green glow from below, 3D Pixar style, cinematic, no people",
            ),
            (
                "a cartoon toy shop interior with shelves of colorful toys, wooden floor, display cases, warm window light, magical sparkles in air, 3D Pixar style, cinematic, no people",
                "a cartoon child's bedroom at night with star projector on ceiling, toy castle on floor, bed with canopy, soft moonlight through curtains, 3D Pixar style, cinematic, no people",
                "inside a cartoon toy box with jumbled toys, dark except for lid crack of light, cramped colorful space, fabric lining walls, 3D Pixar style, cinematic, no people",
            ),
            (
                "a cartoon enchanted forest clearing with giant toadstools, fairy ring of flowers, soft bioluminescent glow, ancient tree with face, magical atmosphere, 3D Pixar style, cinematic, no people",
                "a cartoon dark swamp with gnarled dead trees, green murky water, floating log, will-o-wisps, eerie fog, 3D Pixar style, cinematic, no people",
                "a narrow cartoon root tunnel underground with earthworm holes in walls, packed earth ceiling, tiny glowing crystals, cozy earthy, 3D Pixar style, cinematic, no people",
            ),
        ],
        "actions": [
            ["{A} blinks LED eyes curiously, scanning the laboratory with antenna buzzing",
             "{B} leaps in with oversized eyes wide and sits beside {A}",
             "{A} and {B} march together into the candy factory, mouths agape",
             "{A} rolls alone into the narrow ventilation duct, antenna scraping the ceiling",
             "{A} discovers a hidden button and presses it with one claw hand",
             "{A} zooms back and finds {B} licking a candy conveyor belt",
             "{C} bursts through the wall in a cloud of purple smoke, cackling",
             "{A} puffs up bravely and rolls toward {C} with claws raised",
             "{A} and {B} run away together, leaving cartoon dust clouds behind",
             "{A} does a happy robot dance alone, lightbulb antenna flashing",
            ],
            ["{A} draws the tiny needle-sword and peers cautiously around the village square",
             "{B} hops in with a cheerful ribbit and salutes {A} with a leaf hat tip",
             "{A} and {B} tiptoe together down the dark dungeon stairs",
             "{A} marches alone into the narrow castle corridor, armor clanking",
             "{A} examines a suspicious tapestry, pushing it aside to reveal a passage",
             "{A} rushes back and finds {B} nervously peeking from behind a torch",
             "{C} slithers around a corner, tiny wizard hat glowing with dark magic",
             "{A} points the needle-sword and faces {C} with knees trembling but resolute",
             "{A} and {B} dash together out of the dungeon into the warm sunset",
             "{A} plants the needle-sword in the cobblestones triumphantly, standing tall"],
            ["{A} flutters through the magical garden, leaving a trail of sparkles",
             "{B} toddles along the stone path on six tiny blue-shoed legs",
             "{A} and {B} fly together toward the glowing mushroom village",
             "{A} squeezes alone into the narrow underground root tunnel",
             "{A} examines glowing worm lights on the earthen ceiling with wonder",
             "{A} zips back and finds {B} admiring a dewdrop on a giant tulip",
             "{C} waddles into the clearing, lantern swinging and eyes narrowed",
             "{A} puffs up tiny wings and hovers defiantly before {C}",
             "{A} and {B} zip together through the flower garden at top speed",
             "{A} rests alone on a daisy petal, catching tiny breath with a smile"],
            ["{A} bounces excitedly inside the spaceship, nose pressed against the porthole",
             "{B} oozes in through a vent and wobbles over to {A} on tentacle arms",
             "{A} and {B} step out onto the alien planet's purple sand together",
             "{A} floats alone into the spaceship corridor, tail wagging in zero gravity",
             "{A} paws at blinking control panels, accidentally activating the alarms",
             "{A} bounces back and finds {B} stuck to the airlock window like jelly",
             "{C} bursts up from the sand, tentacles waving weapons in every direction",
             "{A} barks bravely and faces {C}, bubble helmet fogging up with determination",
             "{A} and {B} tumble together back into the spaceship hatch",
             "{A} curls up alone by the porthole, watching colorful planets drift by"],
            ["{A} splashes through a puddle on the rainy cartoon street, boots squeaking",
             "{B} waddles through the rain on oversized feet, flippers spread for balance",
             "{A} and {B} step together into the warm sunny meadow, blinking at the rainbow",
             "{A} squeezes alone into the dark tree hollow, yellow raincoat glowing",
             "{A} explores the cozy interior, touching tiny acorn cups on the table",
             "{A} pops out and finds {B} sliding belly-first down a wet hill nearby",
             "{C} rumbles overhead, angry face darkening the sunny meadow",
             "{A} opens the tiny umbrella and stands bravely beneath {C}'s thundering",
             "{A} and {B} waddle-run together back to the lamplit street",
             "{A} sits alone under the umbrella, watching the storm pass with a smile"],
            ["{A} adjusts the fedora and examines the noir street through narrowed amber eyes",
             "{B} flutters down from a fire escape and perches on {A}'s shoulder",
             "{A} and {B} enter the evidence room together, scanning the bulletin board",
             "{A} pads alone into the narrow sewer tunnel, tail held high above the water",
             "{A} sniffs at footprints in the shallow water, following the trail",
             "{A} climbs back out and finds {B} organizing tiny evidence notes",
             "{C} emerges from the shadows in a pinstripe suit, gold tooth glinting",
             "{A} raises the magnifying glass and faces {C} under the single streetlight",
             "{A} and {B} sprint together through the noir streets, shadows stretching",
             "{A} sits alone at the detective desk, case file closed, a satisfied smirk"],
            ["{A} stomps through the prehistoric landscape, tiny arms waving at pterodactyls",
             "{B} glides in on rainbow wings and lands on {A}'s baseball cap",
             "{A} and {B} wander together into the sparkling crystal cave",
             "{A} squeezes alone into the narrow lava tube, obsidian crunching underfoot",
             "{A} examines glowing magma behind the tunnel wall with big curious eyes",
             "{A} stomps back out and finds {B} admiring crystal reflections",
             "{C} erupts from the crater head smoking, stubby stone feet shaking the ground",
             "{A} opens the big toothy grin and roars right back at {C}",
             "{A} and {B} scramble together across the fern-covered landscape",
             "{A} sits alone on a warm rock, tiny arms folded, proudly victorious"],
            ["{A} rides the tiny broom low over the spooky forest, cackling with glee",
             "{B} slinks out from behind a twisted tree, crescent moon marking glowing",
             "{A} and {B} creep together into the witch's cottage, cauldron bubbling",
             "{A} slides alone into the secret passage behind the bookshelf",
             "{A} descends the stone steps, wand lighting the way with faint green glow",
             "{A} flies back up and finds {B} batting at a floating spell book",
             "{C} lurches into the clearing, stitched pumpkin grin widening unnervingly",
             "{A} aims the wand and casts a spell at {C}, sparks flying everywhere",
             "{A} and {B} zoom together through the spooky forest on the broom",
             "{A} lands in the cottage, hanging the hat up, cozy fire crackling"],
            ["{A} marches stiffly through the toy shop, drum beating with each step",
             "{B} twirls gracefully beside {A}, key in back slowly winding down",
             "{A} and {B} tiptoe together into the child's moonlit bedroom",
             "{A} topples alone into the dark toy box through the lid crack",
             "{A} navigates between jumbled toys, painted eyes wide in the dark",
             "{A} climbs back out and finds {B} frozen mid-pose on the nightstand",
             "{C} springs up from the decorative box, jester bells jingling maniacally",
             "{A} raises drumsticks and stands at attention facing {C}",
             "{A} and {B} march together back to the warm toy shop shelf",
             "{A} stands alone on the shelf, drum still, painted smile content"],
            ["{A} walks cheerfully through the enchanted clearing, twig walking stick tapping",
             "{B} slides slowly along the path, rainbow shell catching bioluminescent glow",
             "{A} and {B} venture together into the dark cartoon swamp",
             "{A} squeezes alone into the narrow root tunnel underground",
             "{A} examines tiny glowing crystals in the packed earth walls",
             "{A} pops back up and finds {B} stuck on a floating log, eyestalks spinning",
             "{C} erupts from the swamp water, thorny vine arms reaching outward",
             "{A} raises the twig walking stick and faces {C} with determined round face",
             "{A} and {B} hurry together back to the enchanted clearing",
             "{A} sits alone on a giant toadstool, cheerful face beaming at the fairy ring"],
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Motion Prompt Templates
# ═══════════════════════════════════════════════════════════════════════

MOTION_TEMPLATES = [
    # S1: Init — single entity, first BG
    "{entity_A_desc} {action} in {bg_D_desc}, slow camera pan, cinematic, high quality, detailed",
    # S2: +entity
    "{action} in {bg_D_desc}, tracking shot, cinematic, high quality, detailed",
    # S3: BG change
    "{action} in {bg_E_desc}, slow dolly forward, cinematic, high quality, detailed",
    # S4: -entity + BG change (bridge)
    "{entity_A_desc} {action} in {bg_F_desc}, POV tracking shot, cinematic, high quality, detailed",
    # S5: D=0 identity lock
    "{entity_A_desc} {action} in {bg_F_desc}, close-up static shot, cinematic, high quality, detailed",
    # S6: long-range routing
    "{action} in {bg_E_desc}, medium wide shot, cinematic, high quality, detailed",
    # S7: extreme swap
    "{entity_C_desc} {action} in {bg_E_desc}, dramatic low-angle shot, cinematic, high quality, detailed",
    # S8: +entity heterogeneous
    "{action} in {bg_E_desc}, wide static shot, cinematic, high quality, detailed",
    # S9: ultra-long routing
    "{action} in {bg_D_desc}, dynamic tracking shot, cinematic, high quality, detailed",
    # S10: identity lock
    "{entity_A_desc} {action} in {bg_D_desc}, slow push-in shot, cinematic, high quality, detailed",
]

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, watermark, text, static, "
    "ugly, bad anatomy, extra limbs, disfigured"
)


# ═══════════════════════════════════════════════════════════════════════
# Builder
# ═══════════════════════════════════════════════════════════════════════

def _short_desc(full_desc: str) -> str:
    """Extract a clean short identifier from a full entity description.

    Takes everything up to the first comma after skipping 'a/an'.
    e.g. 'a brave elven archer with long platinum blonde hair, green...'
         -> 'brave elven archer'
    """
    # Remove 'a/an' prefix
    text = full_desc.strip()
    for prefix in ("a ", "an "):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break

    # Take up to first comma, then keep just the core noun phrase
    first_part = text.split(",")[0].strip()

    # Further trim: take up to "with/wearing/in" to get the core noun
    for stop_word in [" with ", " wearing ", " in a ", " carrying "]:
        idx = first_part.lower().find(stop_word)
        if idx > 0:
            first_part = first_part[:idx]
            break

    return first_part.strip()


def _clean_entity(desc: str) -> str:
    """Remove anchor-generation suffixes from entity descriptions."""
    for suffix in [", full body, standing, white background",
                   ", full body, white background",
                   ", white background"]:
        desc = desc.replace(suffix, "")
    return desc.strip().rstrip(",")


def _clean_bg(bg: str) -> str:
    """Remove cinematic/no-people suffix from background descriptions."""
    return bg.replace(", cinematic, no people", "").strip().rstrip(",")


def build_scenario(
    scenario_id: str,
    domain: str,
    entities: tuple[str, str, str],
    backgrounds: tuple[str, str, str],
    actions: list[str],
    seed: int = 42,
) -> dict:
    """Build a single 10-shot scenario JSON."""
    entity_A, entity_B, entity_C = entities
    bg_D, bg_E, bg_F = backgrounds

    entity_A_short = _short_desc(entity_A)
    entity_B_short = _short_desc(entity_B)
    entity_C_short = _short_desc(entity_C)

    shots = []
    for i, tmpl in enumerate(SHOT_TEMPLATE):
        shot_id = f"S{i + 1}"
        action = actions[i].format(
            A=entity_A_short, B=entity_B_short, C=entity_C_short,
        )

        # Build keyframe prompt
        ent_list = tmpl["entities"]
        bg_sym = tmpl["bg"]
        bg_map = {"D": bg_D, "E": bg_E, "F": bg_F}
        ent_map = {"A": entity_A, "B": entity_B, "C": entity_C}

        # Build keyframe prompt with clean grammar
        kf_parts = []
        ent_sorted = sorted(ent_list)

        if len(ent_sorted) == 2:
            kf_parts.append(f"{_clean_entity(ent_map[ent_sorted[0]])} standing on the left")
            kf_parts.append(f"{_clean_entity(ent_map[ent_sorted[1]])} standing on the right")
        else:
            kf_parts.append(_clean_entity(ent_map[ent_sorted[0]]))

        kf_parts.append(action)
        bg_short = bg_map[bg_sym].split(",")[0]
        kf_parts.append(f"in {bg_short}")
        kf_parts.append("cinematic still frame, high quality, detailed")
        keyframe_prompt = ", ".join(kf_parts)

        # Build motion prompt
        motion = MOTION_TEMPLATES[i].format(
            entity_A_desc=_clean_entity(entity_A),
            entity_B_desc=_clean_entity(entity_B),
            entity_C_desc=_clean_entity(entity_C),
            bg_D_desc=_clean_bg(bg_D),
            bg_E_desc=_clean_bg(bg_E),
            bg_F_desc=_clean_bg(bg_F),
            action=action,
        )

        shots.append({
            "shot_id": shot_id,
            "target_entities": ent_list,
            "target_bg": bg_sym,
            "expected_d": tmpl["expected_d"],
            "keyframe_prompt": keyframe_prompt,
            "motion_prompt": motion,
        })

    return {
        "scenario_id": scenario_id,
        "domain": domain,
        "entities": {"A": entity_A, "B": entity_B, "C": entity_C},
        "backgrounds": {"D": bg_D, "E": bg_E, "F": bg_F},
        "negative_prompt": NEGATIVE_PROMPT,
        "shots": shots,
    }


def build_msr50(out_dir: Path | str = "datasets/MSR-50") -> list[dict]:
    """Build all 50 scenarios and save as JSON files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_scenarios = []
    scenario_idx = 0

    for domain_key, domain_data in DOMAINS.items():
        domain_name = domain_data["name"]
        entities_list = domain_data["entities"]
        backgrounds_list = domain_data["backgrounds"]
        actions_list = domain_data["actions"]

        for i in range(10):
            scenario_id = f"{domain_key}_{i + 1:02d}"
            scenario = build_scenario(
                scenario_id=scenario_id,
                domain=domain_name,
                entities=entities_list[i],
                backgrounds=backgrounds_list[i],
                actions=actions_list[i],
                seed=42 + scenario_idx,
            )
            all_scenarios.append(scenario)
            scenario_idx += 1

            # Save individual scenario
            fpath = out_dir / f"{scenario_id}.json"
            with open(fpath, "w") as f:
                json.dump(scenario, f, indent=2, ensure_ascii=False)

    # Save combined dataset
    combined_path = out_dir / "MSR-50.json"
    with open(combined_path, "w") as f:
        json.dump({
            "name": "MSR-50: Multi-Shot Routing Benchmark",
            "version": "1.0",
            "description": (
                "50 standardized 10-shot scenarios across 5 domains for "
                "evaluating multi-shot video generation consistency. "
                "Each scenario follows the same D-score transition template."
            ),
            "domains": list(DOMAINS.keys()),
            "num_scenarios": len(all_scenarios),
            "num_shots_per_scenario": 10,
            "total_shots": len(all_scenarios) * 10,
            "transition_template": {
                "S1": "Init (D=-)",
                "S2": "+Entity (D=1) → chimera test",
                "S3": "BG Change (D=1)",
                "S4": "-Entity + BG Change (D=2) → bridge needed",
                "S5": "Identity Lock (D=0)",
                "S6": "Long-Range Routing (D=1)",
                "S7": "Extreme Swap (D=3) → bridge chain",
                "S8": "Heterogeneous Pair (D=1)",
                "S9": "Ultra-Long Routing (D=1)",
                "S10": "Long-Term Consistency (D=0)",
            },
            "scenarios": all_scenarios,
        }, f, indent=2, ensure_ascii=False)

    # Save dataset card
    card_path = out_dir / "README.md"
    with open(card_path, "w") as f:
        f.write("# MSR-50: Multi-Shot Routing Benchmark\n\n")
        f.write("## Overview\n")
        f.write(f"- **50 scenarios** across 5 domains\n")
        f.write(f"- **500 total shots** (50 × 10)\n")
        f.write(f"- Standardized D-score transition template\n\n")
        f.write("## Domains\n")
        for dk, dv in DOMAINS.items():
            f.write(f"- **{dv['name']}** (10 scenarios)\n")
        f.write("\n## Transition Template\n")
        f.write("| Shot | D | Test Point |\n|------|---|---|\n")
        for desc in [
            ("S1", "-", "Init: single entity"),
            ("S2", "1", "+Entity: chimera prevention"),
            ("S3", "1", "BG change: identity preservation"),
            ("S4", "2", "Bridge needed: -entity + BG change"),
            ("S5", "0", "Identity lock: action change only"),
            ("S6", "1", "Long-range routing: non-Markovian parent"),
            ("S7", "3", "Extreme swap: complete entity change → bridge chain"),
            ("S8", "1", "Heterogeneous pair: dissimilar entities"),
            ("S9", "1", "Ultra-long routing: 7-shot parent retrieval"),
            ("S10", "0", "Long-term consistency: round-trip identity"),
        ]:
            f.write(f"| {desc[0]} | {desc[1]} | {desc[2]} |\n")

    print(f"MSR-50 dataset built: {out_dir.resolve()}")
    print(f"  Scenarios: {len(all_scenarios)}")
    print(f"  Total shots: {len(all_scenarios) * 10}")
    print(f"  Combined JSON: {combined_path}")

    return all_scenarios


# ═══════════════════════════════════════════════════════════════════════
# Sample Preview
# ═══════════════════════════════════════════════════════════════════════

def preview_sample(domain_key: str = "fantasy", variant: int = 0):
    """Pretty-print one 10-shot scenario for review."""
    domain = DOMAINS[domain_key]
    scenario = build_scenario(
        scenario_id=f"{domain_key}_{variant + 1:02d}",
        domain=domain["name"],
        entities=domain["entities"][variant],
        backgrounds=domain["backgrounds"][variant],
        actions=domain["actions"][variant],
    )

    W = 80
    print("\n" + "=" * W)
    print(f"  SAMPLE SCENARIO: {scenario['scenario_id']}")
    print(f"  Domain: {scenario['domain']}")
    print("=" * W)

    print(f"\n  {'─' * (W - 4)}")
    print("  ENTITIES:")
    for sym, desc in scenario["entities"].items():
        print(f"    [{sym}] {desc[:90]}...")

    print(f"\n  {'─' * (W - 4)}")
    print("  BACKGROUNDS:")
    for sym, desc in scenario["backgrounds"].items():
        print(f"    [{sym}] {desc[:90]}...")

    print(f"\n  {'─' * (W - 4)}")
    print("  10-SHOT SEQUENCE:")
    print(f"  {'─' * (W - 4)}")

    d_labels = {-1: "ROOT", 0: "D=0 (reuse)", 1: "D=1", 2: "D=2 (bridge)", 3: "D=3 (chain)"}

    for shot in scenario["shots"]:
        sid = shot["shot_id"]
        ents = "+".join(shot["target_entities"])
        bg = shot["target_bg"]
        d = shot["expected_d"]
        d_str = d_labels.get(d, f"D={d}")

        print(f"\n  [{sid}]  Entities: {ents}  |  BG: {bg}  |  {d_str}")
        print(f"  ┌─ Keyframe Prompt:")
        # Word-wrap the prompt at ~72 chars
        kp = shot["keyframe_prompt"]
        while kp:
            print(f"  │  {kp[:72]}")
            kp = kp[72:]
        print(f"  ├─ Motion Prompt:")
        mp = shot["motion_prompt"]
        while mp:
            print(f"  │  {mp[:72]}")
            mp = mp[72:]
        print(f"  └─")

    print(f"\n{'=' * W}")
    print(f"  Negative: {scenario['negative_prompt']}")
    print(f"{'=' * W}\n")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys as _sys

    # If --preview flag, show one sample and exit
    if "--preview" in _sys.argv:
        domain = "fantasy"
        variant = 0
        for i, arg in enumerate(_sys.argv):
            if arg == "--domain" and i + 1 < len(_sys.argv):
                domain = _sys.argv[i + 1]
            if arg == "--variant" and i + 1 < len(_sys.argv):
                variant = int(_sys.argv[i + 1])
        preview_sample(domain, variant)
    else:
        # Preview one sample first
        print("Previewing sample scenario before full build...\n")
        preview_sample("fantasy", 0)

        resp = input("\nProceed with full MSR-50 build? [Y/n] ")
        if resp.strip().lower() not in ("n", "no"):
            build_msr50()
