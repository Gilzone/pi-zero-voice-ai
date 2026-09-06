#!/usr/bin/env python3
"""
Prepares high-quality Reasoning (<think>) and QA data for fine-tuning SmolLM2-360M.
Equips the model with internal step-by-step scratchpad reasoning so it correctly
solves logic traps, riddles, and math while maintaining concise voice outputs.
"""

import json
import os
import random

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "distillation_reasoning_train.jsonl")

# High-impact reasoning pairs with explicit internal <think> chain-of-thought
REASONING_PAIRS = [
    # Logic Traps & Riddles
    (
        "Which is heavier: a pound of feathers or two pounds of gold?",
        "<think>Compare weights: A pound of feathers is 1 lb. Two pounds of gold is 2 lbs. 2 lbs is greater than 1 lb, regardless of the material.</think> Two pounds of gold is heavier because two pounds is greater than one pound."
    ),
    (
        "Which is heavier: a pound of feathers or a pound of bricks?",
        "<think>Both weigh exactly one pound. The materials differ in density and volume, but their mass and weight are equal.</think> They weigh the exact same: one pound."
    ),
    (
        "If today is Sunday, what day was it 4 days ago?",
        "<think>Count backwards from Sunday: 1 day ago is Saturday, 2 days ago is Friday, 3 days ago is Thursday, 4 days ago is Wednesday.</think> Four days ago was Wednesday."
    ),
    (
        "If today is Tuesday, what day will it be in 10 days?",
        "<think>7 days from Tuesday is Tuesday. 10 minus 7 is 3 days. Tuesday plus 3 days (Wednesday, Thursday, Friday) is Friday.</think> In 10 days it will be Friday."
    ),
    (
        "If a rooster lays an egg on top of a barn roof, which way does it roll?",
        "<think>Fact check biological reality: Roosters are male chickens. Only female hens lay eggs. Therefore, no egg was laid.</think> Roosters do not lay eggs; only hens do."
    ),
    (
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "<think>Set up algebraic equation: Let ball = x. Bat = x + 1.00. Total = x + (x + 1.00) = 2x + 1.00 = 1.10. 2x = 0.10. x = 0.05.</think> The ball costs 5 cents ($0.05), and the bat costs $1.05."
    ),
    (
        "If you have 3 apples and you take away 2, how many apples do you have?",
        "<think>Pay attention to the phrasing: You took 2 apples from the group. You now physically have the 2 apples you took.</think> You have 2 apples, because those are the ones you took."
    ),
    (
        "Can a man marry his widow's sister?",
        "<think>Analyze the definition of widow: For a woman to be a widow, her husband must already be dead. A dead man cannot marry anyone.</think> No, because for a woman to be a widow, the man must be dead."
    ),
    (
        "How many months have 28 days?",
        "<think>Every month has at least 28 days. February has 28 or 29, and all other 11 months have 30 or 31 days.</think> All 12 months have at least 28 days."
    ),
    (
        "Some months have 31 days, some have 30. How many have 28?",
        "<think>All twelve months contain at least 28 days.</think> All 12 months have 28 days."
    ),
    (
        "If a train travels at 60 miles per hour, how far does it travel in 45 minutes?",
        "<think>45 minutes is 45/60 = 0.75 hours. Distance = Speed * Time = 60 * 0.75 = 45 miles.</think> It travels 45 miles."
    ),
    (
        "What is 15 percent of 80?",
        "<think>10 percent of 80 is 8. 5 percent is 4. 8 + 4 = 12. Alternatively, 0.15 * 80 = 12.</think> 15 percent of 80 is 12."
    ),
    (
        "What is 25 multiplied by 16?",
        "<think>25 * 16 = 25 * 4 * 4 = 100 * 4 = 400.</think> 25 multiplied by 16 is 400."
    ),
    (
        "What is the next prime number after 29?",
        "<think>Check integers greater than 29: 30 is even (divisible by 2, 3, 5). 31 has no divisors other than 1 and 31, so 31 is prime.</think> The next prime number after 29 is 31."
    ),
    (
        "If you are running a race and you pass the person in second place, what place are you in?",
        "<think>If you pass the person in second place, you take their position. You are now in second place, not first.</think> You are in second place."
    ),

    # Science, Nature & Physics
    (
        "Why is the sky blue?",
        "<think>Sunlight contains all wavelengths. Gas molecules in Earth's atmosphere scatter short-wavelength blue light much more strongly than longer red wavelengths (Rayleigh scattering).</think> The sky appears blue because atmospheric molecules scatter short-wavelength blue sunlight more than other colors via Rayleigh scattering."
    ),
    (
        "Why do leaves change color in autumn?",
        "<think>In autumn, decreasing sunlight causes trees to stop producing green chlorophyll, revealing carotenoid (yellow/orange) and anthocyanin (red) pigments.</think> Trees stop producing green chlorophyll as daylight decreases, unmasking yellow, orange, and red pigments."
    ),
    (
        "How does electricity travel through a wire?",
        "<think>An electrical potential difference induces an electric field that causes free valence electrons in the metal lattice to drift collectively.</think> Electricity flows through the coordinated drift of free electrons through the metal wire driven by an electric field."
    ),
    (
        "What causes ocean tides?",
        "<think>Gravitational attraction of the Moon and Sun creates tidal bulges in Earth's oceans as the planet rotates.</think> Ocean tides are caused by the gravitational pull of the Moon and the Sun on Earth's rotating oceans."
    ),
    (
        "Why is ice less dense than liquid water?",
        "<think>Water molecules form hydrogen bonds that expand into an open hexagonal crystal lattice upon freezing, increasing volume and lowering density.</think> As water freezes, hydrogen bonds force molecules into an open hexagonal lattice that occupies more volume than liquid water."
    ),
    (
        "What is the speed of light in a vacuum?",
        "<think>Standard physical constant c = 299,792,458 meters per second.</think> The speed of light in a vacuum is exactly 299,792,458 meters per second."
    ),
    (
        "How does photosynthesis work?",
        "<think>Chlorophyll absorbs solar energy to convert carbon dioxide and water into glucose and oxygen.</think> Plants use chlorophyll to capture sunlight, turning carbon dioxide and water into energy-rich glucose and oxygen."
    ),
    (
        "Why is the ocean salty?",
        "<think>Rainwater dissolves mineral ions from land rocks, which rivers carry into oceans where evaporation concentrates the dissolved salts.</think> Rain dissolves mineral salts from land rocks and rivers transport them to the ocean, where evaporation leaves the salt behind."
    ),
    (
        "Why does thunder happen after lightning?",
        "<think>Lightning superheats air to 30,000 kelvins creating an explosive shockwave. Light travels at 300,000 km/s, while sound travels at only ~343 m/s.</think> Lightning superheats the air causing an explosive shockwave, and sound travels much slower than the flash of light."
    ),
    (
        "Why doesn't the ocean freeze easily?",
        "<think>Dissolved salt lowers the freezing point of water from 0 degrees Celsius to about -2 degrees Celsius.</think> The high salt concentration in seawater depresses its freezing point to about minus 2 degrees Celsius."
    ),

    # Geography & World Knowledge
    (
        "What is the capital of Australia?",
        "<think>Canberra was selected as the purpose-built capital in 1908 as a compromise between Sydney and Melbourne.</think> The capital of Australia is Canberra."
    ),
    (
        "What is the capital of Canada?",
        "<think>Ottawa was chosen by Queen Victoria in 1857.</think> The capital of Canada is Ottawa."
    ),
    (
        "What is the longest river in the world?",
        "<think>The Nile is traditionally measured at ~6,650 km, while the Amazon is measured around 6,400 to 6,992 km.</think> The Nile River is traditionally recognized as the longest river in the world, spanning about 6,650 kilometers."
    ),
    (
        "What is the deepest point in the Earth's oceans?",
        "<think>Challenger Deep in the Mariana Trench reaches ~10,994 meters below sea level.</think> The deepest point is Challenger Deep in the Mariana Trench, reaching approximately 10,994 meters (36,070 feet) deep."
    ),
    (
        "What is the largest desert in the world?",
        "<think>A desert is defined by low precipitation (<250mm/yr). Antarctica receives very little precipitation and covers 14 million sq km.</think> The Antarctic Polar Desert is the largest desert on Earth, covering roughly 14 million square kilometers."
    ),
    (
        "Which country has the longest coastline in the world?",
        "<think>Canada has vast northern archipelagos and coastlines totaling over 202,080 km.</think> Canada has the longest coastline in the world, spanning over 202,080 kilometers."
    ),
    (
        "In which country is Mount Kilimanjaro located?",
        "<think>Mount Kilimanjaro is a dormant volcano in northeastern Tanzania.</think> Mount Kilimanjaro is located in Tanzania."
    ),
    (
        "What is the most spoken native language in the world?",
        "<think>Mandarin Chinese has over 900 million first-language native speakers.</think> Mandarin Chinese is the most spoken native language by number of first-language speakers."
    ),
    (
        "What is the smallest country in the world by land area?",
        "<think>Vatican City covers roughly 0.49 square kilometers inside Rome, Italy.</think> Vatican City is the smallest country in the world, covering approximately 0.49 square kilometers."
    ),
    (
        "Which planet in our solar system has the most moons?",
        "<think>Saturn has 146 confirmed moons, surpassing Jupiter.</think> Saturn currently has the most confirmed moons in our solar system, with 146."
    ),

    # Everyday Life, Health & Voice Assistance
    (
        "What should I do if a minor burn occurs?",
        "<think>Immediate first aid: cool water for 10-20 mins, do not apply ice or butter, cover loosely.</think> Immediately cool the burn under running cool water for 10 to 20 minutes and loosely cover it with a clean, sterile dressing."
    ),
    (
        "What is the difference between baking soda and baking powder?",
        "<think>Baking soda = pure sodium bicarbonate needing acid. Baking powder = sodium bicarbonate + dry acid salts.</think> Baking soda is pure sodium bicarbonate that requires an acid to activate, while baking powder already contains an acid and only needs moisture and heat."
    ),
    (
        "How do noise-canceling headphones work?",
        "<think>Microphones detect external sound; the speaker produces an anti-phase inverted waveform that cancels out the original wave.</think> They use microphones to capture ambient sound and emit an inverted phase-reversed soundwave that cancels noise through destructive interference."
    ),
    (
        "Why shouldn't you put metal in a microwave?",
        "<think>Microwaves reflect off metals, generating high electric charges that cause sparking and electrical arcing.</think> Metal reflects microwaves and concentrates electrical charges on edges, creating hazardous electrical sparks and fire risks."
    ),
    (
        "How many hours of sleep does an adult need on average?",
        "<think>Medical consensus (CDC, National Sleep Foundation) recommends 7 to 9 hours for healthy adults.</think> Most healthy adults need between 7 and 9 hours of sleep per night for optimal cognitive and physical health."
    )
]

def generate_variations(q, a):
    """Generates prompt phrasings to strengthen conversational flexibility."""
    prefixes = [
        "",
        "Can you tell me, ",
        "Quick question: ",
        "Explain: ",
        "Please answer: "
    ]
    variants = []
    for p in prefixes:
        prompt = p + (q[0].lower() + q[1:] if p and not q.startswith("What") and not q.startswith("Why") and not q.startswith("Which") else q)
        variants.append((prompt.strip(), a))
    return variants

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    all_pairs = []

    print(f"Expanding {len(REASONING_PAIRS)} core reasoning and logic pairs...")
    for q, a in REASONING_PAIRS:
        all_pairs.extend(generate_variations(q, a))

    # Add diverse everyday QA from smoltalk dataset to ensure broad conversational breadth
    try:
        from datasets import load_dataset
        print("Loading additional conversational samples from HuggingFace smoltalk...")
        ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train[:800]")
        for row in ds:
            msgs = row.get("messages", [])
            if len(msgs) >= 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
                u = msgs[0]["content"].strip()
                resp = msgs[1]["content"].strip()
                if 5 < len(resp.split()) <= 40:
                    # Synthesize clean reasoning tag
                    thought = f"<think>Direct answer to user query: {u[:40]}</think>"
                    all_pairs.append((u, f"{thought} {resp}"))
        print(f"Total dataset size after smoltalk merge: {len(all_pairs)}")
    except Exception as e:
        print(f"SmolTalk load notice ({e}). Continuing with core reasoning dataset.")

    random.seed(42)
    random.shuffle(all_pairs)

    print(f"Writing {len(all_pairs)} ChatML formatted examples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for user_q, assistant_a in all_pairs:
            chatml_text = (
                f"<|im_start|>system\n"
                f"You are a concise voice assistant. Think step-by-step inside <think> tags, then answer directly in 1 short sentence.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"{user_q}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"{assistant_a}<|im_end|>"
            )
            f.write(json.dumps({"text": chatml_text}) + "\n")

    print(f"Done! {len(all_pairs)} reasoning examples written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
