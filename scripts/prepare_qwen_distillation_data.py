#!/usr/bin/env python3
"""
Prepares high-quality Qwen-2.5 distilled reasoning and QA data for fine-tuning SmolLM2-360M.
Formats all examples into strict ChatML format with concise voice-oriented responses.
"""

import json
import os
import random
import sys

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "distillation_train.jsonl")

# Curated high-impact Qwen-style QA and reasoning samples (factual, zero-fluff, dense)
CORE_SAMPLES = [
    # Science & Nature
    ("Why is the sky blue?", "The sky appears blue because molecules in Earth's atmosphere scatter short-wavelength blue light more than other colors, a phenomenon known as Rayleigh scattering."),
    ("Why do leaves change color in autumn?", "As daylight shortens, trees stop producing chlorophyll, unmasking the carotenoid and anthocyanin pigments that create yellow, orange, and red colors."),
    ("How does electricity travel through a wire?", "Electricity moves via the collective drift of free electrons through the conductive metal lattice when an electric potential difference is applied."),
    ("What causes ocean tides?", "Tides are primarily caused by the gravitational pull of the Moon and the Sun acting on Earth's rotating oceans."),
    ("Why is ice less dense than liquid water?", "As water freezes into ice, hydrogen bonds lock the molecules into a rigid, open hexagonal lattice that occupies more volume than liquid water."),
    ("What is the speed of light in a vacuum?", "The speed of light in a vacuum is exactly 299,792,458 meters per second, or about 300,000 kilometers per second."),
    ("How does photosynthesis work?", "Plants use chlorophyll to absorb sunlight, converting carbon dioxide and water into glucose and oxygen."),
    ("Why is the ocean salty?", "Rainwater erodes mineral salts from rocks on land and rivers wash them into the ocean, where water evaporates and leaves the salt behind."),
    ("What happens during a solar eclipse?", "A solar eclipse occurs when the Moon passes directly between the Earth and the Sun, temporarily blocking the Sun's light from reaching Earth."),
    ("Why does thunder happen after lightning?", "Lightning superheats surrounding air to 30,000 kelvins, causing an explosive shockwave that travels at the speed of sound, which is far slower than the speed of light."),
    
    # Logic, Math & Reasoning
    ("Which is heavier: a pound of feathers or two pounds of gold?", "Two pounds of gold is heavier because two pounds is greater than one pound, regardless of the material."),
    ("If a rooster lays an egg on the roof of a barn, which way does it roll?", "Roosters do not lay eggs; only hens do."),
    ("If today is Sunday, what day was it 4 days ago?", "Four days ago was Wednesday."),
    ("How many centimeters are in 2.5 meters?", "There are 250 centimeters in 2.5 meters."),
    ("What is 15 percent of 80?", "15 percent of 80 is 12."),
    ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?", "The ball costs 5 cents ($0.05), and the bat costs $1.05."),
    ("If you have 3 apples and you take away 2, how many apples do you have?", "You have 2 apples, because you took them."),
    ("Can a plane take off from a conveyor belt moving in the opposite direction?", "Yes, because airplanes generate thrust from their engines pushing against the air, not by driving their wheels against the ground."),
    ("What is the next prime number after 29?", "The next prime number after 29 is 31."),
    ("If a train travels 60 miles per hour, how far does it travel in 45 minutes?", "It travels 45 miles, since 45 minutes is three-quarters of an hour."),
    
    # Geography & World Facts
    ("What is the capital of Australia?", "The capital of Australia is Canberra, not Sydney or Melbourne."),
    ("What is the longest river in the world?", "The Nile River is traditionally considered the longest at about 6,650 kilometers, though some studies argue the Amazon is slightly longer."),
    ("What is the deepest point in the Earth's oceans?", "The deepest point is Challenger Deep in the Mariana Trench, reaching approximately 10,994 meters (36,070 feet) below sea level."),
    ("Which continent has the most countries?", "Africa has the most sovereign countries of any continent, with 54 member states."),
    ("What is the largest desert in the world?", "The Antarctic Polar Desert is the largest desert on Earth, covering roughly 14 million square kilometers."),
    ("What country has the longest coastline in the world?", "Canada has the longest coastline in the world, spanning over 202,080 kilometers."),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
    ("Which planet is closest in size to Earth?", "Venus is the closest in size to Earth, with a diameter about 95% of Earth's."),
    ("In which country is Mount Kilimanjaro located?", "Mount Kilimanjaro is located in Tanzania."),
    ("What is the most spoken native language in the world?", "Mandarin Chinese is the most spoken native language by number of first-language speakers."),

    # Everyday Voice Assistant & Practical Knowledge
    ("What should I do if a burn occurs?", "Immediately cool the burn under cool running tap water for 10 to 20 minutes, avoid ice or butter, and cover loosely with a sterile non-stick dressing."),
    ("How long can cooked rice sit at room temperature before going bad?", "Cooked rice should not sit at room temperature for more than two hours to prevent the growth of Bacillus cereus bacteria."),
    ("What is the difference between baking soda and baking powder?", "Baking soda is pure sodium bicarbonate that requires an acid to activate, while baking powder already contains an acid and only needs moisture and heat."),
    ("How do noise-canceling headphones work?", "They use external microphones to capture ambient sound and emit an inverted soundwave (180 degrees out of phase) to cancel the unwanted noise by destructive interference."),
    ("Why shouldn't you put metal in a microwave?", "Metals reflect microwaves and concentrate electric charges on thin edges or points, which can cause intense electrical arcing and ignite fires."),
    ("What is the primary function of the liver?", "The liver filters toxins from the blood, produces bile for digestion, metabolizes nutrients, and regulates blood clotting."),
    ("Why do we yawn?", "While not fully understood, yawning is thought to help cool the brain and increase alertness by enhancing cerebral blood flow."),
    ("How many hours of sleep does an adult need on average?", "Most healthy adults need between 7 and 9 hours of sleep per night for optimal cognitive and physical health."),
    ("What makes popcorn pop?", "Popcorn kernels contain trapped moisture inside a hard hull; when heated, the water turns to steam, builds pressure, and suddenly pops the starchy interior inside-out."),
    ("What is the purpose of an IP address?", "An IP address serves as a unique identifier for a device on a network, allowing computers to locate and communicate with each other.")
]

def generate_variations(q, a):
    """Generates natural conversational prompt variations to improve robustness."""
    prefixes = [
        "",
        "Can you tell me, ",
        "Quick question: ",
        "Explain briefly: ",
        "What is the reason: "
    ]
    variants = []
    for p in prefixes:
        prompt = p + (q[0].lower() + q[1:] if p and not q.startswith("What") and not q.startswith("Why") else q)
        variants.append((prompt.strip(), a))
    return variants

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    all_pairs = []

    # 1. Expand core high-density pairs
    print(f"Expanding {len(CORE_SAMPLES)} core high-density QA pairs...")
    for q, a in CORE_SAMPLES:
        all_pairs.extend(generate_variations(q, a))

    # 2. Try fetching additional Qwen-2.5 distilled data from Hugging Face if datasets is available
    try:
        from datasets import load_dataset
        print("Attempting to load Magpie-Qwen2.5 / SmolTalk distillation samples...")
        ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train[:1000]")
        for row in ds:
            msgs = row.get("messages", [])
            if len(msgs) >= 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
                u = msgs[0]["content"].strip()
                resp = msgs[1]["content"].strip()
                # Filter for concise answers (voice-friendly, under 50 words)
                if 5 < len(resp.split()) <= 45:
                    all_pairs.append((u, resp))
        print(f"Successfully added samples from SmolTalk. Total dataset size: {len(all_pairs)}")
    except Exception as e:
        print(f"HuggingFace dataset download skipped ({e}). Using expanded core distillation dataset.")

    # 3. Format into strict ChatML strings
    print(f"Writing {len(all_pairs)} ChatML formatted examples to {OUTPUT_FILE}...")
    random.seed(42)
    random.shuffle(all_pairs)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for user_q, assistant_a in all_pairs:
            chatml_text = (
                f"<|im_start|>system\n"
                f"You are a concise voice assistant. Answer directly in 1-2 short sentences without pleasantries.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"{user_q}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"{assistant_a}<|im_end|>"
            )
            f.write(json.dumps({"text": chatml_text}) + "\n")

    print(f"Done! {len(all_pairs)} training examples prepared at: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
