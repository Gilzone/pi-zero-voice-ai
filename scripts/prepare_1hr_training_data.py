#!/usr/bin/env python3
"""
Prepares a comprehensive ~5,000 sample high-density knowledge & reasoning
dataset for 1-hour training of SmolLM2-360M.
Combines:
1. Direct factual world knowledge (superlatives, records, geography, science, tech)
2. Filtered openhermes-100k facts and Q&A
3. Everyday conversational turns from smoltalk
4. Step-by-step logic, math & reasoning pairs
"""

import json
import os
import random
import re
from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(DATA_DIR, "distillation_v2_1hr.jsonl")

# 1. Core Ground-Truth World Knowledge & Superlatives
FACTUAL_QA = [
    # Superlatives & Animal Kingdom
    ("What is the fastest animal in the world?", "The peregrine falcon is the fastest animal in the world, reaching dive speeds over 240 mph, while the cheetah is the fastest land animal at up to 70 mph."),
    ("What is the fastest land animal?", "The cheetah is the fastest land animal, capable of running up to 70 miles per hour in short sprints."),
    ("What is the fastest bird in the world?", "The peregrine falcon is the fastest bird, reaching speeds over 240 miles per hour when diving for prey."),
    ("What is the fastest fish in the ocean?", "The sailfish is considered the fastest fish in the ocean, clocking speeds up to 68 miles per hour."),
    ("What is the biggest animal in the world?", "The blue whale is the biggest animal on Earth, growing up to 100 feet long and weighing up to 200 tons."),
    ("What is the largest land animal?", "The African bush elephant is the largest living land animal, weighing up to 11 tons."),
    ("What is the tallest animal in the world?", "The giraffe is the tallest living animal, reaching heights of up to 19 feet."),
    ("What is the loudest animal in the world?", "The sperm whale is the loudest animal on Earth, producing clicks that reach 230 decibels."),
    ("What is the most venomous snake?", "The inland taipan of Australia is the most venomous snake in the world."),
    ("What is the longest living animal?", "The Greenland shark is the longest-living vertebrate, with an estimated lifespan of up to 400 years."),
    ("What is the smallest bird in the world?", "The bee hummingbird of Cuba is the smallest bird in the world, measuring just over two inches long."),
    ("Why do plants have green leaves?", "Plants have green leaves because of chlorophyll, which absorbs blue and red wavelengths of light for photosynthesis while reflecting green light."),
    ("Why do humans grow hair?", "Humans grow hair to regulate body temperature, protect the skin from UV radiation and debris, and provide sensory feedback."),
    ("Why is the sky blue?", "The sky appears blue because molecules in Earth's atmosphere scatter short blue wavelengths of sunlight more than other colors, a process known as Rayleigh scattering."),
    ("Why is the ocean salty?", "The ocean is salty because rainwater erodes rocks on land and washes mineral salts, primarily sodium chloride, into rivers and oceans."),
    ("Why do we have seasons?", "Earth has seasons because its axis of rotation is tilted relative to its orbital plane around the Sun."),

    # World Geography & Capitals
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Australia?", "The capital of Australia is Canberra."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    ("What is the capital of Spain?", "The capital of Spain is Madrid."),
    ("What is the capital of the United Kingdom?", "The capital of the United Kingdom is London."),
    ("What is the capital of the United States?", "The capital of the United States is Washington, D.C."),
    ("What is the capital of Brazil?", "The capital of Brazil is Brasilia."),
    ("What is the capital of Egypt?", "The capital of Egypt is Cairo."),
    ("What is the capital of China?", "The capital of China is Beijing."),
    ("What is the capital of India?", "The capital of India is New Delhi."),
    ("What is the capital of Russia?", "The capital of Russia is Moscow."),
    ("What is the capital of Mexico?", "The capital of Mexico is Mexico City."),
    ("What is the capital of South Africa?", "South Africa has three capital cities: Pretoria (administrative), Cape Town (legislative), and Bloemfontein (judicial)."),
    ("What is the highest mountain in the world?", "Mount Everest is the highest mountain above sea level, reaching an elevation of 29,032 feet (8,849 meters) in the Himalayas."),
    ("What is the deepest place in the ocean?", "The Mariana Trench's Challenger Deep in the western Pacific Ocean is the deepest point on Earth, reaching nearly 36,000 feet (11,000 meters) deep."),
    ("What is the longest river in the world?", "The Nile River in Africa is widely recognized as the longest river in the world, spanning about 4,132 miles."),
    ("What is the largest ocean in the world?", "The Pacific Ocean is the largest ocean on Earth, covering more than 60 million square miles."),
    ("What is the largest desert in the world?", "The Antarctic Desert is the largest desert on Earth, while the Sahara is the largest hot desert."),

    # Physical Science & Chemistry
    ("What is the speed of light?", "The speed of light in a vacuum is approximately 186,282 miles per second, or roughly 300,000 kilometers per second."),
    ("What is the speed of sound?", "The speed of sound in dry air at room temperature is approximately 767 miles per hour, or 343 meters per second."),
    ("What is the chemical formula for water?", "The chemical formula for water is H2O, meaning two hydrogen atoms bonded to one oxygen atom."),
    ("What is the chemical formula for table salt?", "The chemical formula for table salt is NaCl, sodium chloride."),
    ("What is the atomic number of carbon?", "The atomic number of carbon is 6."),
    ("What is the boiling point of water?", "Water boils at 100 degrees Celsius, which is 212 degrees Fahrenheit at standard sea level pressure."),
    ("What is the freezing point of water?", "Water freezes at 0 degrees Celsius, which is 32 degrees Fahrenheit."),
    ("What is the closest planet to the Sun?", "Mercury is the closest planet to the Sun in our solar system."),
    ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
    ("How many planets are in our solar system?", "There are eight recognized planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."),

    # Computing, Electronics & Raspberry Pi
    ("What is a Raspberry Pi Zero 2 W?", "The Raspberry Pi Zero 2 W is a compact single-board computer featuring a quad-core 64-bit ARM Cortex-A53 processor, 512 MB of RAM, and wireless connectivity."),
    ("How much RAM does a Raspberry Pi Zero 2 W have?", "The Raspberry Pi Zero 2 W has 512 megabytes of LPDDR2 SDRAM."),
    ("What processor does the Raspberry Pi Zero 2 W use?", "It is powered by a Broadcom BCM2710A1 system-on-chip with four 64-bit ARM Cortex-A53 cores running at 1.0 GHz."),
    ("What voltage does a Raspberry Pi operate on?", "A Raspberry Pi typically operates on a 5-volt DC power supply via USB."),
    ("What is Python?", "Python is a high-level, general-purpose interpreted programming language created by Guido van Rossum, known for its clear and readable syntax."),
    ("What is Linux?", "Linux is an open-source Unix-like operating system kernel created by Linus Torvalds in 1991, which powers servers, desktops, Android, and embedded devices."),
    ("What is the difference between RAM and storage?", "RAM is high-speed, volatile memory used by the CPU for active tasks, while storage is non-volatile memory used for permanent data retention."),
    ("What is an operating system?", "An operating system is system software that manages computer hardware and software resources, providing common services for computer programs."),

    # Logic, Traps & Commonsense Reasoning
    ("Which is heavier: a pound of feathers or two pounds of gold?", "Two pounds of gold is heavier because two pounds is twice as much mass as one pound."),
    ("Which is heavier: a pound of feathers or a pound of bricks?", "They weigh the exact same: one pound."),
    ("If today is Sunday, what day was it 4 days ago?", "Four days ago was Wednesday."),
    ("If today is Tuesday, what day will it be in 10 days?", "In 10 days it will be Friday."),
    ("Can a rooster lay an egg?", "No, roosters are male chickens and cannot lay eggs; only hens lay eggs."),
    ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?", "The ball costs 5 cents ($0.05), and the bat costs $1.05."),
    ("If you have 3 apples and you take away 2, how many apples do you have?", "You have 2 apples, because you took them."),
    ("Can a man marry his widow's sister?", "No, because for a woman to be a widow, the man must already be dead."),
    ("How many months have 28 days?", "All 12 months have at least 28 days."),
]

def clean_sentence(text):
    text = text.strip()
    # If text is multi-paragraph, keep first 1-2 sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 2:
        return ' '.join(sentences[:2])
    return text

def format_chatml(user_text, assistant_text):
    return (
        "<|im_start|>system\n"
        "You are a helpful and factual voice assistant. Answer accurately and concisely in 1 to 2 sentences.<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_text}<|im_end|>\n"
    )

def main():
    print("=== Step 1: Compiling High-Density Knowledge Dataset ===")
    dataset = []

    # A. Add curated ground-truth QA (replicated 15x for deep anchor retention)
    print(f"Adding {len(FACTUAL_QA)} core anchor facts (replicated 15x)...")
    for q, a in FACTUAL_QA:
        for _ in range(15):
            dataset.append(format_chatml(q, a))

    # B. Stream high-quality factual OpenHermes QA
    print("Streaming openhermes-100k for science, geography & general trivia...")
    try:
        ds_hermes = load_dataset("HuggingFaceTB/smoltalk", "openhermes-100k", split="train", streaming=True)
        count = 0
        for sample in ds_hermes:
            msgs = sample.get("messages", [])
            if len(msgs) >= 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
                u = msgs[0]["content"].strip()
                a = clean_sentence(msgs[1]["content"])
                # Filter for concise, voice-friendly QA
                if 10 <= len(u) <= 150 and 20 <= len(a) <= 250 and not any(k in a for k in ["```", "http", "JSON", "def "]):
                    dataset.append(format_chatml(u, a))
                    count += 1
                    if count >= 2200:
                        break
        print(f"Added {count} openhermes-100k factual samples.")
    except Exception as e:
        print(f"Warning: OpenHermes stream error: {e}")

    # C. Stream everyday natural conversations
    print("Streaming everyday-conversations for natural voice responses...")
    try:
        ds_everyday = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train", streaming=True)
        count = 0
        for sample in ds_everyday:
            msgs = sample.get("messages", [])
            for i in range(0, len(msgs)-1, 2):
                if msgs[i]["role"] == "user" and msgs[i+1]["role"] == "assistant":
                    u = msgs[i]["content"].strip()
                    a = clean_sentence(msgs[i+1]["content"])
                    if 5 <= len(u) <= 120 and 15 <= len(a) <= 200:
                        dataset.append(format_chatml(u, a))
                        count += 1
                        if count >= 1200:
                            break
            if count >= 1200:
                break
        print(f"Added {count} everyday-conversations samples.")
    except Exception as e:
        print(f"Warning: Everyday conversations stream error: {e}")

    # D. Stream Magpie Ultra high-quality instructions
    print("Streaming smol-magpie-ultra for general instructions...")
    try:
        ds_magpie = load_dataset("HuggingFaceTB/smoltalk", "smol-magpie-ultra", split="train", streaming=True)
        count = 0
        for sample in ds_magpie:
            msgs = sample.get("messages", [])
            if len(msgs) >= 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
                u = msgs[0]["content"].strip()
                a = clean_sentence(msgs[1]["content"])
                if 15 <= len(u) <= 140 and 25 <= len(a) <= 250 and not any(k in a for k in ["```", "http"]):
                    dataset.append(format_chatml(u, a))
                    count += 1
                    if count >= 800:
                        break
        print(f"Added {count} smol-magpie-ultra samples.")
    except Exception as e:
        print(f"Warning: Magpie stream error: {e}")

    # Shuffle thoroughly
    random.seed(42)
    random.shuffle(dataset)

    print(f"\nTotal Dataset Samples: {len(dataset)}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in dataset:
            f.write(json.dumps({"text": line}) + "\n")

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"Dataset successfully written to: {OUTPUT_FILE} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
