#!/usr/bin/env python3
"""
Converts the fine-tuned merged SmolLM2-360M model to GGUF format
and quantizes it to Q3_K_M (targeting <= 224 MB).
"""

import os
import subprocess
import sys

def main():
    base_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(base_dir)
    merged_model_dir = os.path.join(base_dir, "models", "smollm2-360m-qwen-merged")
    f16_gguf = os.path.join(base_dir, "models", "smollm2-360m-qwen-f16.gguf")
    q3_gguf = os.path.join(base_dir, "models", "smollm2-360m-qwen-distill-q3_k_m.gguf")
    convert_script = os.path.join(project_dir, "llama.cpp", "convert_hf_to_gguf.py")
    quantize_exe = os.path.join(project_dir, "bin", "llama-quantize.exe")
    llama_cli = os.path.join(project_dir, "bin", "llama-cli.exe")

    if not os.path.exists(merged_model_dir):
        print(f"Error: Merged model directory not found at: {merged_model_dir}")
        sys.exit(1)

    # 1. Convert to GGUF FP16
    print("=== Step 1: Converting Merged Model to GGUF (FP16) ===")
    cmd_convert = [
        sys.executable,
        convert_script,
        merged_model_dir,
        "--outfile", f16_gguf,
        "--outtype", "f16"
    ]
    print(f"Running: {' '.join(cmd_convert)}")
    res = subprocess.run(cmd_convert)
    if res.returncode != 0:
        print("Conversion failed!")
        sys.exit(res.returncode)

    f16_size_mb = os.path.getsize(f16_gguf) / (1024 * 1024)
    print(f"FP16 GGUF created: {f16_gguf} ({f16_size_mb:.2f} MB)")

    # 2. Quantize to Q3_K_M
    print("\n=== Step 2: Quantizing to Q3_K_M (Target <= 224 MB) ===")
    cmd_quant = [
        quantize_exe,
        f16_gguf,
        q3_gguf,
        "Q3_K_M"
    ]
    print(f"Running: {' '.join(cmd_quant)}")
    res = subprocess.run(cmd_quant)
    if res.returncode != 0:
        print("Quantization failed!")
        sys.exit(res.returncode)

    q3_size_mb = os.path.getsize(q3_gguf) / (1024 * 1024)
    print(f"\n==================================================")
    print(f"SUCCESS! Quantized Model Created:")
    print(f"Path: {q3_gguf}")
    print(f"Final Size: {q3_size_mb:.2f} MB")
    print(f"Target Budget (<= 224 MB): {'PASSED' if q3_size_mb <= 224 else 'EXCEEDED'}")
    print(f"==================================================")

    # 3. Create System Prompt Cache for Instant 0.0s Startup
    print("\n=== Step 3: Generating Persistent Prompt Cache ===")
    cache_file = os.path.join(base_dir, "models", "voice_sys_cache.bin")
    sys_prompt = "<|im_start|>system\nYou are a concise voice assistant. Think step-by-step inside <think> tags, then answer directly in 1 short sentence.<|im_end|>\n"
    cmd_cache = [
        llama_cli,
        "-m", q3_gguf,
        "-p", sys_prompt,
        "--prompt-cache", cache_file,
        "-n", "1",
        "-no-cnv"
    ]
    subprocess.run(cmd_cache)
    if os.path.exists(cache_file):
        print(f"Prompt cache generated: {cache_file} ({os.path.getsize(cache_file)/1024:.1f} KB)")

    # 4. Multi-Question Logic & Reasoning Benchmark
    print("\n=== Step 4: Testing Logic, Math & Factual Reasoning ===")
    test_questions = [
        "Why is the sky blue?",
        "Which is heavier: a pound of feathers or two pounds of gold?",
        "If today is Sunday, what day was it 4 days ago?",
        "What is the capital of Australia?"
    ]

    for q in test_questions:
        print(f"\n[Test Question]: {q}")
        p = f"{sys_prompt}<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        cmd_test = [
            llama_cli,
            "-m", q3_gguf,
            "-p", p,
            "-n", "80",
            "-no-cnv",
            "--temp", "0.2",
            "--top-p", "0.9",
            "--no-display-prompt"
        ]
        res = subprocess.run(cmd_test, capture_output=True, text=True)
        out = res.stdout.strip()
        print(f"[Full Output]: {out}")

if __name__ == "__main__":
    main()
