#!/usr/bin/env python3
"""
Converts the newly trained SmolLM2-360M-Qwen-v2 model to GGUF format
and quantizes it to Q3_K_M (strictly targeting <= 224 MB).
"""

import os
import subprocess
import sys

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    merged_model_dir = os.path.join(base_dir, "models", "smollm2-360m-qwen-v2-merged")
    f16_gguf = os.path.join(base_dir, "models", "smollm2-360m-qwen-v2-f16.gguf")
    q3_gguf = os.path.join(base_dir, "models", "smollm2-360m-qwen-v2-q3_k_m.gguf")

    convert_script = os.path.join(project_dir, "llama.cpp", "convert_hf_to_gguf.py")
    quantize_exe = os.path.join(project_dir, "bin", "llama-quantize.exe")

    if not os.path.exists(merged_model_dir):
        print(f"Error: Merged model directory not found at: {merged_model_dir}")
        sys.exit(1)

    # 1. Convert to GGUF FP16
    print("=== Step 1: Converting Merged Model v2 to GGUF (FP16) ===")
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
    print(f"SUCCESS! Quantized Model v2 Created:")
    print(f"Path: {q3_gguf}")
    print(f"File Size: {q3_size_mb:.2f} MB")
    if q3_size_mb <= 224.0:
        print(f"Constraint PASSED: {q3_size_mb:.2f} MB <= 224 MB")
    else:
        print(f"Warning: Model exceeded 224 MB budget!")
    print(f"==================================================")

if __name__ == "__main__":
    main()
