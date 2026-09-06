#!/usr/bin/env python3
"""
Downloads HuggingFaceTB/SmolLM2-135M-Instruct, converts to GGUF FP16,
and quantizes to Q4_K_M, Q3_K_M, and Q2_K for Speculative Decoding.
"""

import os
import sys
import subprocess
from huggingface_hub import snapshot_download

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    hf_model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    download_dir = os.path.join(models_dir, "smollm2-135m-instruct-hf")
    f16_gguf = os.path.join(models_dir, "smollm2-135m-instruct-f16.gguf")

    convert_script = os.path.join(project_dir, "llama.cpp", "convert_hf_to_gguf.py")
    quantize_exe = os.path.join(project_dir, "bin", "llama-quantize.exe")

    # 1. Download Model
    if not os.path.exists(download_dir) or not os.listdir(download_dir):
        print(f"=== Step 1: Downloading {hf_model_id} ===")
        snapshot_download(
            repo_id=hf_model_id,
            local_dir=download_dir,
            local_dir_use_symlinks=False
        )
        print(f"Downloaded to: {download_dir}")
    else:
        print(f"=== Step 1: Using existing download at {download_dir} ===")

    # 2. Convert to GGUF FP16
    print(f"\n=== Step 2: Converting {hf_model_id} to GGUF FP16 ===")
    cmd_convert = [
        sys.executable,
        convert_script,
        download_dir,
        "--outfile", f16_gguf,
        "--outtype", "f16"
    ]
    print(f"Running: {' '.join(cmd_convert)}")
    res = subprocess.run(cmd_convert)
    if res.returncode != 0:
        print("GGUF FP16 conversion failed!")
        sys.exit(res.returncode)

    f16_mb = os.path.getsize(f16_gguf) / (1024 * 1024)
    print(f"FP16 GGUF created: {f16_gguf} ({f16_mb:.2f} MB)")

    # 3. Quantize into candidates
    quants = [
        ("Q4_K_M", os.path.join(models_dir, "smollm2-135m-instruct-q4_k_m.gguf")),
        ("Q3_K_M", os.path.join(models_dir, "smollm2-135m-instruct-q3_k_m.gguf")),
        ("Q2_K",   os.path.join(models_dir, "smollm2-135m-instruct-q2_k.gguf")),
    ]

    print("\n=== Step 3: Quantizing into candidate draft formats ===")
    for q_type, out_path in quants:
        print(f"\n--- Quantizing to {q_type} ---")
        cmd_quant = [
            quantize_exe,
            f16_gguf,
            out_path,
            q_type
        ]
        res = subprocess.run(cmd_quant)
        if res.returncode != 0:
            print(f"Quantization to {q_type} failed!")
        else:
            mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f"Successfully created: {out_path} ({mb:.2f} MB)")

    print("\n=== Quantization Complete! Summary of Draft Models ===")
    for q_type, out_path in quants:
        if os.path.exists(out_path):
            mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f"  * {q_type:8s}: {mb:.2f} MB -> {os.path.basename(out_path)}")

if __name__ == "__main__":
    main()
