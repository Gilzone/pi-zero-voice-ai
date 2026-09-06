#!/usr/bin/env python3
"""
Benchmarks Speculative Decoding between Target (SmolLM2-360M-Distill Q3_K_M)
and Draft Candidates (SmolLM2-135M Q4_K_M, Q3_K_M, Q2_K).
Measures token throughput (tokens/sec), generation latency, and draft acceptance stats.
"""

import os
import sys
import time
import subprocess
import re

def run_cmd(cmd):
    t0 = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    dur = time.time() - t0
    return res.stdout, dur

def parse_perf(output):
    # Extract eval time and tokens per second
    # Example: eval time =    6049.02 ms /    15 runs   (  403.27 ms per token,     2.48 tokens per second)
    m_eval = re.search(r'eval time\s*=\s*[\d\.]+\s*ms\s*/\s*(\d+)\s*runs\s*\(\s*[\d\.]+\s*ms per token,\s*([\d\.]+)\s*tokens per second\)', output)
    tokens = int(m_eval.group(1)) if m_eval else 0
    tps = float(m_eval.group(2)) if m_eval else 0.0

    # Extract draft acceptance stats if available
    # Example in llama-speculative: n_drafted = ..., n_accepted = ...
    m_draft = re.search(r'draft acceptance rate:\s*([\d\.]+)%', output, re.IGNORECASE)
    draft_rate = float(m_draft.group(1)) if m_draft else None

    # Alternative pattern: accepted = X / Y
    m_acc = re.search(r'accepted\s*=\s*(\d+)\s*/\s*(\d+)', output, re.IGNORECASE)
    if m_acc:
        acc = int(m_acc.group(1))
        tot = int(m_acc.group(2))
        draft_rate = (acc / tot * 100.0) if tot > 0 else 0.0

    return tokens, tps, draft_rate

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    models_dir = os.path.join(base_dir, "models")

    target_model = os.path.join(models_dir, "smollm2-360m-qwen-distill-q3_k_m.gguf")
    draft_models = {
        "Q4_K_M": os.path.join(models_dir, "smollm2-135m-instruct-q4_k_m.gguf"),
        "Q3_K_M": os.path.join(models_dir, "smollm2-135m-instruct-q3_k_m.gguf"),
        "Q2_K":   os.path.join(models_dir, "smollm2-135m-instruct-q2_k.gguf"),
    }

    llama_cli = os.path.join(project_dir, "bin", "llama-cli.exe")
    llama_spec = os.path.join(project_dir, "bin", "llama-speculative.exe")

    prompt = "<|im_start|>system\nAnswer directly in 1 short sentence.<|im_end|>\n<|im_start|>user\nWhy do plants have green leaves?<|im_end|>\n<|im_start|>assistant\n"
    n_predict = 40

    print("=================================================================")
    print("      SPECULATIVE DECODING BENCHMARK: 360M TARGET + 135M DRAFT   ")
    print("=================================================================")
    print(f"Target: {os.path.basename(target_model)}")
    print(f"Prompt tokens ~ 25, Generation tokens: {n_predict}")
    print("-----------------------------------------------------------------")

    # 1. Baseline: Target Only
    print("\n[1/4] Running Baseline (Target Only via llama-cli)...")
    cmd_base = [
        llama_cli,
        "-m", target_model,
        "-p", prompt,
        "-n", str(n_predict),
        "-t", "4",
        "--temp", "0.2",
        "--top-p", "0.9",
        "-no-cnv"
    ]
    out_base, dur_base = run_cmd(cmd_base)
    tok_base, tps_base, _ = parse_perf(out_base)
    print(f"  -> Tokens generated: {tok_base}, TPS: {tps_base:.2f} t/s, Total Duration: {dur_base:.2f}s")

    # 2. Speculative Decoding with each draft model
    results = {}
    idx = 2
    for q_name, draft_path in draft_models.items():
        if not os.path.exists(draft_path):
            print(f"\n[{idx}/4] Skipping {q_name} (file not found)")
            idx += 1
            continue

        draft_mb = os.path.getsize(draft_path) / (1024 * 1024)
        print(f"\n[{idx}/4] Running Speculative Decoding with Draft: {q_name} ({draft_mb:.1f} MB)...")
        cmd_spec = [
            llama_spec,
            "-m", target_model,
            "-md", draft_path,
            "-p", prompt,
            "-n", str(n_predict),
            "-t", "4",
            "--draft-max", "8",
            "--temp", "0.2",
            "--top-p", "0.9"
        ]
        out_spec, dur_spec = run_cmd(cmd_spec)
        tok_spec, tps_spec, draft_rate = parse_perf(out_spec)
        speedup = (tps_spec / tps_base) if tps_base > 0 else 0.0
        results[q_name] = {
            "tokens": tok_spec,
            "tps": tps_spec,
            "speedup": speedup,
            "dur": dur_spec,
            "rate": draft_rate,
            "size_mb": draft_mb
        }
        print(f"  -> Generated: {tok_spec}, TPS: {tps_spec:.2f} t/s ({speedup:.2f}x speedup), Duration: {dur_spec:.2f}s")
        if draft_rate is not None:
            print(f"  -> Draft Acceptance Rate: {draft_rate:.1f}%")
        idx += 1

    print("\n=================================================================")
    print("                    BENCHMARK RESULTS SUMMARY                    ")
    print("=================================================================")
    print(f"{'Configuration':<25} | {'Draft Size':<10} | {'Speed (t/s)':<12} | {'Speedup':<9} | {'Accept Rate':<11}")
    print("-" * 75)
    print(f"{'Baseline (360M Q3_K_M)':<25} | {'None':<10} | {tps_base:<12.2f} | {'1.00x':<9} | {'N/A':<11}")
    for q_name, res in results.items():
        rate_str = f"{res['rate']:.1f}%" if res['rate'] is not None else "N/A"
        print(f"{'Draft ' + q_name:<25} | {res['size_mb']:<8.1f}MB | {res['tps']:<12.2f} | {res['speedup']:<8.2f}x | {rate_str:<11}")
    print("=================================================================")

if __name__ == "__main__":
    main()
