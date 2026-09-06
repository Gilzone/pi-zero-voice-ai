#!/usr/bin/env python3
"""
Reliable chunked SFTP upload to Raspberry Pi Zero 2 W with progress reporting.
"""

import os
import sys
import time
import paramiko

PI_HOST = "192.168.86.34"
PI_USER = "pi"
KEY_FILE = os.path.expanduser("~/.ssh/id_ed25519")

def upload_file(local_path, remote_path):
    print(f"Connecting to {PI_USER}@{PI_HOST} via SFTP...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_HOST, username=PI_USER, key_filename=KEY_FILE, timeout=10)
    
    sftp = ssh.open_sftp()
    total_size = os.path.getsize(local_path)
    total_mb = total_size / (1024 * 1024)
    print(f"Uploading {os.path.basename(local_path)} ({total_mb:.2f} MB) -> {remote_path}...")

    t0 = time.time()
    last_print = 0

    def progress_callback(transferred, total):
        nonlocal last_print
        now = time.time()
        if now - last_print >= 2.0 or transferred == total:
            pct = (transferred / total) * 100
            mb = transferred / (1024 * 1024)
            speed = (transferred / (now - t0)) / (1024 * 1024) if (now - t0) > 0 else 0
            print(f"[{pct:5.1f}%] {mb:.1f} MB / {total_mb:.1f} MB ({speed:.2f} MB/s)", flush=True)
            last_print = now

    sftp.put(local_path, remote_path, callback=progress_callback)
    sftp.close()
    ssh.close()
    elapsed = time.time() - t0
    print(f"Upload complete in {elapsed:.1f} seconds ({total_mb/elapsed:.2f} MB/s)!", flush=True)

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    model_file = os.path.join(base_dir, "models", "smollm2-360m-qwen-distill-q3_k_m.gguf")
    cache_file = os.path.join(base_dir, "models", "voice_sys_cache.bin")
    
    if len(sys.argv) > 2:
        upload_file(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "--draft":
        draft_path = os.path.join(base_dir, "models", "smollm2-135m-instruct-q2_k.gguf")
        upload_file(draft_path, "/home/pi/ai/models/smollm2-135m-instruct-q2_k.gguf")
    elif len(sys.argv) > 1 and sys.argv[1] == "--cache-only":
        upload_file(cache_file, "/home/pi/ai/models/voice_sys_cache.bin")
    else:
        upload_file(model_file, "/home/pi/ai/models/smollm2-360m-qwen-distill-q3_k_m.gguf")
        if os.path.exists(cache_file):
            upload_file(cache_file, "/home/pi/ai/models/voice_sys_cache.bin")
