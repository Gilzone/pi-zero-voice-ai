#!/usr/bin/env python3
import sys
import wave
import struct
import math

def get_rms(filepath):
    try:
        with wave.open(filepath, 'rb') as w:
            n_frames = w.getnframes()
            if n_frames == 0:
                return 0
            frames = w.readframes(n_frames)
            total_samples = len(frames) // 2
            if total_samples == 0:
                return 0
            values = struct.unpack(f"<{total_samples}h", frames)
            # Sample every 4th integer to calculate RMS in <5ms
            sub = values[::4]
            sum_sq = sum(v * v for v in sub)
            return int(math.sqrt(sum_sq / len(sub)))
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 0

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("0")
        sys.exit(0)
    rms = get_rms(sys.argv[1])
    print(rms)
