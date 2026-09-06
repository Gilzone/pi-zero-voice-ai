#!/bin/bash
# ==============================================================================
# Autonomous Voice AI Launcher & Background Service
# Automatically detects USB Audio card, configures mixer, greets user,
# and runs energy-gated hands-free listening loop.
# ==============================================================================

BASE_DIR="/home/pi/ai"
WHISPER_BIN="$BASE_DIR/bin/whisper-cli"
LLAMA_BIN="$BASE_DIR/bin/llama-cli"
WHISPER_LIB="$BASE_DIR/src/whisper.cpp/build/bin"
LLAMA_LIB="$BASE_DIR/src/llama.cpp/build-fast/bin"

WHISPER_MODEL="$BASE_DIR/models/ggml-tiny.en.bin"
LLM_MODEL="$BASE_DIR/models/smollm2-360m-qwen-distill-q3_k_m.gguf"

QUESTION_WAV="$BASE_DIR/current_question.wav"
REPLY_WAV="$BASE_DIR/current_reply.wav"
GREETING_WAV="$BASE_DIR/greeting.wav"
CHIME_WAV="$BASE_DIR/chime.wav"
RAW_LLM_FILE="/tmp/llm_raw_out.txt"
RAW_STT_FILE="/tmp/whisper_raw_out.txt"
ENERGY_SCRIPT="$BASE_DIR/check_audio_energy.py"

RECORD_SECONDS=6
ENERGY_THRESHOLD=250

echo "=== Voice AI Background Daemon Started ==="

while true; do
    # 1. Wait for USB Audio Card to be connected
    echo "Searching for USB Audio hardware..."
    while true; do
        USB_CAP=$(arecord -l 2>/dev/null | grep -i "usb" | head -n 1 | sed -E 's/card ([0-9]+).*/\1/' || true)
        USB_PLAY=$(aplay -l 2>/dev/null | grep -i "usb" | head -n 1 | sed -E 's/card ([0-9]+).*/\1/' || true)
        if [ -n "$USB_CAP" ] && [ -n "$USB_PLAY" ]; then
            MIC_DEV="plughw:$USB_CAP,0"
            SPK_DEV="plughw:$USB_PLAY,0"
            CARD_NUM="$USB_CAP"
            echo "Found USB Audio: Card $CARD_NUM (Mic: $MIC_DEV, Spk: $SPK_DEV)"
            break
        fi
        sleep 2
    done

    # 2. Configure Audio Mixer Levels
    sleep 1
    amixer -c "$CARD_NUM" sset 'Speaker' 90% unmute >/dev/null 2>&1 || true
    amixer -c "$CARD_NUM" sset 'PCM' 85% unmute >/dev/null 2>&1 || true
    amixer -c "$CARD_NUM" sset 'Master' 90% unmute >/dev/null 2>&1 || true
    amixer -c "$CARD_NUM" sset 'Mic' 92% cap unmute >/dev/null 2>&1 || true
    amixer -c "$CARD_NUM" sset 'Capture' 92% cap unmute >/dev/null 2>&1 || true

    # 3. Audio Greeting to signal the device is ready
    echo "Synthesizing boot greeting..."
    flite -voice slt -t "Voice AI online and ready." -o "$GREETING_WAV" >/dev/null 2>&1 || true
    aplay -D "$SPK_DEV" "$GREETING_WAV" >/dev/null 2>&1 || true

    # Pre-generate chime
    flite -voice slt -t "Listening." -o "$CHIME_WAV" >/dev/null 2>&1 || true

    echo "=== Entering Energy-Gated Listening Loop ==="
    JUST_ANSWERED=false

    # 4. Main Hands-Free Conversation Loop
    while true; do
        # Check if USB card is still plugged in
        if ! arecord -l 2>/dev/null | grep -qi "usb"; then
            echo "USB Audio disconnected! Returning to device discovery."
            break
        fi

        if [ "$JUST_ANSWERED" = true ]; then
            aplay -D "$SPK_DEV" "$CHIME_WAV" >/dev/null 2>&1 || true
            JUST_ANSWERED=false
        fi

        # Record audio sample
        rm -f "$QUESTION_WAV"
        arecord -D "$MIC_DEV" -f S16_LE -r 16000 -c 1 -d "$RECORD_SECONDS" "$QUESTION_WAV" 2>/dev/null || {
            sleep 2
            continue
        }

        # Check RMS audio energy
        RMS=$(python3 "$ENERGY_SCRIPT" "$QUESTION_WAV" 2>/dev/null || echo "0")
        
        # If silence / room noise below threshold, skip CPU-intensive inference
        if [ "$RMS" -lt "$ENERGY_THRESHOLD" ]; then
            # Quiet room, do not burn CPU
            continue
        fi

        echo "[Speech Detected! RMS=$RMS >= $ENERGY_THRESHOLD]"
        echo "Transcribing with Whisper..."

        T0=$(date +%s)
        rm -f "$RAW_STT_FILE"
        LD_LIBRARY_PATH="$WHISPER_LIB" "$WHISPER_BIN" \
          -m "$WHISPER_MODEL" \
          -f "$QUESTION_WAV" \
          -t 4 --no-timestamps -nf -sns -bs 1 -bo 1 -ac 512 \
          > "$RAW_STT_FILE" 2>&1 || true
        T1=$(date +%s)
        STT_DUR=$(( T1 - T0 ))

        TRANSCRIBED=$(grep -v 'whisper_' "$RAW_STT_FILE" | grep -v 'system_info' | grep -v 'read_audio' | grep -v 'main:' | sed '/^[[:space:]]*$/d' | tail -n 1 | sed 's/^[ \t]*//')
        CLEAN=$(echo "$TRANSCRIBED" | tr -d '[:punct:][:space:]')

        if [ -z "$CLEAN" ] || [ "${#CLEAN}" -lt 3 ]; then
            echo "No recognizable words (clean length: ${#CLEAN}). Resuming listening."
            continue
        fi

        echo "User said: \"$TRANSCRIBED\" (STT: ${STT_DUR}s)"
        echo "Querying SmolLM2-360M..."

        rm -f "$RAW_LLM_FILE"
        T_LLM_0=$(date +%s)
        echo "$TRANSCRIBED" | LD_LIBRARY_PATH="$LLAMA_LIB" "$LLAMA_BIN" \
          -m "$LLM_MODEL" \
          -t 4 -c 256 -n 36 -b 128 -ub 64 \
          --load-mode mmap --fit off \
          -ctk q8_0 -ctv q8_0 \
          --temp 0.2 --top-p 0.9 --repeat-penalty 1.15 \
          -sys "Answer in 1 short sentence." \
          -cnv > "$RAW_LLM_FILE" 2>&1 || true
        T_LLM_1=$(date +%s)
        LLM_DUR=$(( T_LLM_1 - T_LLM_0 ))

        ANSWER=$(python3 -c "
import re
try:
    with open('$RAW_LLM_FILE', 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
except Exception:
    lines = []
res = ''
for i, l in enumerate(lines):
    if '> EOF by user' in l:
        for j in range(i-1, -1, -1):
            cl = lines[j].strip()
            if cl and not cl.startswith('>') and not cl.startswith('-') and not re.match(r'^\s*\d+\.\d+\.\d+\.\d+\s+[IWE]\s*', cl):
                res = cl
                break
        break
res = re.sub(r'<think>.*?</think>', '', res, flags=re.DOTALL)
res = re.sub(r'<[^>]+>', '', res)
res = re.sub(r'[^\x20-\x7E]', ' ', res)
res = ' '.join(res.split()).strip()
print(res if res else 'I am your offline assistant.')
")

        echo "AI Reply: \"$ANSWER\" (LLM: ${LLM_DUR}s)"

        flite -voice slt -t "$ANSWER" -o "$REPLY_WAV"
        aplay -D "$SPK_DEV" "$REPLY_WAV" 2>/dev/null || true
        JUST_ANSWERED=true
        sleep 1
    done
done
