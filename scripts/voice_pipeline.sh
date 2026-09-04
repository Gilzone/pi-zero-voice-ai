#!/bin/bash
# ==============================================================================
# Unified Voice AI Pipeline for Raspberry Pi Zero 2 W
# Full Autonomous Flow: Live Mic -> Whisper STT -> SmolLM2 LLM -> Flite TTS -> Speaker
# Supports: Single Turn (--talk) and Continuous Hands-Free Loop (--loop)
# ==============================================================================

set -e

BASE_DIR="/home/pi/ai"
WHISPER_BIN="$BASE_DIR/bin/whisper-cli"
LLAMA_BIN="$BASE_DIR/bin/llama-cli"
WHISPER_LIB="$BASE_DIR/src/whisper.cpp/build/bin"
LLAMA_LIB="$BASE_DIR/src/llama.cpp/build-fast/bin"

WHISPER_MODEL="$BASE_DIR/models/ggml-tiny.en.bin"
LLM_MODEL="$BASE_DIR/models/SmolLM2-360M-Instruct-Q3_K_M.gguf"

TEXT_INPUT=""
AUDIO_INPUT=""
STT_MODEL_NAME="tiny"
RUN_LLM=true
LIVE_MIC=false
LOOP_MODE=false
RECORD_SECONDS=6

print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --loop, -c             Continuous Loop Mode: Keeps listening and replying hands-free"
    echo "  --talk, -l             Single Live Mode: Record 1 question and speak answer"
    echo "  -d, --duration SEC     Recording duration in seconds (default: 6)"
    echo "  -a, --ask \"QUESTION\"   Synthesize question audio, transcribe it, prompt LLM, and speak answer"
    echo "  -f, --file FILE.wav    Transcribe existing audio file and pass to LLM"
    echo "  -m, --model tiny|base  Choose Whisper model (default: tiny)"
    echo "  --no-llm               Run STT loopback only (skip LLM reasoning)"
    echo "  -h, --help             Show this help message"
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --loop|-c) LOOP_MODE=true; LIVE_MIC=true ;;
        --talk|-l) LIVE_MIC=true ;;
        -d|--duration) RECORD_SECONDS="$2"; shift ;;
        -a|--ask) TEXT_INPUT="$2"; shift ;;
        -f|--file) AUDIO_INPUT="$2"; shift ;;
        -m|--model) STT_MODEL_NAME="$2"; shift ;;
        --no-llm) RUN_LLM=false ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "Unknown parameter: $1"; print_help; exit 1 ;;
    esac
    shift
done

# Detect USB Audio Cards for Mic and Speaker
detect_usb_cards() {
    USB_CAP=$(arecord -l 2>/dev/null | grep -i "usb" | head -n 1 | sed -E 's/card ([0-9]+).*/\1/' || true)
    USB_PLAY=$(aplay -l 2>/dev/null | grep -i "usb" | head -n 1 | sed -E 's/card ([0-9]+).*/\1/' || true)
    
    MIC_DEV="plughw:${USB_CAP:-1},0"
    SPK_DEV="plughw:${USB_PLAY:-1},0"
}

detect_usb_cards

if [ "$STT_MODEL_NAME" == "base" ]; then
    WHISPER_MODEL="$BASE_DIR/models/ggml-base.en.bin"
else
    WHISPER_MODEL="$BASE_DIR/models/ggml-tiny.en.bin"
fi

QUESTION_WAV="$BASE_DIR/current_question.wav"
CUE_WAV="$BASE_DIR/cue.wav"
REPLY_WAV="$BASE_DIR/current_reply.wav"
RAW_LLM_FILE="/tmp/llm_raw_out.txt"

# Pre-generate friendly voice cue
flite -voice slt -t "Listening." -o "$CUE_WAV" >/dev/null 2>&1 || true

run_cycle() {
    if [ "$LIVE_MIC" = true ]; then
        echo "=================================================================="
        echo "[1/3] LIVE MICROPHONE RECORDING ($RECORD_SECONDS SECONDS)"
        echo "Microphone: $MIC_DEV | Speaker: $SPK_DEV"
        
        # Audio chime/cue
        aplay -D "$SPK_DEV" "$CUE_WAV" >/dev/null 2>&1 || aplay "$CUE_WAV" >/dev/null 2>&1 || true
        
        echo ">>> SPEAK YOUR QUESTION NOW! <<<"
        arecord -D "$MIC_DEV" -f S16_LE -r 16000 -c 1 -d "$RECORD_SECONDS" "$QUESTION_WAV" 2>/dev/null
        AUDIO_INPUT="$QUESTION_WAV"
        echo "Recording captured."
    elif [ -n "$TEXT_INPUT" ]; then
        echo "=================================================================="
        echo "[1/3] USER VOICE SYNTHESIS (Simulating spoken input)"
        echo "Input Text: \"$TEXT_INPUT\""
        flite -voice slt -t "$TEXT_INPUT" -o "$QUESTION_WAV"
        AUDIO_INPUT="$QUESTION_WAV"
    fi

    # Step 2: Speech to Text (Whisper)
    echo ""
    echo "=================================================================="
    echo "[2/3] SPEECH-TO-TEXT (Whisper $STT_MODEL_NAME.en)"
    echo "Transcribing..."
    
    T_STT_0=$(date +%s)
    STT_RAW=$(LD_LIBRARY_PATH="$WHISPER_LIB" "$WHISPER_BIN" -m "$WHISPER_MODEL" -f "$AUDIO_INPUT" -t 4 --no-timestamps -nf -sns 2>&1)
    T_STT_1=$(date +%s)
    STT_DUR=$(( T_STT_1 - T_STT_0 ))

    TRANSCRIBED=$(echo "$STT_RAW" | grep -v 'whisper_' | grep -v 'system_info' | grep -v 'read_audio' | sed '/^[[:space:]]*$/d' | tail -n 1 | sed 's/^[ \t]*//')
    
    # Check if empty or silence
    CLEAN_CHECK=$(echo "$TRANSCRIBED" | tr -d '[:punct:][:space:]')
    if [ -z "$CLEAN_CHECK" ] || [ "${#CLEAN_CHECK}" -lt 3 ]; then
        echo "No clear speech detected. (Length: ${#CLEAN_CHECK})"
        return 0
    fi
    
    echo "You asked: \"$TRANSCRIBED\" (in ${STT_DUR}s)"

    if [ "$RUN_LLM" = false ]; then
        return 0
    fi

    # Step 3: LLM Inference (SmolLM2-360M)
    echo ""
    echo "=================================================================="
    echo "[3/3] LLM REASONING & VOICE REPLY (SmolLM2-360M-Instruct)"
    echo "Thinking..."
    
    rm -f "$RAW_LLM_FILE"
    T_LLM_0=$(date +%s)
    echo "$TRANSCRIBED" | LD_LIBRARY_PATH="$LLAMA_LIB" "$LLAMA_BIN" \
      -m "$LLM_MODEL" \
      -t 4 -c 384 -n 32 -b 128 -ub 64 \
      --load-mode mmap --fit off \
      --temp 0.2 --top-p 0.9 \
      -sys "You are a concise assistant. Answer directly in 1 short sentence." \
      -cnv > "$RAW_LLM_FILE" 2>&1 || true
    T_LLM_1=$(date +%s)
    LLM_DUR=$(( T_LLM_1 - T_LLM_0 ))

    ANSWER=$(python3 -c "
import re
lines = open('$RAW_LLM_FILE').readlines()
res = ''
for i, l in enumerate(lines):
    if '> EOF by user' in l:
        for j in range(i-1, -1, -1):
            cl = lines[j].strip()
            if cl and not cl.startswith('>') and not cl.startswith('-') and not re.match(r'^\s*\d+\.\d+\.\d+\.\d+\s+[IWE]\s*', cl):
                res = cl
                break
        break
print(res if res else 'I am an offline AI assistant.')
")

    echo "AI Answer: \"$ANSWER\" (in ${LLM_DUR}s)"

    # Synthesize answer speech
    flite -voice slt -t "$ANSWER" -o "$REPLY_WAV"
    
    # Play response through speaker
    echo "Speaking response through USB speaker..."
    aplay -D "$SPK_DEV" "$REPLY_WAV" 2>/dev/null || aplay "$REPLY_WAV" 2>/dev/null || true
    
    echo "=================================================================="
    echo "Turn Complete!"
}

if [ "$LOOP_MODE" = true ]; then
    echo "=================================================================="
    echo " 🎙️ CONTINUOUS VOICE AI ASSISTANT RUNNING ON PI ZERO 2 W"
    echo " Speak into your microphone after each 'Listening' chime."
    echo " Press Ctrl+C at any time to exit loop mode."
    echo "=================================================================="
    
    TURN_COUNT=1
    while true; do
        echo ""
        echo ">>> [TURN $TURN_COUNT] <<<"
        run_cycle
        TURN_COUNT=$(( TURN_COUNT + 1 ))
        sleep 1
    done
else
    run_cycle
fi
