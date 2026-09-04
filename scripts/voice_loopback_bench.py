import subprocess
import time
import os
import json
import wave

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence running locally on a Raspberry Pi Zero."
]

TTS_WAV_RAW = '/home/pi/ai/tts_raw.wav'
TTS_WAV_16K = '/home/pi/ai/tts_16k.wav'
WHISPER_BIN = '/home/pi/ai/bin/whisper-cli'
MODEL_PATH = '/home/pi/ai/models/ggml-tiny.en.bin'

def get_temp():
    try:
        out = subprocess.check_output(['vcgencmd', 'measure_temp']).decode().strip()
        return float(out.replace('temp=', '').replace("'C", ''))
    except Exception:
        return 0.0

def get_cpu_freq_mhz():
    try:
        with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq') as f:
            return int(f.read().strip()) // 1000
    except Exception:
        return 1000

def get_audio_meta(filepath):
    with wave.open(filepath, 'rb') as w:
        frames = w.getnframes()
        rate = w.getframerate()
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        duration = frames / float(rate)
    size = os.path.getsize(filepath)
    return {
        'filepath': filepath,
        'size_bytes': size,
        'channels': channels,
        'sample_rate_hz': rate,
        'bit_depth': sampwidth * 8,
        'duration_sec': round(duration, 3)
    }

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def evaluate_accuracy(reference, hypothesis):
    clean_ref = "".join(c.lower() for c in reference if c.isalnum() or c.isspace()).strip()
    clean_hyp = "".join(c.lower() for c in hypothesis if c.isalnum() or c.isspace()).strip()
    
    char_dist = levenshtein_distance(clean_ref, clean_hyp)
    cer = char_dist / max(1, len(clean_ref))
    char_acc = max(0.0, 1.0 - cer)
    
    ref_words = clean_ref.split()
    hyp_words = clean_hyp.split()
    word_dist = levenshtein_distance(ref_words, hyp_words)
    wer = word_dist / max(1, len(ref_words))
    word_acc = max(0.0, 1.0 - wer)
    
    return {
        'clean_reference': clean_ref,
        'clean_hypothesis': clean_hyp,
        'word_error_rate': round(wer, 4),
        'word_accuracy_pct': round(word_acc * 100, 2),
        'char_error_rate': round(cer, 4),
        'char_accuracy_pct': round(char_acc * 100, 2)
    }

runs = []
env = os.environ.copy()
env['LD_LIBRARY_PATH'] = '/home/pi/ai/src/whisper.cpp/build/bin'

for idx, text in enumerate(SENTENCES):
    t_start = get_temp()
    freq_start = get_cpu_freq_mhz()

    # Step 1: TTS Generation (espeak-ng)
    t0 = time.perf_counter()
    cmd_tts = ['/usr/bin/time', '-v', 'espeak-ng', '-w', TTS_WAV_RAW, text]
    p_tts = subprocess.run(cmd_tts, capture_output=True, text=True)
    t1 = time.perf_counter()
    tts_time = t1 - t0
    
    tts_rss_kb = None
    for l in p_tts.stderr.splitlines():
        if 'Maximum resident set size' in l:
            tts_rss_kb = int(l.split(':')[-1].strip())
            
    raw_meta = get_audio_meta(TTS_WAV_RAW)

    # Step 2: Sox Resample to 16kHz mono
    t2 = time.perf_counter()
    subprocess.run(['sox', TTS_WAV_RAW, '-r', '16000', '-c', '1', '-b', '16', TTS_WAV_16K], check=True)
    t3 = time.perf_counter()
    resample_time = t3 - t2
    meta_16k = get_audio_meta(TTS_WAV_16K)

    # Step 3: STT Inference (whisper.cpp)
    t4 = time.perf_counter()
    cmd_stt = [
        '/usr/bin/time', '-v',
        WHISPER_BIN,
        '-m', MODEL_PATH,
        '-f', TTS_WAV_16K,
        '-t', '4',
        '--no-timestamps'
    ]
    p_stt = subprocess.run(cmd_stt, capture_output=True, text=True, env=env)
    t5 = time.perf_counter()
    stt_wall_time = t5 - t4
    
    stt_rss_kb = None
    for l in p_stt.stderr.splitlines():
        if 'Maximum resident set size' in l:
            stt_rss_kb = int(l.split(':')[-1].strip())
            
    transcription = p_stt.stdout.strip()
    
    timings = {}
    for l in p_stt.stderr.splitlines():
        if 'whisper_print_timings:' in l:
            p = l.split('whisper_print_timings:')[-1].strip().split('=')
            if len(p) == 2:
                timings[p[0].strip()] = p[1].strip()
                
    accuracy = evaluate_accuracy(text, transcription)
    
    t_end = get_temp()
    freq_end = get_cpu_freq_mhz()

    runs.append({
        'test_id': idx + 1,
        'input_text': text,
        'transcribed_text': transcription,
        'tts': {
            'engine': 'espeak-ng 1.52',
            'duration_sec': round(tts_time, 4),
            'peak_rss_mb': round(tts_rss_kb / 1024, 2) if tts_rss_kb else None,
            'audio_raw': raw_meta,
            'audio_16k': meta_16k,
            'resample_sec': round(resample_time, 4)
        },
        'stt': {
            'engine': 'whisper.cpp 1.7.4 (ARMv8 NEON 32-bit)',
            'model': 'ggml-tiny.en.bin',
            'model_size_mb': round(os.path.getsize(MODEL_PATH)/(1024*1024), 2),
            'threads': 4,
            'inference_wall_sec': round(stt_wall_time, 3),
            'audio_duration_sec': meta_16k['duration_sec'],
            'real_time_factor': round(stt_wall_time / meta_16k['duration_sec'], 2),
            'peak_rss_mb': round(stt_rss_kb / 1024, 2) if stt_rss_kb else None,
            'timings': timings
        },
        'accuracy': accuracy,
        'telemetry': {
            'temp_start_c': t_start,
            'temp_end_c': t_end,
            'cpu_freq_mhz': freq_end
        }
    })

output_data = {
    'platform': {
        'device': 'Raspberry Pi Zero 2 W Rev 1.0',
        'cpu': 'Quad-Core ARM Cortex-A53 @ 1.0 GHz',
        'arch': 'armv7l (32-bit Raspbian 13 / Trixie)',
        'physical_ram_mb': 512,
        'simd_support': 'ARM NEON + CRC'
    },
    'results': runs
}

with open('/home/pi/ai/voice_metadata_report.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print(json.dumps(output_data, indent=2))
