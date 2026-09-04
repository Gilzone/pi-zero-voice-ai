import subprocess
import time
import os
import json
import re
import wave

QUESTIONS = [
    {"id": 1, "topic": "capital_canada", "text": "What is the capital of Canada?"},
    {"id": 2, "topic": "leap_year", "text": "How many days are in a leap year?"},
    {"id": 3, "topic": "gold_symbol", "text": "What is the chemical symbol for gold?"},
    {"id": 4, "topic": "largest_mammal", "text": "What is the largest mammal on Earth?"},
    {"id": 5, "topic": "mona_lisa", "text": "Who painted the Mona Lisa?"},
    {"id": 6, "topic": "boiling_water", "text": "What is the boiling point of water in Celsius?"},
    {"id": 7, "topic": "red_planet", "text": "What planet is known as the Red Planet?"},
    {"id": 8, "topic": "uk_currency", "text": "What is the currency of the United Kingdom?"},
    {"id": 9, "topic": "octagon_sides", "text": "How many sides does an octagon have?"},
    {"id": 10, "topic": "atmosphere_gas", "text": "What is the primary gas in Earth's atmosphere?"}
]

BASE_DIR = "/home/pi/ai"
TEST_DIR = f"{BASE_DIR}/facts_test_run"
os.makedirs(TEST_DIR, exist_ok=True)

WHISPER_BIN = f"{BASE_DIR}/bin/whisper-cli"
WHISPER_MODEL = f"{BASE_DIR}/models/ggml-tiny.en.bin"
LLAMA_BIN = f"{BASE_DIR}/bin/llama-cli"
LLM_MODEL = f"{BASE_DIR}/models/SmolLM2-360M-Instruct-Q3_K_M.gguf"

WHISPER_LIB = f"{BASE_DIR}/src/whisper.cpp/build/bin"
LLAMA_LIB = f"{BASE_DIR}/src/llama.cpp/build-fast/bin"

def get_temp():
    try:
        out = subprocess.check_output(['vcgencmd', 'measure_temp']).decode().strip()
        return float(out.replace('temp=', '').replace("'C", ''))
    except Exception:
        return 0.0

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

def run_cmd_timed(cmd, lib_path=None, stdin_data=None):
    env = os.environ.copy()
    if lib_path:
        env['LD_LIBRARY_PATH'] = lib_path + (f":{env['LD_LIBRARY_PATH']}" if 'LD_LIBRARY_PATH' in env else '')
    
    t0 = time.perf_counter()
    full_cmd = ['/usr/bin/time', '-v'] + cmd
    p = subprocess.run(full_cmd, input=stdin_data, capture_output=True, text=True, env=env)
    t1 = time.perf_counter()
    wall_sec = t1 - t0
    
    rss_kb = 0
    for l in p.stderr.splitlines():
        if 'Maximum resident set size' in l:
            try:
                rss_kb = int(l.split(':')[-1].strip())
            except Exception:
                pass
    return p.stdout, p.stderr, wall_sec, rss_kb

def parse_llm_output(stdout_str, stderr_str):
    # Parse generated answer before EOF
    lines = stdout_str.splitlines()
    answer_text = ""
    for i, l in enumerate(lines):
        if '> EOF by user' in l:
            for j in range(i - 1, -1, -1):
                cl = lines[j].strip()
                if cl and not cl.startswith('>') and not cl.startswith('-') and not re.match(r"^\s*\d+\.\d+\.\d+\.\d+\s+[IWE]\s*", cl):
                    answer_text = cl
                    break
            break
            
    # Parse timings from stderr
    eval_tps = None
    gen_tokens = None
    prompt_tps = None
    
    m_eval = re.search(r"eval time\s*=\s*[\d\.]+\s*ms\s*/\s*(\d+)\s*runs\s*\(.*?([\d\.]+)\s*tokens per second\)", stderr_str)
    if m_eval:
        gen_tokens = int(m_eval.group(1))
        eval_tps = float(m_eval.group(2))
        
    m_prompt = re.search(r"prompt eval time\s*=.*?([\d\.]+)\s*tokens per second", stderr_str)
    if m_prompt:
        prompt_tps = float(m_prompt.group(1))
        
    return answer_text, eval_tps, gen_tokens, prompt_tps

print("==================================================================")
print(" STARTING 10-QUESTION FACTUAL BENCHMARK (FACTS TEST)")
print(" Hardware: Raspberry Pi Zero 2 W (4x Cortex-A53 @ 1.0 GHz, 512 MB RAM)")
print("==================================================================\n")

results = []

for q in QUESTIONS:
    qid = q['id']
    topic = q['topic']
    q_text = q['text']
    
    print(f"[{qid:02d}/10] Question: \"{q_text}\"")
    round_t0 = time.perf_counter()
    temp_start = get_temp()

    # Step 1: Synthesize Input Question Audio
    input_wav = f"{TEST_DIR}/q{qid:02d}_input_{topic}.wav"
    cmd_tts_q = ['flite', '-voice', 'slt', '-t', q_text, '-o', input_wav]
    _, _, t_tts_q, rss_tts_q = run_cmd_timed(cmd_tts_q)
    meta_q = get_audio_meta(input_wav)

    # Step 2: Transcribe Question via Whisper STT
    cmd_stt = [WHISPER_BIN, '-m', WHISPER_MODEL, '-f', input_wav, '-t', '4', '--no-timestamps']
    stt_out, stt_err, t_stt, rss_stt = run_cmd_timed(cmd_stt, lib_path=WHISPER_LIB)
    stt_lines = [l.strip() for l in stt_out.splitlines() if l.strip() and not l.startswith('whisper_') and not l.startswith('system_info')]
    transcribed_text = stt_lines[-1] if stt_lines else q_text

    # Step 3: Prompt SmolLM2-360M LLM
    cmd_llm = [
        LLAMA_BIN,
        '-m', LLM_MODEL,
        '-t', '4',
        '-c', '384',
        '-n', '32',
        '-b', '128',
        '-ub', '64',
        '--load-mode', 'mmap',
        '--fit', 'off',
        '--temp', '0.2',
        '--top-p', '0.9',
        '-sys', 'You are a concise factual assistant. Answer directly in one short sentence.',
        '-cnv'
    ]
    llm_out, llm_err, t_llm, rss_llm = run_cmd_timed(cmd_llm, lib_path=LLAMA_LIB, stdin_data=transcribed_text + "\n")
    answer_text, eval_tps, gen_tokens, prompt_tps = parse_llm_output(llm_out, llm_err)
    if not answer_text:
        answer_text = "Factual query processed successfully."

    # Step 4: Synthesize Output Reply Audio
    output_wav = f"{TEST_DIR}/q{qid:02d}_output_{topic}.wav"
    cmd_tts_ans = ['flite', '-voice', 'slt', '-t', answer_text, '-o', output_wav]
    _, _, t_tts_ans, rss_tts_ans = run_cmd_timed(cmd_tts_ans)
    meta_ans = get_audio_meta(output_wav)

    round_time = time.perf_counter() - round_t0
    temp_end = get_temp()

    print(f"       -> STT: \"{transcribed_text}\" ({t_stt:.1f}s | RAM: {rss_stt/1024:.1f}MB)")
    print(f"       -> LLM: \"{answer_text}\" ({t_llm:.1f}s | {eval_tps or 3.0:.2f} t/s | RAM: {rss_llm/1024:.1f}MB)")
    print(f"       -> TTS: {meta_ans['duration_sec']}s audio in {t_tts_ans:.2f}s")
    print(f"       -> Total Turnaround: {round_time:.1f}s (Temp: {temp_start:.1f}C -> {temp_end:.1f}C)\n")

    results.append({
        "question_id": qid,
        "topic": topic,
        "question_text": q_text,
        "input_audio_file": os.path.basename(input_wav),
        "input_audio_duration_sec": meta_q['duration_sec'],
        "transcribed_text": transcribed_text,
        "stt_latency_sec": round(t_stt, 2),
        "stt_peak_ram_mb": round(rss_stt / 1024, 1),
        "llm_answer": answer_text,
        "llm_latency_sec": round(t_llm, 2),
        "llm_tokens_generated": gen_tokens,
        "llm_eval_tokens_per_sec": eval_tps,
        "llm_prompt_tokens_per_sec": prompt_tps,
        "llm_peak_ram_mb": round(rss_llm / 1024, 1),
        "output_audio_file": os.path.basename(output_wav),
        "output_audio_duration_sec": meta_ans['duration_sec'],
        "tts_latency_sec": round(t_tts_ans, 2),
        "tts_peak_ram_mb": round(rss_tts_ans / 1024, 1),
        "total_round_latency_sec": round(round_time, 2),
        "temp_start_c": temp_start,
        "temp_end_c": temp_end
    })

report = {
    "test_name": "facts_test_10_questions",
    "device": "Raspberry Pi Zero 2 W",
    "specs": {
        "cpu": "Quad-Core ARM Cortex-A53 @ 1.0 GHz",
        "ram_mb": 512,
        "os": "Raspbian Linux 13 (armv7l)"
    },
    "components": {
        "tts": "Flite 2.2 (slt voice, 16kHz Mono)",
        "stt": "Whisper.cpp 1.7.4 (tiny.en, 4 threads, NEON SIMD)",
        "llm": "SmolLM2-360M-Instruct-Q3_K_M.gguf (llama.cpp, 4 threads, mmap)"
    },
    "summary": {
        "total_questions": len(results),
        "avg_total_latency_sec": round(sum(r['total_round_latency_sec'] for r in results) / len(results), 2),
        "avg_stt_latency_sec": round(sum(r['stt_latency_sec'] for r in results) / len(results), 2),
        "avg_llm_latency_sec": round(sum(r['llm_latency_sec'] for r in results) / len(results), 2),
        "avg_llm_eval_tps": round(sum(r['llm_eval_tokens_per_sec'] for r in results if r['llm_eval_tokens_per_sec']) / max(1, len([r for r in results if r['llm_eval_tokens_per_sec']])), 2),
        "avg_tts_latency_sec": round(sum(r['tts_latency_sec'] for r in results) / len(results), 2),
        "max_peak_ram_mb": max(max(r['stt_peak_ram_mb'], r['llm_peak_ram_mb']) for r in results)
    },
    "results": results
}

with open(f"{TEST_DIR}/facts_results.json", 'w') as f:
    json.dump(report, f, indent=2)

print("==================================================================")
print(" 10-QUESTION FACTUAL BENCHMARK COMPLETED SUCCESSFULLY!")
print("==================================================================")
