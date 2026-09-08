#!/usr/bin/env python3
"""Real microphone/speaker acoustic replay, never a simulated physical pass.

prepare uses CPU Kokoro only. run captures the actual selected microphone
through live.py and plays an external actor through the selected room speaker.
The live runtime owns its own playback reference; actor speech is not injected
into the model or the AEC reference. All agent tools use a temporary workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = Path("/tmp/jaeger-physical-actor")
SCRIPT = [
    (6, "Jaeger, what is seven times eight? Use your calculator."),
    (26, "Jaeger, what time is it? Use your clock."),
    (46, "Jaeger, calculate one hundred and forty four divided by twelve."),
    (66, "Alex, please put the groceries in the kitchen. I am talking to Alex."),
    (86, "Jaeger, explain how a rainbow forms in six sentences."),
    (96, "Jaeger, stop. What is two plus two?"),
    (116, "Jaeger, calculate nineteen times twenty three."),
    (136, "Jaeger, tell me a long story about a rabbit exploring a forest."),
    (149, "Jaeger, stop the story. What color is a clear daytime sky?"),
    (169, "Sam, the package is at the front door. That message is for Sam."),
    (189, "Jaeger, what time is it now?"),
    (209, "Jaeger, calculate three hundred and twenty divided by eight."),
    (229, "Jaeger, explain how plants use sunlight."),
    (249, "Jaeger, count slowly from one to twenty."),
    (259, "Jaeger, stop counting. What is ten minus three?"),
    (279, "Jaeger, thank you. Goodbye."),
]
DEMO_SCRIPT = [
    (6, "Jaeger, create a file called workspace slash demo note dot txt containing the words blue cedar."),
    (26, "Jaeger, read workspace slash demo note dot txt and tell me what it says."),
    (46, "Jaeger, calculate forty seven times twenty three plus twelve using your calculator."),
    (66, "Alex, please put the groceries in the kitchen. I am talking to Alex."),
    (86, "Jaeger, explain how a rainbow forms in six sentences."),
    (96, "Jaeger, stop. What is two plus two?"),
    (116, "Jaeger, what time is it now? Use your clock."),
    (136, "Jaeger, tell me a long story about a rabbit exploring a forest."),
    (149, "Jaeger, stop the story. What color is a clear daytime sky?"),
    (169, "Sam, the package is at the front door. That message is for Sam."),
    (189, "Jaeger, remember that my favorite tree is cedar."),
    (209, "Jaeger, what tree did I say was my favorite?"),
    (229, "Jaeger, explain how plants use sunlight."),
    (249, "Jaeger, count slowly from one to twenty."),
    (259, "Jaeger, stop counting. What is ten minus three?"),
    (279, "Jaeger, thank you. Goodbye."),
]


def actor_schedule(script, seconds, repeat=False):
    """Fixed five-minute blocks; long soaks must continue exercising speech."""
    offset = 0
    while offset < seconds:
        for when, words in script:
            if offset + when < seconds:
                yield offset + when, words
        if not repeat:
            break
        offset += 300


def fixture(words):
    return CACHE / (hashlib.sha256(words.encode()).hexdigest()[:20] + ".wav")


def prepare():
    import torch
    import soundfile as sf
    from jaeger_agent_omni.node_registry import load_factory, load_package
    torch.set_num_threads(2)
    engine = load_factory(load_package(ROOT / "jaeger_agent_omni/nodes/kokoro-tts"))(
        lang="a", device="cpu")
    CACHE.mkdir(parents=True, exist_ok=True)
    for _, words in SCRIPT + DEMO_SCRIPT:
        if not fixture(words).exists():
            sf.write(fixture(words), engine.synthesize(words, voice="am_adam"), 24000)
    print(f"Prepared physical actor fixtures for both scripts on CPU")


def run(args):
    import sounddevice as sd
    import soundfile as sf
    from jaeger_agent_omni.benchmark import _physical_payload
    from training.context_distill import refuse_gpu_contention
    refuse_gpu_contention()
    script = DEMO_SCRIPT if args.script == 'demo-v1' else SCRIPT
    schedule = list(actor_schedule(script, args.seconds, args.repeat_script))
    missing = [str(fixture(words)) for _, words in script if not fixture(words).is_file()]
    if missing:
        raise RuntimeError("Run prepare before the GPU trial")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    native_report = args.output.with_suffix(".live.json")
    log_path = args.output.with_suffix(".log")
    devices = {"input": dict(sd.query_devices(args.input_device)),
               "output": dict(sd.query_devices(args.output_device))}
    events, errors = [], []
    stop = threading.Event()

    def actor():
        started = time.monotonic()
        for offset, words in schedule:
            if offset >= args.seconds or stop.wait(max(0, started+offset-time.monotonic())):
                break
            try:
                audio, rate = sf.read(fixture(words), dtype="float32")
                begin = time.monotonic()
                with sd.OutputStream(device=args.output_device, samplerate=rate,
                                     channels=1, dtype="float32") as stream:
                    stream.write(audio.reshape(-1, 1))
                events.append({"scheduled_s": offset, "actual_s": begin-started,
                               "text": words, "audio": str(fixture(words)),
                               "sha256": hashlib.sha256(fixture(words).read_bytes()).hexdigest()})
            except Exception as exc:
                errors.append(f"actor: {type(exc).__name__}: {exc}")
                break

    with tempfile.TemporaryDirectory(prefix="jaeger-physical-tools-") as workspace:
        # This changes this child's PortAudio default, not the user's OS setting.
        child = (
            "import sounddevice as sd,runpy; "
            f"sd.default.device=({args.input_device},{args.output_device}); "
            "from research.benchmark_agentic_corpus import CorpusPermissionProvider; "
            "from jaeger_agent_omni.core.agentic._vendor.jaeger_os.core.safety.permissions "
            "import PermissionPolicy,use_policy; "
            "policy=use_policy(PermissionPolicy(confirmation=CorpusPermissionProvider())); "
            "policy.__enter__(); runpy.run_module('jaeger_agent_omni.live',run_name='__main__')"
        )
        command = [sys.executable, "-u", "-c", child,
                   "--seconds", str(args.seconds), "--device", str(args.input_device),
                   "--front-end", "jaegeromni-41m-medium", "--vad", "neural",
                   "--max-context", "4500", "--aec-reference-mode", "contiguous",
                   "--duplex-mode", "full", "--barge-policy", "stop",
                   "--interaction-mode", "companion", "--agentic",
                   "--agentic-workspace", workspace,
                   "--record", str(args.output.with_suffix(".audio")),
                   "--report", str(native_report)]
        thread = None
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    log.write(line); log.flush()
                    if "talk over the reply to test barge-in" in line and thread is None:
                        thread = threading.Thread(target=actor, daemon=True)
                        thread.start()
                    if "[soak]" in line or "JAEGER" in line or "report " in line:
                        print(line.strip(), flush=True)
                returncode = process.wait()
            finally:
                stop.set()
                if process.poll() is None:
                    process.terminate(); process.wait(timeout=15)
                if thread:
                    thread.join(timeout=20)
        from research.benchmark_agentic_corpus import artifact_snapshot
        final_files = artifact_snapshot(workspace)
    payload = json.loads(native_report.read_text()) if native_report.exists() else {}
    passed, detail, metrics = _physical_payload(payload)
    report = {"command": [sys.executable, *sys.argv], "devices": devices,
              "kind": "physical_acoustic_replay_not_human_conversation",
              "actor_injected_into_model": False, "native_report": str(native_report),
              "log": str(log_path), "events": events, "actor_errors": errors,
              "child_exit_code": returncode, "physical_checks": metrics.get("checks"),
              "script": args.script, "repeat_script": args.repeat_script,
              "script_sha256": hashlib.sha256(json.dumps(script).encode()).hexdigest(),
              "scheduled_events": len(schedule), "final_tool_files": final_files,
              "speech_delivery": (payload.get('metrics') or {}).get('speech_delivery'),
              "duration_seconds": (payload.get('metrics') or {}).get('wall_seconds'),
              "passed": passed and returncode == 0 and not errors,
              "detail": detail,
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


def calibrate(args):
    import numpy as np
    import sounddevice as sd
    import soundfile as sf
    from scipy.signal import correlate, resample_poly
    audio, rate = sf.read(fixture(SCRIPT[0][1]), dtype="float32")
    chunks = []
    with sd.InputStream(device=args.input_device, samplerate=48000, channels=1,
                        callback=lambda data, *_: chunks.append(data[:, 0].copy())):
        time.sleep(1)
        with sd.OutputStream(device=args.output_device, samplerate=rate,
                             channels=1, dtype="float32") as output:
            output.write(audio[:, None])
        time.sleep(1)
    captured = np.concatenate(chunks)
    reference = resample_poly(audio, 48000, rate)
    correlation = correlate(captured, reference, mode="valid", method="fft")
    offset = int(np.argmax(abs(correlation)))
    similarity = float(correlation[offset] / max(
        1e-10, np.linalg.norm(captured[offset:offset+len(reference)]) * np.linalg.norm(reference)))
    report = {"command": [sys.executable, *sys.argv],
              "kind": "physical_speaker_microphone_coupling", "direct_injection": False,
              "input_device": dict(sd.query_devices(args.input_device)),
              "output_device": dict(sd.query_devices(args.output_device)),
              "played_seconds": len(audio)/rate, "captured_seconds": len(captured)/48000,
              "ambient_rms": float(np.sqrt(np.mean(captured[:48000]**2))),
              "captured_rms": float(np.sqrt(np.mean(captured**2))),
              "peak_normalized_correlation": similarity, "best_offset_s": offset/48000,
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    sf.write(CACHE/"coupling-capture.wav", captured, 48000)
    print(json.dumps(report), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "calibrate", "run"))
    p.add_argument("--input-device", type=int, default=1)
    p.add_argument("--output-device", type=int, default=4)
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument('--script', choices=['original', 'demo-v1'], default='original')
    p.add_argument('--repeat-script', action='store_true',
                   help='repeat the selected fixed script every 300 seconds during long soaks')
    p.add_argument("--output", type=Path, default=ROOT/"research/candidate_evals/physical-acoustic-20260904.json")
    args = p.parse_args()
    if args.action == "prepare":
        prepare(); return 0
    if args.action == "calibrate":
        return calibrate(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
