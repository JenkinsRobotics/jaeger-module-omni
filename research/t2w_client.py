"""Client for OpenBMB's token2wav service — speech codes in, waveform out.

We do not port flow-matching. OpenBMB ships an official Python service
(`llama.cpp-omni/tools/omni/pyt2w/token2wav_service.py`) that their own C++
drives over line-delimited JSON on stdin/stdout. We drive the same service the
same way, from an isolated venv (`.venv-t2w`) because its dependency pins
(torch 2.3.1, transformers 4.51) predate ours by a year.

    protocol   {"cmd": "init", "model_dir": ..., "device": ..., ...}
               {"cmd": "set_ref_audio", "ref_audio_path": ...}     <- the VOICE
               {"cmd": "process", "tokens": [...], "last_chunk": ..., "output_path": ...}
               {"cmd": "reset"} / {"cmd": "quit"}

`set_ref_audio` is voice cloning at the vocoder level — campplus.onnx embeds
the reference speaker (the 192-dim spk vector we measured in the prompt
cache). Load the Soul, you loaded the voice, again.

    python research/t2w_client.py --selfcheck        # protocol only, no models
    python research/t2w_client.py --say "hello"      # audible end-to-end
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
T2W_VENV_PY = REPO / ".venv-t2w" / "bin" / "python"
SERVICE = Path.home() / "GITHUB/llama.cpp-omni/tools/omni/pyt2w/token2wav_service.py"
MODEL_DIR = Path.home() / "models/MiniCPM-o-4_5-token2wav/assets/token2wav"


class Token2WavClient:
    """Line-JSON driver for the service subprocess."""

    def __init__(self, service_cmd: list | None = None, timeout_s: float = 300.0):
        self.timeout_s = timeout_s
        cmd = service_cmd or [str(T2W_VENV_PY), str(SERVICE)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True,
                                     bufsize=1)

    def _call(self, **cmd) -> dict:
        if self.proc.poll() is not None:
            raise RuntimeError(f"service exited with {self.proc.returncode}")
        self.proc.stdin.write(json.dumps(cmd) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("service closed stdout")
        resp = json.loads(line)
        if resp.get("status") != "ok":
            raise RuntimeError(f"{cmd.get('cmd')}: {resp.get('message')}")
        return resp

    def init(self, model_dir=MODEL_DIR, device: str = "cpu",
             float16: bool = False, n_timesteps: int = 5) -> dict:
        # float16 False on cpu — their default True assumes CUDA.
        return self._call(cmd="init", model_dir=str(model_dir), device=device,
                          float16=float16, n_timesteps=n_timesteps)

    def set_ref_audio(self, wav_path: str | Path) -> dict:
        return self._call(cmd="set_ref_audio", ref_audio_path=str(wav_path))

    def process(self, tokens: list, output_path: str | Path,
                last_chunk: bool = True) -> dict:
        return self._call(cmd="process", tokens=[int(t) for t in tokens],
                          last_chunk=last_chunk, output_path=str(output_path))

    def close(self):
        try:
            if self.proc.poll() is None:
                self._call(cmd="quit")
        except Exception:                              # noqa: BLE001
            pass
        finally:
            self.proc.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def speak(text: str, out_wav: str | Path, ref_audio: str | Path | None = None,
          device: str = "cpu", max_codes: int = 400) -> Path:
    """text -> talker codes -> token2wav -> waveform. The whole output side."""
    sys.path.insert(0, str(REPO / "research"))
    from load_minicpm_talker import N_TEXT_VOCAB, load as load_talker
    from transformers import AutoTokenizer

    talker = load_talker()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    ids = tok(text, return_tensors="pt").input_ids.clamp(max=N_TEXT_VOCAB - 1)
    merged = talker.condition(ids)                     # zero-conditioning baseline
    codes = talker.generate_codes(merged, max_codes=max_codes,
                                  temperature=0.8)     # sampling: greedy loops
    code_list = codes[0].tolist()

    with Token2WavClient() as t2w:
        t2w.init(device=device)
        if ref_audio:
            t2w.set_ref_audio(ref_audio)
        t2w.process(code_list, out_wav, last_chunk=True)
    return Path(out_wav)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--say", default="We choose to go to the moon.")
    ap.add_argument("--out", default=str(REPO / "training/runs/t2w_test.wav"))
    ap.add_argument("--ref-audio", default=None, help="voice to clone")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    for path, what in ((T2W_VENV_PY, ".venv-t2w (see pyt2w/requirements.txt)"),
                       (SERVICE, "llama.cpp-omni checkout"),
                       (MODEL_DIR / "flow.pt", "token2wav model download")):
        if not Path(path).exists():
            print(f"missing {path} — need {what}")
            return 1

    out = speak(args.say, args.out, ref_audio=args.ref_audio, device=args.device)
    import wave
    with wave.open(str(out)) as w:
        dur = w.getnframes() / w.getframerate()
    print(f"wrote {out} ({dur:.1f}s) — listen, then loop it through "
          f"whisper for the round-trip check")
    return 0


def demo():
    """Protocol driver against a fake service — no models, no venv."""
    fake = (
        "import sys, json\n"
        "for line in sys.stdin:\n"
        "    c = json.loads(line)\n"
        "    if c['cmd'] == 'quit':\n"
        "        print(json.dumps({'status': 'ok'})); break\n"
        "    if c['cmd'] == 'process' and not c.get('tokens'):\n"
        "        print(json.dumps({'status': 'error', 'message': 'empty'}))\n"
        "    else:\n"
        "        print(json.dumps({'status': 'ok', 'echo': c['cmd']}))\n"
        "    sys.stdout.flush()\n")

    c = Token2WavClient(service_cmd=[sys.executable, "-u", "-c", fake])
    assert c.init(model_dir="/x")["echo"] == "init"
    assert c.set_ref_audio("/v.wav")["echo"] == "set_ref_audio"
    assert c.process([1, 2, 3], "/o.wav")["echo"] == "process"

    # Service errors surface as exceptions naming the command, not as None.
    try:
        c.process([], "/o.wav")
    except RuntimeError as exc:
        assert "process" in str(exc) and "empty" in str(exc)
    else:
        raise AssertionError("service error must raise")
    c.close()
    assert c.proc.wait(timeout=5) is not None, "close must end the subprocess"

    # A dead service is reported, not hung on.
    d = Token2WavClient(service_cmd=[sys.executable, "-c", "pass"])
    import time
    time.sleep(0.3)
    try:
        d.init(model_dir="/x")
    except RuntimeError:
        pass
    else:
        raise AssertionError("dead service must raise")
    d.close()

    print("t2w_client: protocol, error surfacing, lifecycle OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        sys.exit(main())
