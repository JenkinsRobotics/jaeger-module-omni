"""CLI: voice in, cognition out.

    python -m jaeger_agent_omni --audio clip.wav --backbone <hf-id>
    python -m jaeger_agent_omni --audio clip.wav --backbone <hf-id> --path omni
    python -m jaeger_agent_omni --audio clip.wav --backbone <hf-id> \
        --capture-to corpus/           # every turn becomes training data

Both paths share the same weights; only the route differs. The text path works
today. The omni path produces nonsense until its projector is trained — run it
anyway to confirm the plumbing, which is a separate question from the weights.
"""

import argparse
import sys

from .app import PATHS, OmniApp, load_models, read_wav


def pick_device(requested=None):
    import torch
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="jaeger_agent_omni",
        description="Voice in, cognition out.")
    ap.add_argument("--audio", required=True, help="16 kHz mono PCM wav (resampled if not)")
    ap.add_argument("--backbone", required=True,
                    help="HF id or local path for the cognition model")
    ap.add_argument("--whisper", default="openai/whisper-medium")
    ap.add_argument("--path", default="text", choices=PATHS)
    ap.add_argument("--projector", default=None,
                    help="checkpoint from training/train.py (omni path)")
    ap.add_argument("--capture-to", default=None,
                    help="corpus directory — text path only")
    ap.add_argument("--soul", default="mochi")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--language", default="en",
                    help="pinned, not auto-detected: a short clip can be "
                         "mis-detected and transcribed into the wrong language")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    device = pick_device(args.device)
    audio = read_wav(args.audio)
    print(f"audio  {len(audio)/16000:.1f}s  path={args.path}  device={device}")

    processor, whisper, tokenizer, backbone = load_models(
        args.whisper, args.backbone, device=device,
        need_decoder=(args.path == "text"))

    projector = None
    if args.projector:
        from .neural import JaegerAudioProjector
        ck = torch.load(args.projector, map_location=device, weights_only=False)
        projector = JaegerAudioProjector(
            audio_dim=whisper.config.d_model, cognition_dim=ck["cognition_dim"])
        projector.load_state_dict(ck["projector"])
        trained_on = ck.get("config", {}).get("backbone_id", "?")
        if trained_on not in ("?", args.backbone):
            # Not fatal — a local path and an HF id can name the same weights —
            # but silently pairing a projector with a different backbone is the
            # failure mode that looks like a bad model instead of a bad load.
            print(f"WARNING projector was trained against {trained_on!r}, "
                  f"running against {args.backbone!r}")
        print(f"projector loaded: step {ck.get('step')}, dim {ck['cognition_dim']}")
    elif args.path == "omni":
        print("WARNING no --projector: running an untrained projector. Output "
              "will be nonsense; this checks plumbing, not weights.")

    app = OmniApp(processor, whisper, tokenizer, backbone, path=args.path,
                  projector=projector, device=device, soul=args.soul,
                  capture_to=args.capture_to, max_new_tokens=args.max_new_tokens,
                  language=args.language)

    out, timing = app.listen(audio)
    print(f"\n{out.text}\n")
    print(f"timing {timing}")
    if app.capture is not None:
        print(f"captured to {args.capture_to}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
