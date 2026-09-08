"""Immutable heads-free medium causal trial; converse seam only."""
import json
import sys
from pathlib import Path
NODE_ROOT = Path(__file__).resolve().parent
TRANSCRIPT_SYSTEM = "Repeat the spoken words exactly. Output only what was said, with no commentary or explanation."
def build(*, model_path=None, system=None, device="auto"):
    sys.path.insert(0, str(NODE_ROOT.parent.parent / "core"))
    from gemmaomni_adapter import CleanroomTranscriber
    from mlx_audio_streaming import AUDIO_MARKER
    spec = json.loads((NODE_ROOT / "node.json").read_text())
    ear = CleanroomTranscriber(adapter_path=NODE_ROOT / spec["adapter"]["path"],
        model_path=model_path, encoder={"source": str(NODE_ROOT / spec["encoder"]["path"]),
                                      "output_dim": spec["encoder"]["output_dim"]})
    rendered = ear.tokenizer.apply_chat_template(
        [{"role": "system", "content": system or TRANSCRIPT_SYSTEM},
         {"role": "user", "content": AUDIO_MARKER}], tokenize=False, add_generation_prompt=True)
    prefix, suffix = rendered.split(AUDIO_MARKER, 1)
    ear.prefix_ids = ear.tokenizer.encode(prefix, add_special_tokens=False)
    ear.suffix_ids = ear.tokenizer.encode(suffix, add_special_tokens=False)
    return ear
build_converse = build
