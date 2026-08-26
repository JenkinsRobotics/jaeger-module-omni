"""CPU shim for OpenBMB's token2wav service.

stepaudio2 hardcodes `.cuda()` (e.g. `s3tokenizer.load_model(...).cuda()`), so
on a machine without CUDA the service dies in init regardless of its `device`
argument. Rather than fork their package, this bootstrap makes `.cuda()` a
no-op — modules and tensors stay where they are — then runs the unmodified
service.

    .venv-t2w/bin/python research/t2w_bootstrap.py   (launched by t2w_client)
"""

import runpy
import sys
from pathlib import Path

import torch

torch.nn.Module.cuda = lambda self, *a, **k: self
torch.Tensor.cuda = lambda self, *a, **k: self
# Some paths gate on availability rather than calling .cuda() directly; keep
# them honest so they pick CPU code paths.
torch.cuda.is_available = lambda: False

# stepaudio2 also builds tensors with device='cuda' literals
# (torch.zeros(..., device='cuda')), which .cuda() shims never see. Remap the
# device kwarg at the factory functions.
def _remap(kw):
    if "device" in kw and kw["device"] is not None and "cuda" in str(kw["device"]):
        kw["device"] = "cpu"
    return kw

for _name in ("tensor", "zeros", "ones", "empty", "full", "arange",
              "linspace", "randn", "rand", "as_tensor", "randint"):
    _f = getattr(torch, _name)
    def _wrap(f):
        def g(*a, **k):
            return f(*a, **_remap(k))
        return g
    setattr(torch, _name, _wrap(_f))

# And explicit .to("cuda") for completeness.
_orig_to = torch.Tensor.to
def _to(self, *a, **k):
    a = tuple("cpu" if isinstance(x, (str, torch.device)) and "cuda" in str(x)
              else x for x in a)
    return _orig_to(self, *a, **_remap(k))
torch.Tensor.to = _to

SERVICE = (Path(__file__).resolve().parent.parent / "jaeger_agent_omni" /
           "runtime" / "token2wav_service.py")
sys.argv = [str(SERVICE)]
runpy.run_path(str(SERVICE), run_name="__main__")
