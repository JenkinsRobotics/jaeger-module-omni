#!/usr/bin/env python3
"""Replay measured AEC telemetry to screen policy candidates without DSP reruns.

This is selection evidence only. Any candidate must rerun the full waveform
gate and a new room seed before promotion; telemetry holds the AEC fixed.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jaeger_agent_omni.core.runtime.aec import EchoAwareBargeIn


def replay(rows, options):
    delays, false, missed = [], 0, 0
    for row in rows:
        detector = EchoAwareBargeIn(16000, .012, **options)
        first = None
        for trace in row["detector_trace"]:
            aec = SimpleNamespace(playback_generation=1,
                                  block_playback_age=trace["end_s"]-.1,
                                  block_attenuation_db=trace["attenuation_db"],
                                  raw_reference_correlation=trace["raw_correlation"],
                                  residual_reference_correlation=trace["residual_correlation"])
            if detector.update(aec, trace["clean_rms"], 1600):
                first = trace["end_s"]; break
        if first is None:
            missed += 1
        elif first < row["near_onset_s"]:
            false += 1
        else:
            delays.append(1000*(first-row["near_onset_s"]))
    return {"false": false, "missed": missed, "detected": len(delays),
            "p95_ms": sorted(delays)[int(np.ceil(.95*len(delays)))-1] if delays else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); data = json.loads(args.trace.read_text())
    baseline = replay(data["rows"], data["detector"])
    # Verify replay before trusting any hypothetical policy result.
    recorded = data["reference_aware"]
    assert baseline["false"] == recorded["false_barges"]
    assert baseline["detected"] == recorded["detected"]
    assert abs(baseline["p95_ms"]-recorded["p95_delay_ms"]) < 1e-6
    trials = []
    for corr, attenuation, confirm in itertools.product((.35,.4,.45,.5), (2.5,4,6), (100,200,250)):
        options = dict(data["detector"], residual_echo_correlation=corr,
                       strong_attenuation_db=attenuation, strong_confirm_ms=confirm)
        trials.append({"options": options, **replay(data["rows"], options)})
    trials.sort(key=lambda r:(r["false"],r["missed"],r["p95_ms"] or float("inf")))
    report = {"command": [sys.executable,*sys.argv], "baseline": baseline,
              "trials": trials, "selection_only": True, "runtime_changed": False}
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(trials[:3],indent=2))


if __name__ == "__main__":
    main()
