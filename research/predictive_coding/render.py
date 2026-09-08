"""Render recorded experiments as offline HTML; no training occurs in JS."""
import json
from pathlib import Path


def render(report, output):
    template = Path(__file__).with_name("view.html").read_text()
    # Only measured JSON is inserted; template markup stays literal and readable.
    payload = json.dumps(report, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    fragment = template.replace("__MEASURED_DATA__", payload)
    output.with_suffix(".fragment.html").write_text(fragment)
    style = """<style>
    :root {color-scheme:light dark; --foreground:light-dark(#1e293b,#e2e8f0);
    --background:light-dark(#fafafa,#10151e); --border:light-dark(#cbd5e1,#475569);
    --viz-series-1:light-dark(#007c83,#49d5df); --viz-series-2:light-dark(#ae510c,#ffb56b);
    --viz-series-3:light-dark(#7042b2,#c3a2ff); --font-size-base:16px;}
    body {background:var(--background);color:var(--foreground);font:16px/1.5 system-ui;
    max-width:900px;margin:24px auto;padding:0 18px;}
    h2,h3 {font-weight:500;} .viz-controls,.viz-row {display:flex;gap:16px;flex-wrap:wrap;align-items:center;}
    .form-label {display:grid;grid-template-columns:minmax(0,1fr);gap:6px;flex:1;min-width:180px;}
    select,input {font:inherit;max-width:100%;min-width:0;} select {width:100%;}
    .form-range {width:100%;} .text-small {font-size:14px;} .tabular-nums {font-variant-numeric:tabular-nums;}
    [hidden] {display:none!important;}
    </style>"""
    output.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                      '<meta name="viewport" content="width=device-width,initial-scale=1">'
                      '<title>Jaeger predictive coding lab</title>'+style+'</head><body>'
                      +fragment+'</body></html>')
