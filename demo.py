#!/usr/bin/env python3
"""Temporary demo GUI for Jaeger Agent Omni. One file, stdlib server.

    python demo.py --backbone <hf-id> [--whisper openai/whisper-medium.en]
                   [--path text|omni] [--projector ckpt] [--port 7860]

Open http://localhost:7860 — talk (mic stays open, silence ends the turn,
speaking over the reply cancels it), or type. Replies land in the log and are
spoken at the same time.

What is real and what is not, so the demo is honest:

    text input     -> backbone, with conversation history          REAL
    audio input    -> whisper -> backbone (text path, default)     REAL
                   -> their frozen front end -> delta -> backbone  REAL once
                      (--path omni --projector <ckpt>)             delta trains
    speech output  -> browser speechSynthesis                      STAND-IN for
                                                                   Track A TTS
    video input    -> preview only. The SigLIP path is Stage 7;    NOT WIRED
                      nothing fake happens to the frames.
    turn-taking    -> JS RMS VAD, 700 ms hangover — the legacy     STAND-IN for
                      stack's numbers, in the browser              the turn-end
                                                                   head

ponytail: stdlib http.server + inline HTML, no gradio/flask/websockets. Browser
AEC (getUserMedia echoCancellation) instead of Speex. Raw PCM over POST instead
of codec streams. This file is meant to be replaced, not grown.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jaeger_agent_omni.app import OmniApp, load_models          # noqa: E402
from jaeger_agent_omni.capture import SAMPLE_RATE               # noqa: E402

SYSTEM_PROMPT = ("You are Jaeger, a helpful voice companion. Answer in 1-3 "
                 "short conversational sentences. No markdown, no lists.")


class Brain:
    """Model calls + conversation history, behind one lock (single GPU)."""

    def __init__(self, app: OmniApp, tokenizer, backbone, device: str):
        self.app = app
        self.tok = tokenizer
        self.lm = backbone
        self.device = device
        self.lock = threading.Lock()
        self.history: list = []

    def _respond(self, user_text: str) -> tuple:
        import torch
        msgs = ([{"role": "system", "content": SYSTEM_PROMPT}]
                + self.history + [{"role": "user", "content": user_text}])
        prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                              tokenize=False)
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        t0 = time.monotonic()
        with torch.no_grad():
            out = self.lm.generate(ids, max_new_tokens=110, do_sample=False,
                                   pad_token_id=self.tok.pad_token_id
                                   or self.tok.eos_token_id)
        reply = self.tok.decode(out[0][ids.shape[1]:],
                                skip_special_tokens=True).strip()
        gen_ms = (time.monotonic() - t0) * 1000
        self.history += [{"role": "user", "content": user_text},
                         {"role": "assistant", "content": reply}]
        self.history = self.history[-16:]          # bounded context
        return reply, gen_ms

    def chat(self, text: str) -> dict:
        with self.lock:
            reply, gen_ms = self._respond(text)
        return {"reply": reply, "timing": f"generate {gen_ms:.0f}ms"}

    def audio(self, pcm: np.ndarray) -> dict:
        with self.lock:
            if self.app.path == "omni":
                # Single-turn for now: the omni path has no history splicing yet.
                out, t = self.app.listen(pcm)
                return {"reply": out.text, "transcript": "(no transcript — omni)",
                        "timing": str(t)}
            t0 = time.monotonic()
            transcript = self.app.transcribe(pcm)
            asr_ms = (time.monotonic() - t0) * 1000
            if not transcript:
                return {"reply": "", "transcript": "", "timing": "silence"}
            reply, gen_ms = self._respond(transcript)
            return {"reply": reply, "transcript": transcript,
                    "timing": f"transcribe {asr_ms:.0f}ms + generate {gen_ms:.0f}ms"}

    def reset(self):
        with self.lock:
            self.history = []


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Jaeger Agent Omni — demo</title><style>
 body{font:14px -apple-system,sans-serif;background:#111;color:#ddd;margin:0;
      display:grid;grid-template-columns:320px 1fr;height:100vh}
 #side{padding:14px;border-right:1px solid #333;overflow:auto}
 #main{display:flex;flex-direction:column;height:100vh}
 #log{flex:1;overflow-y:auto;padding:16px}
 .u{color:#8cf}.a{color:#dfd}.m{color:#888;font-size:11px;margin:2px 0 12px}
 .row{display:flex;gap:8px;padding:12px;border-top:1px solid #333}
 input[type=text]{flex:1;background:#222;color:#ddd;border:1px solid #444;
      border-radius:6px;padding:10px}
 button{background:#2a6;border:0;color:#fff;border-radius:6px;padding:10px 16px;
      cursor:pointer}button.off{background:#444}button.rec{background:#c33}
 video{width:100%;border-radius:8px;background:#000}
 #vu{height:6px;background:#333;border-radius:3px;margin:8px 0}
 #vu>div{height:6px;width:0;background:#2a6;border-radius:3px}
 .note{color:#777;font-size:11px;margin:6px 0}
 h3{margin:12px 0 4px;font-size:13px;color:#aaa}
</style></head><body>
<div id="side">
 <h3>Jaeger Agent Omni</h3><div class="note" id="cfg">loading…</div>
 <h3>Microphone</h3>
 <button id="mic" class="off">mic off</button>
 <div id="vu"><div></div></div>
 <div class="note">continuous · 700&thinsp;ms silence ends the turn ·
   speaking over the reply cancels it (barge-in)</div>
 <h3>Camera</h3>
 <button id="cam" class="off">camera off</button>
 <video id="vid" muted playsinline></video>
 <div class="note">preview only — vision input lands at Stage&nbsp;7;
   frames are not sent anywhere</div>
 <h3>Session</h3><button id="reset" class="off">reset conversation</button>
</div>
<div id="main">
 <div id="log"></div>
 <div class="row">
  <input id="txt" type="text" placeholder="type, or just talk…">
  <button id="send">send</button>
 </div>
</div>
<script>
const log=document.getElementById('log'),txt=document.getElementById('txt');
function add(cls,who,text,meta){
 const d=document.createElement('div');d.className=cls;
 d.textContent=who+'  '+text;log.appendChild(d);
 if(meta){const m=document.createElement('div');m.className='m';
   m.textContent=meta;log.appendChild(m);}
 log.scrollTop=log.scrollHeight;}
function speak(t){if(!t)return;speechSynthesis.cancel();
 speechSynthesis.speak(new SpeechSynthesisUtterance(t));}
async function post(url,body,hdr){const r=await fetch(url,{method:'POST',
 body:body,headers:hdr||{}});return r.json();}

fetch('/config').then(r=>r.json()).then(c=>{
 document.getElementById('cfg').textContent=
  c.backbone+' · '+c.whisper+' · path='+c.path;});

async function sendText(){const t=txt.value.trim();if(!t)return;txt.value='';
 add('u','you',t);const r=await post('/chat',JSON.stringify({text:t}),
 {'Content-Type':'application/json'});add('a','jaeger',r.reply,r.timing);
 speak(r.reply);}
document.getElementById('send').onclick=sendText;
txt.addEventListener('keydown',e=>{if(e.key==='Enter')sendText();});
document.getElementById('reset').onclick=async()=>{await post('/reset','');
 add('m','','— conversation reset —');};

// ---- mic: continuous capture, RMS VAD, barge-in ----------------------------
const HANGOVER_MS=700,MIN_SPEECH_MS=400,PREROLL_MS=250,THRESH=0.012;
let micOn=false,ac,src,proc,stream;
let voiced=false,silence=0,buf=[],preroll=[],speechMs=0;
const vu=document.querySelector('#vu>div');
document.getElementById('mic').onclick=async function(){
 if(micOn){micOn=false;this.textContent='mic off';this.className='off';
   proc&&proc.disconnect();src&&src.disconnect();
   stream&&stream.getTracks().forEach(t=>t.stop());vu.style.width=0;return;}
 stream=await navigator.mediaDevices.getUserMedia({audio:{
   echoCancellation:true,noiseSuppression:true}});
 ac=new AudioContext();src=ac.createMediaStreamSource(stream);
 proc=ac.createScriptProcessor(4096,1,1);
 src.connect(proc);proc.connect(ac.destination);
 micOn=true;this.textContent='listening…';this.className='rec';
 proc.onaudioprocess=e=>{
  const x=e.inputBuffer.getChannelData(0);
  let s=0;for(let i=0;i<x.length;i++)s+=x[i]*x[i];
  const rms=Math.sqrt(s/x.length),blockMs=x.length/ac.sampleRate*1000;
  vu.style.width=Math.min(100,rms*800)+'%';
  const isVoice=rms>THRESH;
  if(isVoice&&speechSynthesis.speaking)speechSynthesis.cancel(); // barge-in
  if(!voiced){
   preroll.push(new Float32Array(x));
   while(preroll.length*blockMs>PREROLL_MS)preroll.shift();
   if(isVoice){voiced=true;buf=preroll.slice();preroll=[];
     speechMs=0;silence=0;}
  }else{
   buf.push(new Float32Array(x));speechMs+=blockMs;
   silence=isVoice?0:silence+blockMs;
   if(silence>=HANGOVER_MS){
    voiced=false;
    if(speechMs-silence>=MIN_SPEECH_MS)shipTurn(buf,ac.sampleRate);
    buf=[];}
  }};
};
async function shipTurn(blocks,rate){
 let n=0;blocks.forEach(b=>n+=b.length);
 const all=new Float32Array(n);let o=0;
 blocks.forEach(b=>{all.set(b,o);o+=b.length;});
 const m=Math.round(n*16000/rate),out=new Int16Array(m);
 for(let i=0;i<m;i++){const p=i*rate/16000,l=Math.floor(p),
  f=Math.min(l+1,n-1),v=all[l]+(all[f]-all[l])*(p-l);
  out[i]=Math.max(-32768,Math.min(32767,v*32767));}
 add('u','you','🎤 …');
 const r=await post('/audio',out.buffer,{'Content-Type':
  'application/octet-stream','X-Sample-Rate':'16000'});
 log.lastChild.textContent='you  🎤 '+(r.transcript||'(silence)');
 if(r.reply){add('a','jaeger',r.reply,r.timing);speak(r.reply);}}

// ---- camera preview (not wired to cognition) -------------------------------
let camOn=false,camStream;
document.getElementById('cam').onclick=async function(){
 const v=document.getElementById('vid');
 if(camOn){camOn=false;this.textContent='camera off';this.className='off';
   camStream.getTracks().forEach(t=>t.stop());v.srcObject=null;return;}
 camStream=await navigator.mediaDevices.getUserMedia({video:true});
 v.srcObject=camStream;v.play();camOn=true;
 this.textContent='camera on (preview)';this.className='rec';};
</script></body></html>"""


def make_handler(brain: Brain, config: dict):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):                       # quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif self.path == "/health":
                self._send(200, '{"ok": true}')
            elif self.path == "/config":
                self._send(200, json.dumps(config))
            else:
                self._send(404, '{"error": "not found"}')

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n)
            try:
                if self.path == "/chat":
                    text = json.loads(raw)["text"]
                    self._send(200, json.dumps(brain.chat(text)))
                elif self.path == "/audio":
                    rate = int(self.headers.get("X-Sample-Rate", SAMPLE_RATE))
                    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
                    if rate != SAMPLE_RATE:
                        m = int(round(len(pcm) * SAMPLE_RATE / rate))
                        pcm = np.interp(np.linspace(0, len(pcm) - 1, m),
                                        np.arange(len(pcm)), pcm).astype(np.float32)
                    self._send(200, json.dumps(brain.audio(pcm)))
                elif self.path == "/reset":
                    brain.reset()
                    self._send(200, '{"ok": true}')
                else:
                    self._send(404, '{"error": "not found"}')
            except Exception as exc:                     # noqa: BLE001
                self._send(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
    return H


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--whisper", default="openai/whisper-medium.en")
    ap.add_argument("--path", default="text", choices=["text", "omni"])
    ap.add_argument("--projector", default=None)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"loading models on {device}…")
    processor, whisper, tokenizer, backbone = load_models(
        args.whisper, args.backbone, device=device, need_decoder=True)

    projector = None
    if args.projector and Path(args.projector).is_dir():
        # The standard form: an exported artifact directory.
        from jaeger_agent_omni.artifacts import load_audio_path
        enc, projector = load_audio_path(args.projector)
        if enc is not None:
            whisper = enc          # their fine-tuned encoder replaces stock
        print(f"artifact loaded: {args.projector}")
    elif args.projector:
        # Raw training checkpoint — works, but export it: 
        #   python -m jaeger_agent_omni.artifacts export <ckpt> artifacts/<name>
        ck = torch.load(args.projector, map_location=device, weights_only=False)
        if ck.get("config", {}).get("arch") == "minicpm_delta":
            sys.path.insert(0, str(Path(__file__).parent / "research"))
            from load_minicpm import load_front_end
            from jaeger_agent_omni.neural import DeltaAdapter, MiniCPMDeltaPath
            enc, front = load_front_end()
            whisper = enc
            delta = DeltaAdapter(front.cognition_dim, ck["cognition_dim"],
                                 embed_norm=ck["embed_norm"])
            projector = MiniCPMDeltaPath(front, delta)
            projector.load_state_dict(ck["projector"])
            print(f"minicpm_delta checkpoint loaded (step {ck['step']})")
        else:
            from jaeger_agent_omni.neural import JaegerAudioProjector
            projector = JaegerAudioProjector(
                audio_dim=whisper.config.d_model, cognition_dim=ck["cognition_dim"],
                embed_norm=ck.get("embed_norm") or None)
            projector.load_state_dict(ck["projector"])

    app = OmniApp(processor, whisper, tokenizer, backbone, path=args.path,
                  projector=projector, device=device)
    brain = Brain(app, tokenizer, backbone, device)
    config = {"backbone": args.backbone, "whisper": args.whisper,
              "path": args.path}

    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 make_handler(brain, config))
    print(f"demo up: http://localhost:{args.port}  (path={args.path})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
