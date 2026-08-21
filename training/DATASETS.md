# Open Training Data — the reference archive

Every open corpus we use or may use, by modality and by which adapter it
feeds. Licences are as-known and **must be re-verified before a run that
matters** — terms drift, and several "open" sets are research-only or
non-commercial. `training/fetch_dataset.py --list` shows what is actually
wired; ✔ marks schemas verified live against the hub.

Legend: ✔ wired+verified · ○ wired, unverified schema · — candidate only ·
⚠ licence needs attention for a monetised channel

---

## Speech → text (feeds: audio delta, Stage 3/4)

| Corpus | Size | Licence | Status | Notes |
|---|---|---|---|---|
| LibriSpeech clean/other | 960 h | CC BY 4.0 | ✔ | read audiobooks; the standard warmup. train.100/train.500 wired |
| FLEURS en_us | ~10 h | CC BY 4.0 | ✔ | clean, many languages available later |
| VoxPopuli en | ~500 h | CC0 | ○ | parliamentary, accented English, public domain |
| Common Voice | huge | CC0 | ○ | accent/speaker diversity; needs HF token + terms click |
| People's Speech | ~30k h | CC BY / BY-SA | ○ | volume + acoustic variety; big download |
| Multilingual LibriSpeech | 50k h | CC BY 4.0 | — | when multilingual matters |
| GigaSpeech | 10k h | agreement | ⚠ | varied domains; access agreement required |
| TED-LIUM 3 | 450 h | CC BY-**NC**-ND | ⚠ | spontaneous talks; NC — reference/eval only for us |
| SPGISpeech | 5k h | agreement | ⚠ | business speech; agreement |

## Far-field / noise / rooms (Tier 2 robustness)

| Corpus | Size | Licence | Status | Notes |
|---|---|---|---|---|
| AMI (ihm/sdm) | 100 h | CC BY 4.0 | ○ | real meeting rooms, distance mics, overlap |
| CHiME-5/6 | ~50 h | agreement | — | dinner-party noise; the hard case |
| MUSAN | 109 h | CC BY 4.0 | — | noise/music/babble for augmentation mixing |
| OpenSLR-28 RIRs | small | Apache | — | room impulse responses for reverb augmentation |
| WHAM! | 80 h | CC BY-NC | ⚠ | ambient noise mixes |

## Emotion / paralinguistics (feeds: Stage 6 enriched teacher)

| Corpus | Size | Licence | Status | Notes |
|---|---|---|---|---|
| CREMA-D | 7.4k clips | ODbL | ○ | acted, 6 emotions, clean labels |
| MELD | 13k utt | research | ⚠ | conversational emotion from TV dialogue |
| RAVDESS | 7.3k clips | CC BY-**NC**-SA | ⚠ | acted speech+song; NC |
| ESD | 35 h | research | ⚠ | emotional speech, en/zh |
| IEMOCAP | 12 h | restricted | ⚠ | the classic; signed release required |
| MSP-Podcast | 400 h | restricted | ⚠ | natural emotion at scale; academic licence |

## Speech generation / voices (feeds: speech delta, Track A voice work)

| Corpus | Size | Licence | Status | Notes |
|---|---|---|---|---|
| LibriTTS-R | 585 h | CC BY 4.0 | — | restored 24 kHz TTS-grade LibriSpeech |
| LJSpeech | 24 h | public domain | — | single speaker, the TTS classic |
| VCTK | 44 h | CC BY 4.0 | — | 110 speakers, accents |
| Hi-Fi TTS | 292 h | CC BY 4.0 | — | 10 speakers, high quality |
| EARS | 100 h | CC BY-NC | ⚠ | expressive, anechoic, 107 speakers |
| Emilia | 101k h | CC BY-NC | ⚠ | in-the-wild voices at scale; NC |
| AudioSet (speech subsets) | — | CC BY 4.0 | — | YouTube-labelled; clips are links, not files |

## Sound understanding (future: environmental hearing)

| Corpus | Size | Licence | Status | Notes |
|---|---|---|---|---|
| Clotho v2 | 7k clips | open | — | audio captioning |
| AudioCaps | 46k | research | ⚠ | captions over AudioSet clips |
| ESC-50 / UrbanSound8K | small | CC | — | sound-event classification baselines |
| WavCaps | 400k | CC BY-**NC** | ⚠ | LLM-cleaned captions |

## Image → text (feeds: vision delta, same recipe as audio)

| Corpus | Size | Licence | Status | Notes |
|---|---|---|---|---|
| COCO Captions | 330k img | CC BY 4.0 | — | the warmup standard, 5 captions/image |
| Visual Genome | 108k img | CC BY 4.0 | — | dense region descriptions |
| TextVQA / TextCaps | 28k img | CC BY 4.0 | — | reading text in images |
| VQAv2 | 200k img | CC BY 4.0 | — | question answering, the eval standard |
| GQA | 113k img | CC BY 4.0 | — | compositional reasoning |
| DocVQA | 12k doc | research | ⚠ | documents; check terms |
| LLaVA-Instruct | 158k conv | research | ⚠ | GPT-generated — synthetic-teacher caveat |
| ShareGPT4V | 100k | research | ⚠ | detailed captions, GPT-sourced |

## Video (+audio) → text (after vision delta works on stills)

| Corpus | Size | Licence | Status | Notes |
|---|---|---|---|---|
| VATEX | 41k clips | CC BY 4.0 | — | captioned video |
| ActivityNet Captions | 20k vids | research | ⚠ | dense temporal captions |
| MSR-VTT | 10k clips | research | ⚠ | the classic; research terms |
| Panda-70M | 70M clips | research | ⚠ | scale; auto-captioned |
| Ego4D | 3.7k h | agreement | ⚠ | egocentric — closest to robot-eye view |
| LRS3 | 400 h | restricted | ⚠ | lip-reading / AV speech; TED-sourced |

## Conversation & turn-taking (feeds: turn-end head, duplex behaviour)

| Corpus | Size | Licence | Status | Notes |
|---|---|---|---|---|
| our own sessions | growing | ours | ✔ | the only source of OUR turn boundaries; capture.py |
| prefix augmentation | ∞ | — | ✔ | turn-end labels manufactured from any speech corpus |
| CANDOR | 850 h | agreement | ⚠ | natural two-party conversation |
| Fisher / Switchboard | 2k h | LDC ($$) | ⚠ | telephone conversation; paid licence |
| ICSI meetings | 72 h | CC BY 4.0 | — | overlap-heavy meetings |

## Text / instruct (backbone sanity, PerformanceSpec prediction later)

| Corpus | Size | Licence | Status | Notes |
|---|---|---|---|---|
| OASST 1/2 | 160k msg | Apache 2.0 | — | human conversation trees |
| Dolly-15k | 15k | CC BY-SA | — | human-written instructions |
| UltraChat 200k | 200k | MIT | — | synthetic dialogue |
| FineWeb-Edu | 1.3T tok | ODC-By | — | if we ever pretrain anything text-side |

---

## Rules of use

1. **Tier the mix** (`data.py`): warmup on read speech, robustness from
   far-field, realism from our own sessions, emotion from labelled sets.
2. **Human labels beat volume** — a wrong pseudo-label teaches the projector
   to reproduce the mistake faithfully. Ground truth needs no confidence gate.
3. **⚠ NC/research sets never touch a shipped adapter.** They may inform
   evals and papers; monetised-channel artifacts train on CC0/CC-BY/Apache
   only. When in doubt, leave it out.
4. Normalise everything at the adapter (`fetch_dataset.py`): 16 kHz mono
   float32, lowercase text, markers stripped — one convention, every corpus.
5. Hub ids drift. The fetcher fails loudly per-corpus and keeps what landed.
