# Jaeger Agent Omni deep benchmark

**Verdict: NOT READY**

Profile: `release` · commit `57790378f166` · 15 pass / 4 fail / 1 skip · 949.7s

Model-backed gates are free-running. A skipped required gate is INCOMPLETE, never PASS.

| Group | Gate | Status | Required | Seconds | Detail |
|---|---|:---:|:---:|---:|---|
| cpu | `repo_contracts` | **PASS** | yes | 10.56 | exit 0 |
| cpu | `transcript_ban` | **PASS** | yes | 0.91 | exit 0 |
| cpu | `package_tests` | **PASS** | yes | 2.02 | exit 0 |
| agent | `agentic_selfcheck` | **PASS** | yes | 2.12 | exit 0 |
| evidence | `parity_artifacts` | **PASS** | yes | 0.06 | exit 0 |
| output | `live_assembly` | **PASS** | yes | 0.82 | exit 0 |
| output | `voice_contract` | **PASS** | yes | 0.85 | exit 0 |
| companion | `companion_scoring` | **PASS** | yes | 0.85 | exit 0 |
| agent | `agentic_source_parity` | **PASS** | yes | 1.58 | 5/5 exact checks |
| model | `omni_cases12` | **PASS** | yes | 20.41 | 12/12 · 565 ms |
| agent | `agentic_cases12` | **PASS** | yes | 43.84 | 12/12 · 2320 ms · unexpected=0 |
| duplex | `duplex_barge` | **PASS** | yes | 21.67 | interrupt 17.803833994548768 ms · drops 0 |
| output | `kokoro_pcm` | **PASS** | yes | 8.86 | 2.80s · 24000 Hz · 1ch |
| companion | `companion_ambient` | **FAIL** | yes | 25.55 | false responses 1/12 |
| companion | `build_addressed_audio` | **PASS** | yes | 11.85 | exit 0 |
| companion | `companion_addressed` | **FAIL** | yes | 26.71 | address recall 33.3% |
| perception | `neural_vad_cases12` | **PASS** | yes | 1.72 | opened 12/12 · ended 12/12 · false opens 0/5 |
| agent | `agentic_full_corpus` | **FAIL** | yes | 759.70 | 28/162 · routing 20/128 |
| aec | `aec_room_stress` | **FAIL** | yes | 9.67 | failed: reference_aware_p95_at_most_300ms |
| physical | `physical_five_minute` | **SKIP** | yes | 0.00 | pass --physical-report from a real 5-minute run |

Required failures: `companion_ambient`, `companion_addressed`, `agentic_full_corpus`, `aec_room_stress`

Missing required evidence: `physical_five_minute`
