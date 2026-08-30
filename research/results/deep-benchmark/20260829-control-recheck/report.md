# Jaeger Agent Omni deep benchmark

**Verdict: NOT READY**

Profile: `release` · commit `3d653864f7c2` · 15 pass / 4 fail / 1 skip · 976.5s

Model-backed gates are free-running. A skipped required gate is INCOMPLETE, never PASS.

| Group | Gate | Status | Required | Seconds | Detail |
|---|---|:---:|:---:|---:|---|
| cpu | `repo_contracts` | **PASS** | yes | 8.91 | exit 0 |
| cpu | `transcript_ban` | **PASS** | yes | 0.92 | exit 0 |
| cpu | `package_tests` | **PASS** | yes | 1.98 | exit 0 |
| agent | `agentic_selfcheck` | **PASS** | yes | 2.07 | exit 0 |
| evidence | `parity_artifacts` | **PASS** | yes | 0.05 | exit 0 |
| output | `live_assembly` | **PASS** | yes | 0.87 | exit 0 |
| output | `voice_contract` | **PASS** | yes | 0.88 | exit 0 |
| companion | `companion_scoring` | **PASS** | yes | 0.83 | exit 0 |
| agent | `agentic_source_parity` | **PASS** | yes | 1.60 | 5/5 exact checks |
| model | `omni_cases12` | **PASS** | yes | 21.22 | 12/12 · 579 ms |
| agent | `agentic_cases12` | **PASS** | yes | 50.38 | 12/12 · 2391 ms · unexpected=0 |
| duplex | `duplex_barge` | **PASS** | yes | 22.07 | interrupt 18.155374999878404 ms · drops 0 |
| output | `kokoro_pcm` | **PASS** | yes | 8.56 | 2.80s · 24000 Hz · 1ch |
| companion | `companion_ambient` | **FAIL** | yes | 25.55 | false responses 1/12 |
| companion | `build_addressed_audio` | **PASS** | yes | 11.38 | exit 0 |
| companion | `companion_addressed` | **FAIL** | yes | 26.67 | address recall 33.3% |
| perception | `neural_vad_cases12` | **PASS** | yes | 1.43 | opened 12/12 · ended 12/12 · false opens 0/5 |
| agent | `agentic_full_corpus` | **FAIL** | yes | 780.96 | 28/162 · routing 20/128 |
| aec | `aec_room_stress` | **FAIL** | yes | 10.19 | failed: reference_aware_p95_at_most_300ms |
| physical | `physical_five_minute` | **SKIP** | yes | 0.00 | pass --physical-report from a real 5-minute run |

Required failures: `companion_ambient`, `companion_addressed`, `agentic_full_corpus`, `aec_room_stress`

Missing required evidence: `physical_five_minute`
