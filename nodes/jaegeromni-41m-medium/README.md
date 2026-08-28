# JaegerOmni 41M Medium

This is Jaeger's broad-corpus derivative of the frozen imported GemmaOmni 41M
Medium checkpoint. It is a separate model identity and never overwrites the
imported reference.

```text
stock OpenAI Whisper-medium encoder
  -> cleanroom_context_v2 adapter (40,915,456 parameters)
  -> frozen Gemma-4 E4B
```

The selected checkpoint is a 10% interpolation toward Jaeger's independently
retained room/multidomain continuation. The weight was selected on five locked
corpus gates without cases12. It ties the frozen import on two corpora and
removes 24 errors across the other three, then scores 3.97% free-running WER
on the untouched 51,311-word test-clean lockbox.

| gate | result |
|---|---:|
| test-clean lockbox | **3.97%**, 2,603 clips / 51,311 words |
| dev-clean64 | 4.92% |
| clean-360 validation128 | 5.18% |
| MINDS/Vox591 | 15.22% |
| novel Vox128 | 9.48% |
| AMI64 | 34.18% |
| cases12 | not used for selection; not rerun on this SHA |

The 300-second direct AMI duplex soak fed all audio without drops, produced 14
replies, continued listening through 13 playback overlaps, and completed a
real context roll at 6,010 logical tokens. Twelve-word cadence streaming cut
END-to-first-audio from 3,899/8,953 ms median/p95 to 1,044/3,725 ms on the
same input. This is not a physical-room or AEC measurement.

Select it in the full-duplex GUI as **JaegerOmni 41M Medium**, or use:

```bash
./.venv/bin/python demo.py --selftest-jaeger meeting.wav \
  --front-end jaegeromni-41m-medium --seconds 300
```
