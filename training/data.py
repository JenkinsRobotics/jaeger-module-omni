"""Tiered corpus assembly for Stage 3/4.

Four tiers, each doing a different job (see the plan doc's "Data" section).
The tiering exists because they are not interchangeable — mixing them in the
wrong proportion is how a projector ends up excellent on audiobooks and useless
across a room.

    TIER 1  labeled ASR      LibriSpeech, People's Speech, Common Voice
                             Ground truth transcripts. No ASR error enters the
                             teacher path, so no confidence gate is needed.
    TIER 2  far-field        AMI, CHiME. Real rooms, real distance.
                             Stops overfitting to close-mic studio audio.
    TIER 3  emotion-labeled  CREMA-D, RAVDESS, MELD.
                             Ground truth for Stage 6's enriched teacher.
    TIER 4  our sessions     Our room, actors, characters. Lowest volume,
                             highest relevance. Only source of real turn
                             boundaries for the Stage 5 turn-end head.

This module owns the *mixing policy* and nothing else. Loading is deferred to
`datasets` at call time so the policy stays testable offline.

    python training/data.py     # self-check, no download

ponytail: a dict of weights and a sampler, not a data-pipeline framework.
Reach for webdataset when a single machine stops keeping up, not before.
"""

import math
from dataclasses import dataclass
from enum import Enum


class Tier(Enum):
    LABELED_ASR = 1
    FAR_FIELD = 2
    EMOTION = 3
    OURS = 4


@dataclass(frozen=True)
class Source:
    name: str
    tier: Tier
    hf_id: str | None = None      # None = local, e.g. our own captures
    split: str = "train"
    licence_check: bool = True    # True = verify terms before first use
    notes: str = ""


# Candidate sources. hf_id values need verifying against the hub before a real
# run — dataset ids and access terms both drift.
CATALOG = (
    Source("librispeech", Tier.LABELED_ASR, "openslr/librispeech_asr",
           notes="~960h clean read speech; the standard warmup set"),
    Source("common_voice", Tier.LABELED_ASR, "mozilla-foundation/common_voice_17_0",
           notes="accent + speaker diversity"),
    Source("peoples_speech", Tier.LABELED_ASR, "MLCommons/peoples_speech",
           notes="~30k h; volume and acoustic variety"),
    Source("ami", Tier.FAR_FIELD, "edinburghcstr/ami",
           notes="real meeting rooms, mic arrays, overlap"),
    Source("crema_d", Tier.EMOTION, "confit/cremad",
           notes="acted emotion, clean labels"),
    Source("ravdess", Tier.EMOTION, "narad/ravdess",
           notes="acted emotion, speech + song"),
    Source("jaeger_sessions", Tier.OURS, None,
           licence_check=False,
           notes="research/capture.py output; the only real turn boundaries"),
)

# Per-stage mixing weights. Absent tier = not used at that stage, which is a
# statement rather than an oversight:
#   Stage 3  warmup only needs to land the projector in the right region.
#   Stage 4  ours weighted up — it is the distribution actually served.
#   Stage 5  ours ONLY. Nobody else's turn boundaries are ours.
#   Stage 6  emotion ground truth, grounded by our own realism.
STAGE_MIX = {
    "stage3_warmup":  {Tier.LABELED_ASR: 0.8, Tier.FAR_FIELD: 0.2},
    "stage4_distill": {Tier.LABELED_ASR: 0.4, Tier.FAR_FIELD: 0.25, Tier.OURS: 0.35},
    "stage5_turnend": {Tier.OURS: 1.0},
    "stage6_paraling": {Tier.EMOTION: 0.6, Tier.OURS: 0.4},
}


def sources_for(stage: str) -> tuple:
    """Catalog entries participating in a stage, heaviest tier first."""
    if stage not in STAGE_MIX:
        raise KeyError(f"unknown stage {stage!r}; have {sorted(STAGE_MIX)}")
    mix = STAGE_MIX[stage]
    return tuple(sorted((s for s in CATALOG if s.tier in mix),
                        key=lambda s: -mix[s.tier]))


def sample_counts(stage: str, total: int) -> dict:
    """Samples per tier for a run of `total`. Largest-remainder, so the parts
    sum to exactly `total` rather than drifting by a few from rounding."""
    if total < 0:
        raise ValueError("total must be non-negative")
    mix = STAGE_MIX[stage]
    exact = {t: total * w for t, w in mix.items()}
    counts = {t: int(math.floor(v)) for t, v in exact.items()}
    for tier in sorted(mix, key=lambda t: -(exact[t] - counts[t]))[:total - sum(counts.values())]:
        counts[tier] += 1
    return counts


def needs_licence_review(stage: str) -> tuple:
    """Sources whose terms must be checked before first use. Several
    well-known corpora are research-only or need an access agreement."""
    return tuple(s.name for s in sources_for(stage) if s.licence_check)


def demo():
    # Every mix is a probability distribution, or the sampler silently skews.
    for stage, mix in STAGE_MIX.items():
        assert abs(sum(mix.values()) - 1.0) < 1e-9, (stage, sum(mix.values()))
        assert all(w > 0 for w in mix.values()), stage

    # Every tier in every mix has at least one source behind it.
    for stage in STAGE_MIX:
        assert sources_for(stage), stage
        covered = {s.tier for s in sources_for(stage)}
        assert covered == set(STAGE_MIX[stage]), (stage, covered)

    # Stage 5 is ours alone — turn boundaries cannot be borrowed.
    assert [s.name for s in sources_for("stage5_turnend")] == ["jaeger_sessions"]

    # Stage 3 must NOT train on our own sessions: warmup on a tiny in-domain
    # corpus is how a projector memorises instead of aligning.
    assert Tier.OURS not in STAGE_MIX["stage3_warmup"]

    # Stage 4 weights ours up relative to stage 3 (where it is absent).
    assert STAGE_MIX["stage4_distill"][Tier.OURS] > 0.3

    # Counts sum exactly, including awkward totals.
    for total in (0, 1, 7, 100, 999, 1000):
        for stage in STAGE_MIX:
            counts = sample_counts(stage, total)
            assert sum(counts.values()) == total, (stage, total, counts)
            assert all(c >= 0 for c in counts.values())

    # Dominant tier gets the most samples at a realistic size.
    c = sample_counts("stage3_warmup", 1000)
    assert c[Tier.LABELED_ASR] == 800 and c[Tier.FAR_FIELD] == 200

    try:
        sources_for("stage9_nonsense")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown stage must raise")

    try:
        sample_counts("stage3_warmup", -1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative total must raise")

    # Our own data needs no licence review; public corpora do.
    assert "jaeger_sessions" not in needs_licence_review("stage4_distill")
    assert len(needs_licence_review("stage3_warmup")) >= 2

    print(f"data: {len(CATALOG)} sources across {len(Tier)} tiers, "
          f"{len(STAGE_MIX)} stage mixes")
    print(f"data: stage4 = {sample_counts('stage4_distill', 1000)}")
    print("data: mixes normalised, coverage, exact counts, licence flags OK")


if __name__ == "__main__":
    demo()
