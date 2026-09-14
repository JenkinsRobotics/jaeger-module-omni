from jaeger_agent_omni.core.runtime.duplex_mode import DuplexModePolicy


def test_half_duplex_closes_capture_for_entire_response():
    policy = DuplexModePolicy("half")
    assert policy.accepts_capture()

    response_id = policy.begin_response()
    assert not policy.accepts_capture()

    policy.finish_response(response_id)
    assert policy.accepts_capture()


def test_full_duplex_keeps_capture_open_during_response():
    policy = DuplexModePolicy("full")
    policy.begin_response()
    assert policy.accepts_capture()


def test_older_playback_marker_cannot_reopen_a_newer_response():
    policy = DuplexModePolicy("half")
    older = policy.begin_response()
    newer = policy.begin_response()

    policy.finish_response(older)
    assert policy.response_active
    assert not policy.accepts_capture()

    policy.finish_response(newer)
    assert policy.accepts_capture()


def test_cancelled_response_marker_cannot_finish_a_new_response():
    policy = DuplexModePolicy("half")
    cancelled = policy.begin_response()
    policy.clear_responses()
    current = policy.begin_response()

    policy.finish_response(cancelled)
    assert policy.response_active
    assert not policy.accepts_capture()

    policy.finish_response(current)
    assert policy.accepts_capture()


def test_live_switch_applies_to_active_response_and_is_recorded():
    ticks = iter((12.5, 13.0))
    policy = DuplexModePolicy("full", clock=lambda: next(ticks))
    policy.begin_response()

    assert policy.set_mode("half") == ("full", "half")
    assert not policy.accepts_capture()
    assert policy.set_mode("full") == ("half", "full")
    assert policy.accepts_capture()
    assert policy.transitions() == [
        {
            "at_monotonic_s": 12.5,
            "from": "full",
            "to": "half",
            "response_active": True,
        },
        {
            "at_monotonic_s": 13.0,
            "from": "half",
            "to": "full",
            "response_active": True,
        },
    ]


def test_invalid_mode_fails_closed():
    try:
        DuplexModePolicy("speakerphone")
    except ValueError as exc:
        assert "half or full" in str(exc)
    else:
        raise AssertionError("invalid duplex mode was accepted")
