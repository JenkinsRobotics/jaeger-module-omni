from jaeger_agent_omni.core.agentic.observer import AgenticObserver, safe_preview


def test_safe_preview_redacts_secrets_and_bounds_payloads():
    preview = safe_preview({
        "query": "weather",
        "api_key": "do-not-show",
        "nested": {"Authorization": "Bearer hidden"},
        "payload": "x" * 500,
    })

    assert "do-not-show" not in preview
    assert "Bearer hidden" not in preview
    assert preview.count("<redacted>") == 2
    assert len(preview) <= 260


def test_observer_reports_plan_tools_and_results_without_opaque_callback_text():
    events = []
    ticks = iter((1.0, 2.1, 3.2, 4.3, 5.4, 6.5))
    observer = AgenticObserver(events.append, clock=lambda: next(ticks))
    callbacks = observer.callbacks()

    callbacks.on_thinking("the private chain of thought must never appear")
    callbacks.on_step(1, {
        "role": "assistant",
        "tool_calls": [{
            "name": "web_search",
            "arguments": {"query": "weather", "token": "hidden"},
        }],
    })
    callbacks.on_tool_progress(
        "web_search", "start", {"query": "weather", "token": "hidden"})
    callbacks.on_tool_done(
        "web_search", {"query": "weather"}, {"ok": True, "temp": 72},
        True, None, 0.25)

    rendered = " ".join(
        f"{event.title} {event.detail}" for event in events)
    assert "private chain of thought" not in rendered
    assert '"token": "hidden"' not in rendered
    assert any(event.kind == "activity" for event in events)
    assert any(event.kind == "plan" for event in events)
    assert any(event.kind == "tool" and event.phase == "start" for event in events)
    assert any(event.kind == "tool" and event.phase == "done" for event in events)


def test_heartbeat_is_rate_limited_to_new_seconds():
    events = []
    observer = AgenticObserver(events.append, clock=lambda: 10.0)
    callbacks = observer.callbacks()

    for elapsed in (0.1, 1.0, 1.4, 2.0, 2.2):
        callbacks.on_heartbeat(elapsed)

    assert [event.detail for event in events] == ["1.0s elapsed", "2.0s elapsed"]
