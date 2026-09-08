"""Bound device shutdown and preserve failures before a process must exit."""
import threading
import time


def close_audio_stream(stream, *, after_close=None, timeout=5.0):
    errors = []
    started = time.monotonic()

    def close():
        try:
            stream.abort()
        except Exception as exc:
            errors.append(f'audio abort: {type(exc).__name__}: {exc}')
        closed = False
        try:
            stream.close()
            closed = True
        except Exception as exc:
            errors.append(f'audio close: {type(exc).__name__}: {exc}')
        # DSP may still be in use by a callback until the stream is closed.
        if closed and after_close is not None:
            try:
                after_close()
            except Exception as exc:
                errors.append(f'audio DSP close: {type(exc).__name__}: {exc}')

    worker = threading.Thread(target=close, daemon=True)
    worker.start()
    worker.join(timeout=timeout)
    completed = not worker.is_alive()
    result = list(errors)
    if not completed:
        result.append(f'audio shutdown exceeded {timeout:g} seconds')
    return {'completed': completed, 'errors': result,
            'seconds': time.monotonic() - started}
