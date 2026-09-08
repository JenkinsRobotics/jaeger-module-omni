"""Track speech clauses across synthesis, playback and cancellation threads."""
from __future__ import annotations
import threading
import time


class SpeechDelivery:
    def __init__(self):
        # The caller also holds this lock across playback start/stop, closing
        # the race between a validity check and the device starting old audio.
        self.lock = threading.RLock()
        self._rows = {}

    def begin(self):
        with self.lock:
            key = len(self._rows) + 1
            self._rows[key] = {'id': key, 'queued_clauses': 0, 'played_clauses': 0,
                               'closed': False, 'cancel_reason': None, 'error': None,
                               'started_at': time.monotonic(), 'finished_at': None}
            return key

    def valid(self, key):
        with self.lock:
            row = self._rows.get(key)
            return bool(row and row['cancel_reason'] is None and row['error'] is None)

    def enqueue(self, key):
        with self.lock:
            if not self.valid(key) or self._rows[key]['closed']:
                return False
            self._rows[key]['queued_clauses'] += 1
            return True

    def _finish(self, row):
        if (row['closed'] and row['queued_clauses'] > 0 and
                row['played_clauses'] == row['queued_clauses'] and
                row['cancel_reason'] is None and row['error'] is None and
                row['finished_at'] is None):
            row['finished_at'] = time.monotonic()

    def played(self, key):
        with self.lock:
            if self.valid(key):
                row = self._rows[key]
                row['played_clauses'] += 1
                self._finish(row)

    def close(self, key):
        with self.lock:
            self._rows[key]['closed'] = True
            self._finish(self._rows[key])

    def fail(self, key, error):
        with self.lock:
            self._rows[key]['error'] = str(error)

    def cancel_pending(self, reason='barge'):
        with self.lock:
            for row in self._rows.values():
                if row['finished_at'] is None and row['cancel_reason'] is None:
                    row['cancel_reason'] = reason

    def summary(self):
        with self.lock:
            rows = [dict(r) for r in self._rows.values()]
        return {'completed_uninterrupted_replies': sum(r['finished_at'] is not None for r in rows),
                'interrupted_replies': sum(r['cancel_reason'] == 'barge' and r['queued_clauses'] > 0 for r in rows),
                'shutdown_abandoned_replies': sum(r['cancel_reason'] == 'shutdown' and r['queued_clauses'] > 0 for r in rows),
                'failed_replies': sum(r['error'] is not None for r in rows),
                'played_clauses': sum(r['played_clauses'] for r in rows), 'rows': rows}
