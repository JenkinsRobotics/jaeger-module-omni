#!/usr/bin/env python3
"""Small MLX logits processors for deterministic transcript generation."""

from __future__ import annotations

from collections import Counter
import re


def make_ngram_loop_blocker(size: int = 4, max_occurrences: int = 2):
    """Block only tokens that would extend an already repeated token n-gram.

    Unlike a global repetition penalty, ordinary repeated words retain their
    original probabilities. With ``max_occurrences=2`` an n-gram may occur
    twice, but a third occurrence is prevented. This targets decoder loops
    while allowing natural phrases to recur.
    """
    if size < 2:
        raise ValueError("ngram size must be at least 2")
    if max_occurrences < 1:
        raise ValueError("max ngram occurrences must be positive")

    def processor(tokens, logits):
        import mlx.core as mx

        history = [int(token) for token in tokens.tolist()]
        if len(history) < size - 1:
            return logits
        prefix = tuple(history[-(size - 1):])
        counts = Counter(
            tuple(history[index:index + size])
            for index in range(len(history) - size + 1)
        )
        blocked = [
            gram[-1]
            for gram, count in counts.items()
            if count >= max_occurrences and gram[:-1] == prefix
        ]
        if not blocked:
            return logits
        indices = mx.array(sorted(set(blocked)))
        return logits.at[:, indices].add(-1e9)

    return processor


def make_word_ngram_loop_blocker(
    tokenizer, size: int = 4, max_occurrences: int = 1, candidate_count: int = 8
):
    """Veto top token candidates that complete a repeated normalized word n-gram.

    ``mlx_lm.generate_step`` gives processors the final prompt token followed by
    generated tokens, so the first token is excluded before decoding.  This is
    an inference-time logits constraint; it never edits emitted transcript text.
    """
    if size < 2:
        raise ValueError("word ngram size must be at least 2")
    if max_occurrences < 1 or candidate_count < 1:
        raise ValueError("occurrence and candidate counts must be positive")

    def normalized_words(token_ids):
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        return re.findall(r"[a-z0-9]+", text.lower())

    def processor(tokens, logits):
        import mlx.core as mx

        history = [int(token) for token in tokens.tolist()][1:]
        if len(normalized_words(history)) < size - 1:
            return logits
        count = min(candidate_count, int(logits.shape[-1]))
        indices = mx.argpartition(logits, kth=-count, axis=-1)[:, -count:]
        candidate_ids = [int(value) for value in indices.reshape(-1).tolist()]
        blocked = []
        for candidate in candidate_ids:
            words = normalized_words(history + [candidate])
            if len(words) < size:
                continue
            latest = tuple(words[-size:])
            occurrences = sum(
                tuple(words[index:index + size]) == latest
                for index in range(len(words) - size)
            )
            if occurrences >= max_occurrences:
                blocked.append(candidate)
        if not blocked:
            return logits
        return logits.at[:, mx.array(sorted(set(blocked)))].add(-1e9)

    return processor
