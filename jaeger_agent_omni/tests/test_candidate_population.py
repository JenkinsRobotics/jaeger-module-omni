import copy
import unittest

from training.compare_candidate import validate_population


class CandidatePopulationTests(unittest.TestCase):
    def test_same_word_count_cannot_hide_changed_rows_or_protocol(self):
        original = {"reference_words": 2, "rows": [{"id": "a", "reference": "hello world"}],
                    "scorer_sha256": "one", "evaluation_mode": "transcribe"}
        validate_population(original, copy.deepcopy(original), "fixture")
        for field, value in (("rows", [{"id": "b", "reference": "hello world"}]),
                             ("scorer_sha256", "two"),
                             ("rows", [{"id": "a", "reference": "different words"}])):
            candidate = dict(original, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_population(original, candidate, "fixture")
