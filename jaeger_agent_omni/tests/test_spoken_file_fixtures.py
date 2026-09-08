import unittest
from research.benchmark_agentic_corpus import spoken_file_names


class SpokenFileFixtureTest(unittest.TestCase):
    def test_filename_symbols_are_dictated_without_rewriting_arithmetic(self):
        text = 'Calculate 47 times 23, then save to level3_test.txt.'
        self.assertEqual(spoken_file_names(text),
            'Calculate 47 times 23, then save to level three underscore test dot t x t.')
        self.assertEqual(spoken_file_names('Read chain_note.txt and bench.txt.'),
            'Read chain underscore note dot t x t and bench dot t x t.')
