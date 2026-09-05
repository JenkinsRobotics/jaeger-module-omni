import unittest
from research.agentic_numeric_audit import spoken_integer_match


class SpokenIntegerAuditTests(unittest.TestCase):
    def test_complete_cardinals_only(self):
        for answer, expected, valid in [
            ('The result is one thousand eighty-one.', '1081', True),
            ('It is two thousand one hundred and sixty-two.', '2162', True),
            ('The answer is one thousand eighty-one.', '81', False),
            ('The answer is one thousand eighty-two.', '1081', False),
            ('The answer is negative one.', '1', False),
            ('The answer is -one.', '1', False),
            ('The answer is one point five.', '1', False),
            ('The answer is one thousand eighty one million.', '1081', False),
        ]:
            with self.subTest(answer=answer, expected=expected):
                self.assertEqual(spoken_integer_match(answer, expected), valid)
