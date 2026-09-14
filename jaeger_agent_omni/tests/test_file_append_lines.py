"""append_file must line-separate — the frozen a/b:ms_write_append_read
contract wants todo.txt to read back as two lines, not "buy milkwalk dog"."""
import tempfile
import unittest
from jaeger_agent_omni.core.agentic.workspace import ensure_bound
from jaeger_agent_omni.tools.files import append_file, file_read, file_write


class AppendSeparatesLines(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.layout = ensure_bound(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _content(self, name: str) -> str:
        return file_read(path=name)["content"]

    def test_frozen_todo_contract_reads_back_as_two_lines(self):
        file_write(path="todo.txt", content="buy milk")
        append_file(path="todo.txt", content="walk dog")
        self.assertEqual(
            [s.strip() for s in self._content("todo.txt").splitlines()],
            ["buy milk", "walk dog"],
        )

    def test_existing_trailing_newline_is_not_doubled(self):
        file_write(path="log.txt", content="first\n")
        append_file(path="log.txt", content="second")
        self.assertEqual(self._content("log.txt"), "first\nsecond")

    def test_caller_supplied_newline_is_not_doubled(self):
        file_write(path="log2.txt", content="first")
        append_file(path="log2.txt", content="\nsecond")
        self.assertEqual(self._content("log2.txt"), "first\nsecond")

    def test_append_to_new_file_adds_no_leading_newline(self):
        append_file(path="fresh.txt", content="only")
        self.assertEqual(self._content("fresh.txt"), "only")


if __name__ == "__main__":
    unittest.main()
