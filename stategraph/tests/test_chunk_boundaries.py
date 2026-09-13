import unittest

from stategraph.graphiti_adapter.state_extraction import _ordered_observation_chunks


class ChunkBoundaries(unittest.TestCase):
    def test_complete_lines_survive_packing(self):
        for text in (
            'speaker: hello there.\nother: red green blue yellow.\n',
            '- first list entry\n- second longer list entry\n- third entry\n',
            'First paragraph.\n\nSecond paragraph has several words.\n',
        ):
            with self.subTest(text=text):
                chunks = _ordered_observation_chunks(text, max_characters=40)
                self.assertEqual(''.join(chunks), text)
                self.assertTrue(all(len(c) <= 40 for c in chunks))
                for line in text.splitlines(keepends=True):
                    self.assertTrue(any(line in c for c in chunks))

    def test_long_message_uses_sentences_then_bounded_fallback(self):
        text = 'First sentence. Second sentence. ' + 'z' * 95
        chunks = _ordered_observation_chunks(text, max_characters=40)
        self.assertEqual(chunks[0], 'First sentence. Second sentence. ')
        self.assertEqual(''.join(chunks), text)
        self.assertTrue(all(0 < len(c) <= 40 for c in chunks))

    def test_newline_beats_later_space(self):
        text = 'prefix ' + 'x' * 22 + '\ncomplete message remains intact here\n'
        chunks = _ordered_observation_chunks(text, max_characters=40)
        self.assertEqual(chunks[0], 'prefix ' + 'x' * 22 + '\n')
        self.assertEqual(''.join(chunks), text)

    def test_offsets_and_no_overlap(self):
        text = 'alpha line\n' + 'beta gamma delta\n' * 5 + 'tail'
        chunks = _ordered_observation_chunks(text, max_characters=30)
        offset = 0
        for chunk in chunks:
            self.assertEqual(text[offset:offset + len(chunk)], chunk)
            offset += len(chunk)
        self.assertEqual(offset, len(text))
        self.assertEqual(''.join(chunks).count('beta gamma delta'), 5)
