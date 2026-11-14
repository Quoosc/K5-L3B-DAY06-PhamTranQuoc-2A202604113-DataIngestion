"""Unit fixtures only: tiny synthetic vectors below are not lab embeddings/results."""
from __future__ import annotations

from copy import deepcopy
import re
import tempfile
import unittest

from src.chunking import chunk_pages, token_count
from src.store import QdrantStore


class FixtureTokenizer:
    """Deterministic whitespace/punctuation tokenizer for offset unit tests only."""
    def __call__(self, text, **kwargs):
        spans = [(m.start(), m.end()) for m in re.finditer(r'\w+|[^\w\s]', text)]
        return {'input_ids': list(range(len(spans))), 'offset_mapping': spans}

    def encode(self, text, **kwargs):
        return self(text)['input_ids']


def page(text):
    return {'document_id': 'fixture-doc', 'source': 'fixture://unit-test', 'page': 1,
            'text': text, 'metadata': {'section': 'strings', 'version': 'test',
                                      'nested': {'source_start': 20}}}


class ChunkTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FixtureTokenizer()

    def assert_valid(self, source, chunks, budget):
        covered = set()
        for chunk in chunks:
            a, b = chunk['start_offset'], chunk['end_offset']
            self.assertEqual(source['text'][a:b], chunk['text'])
            self.assertLessEqual(token_count(chunk['text'], self.tokenizer), budget)
            self.assertEqual(chunk['metadata'], source['metadata'])
            covered.update(range(a, b))
        self.assertTrue(all(i in covered for i, c in enumerate(source['text']) if not c.isspace()))
        self.assertEqual(len(chunks), len({c['chunk_id'] for c in chunks}))

    def test_fixed_unicode_offsets_coverage_and_determinism(self):
        source = page('  Tiếng Việt có dấu: một hai ba.\n\nSecond paragraph with punctuation! ' * 5)
        chunks = chunk_pages([source], 'A', self.tokenizer, 12, 3)
        self.assert_valid(source, chunks, 12)
        self.assertEqual(chunks, chunk_pages([source], 'A', self.tokenizer, 12, 3))
        self.assertTrue(any(a['end_offset'] > b['start_offset'] for a, b in zip(chunks, chunks[1:])))

    def test_paragraph_sentence_fallback_and_c_reuse(self):
        source = page('Heading\n\nFirst complete sentence. Second complete sentence.\n\n' +
                      ' '.join('oversized' for _ in range(45)) + '\n\nLast paragraph.')
        chunks = chunk_pages([source], 'B', self.tokenizer, 12, 3)
        self.assert_valid(source, chunks, 12)
        self.assertEqual(chunks, chunk_pages([source], 'C', self.tokenizer, 12, 3))
        self.assertTrue(all(a['end_offset'] <= b['start_offset'] for a, b in zip(chunks, chunks[1:])))
        chunks[0]['metadata']['nested']['source_start'] = 999
        self.assertEqual(source['metadata']['nested']['source_start'], 20)

    def test_invalid_parameters_and_duplicate_pages(self):
        source = page('One two three four.')
        for args in [('A', 3, 3), ('D', 12, 3), ('A', 1, 0)]:
            with self.assertRaises(ValueError):
                chunk_pages([source], args[0], self.tokenizer, args[1], args[2])
        with self.assertRaises(ValueError):
            chunk_pages([source, source], 'A', self.tokenizer, 12, 3)


def fixture_chunk(key, section):
    return {'chunk_id': key, 'document_id': 'fixture-doc', 'source': 'fixture://test',
            'page': 1, 'text': 'Synthetic persistence and filtering fixture ' + key,
            'metadata': {'section': section, 'language': 'en'},
            'start_offset': 0, 'end_offset': 10}


class StoreTests(unittest.TestCase):
    def test_persistence_filter_sync_and_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            chunks = [fixture_chunk('b', 'strings'), fixture_chunk('a', 'files')]
            vectors = [[1., 0., 0.], [1., 0., 0.]]
            with QdrantStore(directory, 3) as store:
                first = store.sync('fixture', chunks, vectors)
                self.assertEqual(first['inserted'], 2)
                second = store.sync('fixture', chunks, vectors)
                self.assertEqual(second['after'], 2)
                self.assertEqual(second['inserted'], 0)
                hits = store.search('fixture', [1., 0., 0.])
                self.assertEqual([h['chunk_id'] for h in hits], ['a', 'b'])
                filtered = store.search('fixture', [1., 0., 0.], filters={'section': 'strings'})
                self.assertEqual([h['chunk_id'] for h in filtered], ['b'])
                self.assertEqual(store.search('fixture', [1., 0., 0.], filters={'section': 'absent'}), [])
                with self.assertRaises(ValueError):
                    store.search('fixture', [1., 0., 0.], filters={'gold_source_id': 'fixture-doc'})
                with self.assertRaises(ValueError):
                    store.sync('fixture', chunks, [[1., 0.], [1., 0.]])
                self.assertEqual(store.count('fixture'), 2)
            with QdrantStore(directory, 3) as store:
                self.assertEqual(store.count('fixture'), 2)
                changed = deepcopy(chunks[0])
                changed['chunk_id'] = 'replacement'
                result = store.sync('fixture', [changed], [[0., 1., 0.]])
                self.assertEqual(result['deleted'], 2)
                self.assertEqual(store.search('fixture', [0., 1., 0.])[0]['chunk_id'], 'replacement')
                store.sync('fixture', [], [])
                self.assertEqual(store.count('fixture'), 0)


if __name__ == '__main__':
    unittest.main()
