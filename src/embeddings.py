"""Real local E5 embeddings. Fail visibly; never fall back to random vectors."""
from __future__ import annotations

from importlib.metadata import version
import math
from pathlib import Path


class LocalEmbedder:
    """CPU-first Sentence Transformers adapter with explicit E5 prefixes."""

    def __init__(self, model_name: str = 'intfloat/multilingual-e5-small',
                 revision: str | None = None, cache_folder: str = '.local/model',
                 model_path: str | None = None, batch_size: int = 32,
                 device: str = 'cpu') -> None:
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        Path(cache_folder).mkdir(parents=True, exist_ok=True)
        kwargs = {'device': device, 'cache_folder': cache_folder, 'trust_remote_code': False}
        if revision and not model_path:
            kwargs['revision'] = revision
        self.model = SentenceTransformer(model_path or model_name, **kwargs)
        self.tokenizer = self.model.tokenizer
        self.max_input_tokens = min(512, int(self.model.max_seq_length))
        self.model.max_seq_length = self.max_input_tokens
        self.dimensions = int(self.model.get_sentence_embedding_dimension())
        config = getattr(self.model._first_module(), 'auto_model', None)
        config = getattr(config, 'config', None)
        resolved_revision = getattr(config, '_commit_hash', None)
        self.metadata = {
            'provider': 'local_sentence_transformers', 'model_name': model_name,
            'requested_revision': revision, 'resolved_revision': resolved_revision or revision,
            'model_path': model_path, 'dimensions': self.dimensions, 'device': device,
            'batch_size': batch_size, 'normalize_embeddings': True,
            'document_prefix': 'passage: ', 'query_prefix': 'query: ',
            'max_input_tokens': self.max_input_tokens,
            'sentence_transformers_version': version('sentence-transformers'),
            'transformers_version': version('transformers'), 'torch_version': version('torch'),
            'api_cost_usd': 0.0,
        }
        # Measure an actual output; a declared model dimension alone is not proof.
        probe = self._encode(['query: local embedding dimension check'])
        self.metadata['measured_dimensions'] = len(probe[0])

    def _encode(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        for i, text in enumerate(inputs):
            length = len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False))
            if length > self.max_input_tokens:
                raise ValueError(f'Input {i} has {length} tokens; limit {self.max_input_tokens}. '
                                 'Refusing silent truncation; change chunking before evaluation.')
        vectors = self.model.encode(inputs, batch_size=self.batch_size,
                                    normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False, prompt='').tolist()
        if len(vectors) != len(inputs):
            raise RuntimeError('Embedder returned the wrong number of vectors')
        for vector in vectors:
            if len(vector) != self.dimensions or not all(math.isfinite(float(x)) for x in vector):
                raise RuntimeError('Embedding has invalid dimension or non-finite values')
            if not math.isclose(sum(float(x) ** 2 for x in vector), 1.0, abs_tol=0.001):
                raise RuntimeError('Embedding is not a normalized nonzero vector')
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError('Document texts must be nonempty strings')
        return self._encode(['passage: ' + t for t in texts])

    def embed_query(self, question: str) -> list[float]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Query must be a nonempty string')
        return self._encode(['query: ' + question])[0]
