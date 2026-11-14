"""Persistent Qdrant local store with exact cosine search and safe filters.

For the small lab corpus, retrieve all matching scores before applying a stable
chunk-ID tie-break. The similarity computation remains Qdrant's vector search,
not a separate Python vector index. This also makes a tie at rank 3 reproducible.
"""
from __future__ import annotations

from copy import deepcopy
from importlib.metadata import version
import json
import math
from pathlib import Path
from typing import Any
import uuid

from qdrant_client import QdrantClient, models


ALLOWED_FILTERS = frozenset({'section', 'document_type', 'language', 'version'})


class QdrantStore:
    def __init__(self, path: str | Path, dimensions: int = 384) -> None:
        if dimensions < 1:
            raise ValueError('dimensions must be positive')
        self.path = str(Path(path).resolve())
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        self.client = QdrantClient(path=self.path)
        self.metadata = {
            'backend': 'qdrant-local', 'version': version('qdrant-client'),
            'path': self.path, 'dimensions': dimensions, 'distance': 'cosine',
            'search': 'exact', 'tie_break': 'score descending, chunk_id ascending',
            'persistence': True,
        }

    def _vector(self, value: Any) -> list[float]:
        vector = [float(x) for x in value]
        if len(vector) != self.dimensions:
            raise ValueError(f'Expected {self.dimensions} dimensions, received {len(vector)}')
        if not all(math.isfinite(x) for x in vector) or not any(vector):
            raise ValueError('Vector must be finite and nonzero')
        return vector

    @staticmethod
    def point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, 'lab6:chunk:' + chunk_id))

    def _ensure_collection(self, collection: str) -> None:
        if not collection or not isinstance(collection, str):
            raise ValueError('Collection name must be a nonempty string')
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=self.dimensions, distance=models.Distance.COSINE),
            )
        else:
            vectors = self.client.get_collection(collection).config.params.vectors
            if not isinstance(vectors, models.VectorParams) or vectors.size != self.dimensions:
                raise ValueError('Existing collection has an incompatible vector schema')
            if vectors.distance != models.Distance.COSINE:
                raise ValueError('Existing collection does not use cosine distance')

    def _ids(self, collection: str) -> set[str]:
        ids: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(collection_name=collection, limit=256,
                                                offset=offset, with_payload=False, with_vectors=False)
            ids.update(str(p.id) for p in points)
            if offset is None:
                return ids

    def sync(self, collection: str, chunks: list[dict], vectors: list) -> dict:
        """Synchronize a complete collection snapshot, removing obsolete points.

        Caller must supply every document intended for this collection. All
        payload/vector validation precedes any database mutation. Errors propagate
        so an interrupted write cannot be reported as a successful run.
        """
        if len(chunks) != len(vectors):
            raise ValueError('Chunk/vector counts differ')
        points = []
        desired: set[str] = set()
        for chunk, raw_vector in zip(chunks, vectors):
            for field in ('chunk_id', 'document_id', 'source', 'text'):
                if not isinstance(chunk.get(field), str) or not chunk[field].strip():
                    raise ValueError(f'Chunk {field} must be a nonempty string')
            if not isinstance(chunk.get('page'), int) or isinstance(chunk['page'], bool) or chunk['page'] < 1:
                raise ValueError('Chunk page must be a positive integer')
            if not isinstance(chunk.get('metadata'), dict):
                raise ValueError('Chunk metadata must be a dictionary')
            point_id = self.point_id(chunk['chunk_id'])
            if point_id in desired:
                raise ValueError('Duplicate chunk ID in collection snapshot')
            desired.add(point_id)
            payload = deepcopy(chunk)
            json.dumps(payload, ensure_ascii=False, allow_nan=False)
            points.append(models.PointStruct(id=point_id, vector=self._vector(raw_vector), payload=payload))
        self._ensure_collection(collection)
        existing = self._ids(collection)
        for i in range(0, len(points), 128):
            self.client.upsert(collection_name=collection, points=points[i:i + 128], wait=True)
        obsolete = existing - desired
        if obsolete:
            self.client.delete(collection_name=collection,
                               points_selector=models.PointIdsList(points=sorted(obsolete)), wait=True)
        after = self.count(collection)
        if after != len(desired):
            raise RuntimeError('Collection count differs after synchronization')
        return {'collection': collection, 'before': len(existing), 'after': after,
                'inserted': len(desired - existing), 'updated': len(desired & existing),
                'deleted': len(obsolete), 'input_chunks': len(chunks)}

    @staticmethod
    def _filter(filters: dict[str, Any] | None) -> models.Filter | None:
        if not filters:
            return None
        if not isinstance(filters, dict):
            raise ValueError('Filters must be an object')
        conditions = []
        for key, value in sorted(filters.items()):
            if key not in ALLOWED_FILTERS:
                raise ValueError(f'Unsupported filter field {key!r}; gold/source/page filtering is forbidden')
            if not isinstance(value, (str, int, bool)):
                raise ValueError(f'Filter {key!r} must be a string, integer or boolean equality')
            conditions.append(models.FieldCondition(key='metadata.' + key,
                                                     match=models.MatchValue(value=value)))
        return models.Filter(must=conditions)

    def search(self, collection: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict]:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError('top_k must be a positive integer')
        vector = self._vector(query_vector)
        query_filter = self._filter(filters)
        if not self.client.collection_exists(collection):
            raise ValueError(f'Collection {collection!r} does not exist; ingest it first')
        matching = int(self.client.count(collection_name=collection,
                                        count_filter=query_filter, exact=True).count)
        if matching == 0:
            return []
        response = self.client.query_points(
            collection_name=collection, query=vector, query_filter=query_filter,
            limit=matching, with_payload=True, with_vectors=False,
            search_params=models.SearchParams(exact=True),
        )
        results = []
        for hit in response.points:
            payload = dict(hit.payload or {})
            payload['score'] = float(hit.score)
            results.append(payload)
        results.sort(key=lambda item: (-item['score'], item['chunk_id']))
        return results[:top_k]

    def count(self, collection: str) -> int:
        return int(self.client.count(collection_name=collection, exact=True).count)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> 'QdrantStore':
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
