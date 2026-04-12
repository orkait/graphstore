"""VectorStore: HNSW vector index for semantic similarity search."""

import numpy as np
from usearch.index import Index


class VectorStore:
    """HNSW vector index backed by usearch."""

    def __init__(self, dims: int, capacity: int = 1024):
        self._dims = dims
        self._index = Index(ndim=dims, metric="cos", dtype="f32")
        self._has_vector = np.zeros(capacity, dtype=bool)
        self._capacity = capacity

    @property
    def dims(self) -> int:
        return self._dims

    def add(self, slot: int, vector: np.ndarray) -> None:
        """Add or replace vector for a slot."""
        if slot >= self._capacity:
            self.grow(max(slot + 1, self._capacity * 2))
        vec = np.asarray(vector, dtype=np.float32).ravel()
        if len(vec) != self._dims:
            raise ValueError(f"Expected {self._dims} dims, got {len(vec)}")
        if self._has_vector[slot]:
            self._index.remove(slot)
        self._index.add(slot, vec)
        self._has_vector[slot] = True

    def remove(self, slot: int) -> None:
        """Remove vector for a slot."""
        if slot < self._capacity and self._has_vector[slot]:
            self._index.remove(slot)
            self._has_vector[slot] = False

    def search(self, query: np.ndarray, k: int, mask: np.ndarray | None = None, oversample_factor: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors. Returns (slot_indices, distances).

        If mask provided, only slots where mask[slot]==True are considered.
        Uses oversampling + post-filter since usearch doesn't natively support masks.
        """
        query = np.asarray(query, dtype=np.float32).ravel()
        count = self.count()
        if count == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        if mask is not None:
            # Adaptive oversample and filter
            current_oversample = min(k * oversample_factor, count)
            max_oversample = min(count, max(current_oversample * 16, 10000))
            
            while True:
                results = self._index.search(query, current_oversample)
                valid = []
                for key, dist in zip(results.keys, results.distances):
                    key = int(key)
                    if key < len(mask) and mask[key]:
                        valid.append((key, float(dist)))
                        if len(valid) >= k:
                            break
                
                if len(valid) >= k or current_oversample >= max_oversample or current_oversample >= count:
                    break
                current_oversample = min(current_oversample * 2, count)

            if not valid:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
            slots = np.array([v[0] for v in valid], dtype=np.int64)
            dists = np.array([v[1] for v in valid], dtype=np.float32)
            return slots, dists
        else:
            actual_k = min(k, count)
            results = self._index.search(query, actual_k)
            return np.array(results.keys, dtype=np.int64), np.array(results.distances, dtype=np.float32)

    def has_vector(self, slot: int) -> bool:
        return slot < self._capacity and bool(self._has_vector[slot])

    def get_vector(self, slot: int) -> np.ndarray | None:
        """Get stored vector for a slot."""
        if not self.has_vector(slot):
            return None
        return np.array(self._index[slot], dtype=np.float32)

    def grow(self, new_capacity: int) -> None:
        old = self._has_vector
        self._has_vector = np.zeros(new_capacity, dtype=bool)
        self._has_vector[:len(old)] = old
        self._capacity = new_capacity

    def count(self) -> int:
        return int(np.sum(self._has_vector))

    @property
    def memory_bytes(self) -> int:
        """Approximate memory: vector data + HNSW graph overhead."""
        n = self.count()
        # ~(dims*4 + 64) bytes per vector in HNSW
        return n * (self._dims * 4 + 64) + self._has_vector.nbytes

    def save(self) -> bytes:
        """Serialize index to bytes."""
        return bytes(self._index.save(None))

    def load(self, data: bytes) -> None:
        """Deserialize index from bytes."""
        new_index = Index(ndim=self._dims, metric="cos", dtype="f32")
        new_index.load(data)
        self._index = new_index
        keys = list(new_index.keys)
        if keys:
            max_key = int(max(keys))
            new_capacity = max(self._capacity, max_key + 1)
            self._has_vector = np.zeros(new_capacity, dtype=bool)
            self._capacity = new_capacity
            for k in keys:
                self._has_vector[int(k)] = True
        else:
            self._has_vector = np.zeros(self._capacity, dtype=bool)
