"""Memory consolidation: cluster episodic memories into observations.

TSM-inspired durative memory construction WITHOUT LLM:
    1. Group messages by entity (via graph edges)
    2. Cluster by cosine similarity within entity groups
    3. Pick most representative message per cluster as observation text
    4. Track evidence (source message count, recency)

This is the "sleep-time consolidation" step from the TSM paper,
implemented as a pure numpy/scipy operation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Observation:
    """A consolidated memory - one cluster of related episodic facts."""
    entity: str
    text: str
    evidence_slots: List[int]
    evidence_count: int
    centroid: Optional[np.ndarray] = None
    event_at_ms: Optional[int] = None
    confidence: float = 0.0


def cluster_by_entity(
    entity_to_slots: Dict[str, List[int]],
    vectors: Optional[np.ndarray],
    has_vector: Optional[np.ndarray],
    texts: Dict[int, str],
    event_times: Optional[Dict[int, int]] = None,
    similarity_threshold: float = 0.7,
    min_cluster_size: int = 1,
) -> List[Observation]:
    """Cluster messages by entity, then by semantic similarity.

    Args:
        entity_to_slots: entity_name -> list of message slot indices
        vectors: full vector array (slot-indexed), or None if no embedder
        has_vector: boolean mask of which slots have vectors
        texts: slot -> message text content
        event_times: slot -> __event_at__ epoch ms (optional)
        similarity_threshold: cosine sim threshold for same-cluster
        min_cluster_size: minimum messages to form an observation

    Returns list of Observation objects, one per cluster.
    """
    observations: list[Observation] = []

    for entity, slots in entity_to_slots.items():
        if not slots:
            continue

        # If no vectors, each message is its own observation
        if vectors is None or has_vector is None:
            for s in slots:
                text = texts.get(s, "")
                if not text:
                    continue
                evt = event_times.get(s) if event_times else None
                observations.append(Observation(
                    entity=entity, text=text,
                    evidence_slots=[s], evidence_count=1,
                    event_at_ms=evt,
                    confidence=1.0,
                ))
            continue

        # Get slots that have vectors
        vec_slots = [s for s in slots if s < len(has_vector) and has_vector[s]]
        non_vec_slots = [s for s in slots if s not in set(vec_slots)]

        if not vec_slots:
            # No vectors - each message standalone
            for s in slots:
                text = texts.get(s, "")
                if text:
                    evt = event_times.get(s) if event_times else None
                    observations.append(Observation(
                        entity=entity, text=text,
                        evidence_slots=[s], evidence_count=1,
                        event_at_ms=evt, confidence=1.0,
                    ))
            continue

        # Cluster by cosine similarity (greedy single-linkage)
        slot_vecs = vectors[vec_slots]
        # Normalize for cosine
        norms = np.linalg.norm(slot_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        slot_vecs_norm = slot_vecs / norms

        assigned = [False] * len(vec_slots)
        clusters: list[list[int]] = []

        for i in range(len(vec_slots)):
            if assigned[i]:
                continue
            cluster = [i]
            assigned[i] = True
            centroid = slot_vecs_norm[i].copy()

            for j in range(i + 1, len(vec_slots)):
                if assigned[j]:
                    continue
                sim = float(np.dot(centroid, slot_vecs_norm[j]))
                if sim >= similarity_threshold:
                    cluster.append(j)
                    assigned[j] = True
                    # Update centroid (running mean)
                    centroid = centroid * (len(cluster) - 1) / len(cluster) + slot_vecs_norm[j] / len(cluster)
                    norm = np.linalg.norm(centroid)
                    if norm > 0:
                        centroid /= norm

            clusters.append(cluster)

        # Build observations from clusters
        for cluster_indices in clusters:
            if len(cluster_indices) < min_cluster_size:
                continue

            cluster_slots = [vec_slots[i] for i in cluster_indices]
            cluster_texts = [(s, texts.get(s, "")) for s in cluster_slots]
            cluster_texts = [(s, t) for s, t in cluster_texts if t]

            if not cluster_texts:
                continue

            # Pick most representative: longest text (most informative)
            best_slot, best_text = max(cluster_texts, key=lambda x: len(x[1]))

            # Compute centroid
            centroid = slot_vecs_norm[cluster_indices].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm

            # Earliest event time
            evt = None
            if event_times:
                evts = [event_times[s] for s in cluster_slots if s in event_times]
                if evts:
                    evt = min(evts)

            # Confidence based on evidence count
            confidence = min(1.0, len(cluster_slots) / 5.0)

            observations.append(Observation(
                entity=entity,
                text=best_text,
                evidence_slots=cluster_slots,
                evidence_count=len(cluster_slots),
                centroid=centroid.astype(np.float32),
                event_at_ms=evt,
                confidence=confidence,
            ))

        # Non-vectorized slots as standalone observations
        for s in non_vec_slots:
            text = texts.get(s, "")
            if text:
                evt = event_times.get(s) if event_times else None
                observations.append(Observation(
                    entity=entity, text=text,
                    evidence_slots=[s], evidence_count=1,
                    event_at_ms=evt, confidence=0.5,
                ))

    return observations
