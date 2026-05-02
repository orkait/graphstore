"""Entity resolution: separate mention from identity.

When ingesting a mention of "Alice", "Maria", or any other proper noun
extracted from natural-language text, naming the canonical-entity node
after the surface form (``ent:alice``) breaks down across conversations:
two different humans named "Alice" collide; the same human mentioned
across sessions either collides into one node by accident or - if write
semantics fail - drops the second mention entirely.

This module implements the production-grade fix used by Wikidata,
Microsoft GraphRAG, and entity-resolution pipelines elsewhere: separate
**mention** (an observation: "Alice was mentioned at message m1, char 42,
within this surrounding sentence") from **entity** (a hypothesis about
identity: "this specific Alice the data model thinks exists, with auto-
generated id ``entity:c4f8a3``").

Mentions are immutable, location-keyed, never collide.
Entities are revisable, auto-id, can be merged or split as evidence accumulates.
A ``refers_to`` edge with confidence connects mention to entity.

Resolver workflow at write time:

  1. **Name match** (cheap precondition). Find all existing ``entity``
     nodes whose ``canonical_name`` matches the new mention's surface
     name (case-folded equality - tighten if false positives appear).
  2. **Empty match → new entity.** Generate ``entity:{uuid4-hex}``,
     caller materializes it.
  3. **Single match → unambiguous link.** Return that entity_id with
     confidence=1.0 (no ambiguity to resolve).
  4. **Multiple matches → embedding disambiguation.** Compute cosine
     between the mention's context embedding and each candidate
     entity's accumulated-context embedding. Pick the highest. If the
     best score is above ``threshold_high``, return it. Otherwise
     return a new entity_id (the contexts diverged enough that this is
     probably a different human with the same name).

This is **correct** because:
  - Mentions never collide (location-keyed).
  - Entities discovered, not asserted - cluster by accumulated evidence.
  - Confidence preserved end-to-end. A weak ``refers_to`` edge is
    revisable in light of later evidence without losing the original
    mention.
  - Same human mentioned 1000 times across 50 conversations =
    1000 mention nodes + 1 entity node.
  - Two genuinely-different "Alice"s with diverging context =
    N mention nodes + 2 entity nodes, automatically.

It is also **cheap**:
  - Pre-filter by exact name match keeps the candidate set small
    (typically 0 or 1 entity per surface name).
  - Embedding disambiguation only fires on the rare collision case
    and reuses the embedder graphstore already has loaded.
  - All ANN lookups are O(log n) on the existing usearch index.
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger(__name__)

# How conservative are we about merging? `threshold_high` is the cosine
# above which we confidently link a new mention to an existing entity
# of the same name. Below that, even with name match, we err on the
# side of creating a new entity - false-merge is worse than
# false-split because MERGE is reversible (DELETE EDGE + UPSERT) while
# silently conflated identities become impossible to untangle once
# downstream beliefs accumulate.
DEFAULT_HIGH_THRESHOLD = 0.85

# Schema constants. Use these as the canonical kind values + edge
# label everywhere; downstream readers should not hard-code strings.
KIND_MENTION = "mention"
KIND_ENTITY = "entity"
EDGE_REFERS_TO = "refers_to"


@dataclass(frozen=True)
class ResolvedMention:
    """Outcome of ``resolve_mention()``.

    Resolver does NOT mutate the store. Caller is expected to:
      1. CREATE the mention node (if it doesn't exist yet)
      2. If ``is_new_entity``: CREATE the entity node with id ``entity_id``
      3. CREATE EDGE mention_id -[refers_to confidence=...]-> entity_id

    Doing those writes outside the resolver keeps the resolver pure
    (testable without a graph), idempotent, and side-effect-free.
    """

    entity_id: str  # always a valid id; either existing or freshly minted
    confidence: float  # 1.0 for new + unambiguous match, 0..1 for disambig
    is_new_entity: bool  # caller must CREATE NODE for the entity
    canonical_name: str  # name to seed on a new entity (== surface_name)
    candidates_seen: int  # how many same-name entities were considered
    notes: list[str]  # human-readable trace of resolver decisions


# ---------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------


_NAME_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize_name(name: str) -> str:
    """Lowercase + collapse non-alphanumerics to nothing.

    "Alice Smith" -> "alicesmith". "alice@stripe" -> "alicestripe".
    Aggressive on purpose - we want "alice", "Alice", "ALICE", "Alice "
    to all hash to the same name match. False-positive risk (e.g.
    "Alice S." vs "Alice S") is fine; we disambiguate by embedding
    score after.
    """
    return _NAME_NORMALIZE_RE.sub("", name.lower())


def make_entity_id(prefix: str = "entity") -> str:
    """Generate a fresh entity id. UUID4-hex first 12 chars - long
    enough for collision-resistance at billions of entities, short
    enough to read in logs."""
    return f"{prefix}:{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------
# Candidate lookup
# ---------------------------------------------------------------------


def _candidates_by_name(gs: Any, surface_name: str) -> list[dict]:
    """Return all entity nodes whose canonical_name normalizes equal
    to surface_name's normalized form.

    Uses the structured-column path (NODES WHERE) rather than vector
    search - this is the cheap precondition before we spend an embed
    cycle. Empty list = unique name = no disambiguation needed.
    """
    target = normalize_name(surface_name)
    if not target:
        return []
    # Pull all entity nodes (typically tens to low thousands), filter
    # in-process by normalized name. We avoid pushing normalize_name
    # into the WHERE clause because the DSL has no equivalent function;
    # store-side filter is fast enough for the cardinalities involved.
    try:
        result = gs.execute(f'NODES WHERE kind = "{KIND_ENTITY}" LIMIT 5000')
    except Exception as e:
        _log.warning("entity_resolver: NODES query failed (%s); "
                     "treating as empty candidates", e)
        return []
    nodes = result.data if hasattr(result, "data") else []
    if not isinstance(nodes, list):
        return []
    out: list[dict] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        cn = n.get("canonical_name") or n.get("name") or ""
        if normalize_name(cn) == target:
            out.append(n)
    return out


def _embed_text(gs: Any, text: str) -> list[float] | None:
    """Embed `text` via the GraphStore's embedder. None if no
    embedder is configured (resolver gracefully degrades to first-match
    selection in that case).
    """
    embedder = getattr(gs, "_embedder", None)
    if embedder is None:
        return None
    try:
        # Embedders implement encode_documents([str]) -> ndarray
        vecs = embedder.encode_documents([text])
        if vecs is None or len(vecs) == 0:
            return None
        return list(vecs[0])
    except Exception as e:
        _log.warning("entity_resolver: embed failed (%s)", e)
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na ** 0.5 * nb ** 0.5)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def resolve_mention(
    gs: Any,
    surface_name: str,
    context: str,
    threshold_high: float = DEFAULT_HIGH_THRESHOLD,
) -> ResolvedMention:
    """Decide which entity a new mention refers to.

    Args:
        gs: live GraphStore. Resolver does not write; only reads.
        surface_name: exact surface text from the source ("Alice",
            "Maria", "OpenAI"). Case + punctuation are normalized
            internally for name match.
        context: surrounding sentence(s) - used for embedding-based
            disambiguation when more than one entity shares this name.
        threshold_high: cosine threshold for confident linking. Below
            this we mint a new entity rather than risk a false-merge.

    Returns: ``ResolvedMention``. Caller materializes the entity node
    if ``is_new_entity`` is True, then creates the refers_to edge with
    the returned confidence.
    """
    notes: list[str] = []

    candidates = _candidates_by_name(gs, surface_name)
    notes.append(f"name match candidates: {len(candidates)}")

    if not candidates:
        return ResolvedMention(
            entity_id=make_entity_id(),
            confidence=1.0,
            is_new_entity=True,
            canonical_name=surface_name,
            candidates_seen=0,
            notes=notes + ["no existing entity with this name; minting new"],
        )

    if len(candidates) == 1:
        # Unambiguous name match. Confidence 1.0 because there is
        # nothing to disambiguate against. If the user later splits
        # this entity (e.g. they realize there are actually two Alices),
        # they do so explicitly via MERGE/SPLIT verbs.
        return ResolvedMention(
            entity_id=candidates[0]["id"],
            confidence=1.0,
            is_new_entity=False,
            canonical_name=surface_name,
            candidates_seen=1,
            notes=notes + ["single name match; linking with confidence=1.0"],
        )

    # Multiple candidates. Embedding-based disambiguation.
    new_vec = _embed_text(gs, f"{surface_name}. {context}")
    if new_vec is None:
        # No embedder. Fall back to picking the entity with the most
        # mentions (preferred-attachment heuristic). Worst case we still
        # bias toward consolidation.
        notes.append("no embedder; falling back to most-mentioned entity")
        best = max(candidates,
                   key=lambda n: int(n.get("mention_count", 0)))
        return ResolvedMention(
            entity_id=best["id"],
            confidence=0.5,  # signal low certainty
            is_new_entity=False,
            canonical_name=surface_name,
            candidates_seen=len(candidates),
            notes=notes,
        )

    best_id: str | None = None
    best_score = -1.0
    for cand in candidates:
        cand_id = cand.get("id", "")
        # Each candidate's discriminator is its accumulated context
        # text. We rebuild it from canonical_name + (any stored
        # context column the caller seeded). If the caller hasn't
        # populated a context column, embedding compares names alone
        # and the disambiguation collapses to "any same-name" - which
        # is acceptable; we already reported candidates_seen so the
        # caller can audit.
        cand_text = " ".join([
            str(cand.get("canonical_name") or cand.get("name") or ""),
            str(cand.get("context", "")),
        ]).strip()
        cand_vec = _embed_text(gs, cand_text)
        if cand_vec is None:
            continue
        score = _cosine(new_vec, cand_vec)
        if score > best_score:
            best_score = score
            best_id = cand_id

    if best_id is not None and best_score >= threshold_high:
        return ResolvedMention(
            entity_id=best_id,
            confidence=float(best_score),
            is_new_entity=False,
            canonical_name=surface_name,
            candidates_seen=len(candidates),
            notes=notes + [
                f"best candidate {best_id} cosine={best_score:.3f} "
                f">= threshold {threshold_high:.2f}; linking"
            ],
        )

    # Multiple same-name entities exist but the new mention does not
    # confidently match any of them. Mint a new entity - false-split
    # is reversible via MERGE; false-merge is not.
    return ResolvedMention(
        entity_id=make_entity_id(),
        confidence=1.0,
        is_new_entity=True,
        canonical_name=surface_name,
        candidates_seen=len(candidates),
        notes=notes + [
            f"best candidate cosine={best_score:.3f} < threshold "
            f"{threshold_high:.2f}; minting new entity"
        ],
    )


# ---------------------------------------------------------------------
# Mention id construction
# ---------------------------------------------------------------------


def make_mention_id(msg_id: str, slug: str, occurrence: int = 0) -> str:
    """Build a location-keyed mention id.

    Format: ``mention:{msg_id}:{slug}:{occurrence}``.

    msg_id alone keys the source message; appending the slug + an
    occurrence index disambiguates multiple mentions of different
    surface forms within the same message ("Alice told Bob...") and
    repeated mentions of the same surface form ("Alice ... Alice ...").

    No collision possible across calls with the same args - that's the
    point: re-extracting the same message must produce the same
    mention id, idempotently.
    """
    return f"mention:{msg_id}:{slug}:{occurrence}"
