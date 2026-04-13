from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True, slots=True)
class RetrievalContext:
    query: str
    query_anchor_ms: int | None
    query_time_range: tuple[int, int] | None
    has_entities: bool
    entity_candidates: list[str]
    has_temporal_signal: bool
    has_observations: bool
    has_graph_edges: bool
    has_fts: bool
    has_vectors: bool
    limit: int
    token_budget: int | None


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    candidate_k: int
    use_temporal_filter: bool
    use_graph_expansion: bool
    use_observations: bool
    use_nucleus: bool
    fusion_method: str
    type_weight_override: dict | None
    fallback_chain: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class RetrievalPlanner:
    def build_context(
        self,
        *,
        query: str,
        limit: int,
        token_budget: int | None,
        query_anchor_ms: int | None,
        query_time_range: tuple[int, int] | None,
        has_entities: bool,
        entity_candidates: list[str],
        has_temporal_signal: bool,
        has_observations: bool,
        has_graph_edges: bool,
        has_fts: bool,
        has_vectors: bool,
    ) -> RetrievalContext:
        return RetrievalContext(
            query=query,
            query_anchor_ms=query_anchor_ms,
            query_time_range=query_time_range,
            has_entities=has_entities,
            entity_candidates=entity_candidates,
            has_temporal_signal=has_temporal_signal,
            has_observations=has_observations,
            has_graph_edges=has_graph_edges,
            has_fts=has_fts,
            has_vectors=has_vectors,
            limit=limit,
            token_budget=token_budget,
        )

    def plan(self, ctx: RetrievalContext, explicit_overrides: dict | None = None) -> RetrievalPlan:
        explicit_overrides = explicit_overrides or {}
        if explicit_overrides.get("retrieval_plan") is not None:
            return explicit_overrides["retrieval_plan"]

        use_temporal = bool(
            ctx.has_temporal_signal and (
                ctx.query_anchor_ms is not None or ctx.query_time_range is not None
            )
        )
        use_graph = bool(ctx.has_graph_edges and ctx.has_entities and ctx.entity_candidates)
        prefish = any(
            tok in ctx.query.lower()
            for tok in ("prefer", "preference", "favorite", "favourite", "usually", "for me")
        )
        use_observations = bool(prefish and ctx.has_observations)
        use_nucleus = bool((not use_observations) and ctx.has_graph_edges)

        plan = RetrievalPlan(
            candidate_k=max(ctx.limit * 3, ctx.limit),
            use_temporal_filter=use_temporal,
            use_graph_expansion=use_graph,
            use_observations=use_observations,
            use_nucleus=use_nucleus,
            fusion_method="weighted",
            type_weight_override=None,
            fallback_chain=["lexical_dense"],
            notes=[],
        )

        if explicit_overrides:
            allowed = {
                "candidate_k",
                "use_temporal_filter",
                "use_graph_expansion",
                "use_observations",
                "use_nucleus",
                "fusion_method",
                "type_weight_override",
                "fallback_chain",
                "notes",
            }
            changes = {k: v for k, v in explicit_overrides.items() if k in allowed}
            if changes:
                plan = replace(plan, **changes)
        return plan
