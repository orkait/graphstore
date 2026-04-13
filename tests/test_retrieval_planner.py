from graphstore.retrieval.planner import RetrievalContext, RetrievalPlan
from graphstore.retrieval.planner import RetrievalPlanner
from graphstore import GraphStore
from tests.test_retrieval_improvements import FixedEmbedder


def test_retrieval_plan_has_no_mode_field():
    plan = RetrievalPlan(
        candidate_k=10,
        use_temporal_filter=False,
        use_graph_expansion=False,
        use_observations=False,
        use_nucleus=False,
        fusion_method="weighted",
        type_weight_override=None,
        fallback_chain=["lexical_dense"],
        notes=[],
    )
    assert not hasattr(plan, "mode")


def test_retrieval_context_carries_range_and_anchor():
    ctx = RetrievalContext(
        query="what happened before may 2023",
        query_anchor_ms=1,
        query_time_range=(0, 1),
        has_entities=False,
        entity_candidates=[],
        has_temporal_signal=True,
        has_observations=False,
        has_graph_edges=False,
        has_fts=True,
        has_vectors=True,
        limit=5,
        token_budget=None,
    )
    assert ctx.query_anchor_ms == 1
    assert ctx.query_time_range == (0, 1)


def test_temporal_query_enables_temporal_filter():
    planner = RetrievalPlanner()
    ctx = planner.build_context(
        query="what happened on 2023-05-08",
        limit=5,
        token_budget=None,
        query_anchor_ms=1,
        query_time_range=None,
        has_entities=False,
        entity_candidates=[],
        has_temporal_signal=True,
        has_observations=False,
        has_graph_edges=True,
        has_fts=True,
        has_vectors=True,
    )
    plan = planner.plan(ctx, explicit_overrides={})
    assert plan.use_temporal_filter is True


def test_entity_query_enables_graph_expansion():
    planner = RetrievalPlanner()
    ctx = planner.build_context(
        query="what did alice say about migration",
        limit=5,
        token_budget=None,
        query_anchor_ms=None,
        query_time_range=None,
        has_entities=True,
        entity_candidates=["alice"],
        has_temporal_signal=False,
        has_observations=False,
        has_graph_edges=True,
        has_fts=True,
        has_vectors=True,
    )
    plan = planner.plan(ctx, explicit_overrides={})
    assert plan.use_graph_expansion is True


def test_explicit_override_beats_policy():
    planner = RetrievalPlanner()
    ctx = planner.build_context(
        query="what happened on 2023-05-08",
        limit=5,
        token_budget=None,
        query_anchor_ms=1,
        query_time_range=None,
        has_entities=False,
        entity_candidates=[],
        has_temporal_signal=True,
        has_observations=False,
        has_graph_edges=False,
        has_fts=True,
        has_vectors=True,
    )
    plan = planner.plan(ctx, explicit_overrides={"use_temporal_filter": False, "fusion_method": "rrf"})
    assert plan.use_temporal_filter is False
    assert plan.fusion_method == "rrf"


def test_graphstore_exposes_retrieval_planner():
    gs = GraphStore(embedder=FixedEmbedder())
    assert isinstance(gs._executor._retrieval_planner, RetrievalPlanner)
    gs.close()


def test_planner_temporal_filter_is_reflected_in_result_meta():
    from tests.test_retrieval_improvements import _make_gs

    gs = _make_gs()
    gs.execute('CREATE NODE "a" kind = "fact" claim = "museum trip" EVENT_AT "2023-05-08"')
    result = gs.execute('REMEMBER "museum trip" AT "2023-05-08" LIMIT 5')
    assert "planner" in result.meta
    assert result.meta["planner"]["use_temporal_filter"] is True
    gs.close()


def test_planner_observation_mode_surfaces_in_result_meta():
    from tests.test_retrieval_improvements import FixedEmbedder

    gs = GraphStore(embedder=FixedEmbedder())
    gs.execute('SYS REGISTER NODE KIND "fact" REQUIRED claim:string EMBED claim')
    gs.execute('SYS REGISTER NODE KIND "observation" REQUIRED claim:string EMBED claim')
    gs.execute('CREATE NODE "msg1" kind = "fact" claim = "the user prefers premiere pro for advanced editing"')
    gs.execute('CREATE NODE "obs1" kind = "observation" claim = "user prefers premiere pro"')
    result = gs.execute('REMEMBER "what do I prefer for video editing?" LIMIT 5')
    assert "planner" in result.meta
    assert result.meta["planner"]["use_observations"] is True
    gs.close()


def test_planner_can_increase_candidate_k():
    planner = RetrievalPlanner()
    ctx = planner.build_context(
        query="what happened on 2023-05-08",
        limit=5,
        token_budget=None,
        query_anchor_ms=1,
        query_time_range=None,
        has_entities=False,
        entity_candidates=[],
        has_temporal_signal=True,
        has_observations=False,
        has_graph_edges=False,
        has_fts=True,
        has_vectors=True,
    )
    plan = planner.plan(ctx, explicit_overrides={})
    assert plan.candidate_k >= 15
