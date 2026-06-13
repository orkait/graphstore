"""Namespace isolation: intelligence written under a namespace must NOT pollute
the default/general view, and must be visible only when that namespace is bound.

Contract (the anti-pollution guarantee):
  - a node tagged with __namespace__ is EXCLUDED from default reads (no namespace active)
  - BIND NAMESPACE "X" -> reads show ONLY namespace X; new writes tag __namespace__=X
  - namespaces are mutually isolated
"""
from graphstore import GraphStore


def _ids(result):
    return [n["id"] for n in (result.data or [])]


def test_namespaced_node_excluded_from_default_view():
    gs = GraphStore(embedder="none", enable_sentence_nodes=False)
    gs.execute('CREATE NODE "general1" kind = "memory" content = "general fact"')
    gs.execute('BIND NAMESPACE "intel:acme"')
    gs.execute('CREATE NODE "intel1" kind = "evidence" content = "secret intel"')
    gs.execute('DISCARD NAMESPACE')
    ids = _ids(gs.execute('NODES'))
    assert "general1" in ids          # general memory stays visible
    assert "intel1" not in ids        # ANTI-POLLUTION: intel hidden from default view


def test_namespaced_node_visible_only_when_bound():
    gs = GraphStore(embedder="none", enable_sentence_nodes=False)
    gs.execute('BIND NAMESPACE "intel:acme"')
    gs.execute('CREATE NODE "intel1" kind = "evidence" content = "x"')
    assert "intel1" in _ids(gs.execute('NODES'))      # visible within namespace
    gs.execute('DISCARD NAMESPACE')
    assert "intel1" not in _ids(gs.execute('NODES'))  # invisible outside


def test_namespaces_are_mutually_isolated():
    gs = GraphStore(embedder="none", enable_sentence_nodes=False)
    gs.execute('BIND NAMESPACE "a"')
    gs.execute('CREATE NODE "na" kind = "evidence" content = "a-fact"')
    gs.execute('DISCARD NAMESPACE')
    gs.execute('BIND NAMESPACE "b"')
    ids = _ids(gs.execute('NODES'))
    gs.execute('DISCARD NAMESPACE')
    assert "na" not in ids            # namespace a not visible from namespace b


def test_count_nodes_excludes_namespaced_by_default():
    gs = GraphStore(embedder="none", enable_sentence_nodes=False)
    gs.execute('CREATE NODE "g" kind = "memory"')
    gs.execute('BIND NAMESPACE "intel"')
    gs.execute('CREATE NODE "i" kind = "evidence"')
    gs.execute('DISCARD NAMESPACE')
    assert gs.execute('COUNT NODES').count == 1   # only the general node counts


def test_lexical_retrieval_respects_namespace():
    # the real intelligence read path (LEXICAL/REMEMBER/SIMILAR) must not leak
    gs = GraphStore(embedder="none", enable_sentence_nodes=False)
    gs.execute('CREATE NODE "g1" kind = "memory" content = "public notes" DOCUMENT "public notes"')
    gs.execute('BIND NAMESPACE "intel"')
    gs.execute('CREATE NODE "i1" kind = "evidence" content = "secret breach" DOCUMENT "secret breach"')
    gs.execute('DISCARD NAMESPACE')
    assert "i1" not in _ids(gs.execute('LEXICAL SEARCH "secret breach" LIMIT 10'))   # no leak
    gs.execute('BIND NAMESPACE "intel"')
    hit = _ids(gs.execute('LEXICAL SEARCH "secret breach" LIMIT 10'))
    gs.execute('DISCARD NAMESPACE')
    assert "i1" in hit                                                                # visible when bound
