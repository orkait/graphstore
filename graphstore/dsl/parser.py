from pathlib import Path
from collections import OrderedDict
from lark import Lark
from graphstore.dsl.transformer import DSLTransformer
from graphstore.core.errors import QueryError

_grammar_path = Path(__file__).parent / "grammar.lark"
_grammar = _grammar_path.read_text()
_parser = Lark(_grammar, parser="lalr", start="start")

# Keywords that resolve to the current time - must bypass LRU cache
_TIME_KEYWORDS = {"NOW()", "TODAY", "YESTERDAY"}


class PlanCache:
    def __init__(self, maxsize: int = 256):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get_or_parse(self, query: str):
        # Bypass cache for queries with time expressions so timestamps are fresh
        if any(kw in query for kw in _TIME_KEYWORDS):
            return _parse_internal(query)
        key = " ".join(query.split())  # normalize whitespace
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        ast = _parse_internal(query)
        self._cache[key] = ast
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return ast

    def clear(self):
        self._cache.clear()

    def __len__(self):
        return len(self._cache)


def _parse_internal(query: str):
    try:
        tree = _parser.parse(query)
        return DSLTransformer().transform(tree)
    except Exception as e:
        raise QueryError(message=str(e), query=query)


# Module-level cache
_plan_cache = PlanCache()


def set_cache_size(maxsize: int):
    """Resize the plan cache (clears existing entries)."""
    _plan_cache._maxsize = maxsize
    _plan_cache.clear()


def parse(query: str):
    """Parse a DSL query string into an AST node. Uses plan cache."""
    return _plan_cache.get_or_parse(query)


def parse_uncached(query: str):
    """Parse without caching."""
    return _parse_internal(query)


def clear_cache():
    """Clear the plan cache."""
    _plan_cache.clear()
