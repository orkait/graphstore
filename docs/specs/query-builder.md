# Query Builder Spec (v1)

Status: **draft, pre-implementation.** Do not merge code until this doc is approved.

Addresses the "100% DSL coverage" ask. Based on the behaviour audit in this PR description: ~80 DSL verbs, 10 open decisions resolved below, 4 critical escape / injection tests gating the first commit.

## 1. Goals

- **100% DSL coverage** — every verb accessible via a typed Python function
- **Escape-safe** by construction — user-supplied values cannot break out of their DSL slot
- **IDE autocomplete** — kwargs match DSL keywords; stage-2 typegen emits per-kind field types
- **Zero surprise** — builder output is a DSL string the real parser accepts; no second interpretation layer
- **Immutable** — `Query` objects are value types; modifiers return new instances

## 2. Non-goals

- ORM (no class-per-kind, no session/unit-of-work, no lazy relationships)
- Async (graphstore core is sync)
- Chained fluent-everything (we prefer kwargs; chaining reserved for modifiers)
- Hiding the DSL (`.dsl()` is always available; users learn the DSL anyway)

## 3. Package layout

```
src/graphstore/query/
  __init__.py          # exports `q`
  runtime.py           # Query class, compose (`|`), execute(gs), dsl()
  escape.py            # value -> DSL literal (str, int, float, bool, None, list, datetime)
  where.py             # WHERE dict -> DSL expression
  verbs/
    __init__.py
    reads.py
    traversal.py
    writes.py
    ingest.py
    contexts.py
    sys.py
    cron.py
    evolve.py
    vault.py
  types/
    __init__.py        # stage 2: generated TypedDicts (placeholder in v1)
```

Public import: `from graphstore import q` (re-exported from `__init__.py`).

## 4. Core types

### `Query`

```python
@dataclass(frozen=True, slots=True)
class Query:
    _dsl: str                  # compiled DSL fragment
    _kind: Literal["read", "write", "sys", "vault", "batch", "raw"]

    def dsl(self) -> str: ...
    def execute(self, gs) -> "Result": ...
    def __or__(self, other: "Query") -> "Query": ...   # batch compose
    def __repr__(self) -> str: ...                      # f"<Query: {first 80 chars}>"
```

Immutable. One verb per instance. `|` produces a new `Query` with `_kind="batch"` that wraps children in `BEGIN ... COMMIT`.

### `LLMResponse` is unrelated to this spec — separate subsystem.

## 5. Open decisions (locked)

| # | Decision | Resolution |
|---|---|---|
| R7 | datetime vs string for date fields | Accept both. `datetime` → `.strftime("%Y-%m-%d")`. Strings passed as-is (assumed ISO). |
| R11 | Empty `__in` list | `ValueError("__in requires a non-empty list")` at build. |
| R12 | Default `depth` | **No default.** User must pass it; omission is a `TypeError`. |
| W6 | `event_at` integer | Accept int (ms-since-epoch) AND string. `int` emitted as number literal; string emitted quoted. |
| W9 | ASSERT require `kind=` | Yes. Build-time `ValueError` if missing. |
| S3 | SYS EVOLVE `when`/`then` raw vs string | `when=` and `then=` are always **raw DSL expressions** (no auto-quoting); documented as such. Use `when_raw=` is redundant. |
| W12 | INGEST without `using=` | Builder omits USING clause; router decides tier. |
| X1 | Re-calling verb on same Query | Not possible: builder functions return new Query. Re-entry bug is unreachable by design. |
| X7 | Query immutable or one-shot | **Immutable.** Same `Query` can be executed many times. `execute(gs)` is not consumed. |
| X2 | `execute(None)` | `TypeError("execute() requires a GraphStore instance")`. |

## 6. Escape layer (`escape.py`)

Pure functions; no state.

| Python type | DSL emission | Test case |
|---|---|---|
| `str` | `"..."` with `\` and `"` backslash-escaped | `'a"b'` → `"a\"b"` |
| `int` | decimal literal | `42` → `42` |
| `float` | decimal literal | `0.5` → `0.5` |
| `bool` | `true` / `false` | `True` → `true` |
| `None` | `null` (or clause-omit depending on caller) | |
| `list[str | int | float]` | `("a", "b", 1)` | for `__in` |
| `datetime.date` / `datetime.datetime` | `"YYYY-MM-DD"` | |
| `list[float]` (vector) | `[0.1, 0.2, ...]` | for `SIMILAR TO [vec]` |
| `Query` | as raw sub-DSL (used by EVOLVE) | via `when=q.raw(...)` |
| any other | `TypeError("unsupported escape type: {type}")` | |

**Critical tests** (gate the first commit):

- R3: `q.nodes(kind='memory"; DROP ALL; --')` — kind escaped, doesn't break DSL
- R4: `q.remember('my "quoted" query')` — quote escaped
- W3: `q.create_node("m1", kind="memory", document='text with "quotes"')` — document escaped
- W4: `q.create_node("m1", kind="memory", document=None)` — omit DOCUMENT clause entirely

## 7. WHERE compiler (`where.py`)

Dict → DSL expression. Field-operator syntax is `field__op`; bare `field` implies `__eq`.

### Op whitelist

| Python op | DSL | Example |
|---|---|---|
| `__eq` (default) | `=` | `{"kind": "memory"}` → `kind = "memory"` |
| `__ne` | `!=` | `{"kind__ne": "test"}` → `kind != "test"` |
| `__gt` | `>` | `{"importance__gt": 0.5}` → `importance > 0.5` |
| `__gte` | `>=` | |
| `__lt` | `<` | |
| `__lte` | `<=` | |
| `__in` | `IN` | `{"topic__in": ["a", "b"]}` → `topic IN ("a", "b")` |
| `__not_in` | `NOT IN` | |
| `__startswith` | `STARTSWITH` | (grammar-dependent; verify exists) |
| `__contains` | `LIKE '%..%'` | (grammar-dependent) |
| `__is_null` | `IS NULL` | value ignored |

Unknown op → `ValueError("unsupported op: {op}. Valid: {list}")`.

### Compound expressions

```python
where = {"kind": "memory", "importance__gt": 0.5}
# -> kind = "memory" AND importance > 0.5

where = {"__or__": [{"kind": "memory"}, {"kind": "fact"}]}
# -> (kind = "memory" OR kind = "fact")

where = {"__not__": {"retracted": True}}
# -> NOT (retracted = true)

# Raw escape hatch for expressions beyond the op whitelist
where_raw = "importance > 0.5 AND (topic LIKE 'trav%' OR topic IS NULL)"
```

Mixing `where=` and `where_raw=` on the same call → `ValueError`.

## 8. Compose / batch semantics

```python
batch = (
    q.begin()
    | q.create_node("m1", kind="memory", document="x")
    | q.create_node("m2", kind="memory", document="y")
    | q.create_edge("m1", "m2", kind="rel")
    | q.commit()
)
batch.execute(gs)
```

Compiled DSL:

```
BEGIN
CREATE NODE "m1" kind = "memory" DOCUMENT "x"
CREATE NODE "m2" kind = "memory" DOCUMENT "y"
CREATE EDGE "m1" -> "m2" kind = "rel"
COMMIT
```

`|` left-folds; nested compositions flatten. A batch with no `begin()` / `commit()` is a sequence of separate statements (graphstore executes them one by one, each atomic).

**Shorthand:** `q.batch(stmt1, stmt2, ...)` wraps a list in BEGIN..COMMIT.

## 9. Verb specs (abbreviated)

All verbs return `Query`. Full signatures in code; below lists only decisions that affect behaviour.

### Reads (17 verbs)

- `q.node(id: str, *, with_document: bool = False)`
- `q.nodes(*, kind=None, where=None, where_raw=None, limit=None, order_by=None)`
- `q.remember(text: str, *, limit=None, tokens=None, at=None, where=None)` — `text` must be non-empty
- `q.recall(from_id: str, *, depth: int, limit=None)` — `depth` required
- `q.similar(*, text=None, node=None, vec=None, limit=None, where=None)` — exactly one of text/node/vec
- `q.lexical(text: str, *, limit=None, where=None)`
- `q.traverse(from_id: str, *, depth: int)`
- `q.subgraph(from_id: str, *, depth: int)`
- `q.path(a: str, b: str, *, max_depth=None, shortest=False)`
- `q.ancestors(id: str, *, depth: int)` / `q.descendants(id: str, *, depth: int)`
- `q.common_neighbors(a: str, b: str)`
- `q.match(pattern: str)` — pattern is raw; Cypher-esque
- `q.count_nodes(*, where=None)`
- `q.aggregate_nodes(*, group_by: list[str], select: list[str], where=None)`
- `q.what_if_retract(id: str)`
- `q.edges(from_id: str, *, where=None)`

### Writes (17 verbs)

- `q.create_node(id: str, *, kind: str, event_at=None, expires_in=None, document=None, **fields)` — clause emission order: `kind` → typed fields → `EVENT_AT` → `EXPIRES IN` → `DOCUMENT` (locked by grammar; tested)
- `q.update_node(id: str, **set)` — at least one set kwarg
- `q.upsert_node(id: str, *, kind=None, **fields)`
- `q.delete_node(id: str)`
- `q.delete_nodes(*, where)` — where required
- `q.update_nodes(*, where, set: dict)` — both required
- `q.create_edge(src: str, tgt: str, *, kind: str, **fields)`
- `q.increment(id: str, field: str, by: int | float)`
- `q.assert_(id: str, *, kind: str, value=None, confidence=None, source=None, event_at=None, **fields)` — `kind` required
- `q.retract(id: str, *, reason=None)`
- `q.merge(old: str, into: str)`
- `q.propagate(id: str, *, field: str, depth: int)`
- `q.ingest(file: str, *, as_id=None, kind=None, using=None, vision_model=None, chunker=None, max_chunk_size=None)`
- `q.forget(id: str)`
- `q.bind_context(name: str)` / `q.discard_context(name: str)`
- `q.begin()` / `q.commit()`

### System (~30 verbs)

Grouped under `q.sys.*`:

- `q.sys.status()`, `.stats()`, `.health()`, `.kinds()`, `.edge_kinds()`, `.describe_node(kind)`
- `q.sys.register_node_kind(name, *, required: dict[str, str] = None, optional: dict[str, str] = None, embed: str = None)`
- `q.sys.register_edge_kind(name, *, from_kind, to_kind)`
- `q.sys.connect(*, threshold=None)`
- `q.sys.consolidate(*, threshold)`
- `q.sys.duplicates(*, threshold)`
- `q.sys.contradictions(*, where=None, field=None, group_by=None)`
- `q.sys.expire(*, where=None)`
- `q.sys.snapshot(name)`, `.rollback_to(name)`
- `q.sys.embedders()`, `.reembed()`
- `q.sys.retain()`, `.evict()`
- `q.sys.checkpoint()`, `.rebuild_indices()`, `.clear_cache()`
- `q.sys.optimize(*, compact=False)`
- `q.sys.log(*, limit=None, trace=None)`
- Nested `q.sys.cron.*`: `add(name, schedule, query)`, `list()`, `remove(name)`
- Nested `q.sys.evolve.*`: `rule(name, *, when, then, cooldown=None)`, `list()`, `show(name)`, `enable(name)`, `disable(name)`, `delete(name)`, `history(name)`

### Vault (~8 verbs)

Grouped under `q.vault.*`:

- `q.vault.new(title, *, kind=None)`, `.search(text, *, limit=None)`, `.sync()`, `.update(id, **fields)`, `.read(id)`, `.delete(id)`

### Escape hatch

- `q.raw(dsl: str, **params)` — substitutes `:name` → `escape(value)`; strict binding (missing param → `ValueError`)

## 10. Testing plan

### Unit (per verb, ~80 test cases)

Each verb has at least one test:
1. Happy-path emission matches expected DSL literal.
2. Roundtrip: emit → parse → `Result` on synthetic GraphStore (no embedder).
3. At least one escape/injection attempt on string args.

### Property tests (hypothesis)

- For every verb × every string kwarg: generated strings never break DSL parse.
- For every verb × every numeric kwarg: generated numbers never break DSL parse.
- Round-trip DSL: `parse(q.verb(...).dsl())` never raises `QueryError`.

### Integration

- `tests/test_query_roundtrip.py` — exhaustive: for every verb, emit 3 variants, parse, execute on fresh GraphStore, assert `Result.kind` is the expected one.
- `tests/test_query_batch.py` — compose 20+ statements in BEGIN..COMMIT, execute, verify atomicity.
- `tests/test_query_injection.py` — the 4 critical cases from audit (R3 / R4 / W3 / W4) plus 20 fuzz inputs.

Target: **every verb has at least one passing test before the verb ships in its PR.**

## 11. Delivery plan (PR-by-PR)

| PR | Scope | Gate |
|---|---|---|
| 1 | `query/runtime.py` + `query/escape.py` + `query/where.py` + 10 highest-traffic verbs (REMEMBER, RECALL, SIMILAR, LEXICAL, NODES, CREATE NODE, CREATE EDGE, INGEST, ASSERT, RETRACT) + unit + injection tests | Critical tests green; 80% of daily-use coverage |
| 2 | All remaining read/traversal verbs (TRAVERSE, PATH, ANCESTORS, MATCH, COUNT, AGGREGATE, WHAT IF, EDGES, NODE, SUBGRAPH, DESCENDANTS, COMMON, LEXICAL variants) | Read-side 100% |
| 3 | Remaining write verbs (UPDATE, UPSERT, DELETE, INCREMENT, MERGE, PROPAGATE, FORGET, BIND/DISCARD CONTEXT, BEGIN/COMMIT) + batch compose | Write-side 100%; batch semantics locked |
| 4 | `q.sys.*` (30+ verbs) + `q.vault.*` | System/vault 100% |
| 5 | Docs pass: README section, SKILL.md update, `docs/query-builder.md` user guide | Public launch |
| 6 (optional, later) | `graphstore typegen` CLI — emits TypedDict per registered kind for mypy | Stage 2 — type safety level up |

Total estimate: ~7-8 focused days. Parallelizable with LLM work (no shared files).

## 11.5 Composability / modularity

Locked primitives, in order of priority:

### A. Predicate algebra `F` (Django-Q-inspired)

First-class WHERE objects. Immutable, composable with `&`, `|`, `~`.

```python
from graphstore.query import F

travel    = F.eq("topic", "travel")
recent    = F.gte("__event_at__", "2024-01-01")
important = F.gt("importance", 0.5)
live      = ~F.eq("__retracted__", True)

q.nodes(where=travel & recent & important & live, limit=10)
```

**Dict shorthand stays as sugar that compiles to `F` internally.** Users mix freely.

```python
q.nodes(where={"topic": "travel", "importance__gt": 0.5})
# ≡
q.nodes(where=F.eq("topic", "travel") & F.gt("importance", 0.5))
```

**Operators:** `&` AND, `|` OR, `~` NOT, `F.raw(expr)` unescaped escape hatch.

**`F` builders:** `F.eq(field, val)`, `F.ne`, `F.gt`, `F.gte`, `F.lt`, `F.lte`, `F.in_(field, list)`, `F.not_in`, `F.startswith`, `F.contains`, `F.is_null(field)`, `F.raw(dsl_expr)`, `F.from_dict(d)`.

**Algebra laws** (tested via hypothesis):
- Associativity: `(a & b) & c == a & (b & c)`
- Commutativity: `a & b == b & a` (DSL may differ; semantic equivalence)
- De Morgan: `~(a & b) == ~a | ~b`
- Double negation: `~~a == a`
- Identity: `a & F.true() == a`, `a | F.false() == a`

### B. Query modifiers (Kysely-inspired immutable clone-with-override)

Read verbs only. Each modifier returns a new `Query`; input `Query` never mutates.

```python
base  = q.nodes(kind="memory")
top10 = base.limit(10)                          # adds LIMIT 10
hot   = base.where(F.gt("importance", 0.5))     # AND-combines with existing WHERE
named = base.with_(limit=20, order_by="__event_at__ DESC")   # replaces named args
```

Modifiers: `.limit(n)`, `.where(f)`, `.tokens(n)`, `.at(date)`, `.order_by(expr)`, `.with_(**kw)`.

Write verbs have no modifier chain. Users pass kwargs at construction.

### C. `.pipe()` helper (Ibis-inspired functional composition)

Every `Query` has `.pipe(fn, *a, **kw)` that calls `fn(self, *a, **kw)` and returns the result. Enables reusable transformations as plain Python functions.

```python
def with_recency(q: Query, days: int = 30) -> Query:
    return q.where(F.gte("__event_at__", _days_ago(days)))

def for_user(q: Query, uid: str) -> Query:
    return q.where(F.eq("owner", uid))

(q.nodes(kind="memory")
  .pipe(with_recency, days=7)
  .pipe(for_user, uid="u42")
  .limit(10))
```

Implementation: 5 LoC on `Query`.

### D. Batch compose `|` (already in §8)

`Query | Query` flattens into `BEGIN ... COMMIT` block. Unambiguous vs `F | F` by type.

### E. Plugin registry stub (forward-compat)

```python
from graphstore.query import register_verb

@register_verb("ts_downsample")
def ts_downsample(series_id: str, *, window: str, agg: str) -> Query: ...

q.ts_downsample(...)  # attribute fallthrough finds it
```

**v1 ships the stub (~30 LoC) without using it internally.** Third-party packages (e.g. future `graphstore-timeseries`) can extend DSL without waiting on core release.

### What's explicitly NOT in composability

| Skipped | Reason |
|---|---|
| Column objects (`c.kind == "memory"`) | Requires static schema per kind; graphstore schema is dynamic (SYS REGISTER at runtime). Dict + F whitelist covers the same DX without the 300 LoC. |
| Ibis-style Table pipelines (`t.filter().group_by().agg()`) | Verb model mismatch — REMEMBER/RECALL/SIMILAR aren't pipelineable the same way. Forcing the shape makes both sides uglier. |
| Gremlin-style traversal steps | Our TRAVERSE/PATH/ANCESTORS/RECALL already cover graph walks. Adding a Gremlin shim duplicates surface for no value. |
| CTE / subquery naming | Grammar doesn't support it. Defer to v2 after grammar work. |

### Impact on v1 scope

- Add `src/graphstore/query/filters.py` (new, ~80 LoC)
- Add `src/graphstore/query/pipe.py` (new, ~5 LoC)
- Add `src/graphstore/query/plugins.py` (new, ~30 LoC)
- `Query` gains `.limit/.where/.with_/.tokens/.at/.order_by/.pipe` (~40 LoC on runtime.py)
- `where=` parameter on every read verb accepts `dict | F | None`
- Top-level `graphstore.__init__` exports `q, F, register_verb`

**Extra ~160 LoC on top of the base builder.** Absorbed into PR 1 (core runtime pre-verbs).

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Grammar drift breaks builder | `tests/test_query_roundtrip.py` runs parser on every builder output — CI catches it next run |
| 80 verbs miss one | `tests/test_query_coverage.py` — enumerates grammar rules, asserts each has a builder function |
| Public API cements too early | Ship as `graphstore.query` but document "experimental" in v1 release notes; lock API after 2 minor versions with no breaking changes |
| User confusion between `q.x(...)` and `gs.execute('...')` | Both work side-by-side; README picks builder for all examples post-ship |

## 13. Out of scope (v1)

- ORM-style class mapping (no `Memory(gs)` models)
- Async variants
- Streaming result iteration (graphstore handlers return full result sets anyway)
- `graphstore typegen` (ships as v2 / PR 6)
- Auto-generated docs from grammar (nice-to-have, not v1)

## 14. Approval gates

Before PR 1 code starts:
- [ ] All 10 open decisions in §5 signed off
- [ ] Escape layer spec in §6 reviewed
- [ ] WHERE op whitelist in §7 finalized
- [ ] Package layout in §3 approved

Once PR 1 lands: iterate in PRs 2-6 without re-approval unless the spec changes.
