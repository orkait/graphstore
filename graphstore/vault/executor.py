"""VaultExecutor: handles VAULT * DSL commands."""
from graphstore.core.types import Result
from graphstore.core.errors import GraphStoreError
from graphstore.dsl.ast_nodes import (
    VaultNew,
    VaultRead,
    VaultWrite,
    VaultAppend,
    VaultSearch,
    VaultBacklinks,
    VaultList,
    VaultSync,
    VaultDaily,
    VaultArchive,
)


class VaultExecutor:
    def __init__(self, vault_manager, vault_sync, store, embedder=None, vector_store=None):
        self._manager = vault_manager
        self._sync = vault_sync
        self._store = store
        self._embedder = embedder
        self._vector_store = vector_store

    def dispatch(self, ast) -> Result:
        handlers = {
            VaultNew: self._new,
            VaultRead: self._read,
            VaultWrite: self._write,
            VaultAppend: self._append,
            VaultSearch: self._search,
            VaultBacklinks: self._backlinks,
            VaultList: self._list,
            VaultSync: self._sync_cmd,
            VaultDaily: self._daily,
            VaultArchive: self._archive,
        }
        handler = handlers.get(type(ast))
        if handler is None:
            raise GraphStoreError(f"Unknown vault command: {type(ast).__name__}")
        return handler(ast)

    def _new(self, q) -> Result:
        tags = q.tags.split(",") if q.tags else []
        slug = self._manager.new(q.title, kind=q.kind, tags=tags)
        self._sync.sync_file(slug)
        return Result(kind="ok", data={"slug": slug, "file": f"{slug}.md"}, count=0)

    def _read(self, q) -> Result:
        content = self._manager.read(q.title)
        from graphstore.vault.parser import parse_frontmatter, parse_sections
        fm = parse_frontmatter(content)
        sections = parse_sections(content)
        return Result(kind="note", data={
            "content": content,
            "frontmatter": fm,
            "sections": sections,
        }, count=1)

    def _write(self, q) -> Result:
        self._manager.write_section(q.title, q.section, q.content)
        from graphstore.vault.parser import title_to_slug
        slug = title_to_slug(q.title)
        self._sync.sync_file(slug)
        return Result(kind="ok", data={"slug": slug, "section": q.section}, count=0)

    def _append(self, q) -> Result:
        self._manager.append_section(q.title, q.section, q.content)
        from graphstore.vault.parser import title_to_slug
        slug = title_to_slug(q.title)
        self._sync.sync_file(slug)
        return Result(kind="ok", data={"slug": slug, "section": q.section}, count=0)

    def _search(self, q) -> Result:
        if not self._embedder or not self._vector_store:
            # Fallback: text search on summary column
            results = []
            for slug in self._manager.list_files():
                node_id = f"note:{slug}"
                node = self._store.get_node(node_id)
                if node and q.query.lower() in node.get("summary", "").lower():
                    results.append(node)
            limit = q.limit.value if q.limit else 10
            return Result(kind="nodes", data=results[:limit], count=min(len(results), limit))

        # Vector search
        import numpy as np
        query_vec = self._embedder.encode_queries([q.query])[0]
        n = self._store._next_slot
        mask = self._store.compute_live_mask(n)
        # Filter to note nodes only
        kind_mask = self._store.columns.get_mask("kind", "=", "note", n)
        if kind_mask is not None:
            mask = mask & kind_mask

        vs_mask = (self._vector_store._has_vector[:n]
                   if n <= self._vector_store._capacity
                   else np.zeros(n, dtype=bool))
        combined = mask & vs_mask[:n]

        limit = q.limit.value if q.limit else 10
        slots, dists = self._vector_store.search(query_vec, k=limit * 3, mask=combined)

        results = []
        for slot, dist in zip(slots, dists):
            slot = int(slot)
            node = self._store._materialize_slot(slot)
            if node:
                node["_similarity_score"] = round(1.0 - float(dist), 4)
                results.append(node)
                if len(results) >= limit:
                    break

        return Result(kind="nodes", data=results, count=len(results))

    def _backlinks(self, q) -> Result:
        from graphstore.vault.parser import title_to_slug
        slug = title_to_slug(q.title)
        node_id = f"note:{slug}"
        edges = self._store.get_edges_to(node_id, kind="links")
        return Result(kind="edges", data=edges, count=len(edges))

    def _list(self, q) -> Result:
        # Get all note nodes, apply WHERE/ORDER/LIMIT
        nodes = self._store.get_all_nodes(kind="note")

        if q.where:
            from graphstore.dsl.executor_base import ExecutorBase
            base = ExecutorBase(self._store)
            nodes = [n for n in nodes if base._eval_where(q.where.expr, n)]

        if q.order:
            field = q.order.field
            desc = q.order.direction == "DESC"
            nodes.sort(key=lambda n: n.get(field, ""), reverse=desc)

        if q.limit:
            nodes = nodes[:q.limit.value]

        return Result(kind="nodes", data=nodes, count=len(nodes))

    def _sync_cmd(self, q) -> Result:
        result = self._sync.sync_all()
        return Result(kind="ok", data=result, count=0)

    def _daily(self, q) -> Result:
        slug = self._manager.daily()
        self._sync.sync_file(slug)
        content = self._manager.read(slug)
        return Result(kind="note", data={"slug": slug, "file": f"{slug}.md", "content": content}, count=1)

    def _archive(self, q) -> Result:
        self._manager.archive(q.title)
        from graphstore.vault.parser import title_to_slug
        slug = title_to_slug(q.title)
        self._sync.sync_file(slug)
        return Result(kind="ok", data={"slug": slug, "status": "archived"}, count=0)
