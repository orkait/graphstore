"""Persistent CRON scheduler for GraphStore.

Registers DSL queries to run on cron schedules. Jobs persist in SQLite
and survive restarts. Uses croniter for full cron expression support.
Requires threaded=True (submits via command queue).
"""

import time
import threading
import logging
import sqlite3
from concurrent.futures import Future

logger = logging.getLogger(__name__)


class CronScheduler:
    """Persistent cron scheduler with daemon timer thread."""

    def __init__(self, conn: sqlite3.Connection | None, submit_fn):
        self._conn = conn
        self._submit = submit_fn
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        """Start the cron timer thread."""
        if self._conn is None:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._tick_loop, daemon=True, name="graphstore-cron"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the cron timer thread. Idempotent."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def add(self, name: str, schedule: str, query: str) -> dict:
        """Register a cron job. Validates cron expression."""
        from croniter import croniter
        if not croniter.is_valid(schedule):
            raise ValueError(f"Invalid cron expression: {schedule!r}")
        now = time.time()
        next_run = croniter(schedule, now).get_next(float)
        conn = self._conn
        if conn is None:
            raise RuntimeError("CRON requires persistence (GraphStore with path)")
        try:
            conn.execute(
                "INSERT INTO cron_jobs (name, schedule, query, enabled, created_at, next_run) "
                "VALUES (?, ?, ?, 1, ?, ?)",
                (name, schedule, query, now, next_run),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Cron job already exists: {name!r}")
        return {"name": name, "schedule": schedule, "query": query, "next_run": next_run}

    def delete(self, name: str) -> None:
        """Remove a cron job."""
        conn = self._conn
        if conn is None:
            return
        r = conn.execute("DELETE FROM cron_jobs WHERE name = ?", (name,))
        conn.commit()
        if r.rowcount == 0:
            raise ValueError(f"Cron job not found: {name!r}")

    def enable(self, name: str) -> None:
        """Enable a cron job."""
        self._set_enabled(name, 1)

    def disable(self, name: str) -> None:
        """Disable a cron job."""
        self._set_enabled(name, 0)

    def _set_enabled(self, name: str, value: int) -> None:
        conn = self._conn
        if conn is None:
            return
        r = conn.execute(
            "UPDATE cron_jobs SET enabled = ? WHERE name = ?", (value, name)
        )
        conn.commit()
        if r.rowcount == 0:
            raise ValueError(f"Cron job not found: {name!r}")

    def list_jobs(self) -> list[dict]:
        """List all cron jobs."""
        conn = self._conn
        if conn is None:
            return []
        rows = conn.execute(
            "SELECT name, schedule, query, enabled, created_at, last_run, next_run, "
            "run_count, error_count, last_error FROM cron_jobs ORDER BY name"
        ).fetchall()
        return [
            {
                "name": r[0], "schedule": r[1], "query": r[2],
                "enabled": bool(r[3]), "created_at": r[4],
                "last_run": r[5], "next_run": r[6],
                "run_count": r[7], "error_count": r[8], "last_error": r[9],
            }
            for r in rows
        ]

    def run_now(self, name: str) -> Future:
        """Manually trigger a cron job. Returns Future."""
        conn = self._conn
        if conn is None:
            raise RuntimeError("CRON requires persistence")
        row = conn.execute(
            "SELECT query FROM cron_jobs WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Cron job not found: {name!r}")
        query = row[0]
        future = self._submit(query)
        future.add_done_callback(lambda f: self._on_done(f, name))
        return future

    def _tick_loop(self) -> None:
        """Check every 60s for due jobs."""
        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.debug("cron tick error: %s", e)
            for _ in range(60):
                if not self._running:
                    return
                time.sleep(1)

    def _tick(self) -> None:
        """Find and submit all due jobs."""
        conn = self._conn
        if conn is None:
            return
        now = time.time()
        rows = conn.execute(
            "SELECT name, query, schedule FROM cron_jobs "
            "WHERE enabled = 1 AND next_run <= ?",
            (now,),
        ).fetchall()
        for name, query, schedule in rows:
            try:
                future = self._submit(query)
                future.add_done_callback(lambda f, n=name: self._on_done(f, n))
                from croniter import croniter
                next_run = croniter(schedule, now).get_next(float)
                conn.execute(
                    "UPDATE cron_jobs SET last_run = ?, next_run = ?, "
                    "run_count = run_count + 1 WHERE name = ?",
                    (now, next_run, name),
                )
                conn.commit()
            except Exception as e:
                logger.debug("cron job %s submission failed: %s", name, e)
                conn.execute(
                    "UPDATE cron_jobs SET error_count = error_count + 1, "
                    "last_error = ? WHERE name = ?",
                    (str(e), name),
                )
                conn.commit()

    def _on_done(self, future: Future, job_name: str) -> None:
        """Callback after cron job completes."""
        exc = future.exception()
        if exc is not None:
            logger.warning("cron job %s failed: %s", job_name, exc)
            conn = self._conn
            if conn is not None:
                try:
                    conn.execute(
                        "UPDATE cron_jobs SET error_count = error_count + 1, "
                        "last_error = ? WHERE name = ?",
                        (str(exc), job_name),
                    )
                    conn.commit()
                except Exception:
                    pass
