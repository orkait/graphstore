"""CLI entry point for graphstore."""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path


def _open_browser(url: str) -> None:
    """Open browser after a short delay to let the server start."""
    import time

    time.sleep(1)
    webbrowser.open(url)


def cmd_playground(args: argparse.Namespace) -> None:
    """Run the playground web UI."""
    try:
        import uvicorn
    except ImportError:
        print(
            "Missing dependencies. Install with:\n"
            "  pip install graphstore[playground]",
            file=sys.stderr,
        )
        sys.exit(1)

    from graphstore.server import app, mount_static

    # Try dev path first (repo checkout), then installed package path
    repo_root = Path(__file__).resolve().parent.parent
    dev_dist = repo_root / "playground" / "dist"
    pkg_dist = Path(__file__).resolve().parent / "playground_dist"

    if dev_dist.is_dir():
        mount_static(app, dev_dist)
    elif pkg_dist.is_dir():
        mount_static(app, pkg_dist)

    if not args.no_browser:
        url = f"http://{args.host}:{args.port}"
        threading.Thread(target=_open_browser, args=(url,), daemon=True).start()

    import os
    if args.db_path:
        os.environ["GRAPHSTORE_DB_PATH"] = args.db_path

    uvicorn.run(app, host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="graphstore", description="graphstore CLI")
    sub = parser.add_subparsers(dest="command")

    pg = sub.add_parser("playground", help="Launch the playground web UI")
    pg.add_argument("--port", type=int, default=7200, help="Port (default 7200)")
    pg.add_argument("--host", default="127.0.0.1", help="Host (default 127.0.0.1)")
    pg.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open browser automatically",
    )
    pg.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to persist playground database",
    )
    pg.set_defaults(func=cmd_playground)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
