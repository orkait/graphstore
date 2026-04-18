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


def cmd_install_embedder(args: argparse.Namespace) -> None:
    """Download and install an embedder model."""
    from graphstore.registry.installer import install_embedder

    try:
        install_embedder(args.name, variant=args.variant)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


def cmd_list_embedders(args: argparse.Namespace) -> None:
    """List available and installed embedder models."""
    from graphstore.registry.models import list_models
    from graphstore.registry.installer import is_installed

    models = list_models()
    print(f"{'NAME':<30} {'STATUS':<12} {'DIMS':<8} DESCRIPTION")
    print("-" * 80)
    for m in models:
        status = "installed" if is_installed(m["name"]) else "available"
        print(f"{m['name']:<30} {status:<12} {m['base_dims']:<8} {m['description']}")


def cmd_uninstall_embedder(args: argparse.Namespace) -> None:
    """Remove an installed embedder model."""
    from graphstore.registry.installer import uninstall_embedder

    uninstall_embedder(args.name)


def _is_loopback_host(host: str) -> bool:
    """Return True iff the given bind host is a loopback-only address.

    Accepts the common forms: ``127.0.0.1``, ``localhost``, ``::1``. Anything
    else (``0.0.0.0``, ``::``, an explicit LAN IP, or a DNS name) is treated
    as "potentially exposed to the network" and requires an auth token.
    """
    if not host:
        return False
    lo = host.strip().lower()
    # IPv4 loopback: strict match. Technically 127.0.0.0/8 is all loopback,
    # but we only accept the canonical form to keep the check conservative.
    if lo in ("127.0.0.1", "localhost", "::1", "[::1]"):
        return True
    return False


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

    import os

    # Refuse to start without auth when binding to anything other than
    # loopback. The playground execute endpoint accepts arbitrary DSL
    # including VAULT READ, INGEST, and SYS *; exposing that to a LAN or the
    # internet without a token is a remote-execute vulnerability.
    # Escape hatch: GRAPHSTORE_ALLOW_UNAUTH_BIND=1 for users who know what
    # they are doing (e.g. a segregated network).
    auth_token_set = bool(os.environ.get("GRAPHSTORE_AUTH_TOKEN"))
    allow_unauth = os.environ.get("GRAPHSTORE_ALLOW_UNAUTH_BIND") == "1"
    if not _is_loopback_host(args.host) and not auth_token_set and not allow_unauth:
        print(
            f"Refusing to bind playground to {args.host!r} without authentication.\n"
            "\n"
            "The playground accepts arbitrary DSL including VAULT READ, INGEST,\n"
            "and SYS commands. Exposing it to a non-loopback address without an\n"
            "auth token lets anyone on the network execute queries against your\n"
            "graphstore.\n"
            "\n"
            "Fix: set an auth token before starting\n"
            "    export GRAPHSTORE_AUTH_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')\n"
            "    graphstore playground --host 0.0.0.0\n"
            "\n"
            "Clients must then send ``Authorization: Bearer <token>`` on\n"
            "every /api/* request.\n"
            "\n"
            "If you really need to disable this check (e.g. inside a private\n"
            "network), set GRAPHSTORE_ALLOW_UNAUTH_BIND=1.",
            file=sys.stderr,
        )
        sys.exit(2)

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

    if args.db_path:
        os.environ["GRAPHSTORE_DB_PATH"] = args.db_path

    uvicorn.run(app, host=args.host, port=args.port)


def cmd_config(args: argparse.Namespace) -> None:
    """Show config schema, defaults, or resolved values."""
    import json
    import msgspec
    from graphstore.config import GraphStoreConfig, load_config

    if args.schema:
        schema = msgspec.json.schema(GraphStoreConfig)
        print(json.dumps(schema, indent=2))
    elif args.defaults:
        data = msgspec.json.decode(msgspec.json.encode(GraphStoreConfig()))
        print(json.dumps(data, indent=2))
    elif args.path:
        config = load_config(args.path)
        data = msgspec.json.decode(msgspec.json.encode(config))
        print(json.dumps(data, indent=2))
    else:
        data = msgspec.json.decode(msgspec.json.encode(GraphStoreConfig()))
        print(json.dumps(data, indent=2))


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

    # install-embedder subcommand
    ie = sub.add_parser("install-embedder", help="Download and install an embedder model")
    ie.add_argument("name", help="Model name (e.g. embeddinggemma-300m)")
    ie.add_argument(
        "--variant",
        default=None,
        help="Model variant (e.g. fp32, q4). Defaults to model's default variant.",
    )
    ie.set_defaults(func=cmd_install_embedder)

    # list-embedders subcommand
    le = sub.add_parser("list-embedders", help="List available and installed embedder models")
    le.set_defaults(func=cmd_list_embedders)

    # uninstall-embedder subcommand
    ue = sub.add_parser("uninstall-embedder", help="Remove an installed embedder model")
    ue.add_argument("name", help="Model name to uninstall")
    ue.set_defaults(func=cmd_uninstall_embedder)

    # config subcommand
    cfg = sub.add_parser("config", help="Show config defaults, schema, or current values")
    cfg.add_argument("--schema", action="store_true", help="Output JSON Schema for graphstore.json")
    cfg.add_argument("--defaults", action="store_true", help="Output all default values as JSON")
    cfg.add_argument("--path", type=str, default=None, help="Path to graphstore.json to show resolved config")
    cfg.set_defaults(func=cmd_config)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
