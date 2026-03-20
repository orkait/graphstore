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


def cmd_install_voice(args: argparse.Namespace) -> None:
    """Install voice dependencies (Moonshine STT + Piper TTS)."""
    import subprocess

    packages = ["piper-tts", "moonshine"]
    print(f"Installing voice dependencies: {', '.join(packages)}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install"] + packages,
        check=False,
    )
    if result.returncode == 0:
        print("Voice dependencies installed. You can now use GraphStore(voice=True).")
    else:
        print("Installation failed. Check the output above for details.", file=sys.stderr)
        sys.exit(result.returncode)


def cmd_list_voice(args: argparse.Namespace) -> None:
    """Show voice (STT/TTS) status."""
    stt_status = "not installed"
    tts_status = "not installed"

    try:
        import moonshine  # noqa: F401
        stt_status = "installed"
    except ImportError:
        pass

    try:
        import piper  # noqa: F401
        tts_status = "installed"
    except ImportError:
        pass

    print(f"{'COMPONENT':<20} {'STATUS':<12} PACKAGE")
    print("-" * 50)
    print(f"{'Moonshine STT':<20} {stt_status:<12} moonshine")
    print(f"{'Piper TTS':<20} {tts_status:<12} piper-tts")
    print()
    if stt_status == "installed" and tts_status == "installed":
        print("Voice is ready. Use GraphStore(voice=True) to enable.")
    else:
        print("Run: graphstore install-voice")


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

    # install-voice subcommand
    iv = sub.add_parser("install-voice", help="Install voice dependencies (Moonshine STT + Piper TTS)")
    iv.set_defaults(func=cmd_install_voice)

    # list-voice subcommand
    lv = sub.add_parser("list-voice", help="Show voice (STT/TTS) installation status")
    lv.set_defaults(func=cmd_list_voice)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
