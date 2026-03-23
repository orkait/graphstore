#!/usr/bin/env python3
"""Download test fixtures for graphstore mock testing.

Run once:  python scripts/download_fixtures.py
Total: ~100MB, gitignored under tests/fixtures/
"""

import json
import os
import ssl
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"

# Relaxed SSL for old certs on some mirrors
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def dl(url, dest, label=""):
    p = Path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.stat().st_size > 0:
        return
    tag = label or p.name
    print(f"  {tag}", end="", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "graphstore-fixtures/1.0"})
        with urllib.request.urlopen(req, context=CTX) as r, open(p, "wb") as f:
            f.write(r.read())
        kb = p.stat().st_size // 1024
        print(f"  ({kb}KB)")
    except Exception as e:
        print(f"  FAILED: {e}")
        p.unlink(missing_ok=True)


# ─── Text: Gutenberg ──────────────────────────────────────────────────────────
def download_text():
    print("\n[text] Gutenberg books")
    books = [
        # Fiction
        (1342, "pride-and-prejudice.txt"),
        (11,   "alice-in-wonderland.txt"),
        (84,   "frankenstein.txt"),
        (1661, "sherlock-holmes.txt"),
        (345,  "dracula.txt"),
        (2701, "moby-dick.txt"),
        (98,   "tale-of-two-cities.txt"),
        (174,  "dorian-gray.txt"),
        (1260, "jane-eyre.txt"),
        (1400, "great-expectations.txt"),
        # Non-fiction
        (1232, "the-prince.txt"),
        (3207, "leviathan.txt"),
        (2488, "origin-of-species.txt"),
        (521,  "flatland.txt"),
        (2009, "art-of-war.txt"),
        (1404, "meditations-aurelius.txt"),
        # Short stories
        (1064, "yellow-wallpaper.txt"),
        (2148, "bartleby.txt"),
        (1268, "metamorphosis.txt"),
        (219,  "heart-of-darkness.txt"),
    ]
    d = FIXTURES / "text"
    for eid, name in books:
        dl(f"https://www.gutenberg.org/cache/epub/{eid}/pg{eid}.txt", d / name)


# ─── PDFs: arXiv (lightweight, <2MB each) ─────────────────────────────────────
def download_pdfs():
    print("\n[pdf] arXiv papers (6 short papers)")
    papers = [
        ("1706.03762", "attention-is-all-you-need.pdf"),
        ("1810.04805", "bert.pdf"),
        ("2005.11401", "rag.pdf"),
        ("2302.13971", "toolformer.pdf"),
        ("1607.00653", "node2vec.pdf"),
        ("1301.3666",  "word2vec.pdf"),
    ]
    d = FIXTURES / "pdf"
    for arxiv_id, name in papers:
        dl(f"https://arxiv.org/pdf/{arxiv_id}", d / name)


# ─── HTML: Wikipedia (mobile = lighter) ───────────────────────────────────────
def download_html():
    print("\n[html] Wikipedia articles")
    articles = [
        ("Transformer_(deep_learning_architecture)", "transformer.html"),
        ("Large_language_model",                     "llm.html"),
        ("Knowledge_graph",                          "knowledge-graph.html"),
        ("Retrieval-augmented_generation",           "rag.html"),
        ("Vector_database",                          "vector-db.html"),
    ]
    d = FIXTURES / "html"
    for slug, name in articles:
        dl(f"https://en.m.wikipedia.org/wiki/{slug}", d / name)


# ─── Markdown: vector DB READMEs ──────────────────────────────────────────────
def download_markdown():
    print("\n[markdown] GitHub READMEs")
    repos = [
        ("facebookresearch/faiss", "main",   "faiss.md"),
        ("chroma-core/chroma",     "main",   "chroma.md"),
        ("qdrant/qdrant",          "master", "qdrant.md"),
        ("milvus-io/milvus",       "master", "milvus.md"),
        ("pgvector/pgvector",      "master", "pgvector.md"),
    ]
    d = FIXTURES / "markdown"
    for repo, branch, name in repos:
        dl(f"https://raw.githubusercontent.com/{repo}/{branch}/README.md", d / name)


# ─── CSV: tabular datasets ────────────────────────────────────────────────────
def download_csv():
    print("\n[csv] Tabular datasets")
    files = [
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
         "iris.csv"),
        ("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
         "tips.csv"),
        ("https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv",
         "countries.csv"),
        ("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv",
         "penguins.csv"),
        ("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv",
         "mpg.csv"),
    ]
    d = FIXTURES / "csv"
    for url, name in files:
        dl(url, d / name)


# ─── Images: HuggingFace beans dataset (ungated, real photos, 3 classes) ──────
def download_images():
    print("\n[images] HuggingFace beans dataset (20 images, 3 classes)")
    d = FIXTURES / "images"
    d.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
    except ImportError:
        print("  SKIP — need: pip install datasets pillow")
        return

    ds = load_dataset("beans", split="test", streaming=True)
    label_names = ["angular_leaf_spot", "bean_rust", "healthy"]
    manifest = []
    counts = {l: 0 for l in label_names}
    for sample in ds:
        label = label_names[sample["labels"]]
        if counts[label] >= 7:
            continue
        if len(manifest) >= 20:
            break
        idx = len(manifest)
        fname = f"img_{idx:02d}_{label}.jpg"
        path = d / fname
        if not path.exists():
            sample["image"].save(path)
        kb = path.stat().st_size // 1024
        print(f"  {fname} ({kb}KB)")
        manifest.append({"file": fname, "label": label})
        counts[label] += 1

    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  manifest.json ({len(manifest)} images)")


# ─── Voice: AI4Bharat Svarah ──────────────────────────────────────────────────
def download_voice():
    print("\n[voice] OpenSLR 118 — real Hindi speech (Gram Vaani)")
    d = FIXTURES / "voice"
    d.mkdir(parents=True, exist_ok=True)

    import subprocess, tarfile, shutil

    archive = d / "_eval.tar.gz"
    extract_dir = d / "_extract"
    manifest_path = d / "manifest.json"

    if manifest_path.exists():
        print("  skip (manifest exists)")
        return

    # Download the 62MB eval set
    dl("https://openslr.org/resources/118/GV_Eval_3h.tar.gz", archive, "GV_Eval_3h.tar.gz")
    if not archive.exists():
        print("  FAILED to download archive")
        return

    # Extract
    print("  extracting...", end="", flush=True)
    extract_dir.mkdir(exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extract_dir, filter="data")
    print(" done")

    # Find audio files + transcript (format: "utt_id transcript_text")
    audio_files = sorted(extract_dir.rglob("*.mp3"))[:15]
    transcripts = {}
    for tf in extract_dir.rglob("text"):
        for line in tf.read_text(encoding="utf-8", errors="replace").strip().splitlines():
            idx = line.find(" ")
            if idx > 0:
                utt_id = line[:idx].strip()
                text = line[idx + 1:].strip()
                transcripts[utt_id] = text

    manifest = []
    for i, src in enumerate(audio_files):
        fname = f"clip_{i:02d}.mp3"
        dest = d / fname
        if not dest.exists():
            shutil.copy2(src, dest)
        kb = dest.stat().st_size // 1024
        transcript = transcripts.get(src.stem, "")
        print(f"  {fname} ({kb}KB)")
        manifest.append({"file": fname, "lang": "hi", "transcript": transcript})

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  manifest.json ({len(manifest)} clips)")

    # Cleanup archive and extract dir
    archive.unlink(missing_ok=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Downloading fixtures to {FIXTURES}/")
    download_text()
    download_pdfs()
    download_html()
    download_markdown()
    download_csv()
    download_images()
    download_voice()

    # Ensure gitignored
    gi = ROOT / "tests" / "fixtures" / ".gitkeep"
    gi.parent.mkdir(parents=True, exist_ok=True)

    total = sum(f.stat().st_size for f in FIXTURES.rglob("*") if f.is_file()) // (1024 * 1024)
    print(f"\nDone. Total: ~{total}MB")
