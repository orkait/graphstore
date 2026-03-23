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
    print("\n[voice] espeak-ng generated clips (English + Hindi)")
    d = FIXTURES / "voice"
    d.mkdir(parents=True, exist_ok=True)

    import subprocess
    clips = [
        # English clips (various lengths and content types)
        ("en", "The quick brown fox jumps over the lazy dog"),
        ("en", "Graphstore is an agent memory substrate"),
        ("en", "Please create a new node with kind memory"),
        ("en", "What is the shortest path from Paris to London"),
        ("en", "The system returned forty two results"),
        # Hindi clips (Indian language, tests non-ASCII transcripts)
        ("hi", "नमस्ते यह एक परीक्षा है"),
        ("hi", "भारत एक विविधताओं वाला देश है"),
        ("hi", "कृपया मुझे रास्ता दिखाइए"),
        ("hi", "आज का मौसम बहुत अच्छा है"),
        ("hi", "ग्राफ डेटाबेस में नोड बनाएं"),
    ]
    manifest = []
    for i, (lang, text) in enumerate(clips):
        fname = f"clip_{i:02d}_{lang}.wav"
        wav_path = d / fname
        if not wav_path.exists():
            try:
                subprocess.run(
                    ["espeak-ng", "-v", lang, "-w", str(wav_path), text],
                    check=True, capture_output=True
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"  {fname} FAILED: {e}")
                continue
        kb = wav_path.stat().st_size // 1024
        print(f"  {fname} ({kb}KB)")
        manifest.append({"file": fname, "lang": lang, "transcript": text})

    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  manifest.json ({len(manifest)} clips)")
    print("  NOTE: These are TTS-generated. Replace with real human recordings for production STT testing.")


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
