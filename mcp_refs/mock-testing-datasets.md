# Mock Testing Dataset Decisions

Datasets for testing graphstore's ingest pipeline (vision, voice, document).
Not for training — sized for pipeline correctness and edge case coverage.

---

## Images — Tiny ImageNet subset

**URL:** https://huggingface.co/datasets/zh-plus/tiny-imagenet
**License:** CC0 / Public Domain
**Target:** 20-30 images across 5-6 categories

```bash
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('zh-plus/tiny-imagenet', split='valid')
# Pick 5 representative categories, 5 images each
categories = ['goldfish', 'tabby', 'school bus', 'desktop computer', 'pizza']
subset = [x for x in ds if x['label_str'] in categories][:30]
from datasets import Dataset
Dataset.from_list(subset).save_to_disk('tests/fixtures/images')
"
```

**Why:** Known labels per image. Test: ingest image → description contains expected category word.

---

## Voice — AI4Bharat Svarah (Indian English)

**URL:** https://huggingface.co/datasets/ai4bharat/Svarah
**License:** CC BY 4.0
**Target:** 10-15 clips across different speakers and domains

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('ai4bharat/Svarah', split='test')
# Take clips across different domains: govt, commerce, sports, general
subset = ds.select(range(15))
subset.save_to_disk('tests/fixtures/voice')
"
```

**Why:** 117 Indian English speakers, diverse accents, ground-truth transcripts included.
Test: STT transcription → fuzzy match against dataset's `sentence` field.

---

## Text — Project Gutenberg books

**URL:** https://www.gutenberg.org
**License:** Public domain
**Target:** 25-30 books across genres — enough for BM25 ranking, cross-document recall, chunk deduplication, and topic clustering to all be exercised meaningfully

```bash
# Download script: scripts/download_text_fixtures.sh
mkdir -p tests/fixtures/text

# Fiction
curl -s "https://www.gutenberg.org/cache/epub/1342/pg1342.txt" -o tests/fixtures/text/pride-and-prejudice.txt
curl -s "https://www.gutenberg.org/cache/epub/11/pg11.txt"     -o tests/fixtures/text/alice-in-wonderland.txt
curl -s "https://www.gutenberg.org/cache/epub/84/pg84.txt"     -o tests/fixtures/text/frankenstein.txt
curl -s "https://www.gutenberg.org/cache/epub/1260/pg1260.txt" -o tests/fixtures/text/jane-eyre.txt
curl -s "https://www.gutenberg.org/cache/epub/98/pg98.txt"     -o tests/fixtures/text/tale-of-two-cities.txt
curl -s "https://www.gutenberg.org/cache/epub/1400/pg1400.txt" -o tests/fixtures/text/great-expectations.txt
curl -s "https://www.gutenberg.org/cache/epub/2701/pg2701.txt" -o tests/fixtures/text/moby-dick.txt
curl -s "https://www.gutenberg.org/cache/epub/174/pg174.txt"   -o tests/fixtures/text/picture-of-dorian-gray.txt
curl -s "https://www.gutenberg.org/cache/epub/1661/pg1661.txt" -o tests/fixtures/text/sherlock-holmes.txt
curl -s "https://www.gutenberg.org/cache/epub/345/pg345.txt"   -o tests/fixtures/text/dracula.txt

# Non-fiction / essays
curl -s "https://www.gutenberg.org/cache/epub/1232/pg1232.txt" -o tests/fixtures/text/the-prince-machiavelli.txt
curl -s "https://www.gutenberg.org/cache/epub/7370/pg7370.txt" -o tests/fixtures/text/discourse-on-method.txt
curl -s "https://www.gutenberg.org/cache/epub/4280/pg4280.txt" -o tests/fixtures/text/pragmatism-james.txt
curl -s "https://www.gutenberg.org/cache/epub/3207/pg3207.txt" -o tests/fixtures/text/leviathan-hobbes.txt
curl -s "https://www.gutenberg.org/cache/epub/5827/pg5827.txt" -o tests/fixtures/text/wealth-of-nations.txt

# Science / technical
curl -s "https://www.gutenberg.org/cache/epub/2488/pg2488.txt" -o tests/fixtures/text/origin-of-species.txt
curl -s "https://www.gutenberg.org/cache/epub/37729/pg37729.txt" -o tests/fixtures/text/relativity-einstein.txt
curl -s "https://www.gutenberg.org/cache/epub/521/pg521.txt"   -o tests/fixtures/text/flatland.txt

# History
curl -s "https://www.gutenberg.org/cache/epub/2009/pg2009.txt" -o tests/fixtures/text/art-of-war.txt
curl -s "https://www.gutenberg.org/cache/epub/1404/pg1404.txt" -o tests/fixtures/text/meditations-aurelius.txt
curl -s "https://www.gutenberg.org/cache/epub/3600/pg3600.txt" -o tests/fixtures/text/history-of-rome.txt

# Short stories (good for chunk boundary testing)
curl -s "https://www.gutenberg.org/cache/epub/1064/pg1064.txt" -o tests/fixtures/text/yellow-wallpaper.txt
curl -s "https://www.gutenberg.org/cache/epub/910/pg910.txt"   -o tests/fixtures/text/turn-of-the-screw.txt
curl -s "https://www.gutenberg.org/cache/epub/2148/pg2148.txt" -o tests/fixtures/text/bartleby.txt
curl -s "https://www.gutenberg.org/cache/epub/1268/pg1268.txt" -o tests/fixtures/text/metamorphosis.txt
curl -s "https://www.gutenberg.org/cache/epub/219/pg219.txt"   -o tests/fixtures/text/heart-of-darkness.txt
```

**Why:** Genre diversity ensures BM25 doesn't collapse to one domain. Short stories test chunk boundaries. Overlapping themes (war, nature, society) test cross-document recall. Known phrases per book enable deterministic assertions.

---

---

## PDFs — arXiv papers (semantic-rich, embedding-friendly)

**URL:** https://arxiv.org
**License:** CC BY 4.0 (most ML/CS papers)
**Target:** 15-20 papers across 4-5 topic clusters — intentionally grouped so embedding similarity between same-topic papers is testable

```bash
mkdir -p tests/fixtures/pdf

# ML / deep learning cluster
curl -sL "https://arxiv.org/pdf/1706.03762" -o tests/fixtures/pdf/attention-is-all-you-need.pdf
curl -sL "https://arxiv.org/pdf/1810.04805" -o tests/fixtures/pdf/bert.pdf
curl -sL "https://arxiv.org/pdf/2005.14165" -o tests/fixtures/pdf/gpt3.pdf
curl -sL "https://arxiv.org/pdf/2303.08774" -o tests/fixtures/pdf/gpt4.pdf

# Retrieval / RAG cluster
curl -sL "https://arxiv.org/pdf/2005.11401" -o tests/fixtures/pdf/rag-lewis.pdf
curl -sL "https://arxiv.org/pdf/2212.10560" -o tests/fixtures/pdf/self-ask.pdf
curl -sL "https://arxiv.org/pdf/2210.11610" -o tests/fixtures/pdf/react.pdf

# Graph / knowledge cluster
curl -sL "https://arxiv.org/pdf/1301.3666" -o tests/fixtures/pdf/word2vec.pdf
curl -sL "https://arxiv.org/pdf/1607.00653" -o tests/fixtures/pdf/node2vec.pdf
curl -sL "https://arxiv.org/pdf/2012.09561" -o tests/fixtures/pdf/graphrag.pdf

# Agent / memory cluster
curl -sL "https://arxiv.org/pdf/2304.03442" -o tests/fixtures/pdf/generative-agents.pdf
curl -sL "https://arxiv.org/pdf/2305.10601" -o tests/fixtures/pdf/memgpt.pdf
curl -sL "https://arxiv.org/pdf/2308.00352" -o tests/fixtures/pdf/longmem.pdf

# Vision / multimodal cluster
curl -sL "https://arxiv.org/pdf/2103.00020" -o tests/fixtures/pdf/clip.pdf
curl -sL "https://arxiv.org/pdf/2204.14198" -o tests/fixtures/pdf/flamingo.pdf
```

**Why:** Topic clusters test that `SIMILAR TO NODE "rag-lewis"` returns other RAG papers, not graph papers. Known abstracts enable LEXICAL SEARCH assertions. PDFs exercise the full pymupdf4llm/docling parse path including page boundaries and headings.

---

## Other formats — HTML, Markdown, CSV

**Target:** 5 files per format — enough to test each parser path without redundancy

### HTML — Wikipedia article exports

```bash
mkdir -p tests/fixtures/html

# Well-structured articles with headings, tables, infoboxes
curl -sL "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)?action=raw" \
    -o tests/fixtures/html/transformer.html
curl -sL "https://en.wikipedia.org/wiki/Large_language_model?action=raw" \
    -o tests/fixtures/html/llm.html
curl -sL "https://en.wikipedia.org/wiki/Knowledge_graph?action=raw" \
    -o tests/fixtures/html/knowledge-graph.html
curl -sL "https://en.wikipedia.org/wiki/Retrieval-augmented_generation?action=raw" \
    -o tests/fixtures/html/rag.html
curl -sL "https://en.wikipedia.org/wiki/Vector_database?action=raw" \
    -o tests/fixtures/html/vector-db.html
```

### Markdown — technical READMEs / docs (GitHub raw)

```bash
mkdir -p tests/fixtures/markdown

curl -sL "https://raw.githubusercontent.com/facebookresearch/faiss/main/README.md" \
    -o tests/fixtures/markdown/faiss.md
curl -sL "https://raw.githubusercontent.com/chroma-core/chroma/main/README.md" \
    -o tests/fixtures/markdown/chroma.md
curl -sL "https://raw.githubusercontent.com/qdrant/qdrant/master/README.md" \
    -o tests/fixtures/markdown/qdrant.md
curl -sL "https://raw.githubusercontent.com/milvus-io/milvus/master/README.md" \
    -o tests/fixtures/markdown/milvus.md
curl -sL "https://raw.githubusercontent.com/pgvector/pgvector/master/README.md" \
    -o tests/fixtures/markdown/pgvector.md
```

**Why:** All vector DB READMEs — embedding them should cluster tightly. Tests that `SIMILAR TO "text: what is a vector database"` returns all five nodes.

### CSV — structured tabular data

```bash
mkdir -p tests/fixtures/csv

# UCI ML repo + Kaggle open datasets (no login required)
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data" \
    -o tests/fixtures/csv/iris.csv
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data" \
    -o tests/fixtures/csv/wine.csv
curl -sL "https://people.sc.fsu.edu/~jburkardt/data/csv/cities.csv" \
    -o tests/fixtures/csv/cities.csv
curl -sL "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv" \
    -o tests/fixtures/csv/heights-weights.csv
curl -sL "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv" \
    -o tests/fixtures/csv/countries.csv
```

**Why:** Tests that graphstore can ingest structured row data as nodes (one node per row), assert on field-level column queries (`NODES WHERE species = "Iris-setosa"`), and handle numeric columns correctly.

---

## Fixture structure

```
tests/
└── fixtures/           # gitignored — run scripts/download_fixtures.sh once
    ├── images/         # 25-30 images, 5-6 categories, known labels
    ├── voice/          # 10-15 Svarah clips, ground-truth transcripts included
    ├── text/           # 25-27 Gutenberg books across 4 genres
    ├── pdf/            # 15 arXiv papers in 4 topic clusters
    ├── html/           # 5 Wikipedia articles (same topic area)
    ├── markdown/       # 5 vector DB READMEs (should cluster together)
    └── csv/            # 5 tabular datasets (row-as-node ingest)
```

`.gitignore` entry:
```
tests/fixtures/
```

---

## Setup script

`scripts/download_fixtures.sh` — run once per dev environment:

```bash
#!/usr/bin/env bash
set -e
pip install -q datasets huggingface_hub
python scripts/download_image_fixtures.py
python scripts/download_voice_fixtures.py
bash scripts/download_text_fixtures.sh
echo "Fixtures ready."
```

---

## Test assertions per type

| Fixture | Ingest path | Assertion |
|---------|-------------|-----------|
| Image | `INGEST "img.jpg" USING VISION "model"` | Description contains expected category label |
| Voice | `gs.listen(on_text=cb)` | Transcript fuzzy-matches `sentence` field from dataset |
| Text | `INGEST "book.txt"` | `LEXICAL SEARCH "known phrase"` returns expected node |
| Text (cross-doc) | Ingest 5+ books | `RECALL FROM "concept:war" DEPTH 2` reaches nodes from multiple books |
| Text (chunking) | Ingest short story | Chunk count matches expected, headings preserved |
| PDF | `INGEST "paper.pdf"` | Page count correct, `LEXICAL SEARCH "attention mechanism"` hits transformer paper |
| PDF (similarity) | Ingest all 15 papers | `SIMILAR TO NODE "rag-lewis"` top-5 are all RAG/retrieval papers |
| HTML | `INGEST "article.html"` | Headings extracted as section nodes, tables preserved |
| Markdown | `INGEST "readme.md"` | `SIMILAR TO "text: vector database"` returns all 5 READMEs |
| CSV | `INGEST "iris.csv"` | Row count matches, `NODES WHERE species = "Iris-setosa"` returns 50 nodes |

Tests tagged `@pytest.mark.fixture` — skipped unless `GRAPHSTORE_FIXTURE_TESTS=1`.
