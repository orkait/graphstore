
import time
import sys
import os
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.framework.datasets import load_longmemeval
from benchmarks.framework.adapters.graphstore_ import GraphStoreAdapter

def test_speed():
    data_path = "data/longmemeval"
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}, skipping speed test")
        return

    dataset = load_longmemeval(data_path, variant="s", max_records=1)
    record = dataset.records[0]
    
    config = {
        "embedder": "onnx",
        "embedder_model_dir": "models/jina-v5-small-retrieval",
        "embedder_pooling": "last_token",
        "embedder_output_dims": 1024,
        "entity_extractor": "tinybert_onnx", 
        "entity_model_dir": "models/tinybert-ner",
        "entities": True,
        "embedder_gpu": True,
        "entity_gpu": True,
        "embed_batch_size": 256
    }
    
    adapter = GraphStoreAdapter(config)
    adapter.reset()
    
    print("Warming up AI models (Loading 1.2GB into GPU)...")
    st = time.time()
    adapter.warmup()
    print(f"Done warming up ({time.time()-st:.2f}s). Starting ingestion...")
    
    print(f"Ingesting 1 record ({len(record.sessions)} sessions)...")
    
    t0 = time.time()
    for i, sess in enumerate(record.sessions):
        if i >= 10: # Show 10 sessions to prove speed
            break
        st = time.time()
        adapter.ingest(sess)
        elapsed = time.time() - st
        avg = (time.time() - t0) / (i + 1)
        print(f"  session {i+1:02d}/53 took {elapsed:.3f}s (avg: {avg:.3f}s)")
    
    total = time.time() - t0
    print(f"Total time: {total:.2f}s")
    print(f"Avg per session: {total/len(record.sessions):.3f}s")

if __name__ == "__main__":
    test_speed()
