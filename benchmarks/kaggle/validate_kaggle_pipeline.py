import os
import sys
from pathlib import Path
import shutil

def validate_pipeline():
    print("--- Validation 1: Directory Structure Simulation ---")
    base_dir = Path("tmp_kaggle_sim")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir()
    
    jina_dir = base_dir / "jina-small"
    jina_dir.mkdir()
    # Mocking onnx folder structure
    (jina_dir / "onnx").mkdir()
    (jina_dir / "onnx" / "model.onnx").touch()
    (jina_dir / "tokenizer.json").touch()
    (jina_dir / "config.json").touch()
    
    ner_dir = base_dir / "models" / "tinybert-ner"
    ner_dir.mkdir(parents=True)
    (ner_dir / "tokenizer.json").touch()
    (ner_dir / "onnx").mkdir()
    (ner_dir / "onnx" / "model_int8.onnx").touch()
    (ner_dir / "config.json").touch()

    print(f"Simulated structure at {base_dir}")
    
    print("\n--- Validation 2: Embedder Path Logic ---")
    # Mocking the onnx_hf_embedder search logic
    def mock_check_embedder(model_dir):
        model_dir = Path(model_dir)
        tokenizer_path = model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
             tokenizer_path = model_dir / "onnx" / "tokenizer.json"
        
        onnx_path = model_dir / "onnx" / "model_fp16.onnx"
        if not onnx_path.exists():
            onnx_path = model_dir / "onnx" / "model.onnx"
        if not onnx_path.exists():
            onnx_path = model_dir / "model.onnx"
            
        print(f"Tokenizer found: {tokenizer_path.exists()} at {tokenizer_path}")
        print(f"ONNX found: {onnx_path.exists()} at {onnx_path}")
        return tokenizer_path.exists() and onnx_path.exists()

    assert mock_check_embedder(jina_dir) == True
    
    print("\n--- Validation 3: NER Path Logic ---")
    def mock_check_ner(model_dir):
        model_dir = Path(model_dir)
        tokenizer_path = model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            tokenizer_path = model_dir / "onnx" / "tokenizer.json"
            
        onnx_path = model_dir / "onnx" / "model_int8.onnx"
        if not onnx_path.exists():
            onnx_path = model_dir / "onnx" / "model.onnx"
        if not onnx_path.exists():
            onnx_path = model_dir / "model.onnx"
            
        print(f"NER Tokenizer found: {tokenizer_path.exists()} at {tokenizer_path}")
        print(f"NER ONNX found: {onnx_path.exists()} at {onnx_path}")
        return tokenizer_path.exists() and onnx_path.exists()

    assert mock_check_ner(ner_dir) == True
    
    print("\nPipeline Path Validation PASSED.")
    shutil.rmtree(base_dir)

if __name__ == "__main__":
    validate_pipeline()
