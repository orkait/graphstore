from graphstore import GraphStore
from graphstore.core.errors import OptimizationInProgress
import os
import pytest

def test_sys_evict_limit():
    db = GraphStore(ceiling_mb=100)
    # Ensure some protected schema to demonstrate they aren't evicted
    db.execute('SYS REGISTER NODE KIND "testing" REQUIRED field1:string')
    
    # Add nodes
    db.execute('CREATE NODE "x1" kind="testing" field1="a"')
    db.execute('CREATE NODE "x2" kind="testing" field1="b"')
    db.execute('CREATE NODE "x3" kind="testing" field1="c"')
    db.execute('CREATE NODE "x4" kind="testing" field1="d"')
    
    res = db.execute("SYS EVICT LIMIT 2")
    assert res.data["evicted"] == 2
    assert db.node_count == 2
    
def test_reset_store_semantics(tmp_path):
    # Test strict reset behaviors ensuring we wipe, but allow recreation seamlessly
    # Ensure VectorStore is fully wiped, SQLite tables wiped, but schema and DB reconnects correctly.
    db_dir = tmp_path / "test_reset"
    db_dir.mkdir()
    
    db = GraphStore(path=str(db_dir))
    db.execute('SYS REGISTER NODE KIND "animal" REQUIRED name:string')
    db.execute('CREATE NODE "dog" kind="animal" name="fido"')
    db.execute('CREATE NODE "cat" kind="animal" name="felix"')
    
    assert db.node_count == 2
    db.set_script('CREATE NODE "parrot" kind="animal" name="polly"')
    
    # Do reset
    res = db.reset_store(preserve_config=True)
    assert res.kind == "ok"
    
    # Store should be empty
    assert db.node_count == 0
    
    # Script should be preserved
    assert db.get_script() == 'CREATE NODE "parrot" kind="animal" name="polly"'
    
    # Can still run things
    db.execute('CREATE NODE "mouse" kind="animal" name="mickey"')
    assert db.node_count == 1
    
def test_config_semantics(tmp_path):
    db_dir = tmp_path / "test_config"
    db_dir.mkdir()
    
    db = GraphStore(path=str(db_dir))
    
    res = db.get_runtime_config()
    assert res.data["core"]["ceiling_mb"] == 256
    
    # Update runtime config
    res = db.update_runtime_config({"ceiling_mb": 512, "cost_threshold": 2000})
    assert db.ceiling_mb == 512
    assert db.cost_threshold == 2000
    
    # Update persisted config
    res = db.update_persisted_config({"ceiling_mb": 1024})
    
    # Runtime should still be 512
    assert db.ceiling_mb == 512
    
    # But read persisted should show 1024
    res = db.get_persisted_config()
    assert res.data["core"]["ceiling_mb"] == 1024
    
    # New instance loading that config should start with 1024
    db2 = GraphStore(path=str(db_dir))
    assert db2.ceiling_mb == 1024
