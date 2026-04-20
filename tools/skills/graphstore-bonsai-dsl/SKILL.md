---
name: graphstore-bonsai-dsl
description: Minimal ingest-only DSL cheat sheet for tiny local LLMs (Ternary-Bonsai 1.7B / 4B / 8B GGUF). Uses few-shot examples instead of reference text. Pair with GBNF grammar-constrained decoding for guaranteed-parseable output.
compatibility: graphstore >= 0.4.0
metadata:
  author: orkait
  version: "2.0"
  target_tokens: 400
---

Convert one user message into GraphStore DSL. Output **only** DSL lines. No prose, no markdown, no `<think>` tags.

Valid verbs:
```
CREATE NODE "id" kind = "K" f = v ... [DOCUMENT "text"]
UPSERT NODE "id" kind = "K" name = "N"
CREATE EDGE "a" -> "b" kind = "K"
ASSERT "fact:X" kind = "belief" value = "V" CONFIDENCE 0.9 SOURCE "msg:id"
RETRACT "fact:X" REASON "why"
```

Escape `"` as `\"` inside strings.

Emit each entity once per message. Entity id = `ent:<lower_slug>`. Belief id = `fact:<topic>`.

---

**Input:**
Session s1, msg m:s1:0, user: "Kailash joined OpenAI as DB engineer."

**Output:**
```
CREATE NODE "m:s1:0" kind = "message" session = "s1" role = "user" DOCUMENT "Kailash joined OpenAI as DB engineer."
UPSERT NODE "ent:kailash" kind = "entity" name = "Kailash"
UPSERT NODE "ent:openai" kind = "entity" name = "OpenAI"
CREATE EDGE "m:s1:0" -> "ent:kailash" kind = "mentions"
CREATE EDGE "m:s1:0" -> "ent:openai" kind = "mentions"
```

---

**Input:**
Session s2, msg m:s2:0, user: "My favorite color is blue."

**Output:**
```
CREATE NODE "m:s2:0" kind = "message" session = "s2" role = "user" DOCUMENT "My favorite color is blue."
ASSERT "fact:favorite_color" kind = "belief" value = "blue" CONFIDENCE 0.9 SOURCE "m:s2:0"
```

---

**Input:**
Session s2, msg m:s2:1, user: "Actually my favorite color is green now."

**Output:**
```
CREATE NODE "m:s2:1" kind = "message" session = "s2" role = "user" DOCUMENT "Actually my favorite color is green now."
RETRACT "fact:favorite_color" REASON "superseded by m:s2:1"
ASSERT "fact:favorite_color" kind = "belief" value = "green" CONFIDENCE 0.9 SOURCE "m:s2:1"
```
