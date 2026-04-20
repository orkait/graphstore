---
name: graphstore-bonsai-dsl-compact
description: Unified verb-positional caveman grammar covering every common GraphStore DSL operation. LLM emits 2-letter verbs + positional args, Python expands to full DSL. ~3-5x fewer output tokens than raw DSL on every path - ingest, query, walk, ops.
compatibility: graphstore >= 0.4.0
metadata:
  author: orkait
  version: "5.0"
  target_tokens: 900
  mode: unified-positional
---

Read the user turn. Output zero or more ops, one per line. No prose, no quotes (unless required inside a query string), no `<think>` tags, no fences.

Each line: `<verb> <arg1> [arg2...]`. Multi-word trailing args (names, query text) are allowed; the verb's shape fixes how Python splits the tokens.

## Ingest (user said something about an entity / themselves)

```
U <slug> <Name>              Upsert entity. Python auto-wires mentions edge from msg.
F <topic> <value>            User's first-person fact ("I", "my"). topic=snake_case.
D <topic>                    Drop a fact (retract). Requires matching known fact.
```

## Graph edges (explicit relationships between entities)

```
E <from_id> <to_id> <kind>   Create edge with given kind. IDs include their prefix (ent:X or fact:X).
```

## Semantic retrieval (user asked a question)

```
RM <query text>              REMEMBER (4-signal NL retrieval, default LIMIT 10)
SM <query text>              SIMILAR TO (vector only, default LIMIT 10)
LX <query text>              LEXICAL SEARCH (BM25 only, default LIMIT 10)
AQ <question text>           ANSWER (LLM-answered recall)
```

## Structural walks (from a known anchor id)

```
RL <anchor_id>               RECALL FROM anchor DEPTH 2 (spreading activation)
TR <anchor_id>               TRAVERSE FROM anchor DEPTH 2 (deterministic walk)
AN <anchor_id>               ANCESTORS OF anchor DEPTH 3
SG <anchor_id>               SUBGRAPH FROM anchor DEPTH 2
```

## SYS / vault ops

```
SS                           SYS SNAPSHOT
SC                           SYS COMPACT
SH                           SYS HEALTH
ST                           SYS STATS
SX <query>                   SYS EXPLAIN REMEMBER (dry-run a retrieval)
VS                           VAULT SYNC
```

## Rules

- Third-person observations emit `U`, NOT `F`. Beliefs require first-person pronouns.
- Empty output is valid - emit nothing if nothing applies.
- If `### KNOWN FACTS` appears above, reuse those topic names exactly when updating same concept.
- Slugs and topics must be single tokens (lowercase + underscores). Names / values / query text can be multi-word.
- For query verbs, write the question as free text - no quotes, Python adds them.

---

**Input:** "Kailash joined OpenAI."

**Output:**
```
U kailash Kailash
U openai OpenAI
```

---

**Input:** "Priya works at Flipkart since 2023 as a frontend engineer."

**Output:**
```
U priya Priya
U flipkart Flipkart
E ent:priya ent:flipkart works_at
```

---

**Input:** "My favorite color is blue."

**Output:**
```
F favorite_color blue
```

---

**Input (correction; known fact exists):**
```
### KNOWN FACTS
[fact:favorite_drink] kind="belief" value="coffee"

user: "Actually I prefer tea now."
```
**Output:**
```
D favorite_drink
F favorite_drink tea
```

---

**Input:** "Remember what I said about coffee."

**Output:**
```
RM what I said about coffee
```

---

**Input:** "Find messages similar to 'joining a startup'."

**Output:**
```
SM joining a startup
```

---

**Input:** "How is Priya connected to OpenAI?"

**Output:**
```
RL ent:priya
```

---

**Input:** "Take a snapshot."

**Output:**
```
SS
```
