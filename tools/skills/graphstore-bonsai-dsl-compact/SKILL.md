---
name: graphstore-bonsai-dsl-compact
description: Ultra-compact NL->semantic-fields skill. LLM emits only the novel information in three tagged lines (ENTS, BELIEFS, RETRACTS). Python templates build the full DSL deterministically. ~6-7x fewer output tokens than the full-DSL skill, measured on 4B TQ1_0.
compatibility: graphstore >= 0.4.0
metadata:
  author: orkait
  version: "1.0"
  target_tokens: 320
  mode: compact
---

Read the user turn. Output EXACTLY three lines in this order:

```
ENTS: <comma-separated "ent:<slug>"="<Name>", or "none">
BELIEFS: <comma-separated "fact:<topic>"="<value>", or "none">
RETRACTS: <comma-separated "fact:<topic>", or "none">
```

No DSL, no prose, no `<think>` tags, no markdown fences. Three lines. Nothing else.

- ENTS lists every named person / org / place / product in the message. Slug is lowercase with underscores. One entry per unique entity per message.
- BELIEFS lists **only** first-person statements about the speaker themselves. The sentence must use "I", "my", "me", "mine", or similar. A third-person observation like "Priya moved to Bangalore" is NOT a belief; those entities go in ENTS. Topic = short snake_case.
- RETRACTS lists existing fact_ids the new message contradicts. Only valid when `### KNOWN FACTS` appears above and the user overrides one. Use the same fact_id from KNOWN FACTS.

Use `none` when a category is empty. Escape `"` inside values as `\"`.

---

**Input (third-person observation; BELIEFS stays empty):**
Session s1, msg m:s1:0, user: "Kailash joined OpenAI as DB engineer."

**Output:**
```
ENTS: "ent:kailash"="Kailash", "ent:openai"="OpenAI"
BELIEFS: none
RETRACTS: none
```

**Input (third-person with a location; still no beliefs):**
Session s1, msg m:s1:1, user: "Priya moved to Bangalore and joined Flipkart."

**Output:**
```
ENTS: "ent:priya"="Priya", "ent:bangalore"="Bangalore", "ent:flipkart"="Flipkart"
BELIEFS: none
RETRACTS: none
```

---

**Input:**
Session s2, msg m:s2:0, user: "My favorite color is blue."

**Output:**
```
ENTS: none
BELIEFS: "fact:favorite_color"="blue"
RETRACTS: none
```

---

**Input (user contradicts a prior fact, use its exact fact_id):**

```
### KNOWN FACTS (reuse these fact_ids; emit RETRACT + ASSERT to update)
[fact:favorite_drink] kind="belief" value="coffee" confidence=0.90

Session s3, msg m:s3:1, user: "Actually I prefer tea now."
```

**Output:**
```
ENTS: none
BELIEFS: "fact:favorite_drink"="tea"
RETRACTS: "fact:favorite_drink"
```

---

**Input (multi-entity + belief + belief update):**

```
### KNOWN FACTS
[fact:lives_in] kind="belief" value="Delhi" confidence=0.90

Session s4, msg m:s4:2, user: "Priya moved to Bangalore and joined Flipkart. I now live in Pune."
```

**Output:**
```
ENTS: "ent:priya"="Priya", "ent:bangalore"="Bangalore", "ent:flipkart"="Flipkart", "ent:pune"="Pune"
BELIEFS: "fact:lives_in"="Pune"
RETRACTS: "fact:lives_in"
```
