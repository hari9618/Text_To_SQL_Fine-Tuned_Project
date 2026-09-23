# Ablation study

Phase 14.

Every configuration below is scored on the **same 453 unseen test examples**, through the same execution-based harness, against the same database. The only thing that changes between rows is the one component named.

## Configurations

| # | configuration | status | what changes |
|---|---|---|---|
| 1 | Base model | measured | Qwen3-8B Instruct, full schema in the prompt. The frozen Phase 4 baseline everything else is compared against. |
| 2 | Base + schema retrieval | **not measured** | Only the tables a lexical retriever selects are shown (k=4, FK expansion 2, tuned on validation). |
| 3 | Fine-tuned | measured | QLoRA adapter (r=16, 1 epoch, completion-only loss), full schema. |
| 4 | Fine-tuned + schema retrieval | **not measured** | The adapter with a retrieved schema subset. |
| 5 | Fine-tuned + repair | measured | The adapter, full schema, one repair attempt when the SQL fails to validate or execute. |

## Results

| metric | **1** | **3** | **5** |
|---|---:|---:|---:|
| strict execution accuracy % <br><sub>higher is better</sub> | 43.71 | 68.43 | **70.86** |
| projection-tolerant % <br><sub>higher is better</sub> | 47.68 | 68.43 | **70.86** |
| executable SQL % <br><sub>higher is better</sub> | 95.58 | 94.48 | **98.23** |
| schema hallucination % <br><sub>lower is better</sub> | **0.22** | 3.75 | 0.66 |
| syntax error % <br><sub>lower is better</sub> | **0.00** | 0.22 | 0.22 |

## Strict execution accuracy by difficulty

| tier | n | **1** | **3** | **5** |
|---|---:|---:|---:|---:|
| easy | 126 | 92.9 | 100.0 | 100.0 |
| medium | 63 | 69.8 | 96.8 | 96.8 |
| hard | 144 | 7.6 | 54.9 | 62.5 |
| very_hard | 36 | 0.0 | 13.9 | 13.9 |
| enterprise | 84 | 31.0 | 46.4 | 46.4 |

## Repair loop

| | |
|---|---:|
| queries that failed and were retried | 25 |
| now execute | 17 (68.0 %) |
| now return the gold rows | 11 (44.0 %) |
| model returned identical SQL | 5 |
| mean repair latency | 15448 ms |

*Repair success* means the query now executes — the only signal available in production, where there is no gold answer to compare against. *Repair correctness* means it now returns the right rows. A repair that turns a crash into a confident wrong answer counts on the first and not the second, which is why both are reported.

## Not measured

- **2. Base + schema retrieval** — no retrieval/summary.json
- **4. Fine-tuned + schema retrieval** — no finetuned_retrieval/summary.json

These are listed rather than omitted. An ablation with a hidden gap invites the reader to assume the missing row would have agreed with the others.

## Provenance

| # | model | schema mode | prompt | database |
|---|---|---|---|---|
| 1 | `base_model_4bit_replayed` | `full_schema_no_retrieval` | `4e72cc5f722ce436` | `5a018e291c3df4ad` |
| 3 | `qlora_adapter_replayed` | `full_schema_no_retrieval` | `4e72cc5f722ce436` | `5a018e291c3df4ad` |
| 5 | `qlora_adapter_replayed` | `full_schema_no_retrieval` | `4e72cc5f722ce436` | `5a018e291c3df4ad` |

