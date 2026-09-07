# Ablation study

Phase 14.

Every configuration below is scored on the **same 453 unseen test examples**, through the same execution-based harness, against the same database. The only thing that changes between rows is the one component named.

## Configurations

| # | configuration | status | what changes |
|---|---|---|---|
| 1 | Base model | measured | Qwen3-8B Instruct, full schema in the prompt. The frozen Phase 4 baseline everything else is compared against. |
| 2 | Base + schema retrieval | measured | Only the tables a lexical retriever selects are shown (k=4, FK expansion 2, tuned on validation). |
| 3 | Fine-tuned | measured | QLoRA adapter (r=16, 1 epoch, completion-only loss), full schema. |
| 4 | Fine-tuned + schema retrieval | measured | The adapter with a retrieved schema subset. |
| 5 | Fine-tuned + repair | measured | The adapter, full schema, one repair attempt when the SQL fails to validate or execute. |

## Results

| metric | **1** | **2** | **3** | **4** | **5** |
|---|---:|---:|---:|---:|---:|
| strict execution accuracy % <br><sub>higher is better</sub> | 10.82 | 9.27 | 50.99 | 41.72 | **52.10** |
| projection-tolerant % <br><sub>higher is better</sub> | 45.92 | 43.93 | 50.99 | 41.72 | **52.10** |
| executable SQL % <br><sub>higher is better</sub> | **98.90** | 92.27 | 95.81 | 94.92 | 98.23 |
| schema hallucination % <br><sub>lower is better</sub> | **0.66** | 3.31 | 1.99 | 3.53 | **0.66** |
| syntax error % <br><sub>lower is better</sub> | **0.00** | **0.00** | 0.22 | **0.00** | 0.22 |

## Strict execution accuracy by difficulty

| tier | n | **1** | **2** | **3** | **4** | **5** |
|---|---:|---:|---:|---:|---:|---:|
| easy | 126 | 0.0 | 0.0 | 23.0 | 19.0 | 23.0 |
| medium | 63 | 52.4 | 44.4 | 52.4 | 60.3 | 60.3 |
| hard | 144 | 3.5 | 2.1 | 84.7 | 63.9 | 84.7 |
| very_hard | 36 | 0.0 | 0.0 | 8.3 | 5.6 | 8.3 |
| enterprise | 84 | 13.1 | 13.1 | 52.4 | 39.3 | 52.4 |

## Repair loop

| | |
|---|---:|
| queries that failed and were retried | 19 |
| now execute | 11 (57.9 %) |
| now return the gold rows | 5 (26.3 %) |
| model returned identical SQL | 4 |
| mean repair latency | 9297 ms |

*Repair success* means the query now executes — the only signal available in production, where there is no gold answer to compare against. *Repair correctness* means it now returns the right rows. A repair that turns a crash into a confident wrong answer counts on the first and not the second, which is why both are reported.

## Provenance

| # | model | schema mode | prompt | database |
|---|---|---|---|---|
| 1 | `hf_inference_providers` | `full_schema_no_retrieval` | `8288e41a496531a9` | `5a018e291c3df4ad` |
| 2 | `hf_inference_providers` | `retrieved_keyword_v1` | `8288e41a496531a9` | `5a018e291c3df4ad` |
| 3 | `qlora_adapter_replayed` | `full_schema_no_retrieval` | `8288e41a496531a9` | `5a018e291c3df4ad` |
| 4 | `qlora_adapter_replayed` | `retrieved_keyword_v1` | `8288e41a496531a9` | `5a018e291c3df4ad` |
| 5 | `qlora_adapter_replayed` | `full_schema_no_retrieval` | `8288e41a496531a9` | `5a018e291c3df4ad` |

