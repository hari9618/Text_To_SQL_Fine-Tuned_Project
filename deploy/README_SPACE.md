---
title: Enterprise Text-to-SQL
emoji: 🗄️
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Natural-language questions answered as executed PostgreSQL
---

# Enterprise Text-to-SQL

Ask a business question in plain English. It becomes PostgreSQL, is validated,
executed against a read-only session, and the rows come back — with the SQL and
the full pipeline trace shown, including any self-correction.

**Fine-tuning a Qwen3-8B adapter took strict execution accuracy from 43.71 % to
68.43 %, and 70.86 % with self-correction** on 453 unseen questions. The
*Benchmark* tab has the ablation, the
difficulty breakdown, and the honest reading of where the gain came from.

## What this demo serves

The **base** model, not the fine-tuned one — the adapter needs a GPU and this
Space is on CPU. So the demo shows the *pipeline* (schema context → generation →
static validation → execution → repair); the fine-tuning result is established
by the benchmark, measured separately with the adapter loaded on a GPU.

## Safety

The service executes text a language model wrote. Three layers hold regardless
of what it writes: a non-superuser role, read-only sessions with a statement
timeout applied on every pool checkout, and static rejection of anything that is
not a single read-only statement over tables that exist. Rows are capped
server-side. All data is synthetic.
