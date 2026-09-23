---
title: Enterprise Text-to-SQL
emoji: 🗄️
colorFrom: gray
colorTo: red
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Ask the live pipeline; explore 453 scored answers
models:
  - hari-krishna-ai/qwen3-8b-text2sql-qlora-v2
  - hari-krishna-ai/qwen3-8b-text2sql-qlora
datasets:
  - hari-krishna-ai/enterprise-text-to-sql-benchmark
tags:
  - text-to-sql
  - qlora
  - evaluation
---

# Enterprise Text-to-SQL — fine-tuning, measured

A QLoRA-fine-tuned **Qwen3-8B** turns business questions into PostgreSQL against
a 12-table enterprise schema.

**43.71 % → 68.43 % strict execution accuracy from fine-tuning, and 70.86 % with
a self-correction loop** — measured on 453 held-out questions by *executing*
every query against a real database, never by string similarity.

## What this Space is

**Try it live** — type a business question and watch the full pipeline run:
SQL generation, static validation, execution against a read-only PostgreSQL
session, and one self-correction attempt if it fails. Every step is shown,
including refusals (ask it to delete something).

The live box calls the deployed API, which serves the **base** Qwen3-8B — free
hosting has no GPU for the adapter. So the page is honest about it: for any of
the 453 benchmark questions, the fine-tuned model's *recorded* answer is shown
next to the live one.

**Explorer** — every one of the 453 scored predictions. For each question: the
gold SQL, what each measured configuration actually produced, and why the
harness marked it right or wrong. Filter to the ones fine-tuning fixed, the
ones it broke, or the ones that never executed. That is the part that carries
the claim: not that a model emits SQL, but *whether the SQL is right*, and how
that was established.

## Two results that did not go as expected

- **Schema retrieval made things worse**, and worse for the fine-tuned model
  (−9.27 pp) than for the base one (−1.55 pp).
- **Most of the accuracy gain is column-convention learning, not better SQL
  reasoning.** "Right rows, wrong columns" went 159 → 0 while genuinely wrong
  row sets only fell 240 → 203.

Both are reported rather than buried.

## Why there is no live demo

Free Hugging Face accounts can host **static** Spaces only; Gradio and Docker
Spaces now require PRO. The model itself is public and runnable — see the
[model card](https://huggingface.co/hari-krishna-ai/qwen3-8b-text2sql-qlora) for
a copy-pasteable loading snippet, including the `enable_thinking=False` argument
that is required and easy to miss.

📦 [Model](https://huggingface.co/hari-krishna-ai/qwen3-8b-text2sql-qlora)
