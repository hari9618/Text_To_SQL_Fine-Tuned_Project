---
title: Enterprise Text-to-SQL
emoji: 🗄️
colorFrom: gray
colorTo: red
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Did fine-tuning help? 453 held-out questions
models:
  - hari-krishna-ai/qwen3-8b-text2sql-qlora
tags:
  - text-to-sql
  - qlora
  - evaluation
---

# Enterprise Text-to-SQL — fine-tuning, measured

A QLoRA-fine-tuned **Qwen3-8B** turns business questions into PostgreSQL against
a 12-table enterprise schema.

**10.82 % → 50.99 % strict execution accuracy from fine-tuning, and 52.10 % with
a self-correction loop** — measured on 453 held-out questions by *executing*
every query against a real database, never by string similarity.

## What this Space is

An explorer over **every one of the 453 scored predictions**. For each question:
the gold SQL, what each of the four measured configurations actually produced,
and why the harness marked it right or wrong. Filter to the ones fine-tuning
fixed, the ones it broke, or the ones that never executed.

That is deliberately more useful than a live text box. The interesting claim
here is not that a model emits SQL — it is *whether the SQL is right*, and how
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
