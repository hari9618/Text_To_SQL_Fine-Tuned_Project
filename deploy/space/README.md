---
title: Text-to-SQL - the fine-tuned model, live
emoji: 🗄️
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: The fine-tuned adapter itself, live on ZeroGPU
models:
  - hari-krishna-ai/qwen3-8b-text2sql-qlora-v2
  - Qwen/Qwen3-8B
tags:
  - text-to-sql
  - qlora
  - peft
---

# Enterprise Text-to-SQL

Ask a business question in plain English; a **QLoRA-fine-tuned Qwen3-8B** turns
it into PostgreSQL against a 12-table enterprise schema, and the SQL is
statically validated before you see it.

**43.71 % → 68.43 % strict execution accuracy from fine-tuning, 70.86 % with a
self-correction loop** — measured on 453 held-out questions by *executing* every
query against a real database, never by string similarity.

The *Benchmark* tab carries the full five-configuration ablation, including the
two results that did not go the expected way: schema retrieval made things
worse, and most of the accuracy gain turns out to be column-convention learning
rather than better SQL reasoning.

**This Space does not execute SQL** — it has no database attached, so the
pipeline stops at static validation and says so. Valid SQL is not a correct
answer.

📦 [Model](https://huggingface.co/hari-krishna-ai/qwen3-8b-text2sql-qlora)
