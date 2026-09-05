"""Schema retrieval.

    schema_index.py  searchable index over tables, columns and their values
    keyword.py       lexical retriever with foreign-key path expansion

Phase 6 in CLAUDE.md's numbering. Answers a different question from
fine-tuning: *what database information should the model see?* rather than
*how should it use that information?* Both are needed, and each is measured
separately so their contributions never get conflated.
"""
