"""Model interfaces for Text-to-SQL generation.

    schema_context.py  renders the database schema shown to the model
    prompt.py          the versioned prompt template
    base_model.py      the generation interface and its implementations

Phase 4 uses these for the un-fine-tuned baseline. Phase 10 reuses them
unchanged against the fine-tuned adapter, which is what makes the two numbers
comparable.
"""
