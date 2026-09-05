"""Enterprise Text-to-SQL system.

Application source code for the Text-to-SQL pipeline. Subpackages are added as
their phase begins:

    retrieval/  schema retrieval        (Phase 6)
    sql/        validation, execution, repair (Phase 1, 11)
    api/        FastAPI service         (Phase 12)
"""

__version__ = "0.1.0"
