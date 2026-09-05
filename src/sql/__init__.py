"""Database connection, SQL validation, execution, and repair.

    config.py     connection settings loaded from the environment  (Phase 1)
    executor.py   run queries against PostgreSQL                   (Phase 1)
    validator.py  syntax + schema grounding checks                 (Phase 3)
    repair.py     feed execution errors back to the model          (Phase 11)
"""
