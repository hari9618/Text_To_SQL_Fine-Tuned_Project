-- =====================================================================
-- Enterprise Text-to-SQL — reference data
--
-- Small, stable lookup rows only: order statuses, product categories,
-- country codes, payment methods. Data that is part of the schema's
-- MEANING rather than part of the generated dataset.
--
-- Bulk synthetic data (customers, orders, order_items, ...) is NOT here.
-- A Phase 3 Python script generates it, because realistic distributions,
-- date spreads, and foreign-key consistency cannot be hand-authored.
--
-- Usage (from project root, AFTER schema.sql):
--   psql -U postgres -d enterprise_sales -f database/seed.sql
--
-- This file is IDEMPOTENT — re-running must not duplicate rows. Use
-- ON CONFLICT DO NOTHING against a natural key on every INSERT:
--
--   INSERT INTO categories (category_name) VALUES ('Electronics')
--   ON CONFLICT (category_name) DO NOTHING;
--
-- Status: Phase 3 not started. Depends on Phase 2 schema.
-- =====================================================================

BEGIN;

-- ---------------------------------------------------------------------
-- Insert order matters: parents before children, or foreign keys fail.
-- ---------------------------------------------------------------------

-- TODO(Phase 3): reference rows, once Phase 2 defines the tables.

COMMIT;
