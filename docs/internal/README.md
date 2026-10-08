# Internal maintainer notes

Point-in-time analysis and audit documents. These are **not** user-facing docs —
the [user documentation](../README.md) is one level up. Keep them here (not in
`docs/`) so the user-docs folder stays focused on explaining how the system works.

- [`BACKLOG.md`](./BACKLOG.md) — detailed engineering backlog (P0–P3 + per-item write-ups).
- [`BUG_REPORT.md`](./BUG_REPORT.md) — 2026-07-08 full-repo bug & design-finding report.
- [`Todo_Reranker.md`](./Todo_Reranker.md) — TEI reranker implementation plan.
- [`RAG_ANALYSIS_2026-09-27.md`](./RAG_ANALYSIS_2026-09-27.md) — German-query retrieval incident: root causes, probe data, fixes. Cited from `kb/` comments as the rationale for specific tuning choices.
- [`CODE_ANALYSIS_2026-09-27.md`](./CODE_ANALYSIS_2026-09-27.md) — one-time code audit (bugs B1–B4, backlog improvements I1–I3). Actionable items live in [`./BACKLOG.md`](./BACKLOG.md).
