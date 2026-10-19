# AI Usage Log

> **Purpose:** Transparent disclosure of all AI tools used during development, as required by the project report. Each entry records the tool, what it was used for, and how its output was verified.

---

## Usage Log

| Date | Member | AI Tool | Purpose | Files Affected | Verification Method |
|:--|:--|:--|:--|:--|:--|
| 2026-10-19 | M2 | Antigravity IDE (Claude) | Generate PDF/DOCX/TXT text extraction module (`extractor.py`) with PyMuPDF primary, pdfplumber fallback, and Tesseract OCR fallback | `src/termsheet_ai/ingest/extractor.py` | Code reviewed line-by-line; tested against real PDF (`Ai_project-2.pdf`) and synthetic TXT files; all 16 ingest tests pass |
| 2026-10-19 | M2 | Antigravity IDE (Claude) | Generate clause segmenter (`segmenter.py`) with heading/numbering pattern matching and sentence-window fallback | `src/termsheet_ai/ingest/segmenter.py` | Verified regex patterns against NVCA-style numbered sections; tested pattern vs. fallback paths; post-processing merge/split logic validated with edge cases |
| 2026-10-19 | M2 | Antigravity IDE (Claude) | Generate S1 Ingestor (`ingestor.py`) wiring extractor + segmenter into `BaseIngestor` contract | `src/termsheet_ai/ingest/ingestor.py` | Confirmed output matches `ClauseSpan` and `DocumentMetadata` schemas; deterministic `doc_id` hash verified; integration test with real PDF passes |
| 2026-10-19 | M2 | Antigravity IDE (Claude) | Generate test suite for ingest module | `tests/test_ingest.py` | All 16 tests pass; covers extraction (txt, empty, missing, unsupported), segmentation (pattern, fallback, merge, headers), and end-to-end ingestion; ruff lint clean |
| 2026-10-19 | M2 | Antigravity IDE (Claude) | Generate corpus manifest template | `corpus/manifest.json` | Reviewed structure against project spec; confirms NVCA/YC sources, 50-doc eval target (20 dev / 30 test), 13 clause families |
| 2026-10-19 | M2 | Antigravity IDE (Claude) | Extract project details from `Ai_project-2.pdf` into Markdown (excluding personal info) | `Ai_project-2.md` | Cross-checked against original PDF; confirmed no personal information (names, IDs, emails) leaked; all sections, tables, references preserved |

---

## Guidelines for Future Entries

Each time an AI tool is used during development, add a row with:

1. **Date** — When the tool was used
2. **Member** — Which team member used it (M1–M7)
3. **AI Tool** — Name and model (e.g., "ChatGPT-4o", "GitHub Copilot", "Claude")
4. **Purpose** — Specific task the tool assisted with (be precise, not "helped with code")
5. **Files Affected** — Which source files were created or modified
6. **Verification Method** — How the AI output was validated (tests, manual review, comparison to docs, etc.)

> [!IMPORTANT]
> AI-generated code must always be reviewed, tested, and understood by the team member before merging. The AI tool is an assistant, not an author.
