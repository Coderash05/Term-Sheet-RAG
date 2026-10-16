# TermSheet AI: A Smart Legal Assistant for Startup Founders

A neuro-symbolic legal assistant combining Deep Learning (SBERT embeddings and LLM fact extraction) with Symbolic Knowledge Representation & Reasoning (NetworkX ontology and forward-chaining Horn-clause rules) to evaluate venture-capital term sheets with deterministic, verifiable proof trees.

---

## Prerequisites

- **Python**: Version `3.10` or higher (`3.11` recommended)
- **Git**

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Coderash05/Term-Sheet-RAG.git
cd Term-Sheet-RAG
```

### 2. Create and Activate a Virtual Environment

**On Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

You can install only what you need using optional dependency targets:

- **Core & Development (Linter + Tests):**

  ```bash
  pip install -e ".[dev]"
  ```
- **Full Installation (NLP, Document Parsing, Streamlit UI, Dev):**

  ```bash
  pip install -e ".[dev,nlp,parsing,ui]"
  ```

---

## Testing & Code Quality

Before pushing code or opening a pull request, run the automated checks locally:

### Run Linter & Formatter (`ruff`)

```bash
# Check code style and common bugs
ruff check .

# Automatically fix issues where possible
ruff check . --fix

# Verify code formatting
ruff format --check .
```

### Run Unit & Integration Tests (`pytest`)

```bash
pytest
```

---

## Interface Contracts & Schemas

The pipeline interface contracts are defined using Pydantic in [`src/termsheet_ai/schemas/contracts.py`](src/termsheet_ai/schemas/contracts.py).

To export or regenerate standalone JSON Schema files for annotators or external consumers into the `contracts/json_schemas/` directory:

```bash
python -m termsheet_ai.schemas.export_schemas
```

---

## Quickstart Pipeline Execution

Run the end-to-end pipeline on any sample text file:

```python
from pathlib import Path
from termsheet_ai.pipeline import TermSheetPipeline

pipeline = TermSheetPipeline()
payload = pipeline.run("path/to/term_sheet.txt")

# View document risk score and flags
if payload.audit_trace:
    print(f"Risk Score: {payload.audit_trace.risk_score.overall_score}/100")
    for flag in payload.audit_trace.flags:
        print(f"[{flag.severity.value.upper()}] {flag.title}: {flag.explanation}")
```

---

Branch Policy & Contributing

1. **Main Branch Protection**: Direct commits to `main` are restricted in production.
2. **Feature Branches**: Create a dedicated branch for each feature or bug fix:
   ```bash
   git checkout -b feature/clause-segmentation
   ```
3. **Pull Requests**: Open a PR targeting `main`. All CI checks (`ruff`, `pytest`) must pass before merging.
