# CS F407 / U407 Artificial Intelligence
## TermSheet AI: A Smart Legal Assistant for Startup Founders

---

### Project Overview & Metadata

- **Project Title:** TermSheet AI: A Smart Legal Assistant for Startup Founders
- **Team Name:** Secret Agents
- **Declared Track:** Application
- **Topic Area:** Knowledge Representation & Reasoning (KR&R) + Applications of Deep Learning (combined project)
- **Team Size:** 7 Members *(Personal information such as names, student ID numbers, and email addresses omitted per privacy guidelines)*

---

### Short Project Description

First-time founders often receive venture-capital term sheets full of dense legal terms and must respond within days, without access to affordable startup counsel. Terms such as a > 1× liquidation multiple, participating preferred with no cap, full-ratchet anti-dilution, or investor-controlled boards can quietly shift economics and control away from the founders.

**TermSheet AI** is a web application where a founder uploads a term sheet (PDF/DOCX) and receives:
1. A plain-English breakdown of every clause,
2. Risk flags colour-coded by severity, and
3. A step-by-step audit trace showing which explicit rule, and which market-norm source, produced each flag.

A deep-learning front end (Sentence-BERT embeddings and an instruction-tuned language model) turns messy text into structured clause facts. A symbolic KR&R back end (a knowledge graph of market-standard terms plus a forward-chaining rule engine) makes every verdict deterministic and explainable, rather than a free-form language-model opinion. An optional module looks up investing entities in public SEC EDGAR Form D records.

---

### Table of Contents

1. [Problem & Motivation](#1-problem-statement--motivation)
2. [Connection to Course Concepts](#2-connection-to-course-concepts)
3. [Proposed Approach & Methodology](#3-proposed-approach--methodology)
4. [Current Work in the Area](#4-current-work-in-the-area)
5. [Feasibility, Timeline & Roles](#5-feasibility-timeline--roles)
6. [References](#references)

---

### 1. Problem Statement & Motivation

#### Problem
A priced-round term sheet is mostly non-binding, but it is the blueprint for every number and right in the definitive documents that follow, so mistakes made at this stage are expensive to undo later. First-time founders negotiate it under time pressure and typically cannot afford specialist counsel. Four provisions drive most of the long-term damage:
- **Liquidation preference** (multiple and participation)
- **Anti-dilution** (broad-based weighted average vs. full ratchet)
- **Board and protective-provision control**
- **Founder vesting**

Empirical work on real VC contracts shows these terms are systematically negotiated and allocate cash-flow and control rights between investors and founders [4]. Founders need to know which term is unusual, by how much, and why it matters.

#### Why Generic AI is Not Enough
General-purpose chatbots summarise well but give no verifiable audit trail. Even commercial retrieval-augmented legal research tools were found to hallucinate between 17% and 33% of the time in a pre-registered study [12]. For a contract, a plausible but wrong statement (e.g., misreading "1× non-participating" as "participating") is worse than no answer. Our design therefore confines the neural model to perception tasks (segmenting, classifying, extracting values with source spans, paraphrasing) and delegates every risk judgement to an explicit, inspectable rule base.

#### Scope and Non-Goals
- **In Scope:** Equity-financing term sheets (priced-round preferred stock; YC post-money SAFE as a secondary format).
- **Out of Scope (Non-goals):** Legal advice, negotiating on the founder's behalf, debt financing, M&A, or jurisdiction-specific statutory analysis.
- **Norms:** The rule base encodes NVCA / US market norms; Indian-specific norms are treated as a stretch goal (Section 5).

---

### 2. Connection to Course Concepts

| Technique | How it is used in TermSheet AI | Module |
| :--- | :--- | :--- |
| **Knowledge representation: ontology / semantic network (frames)** | Typed graph (NetworkX) with nodes `ClauseFamily`, `Attribute`, `MarketNorm`, `Rule`, `SourceDoc`; edges such as `hasAttribute`, `deviatesFrom`, `citedIn`. Each norm cites its section in the NVCA model term sheet [2] or a market-practice study. | KR&R |
| **First-order / definite-clause (Horn) rules** | Risk logic written as rules over predicates, e.g., $\text{Liq}(c)$, $\text{Multiple}(c, m)$, $\text{Participating}(c, \text{cap})$, $\text{BoardSeats}(c, i, f, x)$, with auxiliary derived predicates. | KR&R [1] |
| **Forward chaining (data-driven inference)** | Own implementation (not a black-box library) with working memory, an agenda, and justification records: each derived fact stores the rule and premises that produced it. Single-pass matching is sufficient for our rule-set size, so a full Rete network [7] is skipped and noted as a possible optimisation. | KR&R [1] |
| **Explanation by proof tree** | The recorded justifications are unrolled into a rule-trace tree shown in the UI for every flag. | KR&R |
| **Transformer sentence embeddings (Sentence-BERT)** | Clause text $\rightarrow$ 384-d vector (`all-MiniLM-L6-v2`); cosine-similarity $k$-NN against an exemplar bank, plus a logistic-regression head. Compared with a TF–IDF baseline. | Deep Learning [5] |
| **Instruction-tuned LLM, retrieval-augmented generation** | (a) JSON-schema attribute extraction at temperature 0 with span validation; (b) plain-English summaries generated only from retrieved clause text and KB facts [6]. | Deep Learning |

#### Example of Chained Rules (Two Inference Steps)
*(R- IDs are rule identifiers stored in the knowledge base)*

- `R-LP-01`: $\text{Liq}(c) \land \text{Multiple}(c, m) \land m > 1 \implies \text{Flag}(c, \text{High})$
- `R-CT-02`: $\text{BoardSeats}(c, i, f, x) \land i > f + x \implies \text{InvestorControl}(c)$
- `R-CT-05`: $\text{InvestorControl}(c) \land \text{VestingReset}(c) \implies \text{Flag}(c, \text{High}, \text{"founder removable with loss of equity"})$

*Note:* The third rule fires only after the second has derived `InvestorControl`, demonstrating multi-step reasoning that a single-threshold checker or a plain LLM does not provide.

---

### 3. Proposed Approach & Methodology

#### 3.1 Data and Environment

- **Knowledge Sources (Rule Base):**
  - NVCA Model Legal Documents, including the Model Term Sheet and the Enhanced Model Term Sheet with embedded deal-term frequency data [2].
  - Y Combinator SAFE documents [3].
  - Empirical studies of VC contract terms [4].
  - Every rule stores a source field; thresholds are configurable and reviewed with a faculty mentor or a practising startup lawyer if one can be reached.

- **Classifier Training Data:**
  - Clause exemplars from NVCA/YC documents for $\approx 12$ clause families: liquidation preference, anti-dilution, board, protective provisions, vesting, information rights, ROFR/co-sale, drag-along, option pool, exclusivity/no-shop, conversion, redemption, plus an "other" class.
  - LEDGAR provisions [8] supply optional out-of-domain negatives.

- **Evaluation Set (50 Term Sheets):**
  - $\approx 20$ real or public templates and anonymised founder-shared sheets (provenance and licence logged in a manifest).
  - $\approx 30$ synthetic sheets generated by sampling term values over the ontology and paraphrasing surface text, including at least 15 adversarial cases (e.g., $3\times$ participating preferred, full ratchet, investor-majority board). Openly disclosed that part of the corpus is synthetic.

- **Labelling Protocol (Avoids Circularity):**
  - Gold clause spans, attribute values, and expected flags are labelled by two annotators who do not write rules (annotator roles assigned to project members); disagreements are reconciled and Cohen’s $\kappa$ is reported.
  - Split: **20 dev / 30 test**. All tuning (thresholds, prompts, rules) uses dev only; the test set is evaluated once at the conclusion.

- **Investor Lookup (Optional Module):**
  - SEC EDGAR JSON APIs and Form D data sets [14, 15].
  - Client sends a declared User-Agent, stays under documented limit of 10 requests/sec, and caches responses.
  - Output: `"record found / not found / ambiguous match"` with links to filings. Absence is never reported as a red flag (many funds do not file Form D).
  - Indian MCA records lack an open reliable API and serve as a manual-link stretch goal.

- **Technology Stack:**
  - Language: Python 3.11
  - Document & Text Extraction: PyMuPDF / pdfplumber, python-docx, Tesseract OCR fallback
  - Embeddings & Vector Search: sentence-transformers (`all-MiniLM-L6-v2`), ChromaDB
  - Knowledge Representation & Logic: NetworkX, YAML rule files, custom forward-chaining Horn-clause engine
  - LLM Integration: Instruction-tuned LLM (local via Ollama, or hosted API behind a unified interface)
  - Frontend: Streamlit
  - Testing & CI: pytest, ruff, GitHub Actions CI

#### 3.2 System Pipeline

```
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│      S1: Ingest     │      │     S2: Classify    │      │     S3: Extract     │
│  PDF/DOCX, OCR,     │ ───► │  SBERT +            │ ───► │  LLM → JSON facts   │
│  clause segmentation│      │  Chroma k-NN        │      │  + source spans     │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘
     [Deep Learning]              [Deep Learning]              [Deep Learning]
                                                                      │
                                                                      ▼
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│      Dashboard      │      │     S5: Explain     │      │      S4: Reason     │
│  Severity colours,  │ ◄─── │  RAG summary        │ ◄─── │  KG match, forward  │
│  rule-trace tree,   │      │  + trace            │      │  chaining, score    │
│  investor panel     │      └─────────────────────┘      └─────────────────────┘
└─────────────────────┘          [Deep Learning]               [Symbolic KR&R]
```

- **S1 (Ingest):** Extracts text and splits it into clauses using numbering/heading patterns, falling back to sentence windows.
- **S2 (Classify):** Embeds each clause and assigns a family. If confidence is below threshold, the clause is marked `"unclassified"` and shown to the user rather than guessed.
- **S3 (Extract):** Extracts attributes (multiple, participation, cap, vesting years, cliff, board seats) as JSON. A validator requires every value to be backed by a source span (numbers must literally appear in the span), otherwise marked `unknown`.
- **S4 (Reason):** Asserts validated attributes as facts, forward-chains over rules, and computes a deterministic severity-weighted risk score (weights calibrated on dev data).
- **S5 (Explain):** Writes plain-English text from retrieved clause text and KB rationale only; an automatic consistency check confirms every number in a summary matches the fact base. The UI always displays the original clause beside the summary.

---

### 4. Current Work in the Area

- **Datasets and Models for Contract NLP:**
  - LEDGAR labels $\approx 100{,}000$ provisions from SEC filings [8].
  - CUAD provides expert-annotated clause extraction for commercial contracts [9].
  - Legal-BERT demonstrates domain pre-training value [10].
  - LegalBench benchmarks LLM legal reasoning [11].
  - *Gap:* These focus on commercial/general contracts and classification/extraction; none encode VC term-sheet norms or perform logical reasoning over them.
- **Reliability:**
  - Hallucination rates in leading commercial legal AI tools range between 17% and 33% [12], necessitating verifiable reasoning.
- **Neuro-symbolic Reasoning:**
  - Logic-LM translates problems into symbolic form and uses external solvers to improve faithfulness [13]; TermSheet AI applies this principle to startup financing contracts.
- **Free VC Resources:**
  - NVCA model documents [2] and YC SAFEs [3] offer free standardized templates, but do not analyze or audit a founder's incoming offer.
- **Commercial Tools:**
  - Harvey [16] and Ironclad [17] target enterprise legal departments and law firms and remain largely inaccessible/expensive for early-stage founders.
- **Identified Gap:**
  - A free, founder-facing tool combining neural clause comprehension with an explicit, source-cited VC rule base and a verifiable proof trace.
- **Accepted Limitations:**
  - NVCA/US norms only.
  - Manual rule curation.
  - "Market standard" represents an evolving, negotiated concept.

---

### 5. Feasibility, Timeline & Roles

#### Feasibility & Scope Tiers
Every component relies on mature open-source tooling. The architecture is developed as a CLI first and a UI second, ensuring an end-to-end working system by Week 3.
- **MVP Scope:** S1–S5 covering $\approx 12$ clause families, $\approx 25$ rules, Streamlit UI, 50-document evaluation. MVP is frozen at the end of Week 5.
- **Stretch Scope:** SEC EDGAR investor lookup panel, Indian VC norms pack, SAFE support (dropped first if time constraints arise).
- **Week 6 Focus:** Demo and viva preparation.

#### 5.1 Week-by-Week Plan (Starting Sat 11 Oct 2026)

- **Week 1 (11–17 Oct):**
  - Set up repository, CI pipeline (`ruff`, `pytest`), and branch policy.
  - Collect NVCA/YC documents and $\ge 10$ public term sheets; create corpus manifest.
  - Draft clause families, attribute schema, and annotation guide (v0).
  - Build PDF/DOCX text extraction and clause segmentation.
- **Week 2 (18–24 Oct):**
  - Build ontology v1 and author first 15 rules in YAML.
  - Implement forward-chaining engine v0 with justification records.
  - Create exemplar bank and TF–IDF baseline classifier.
  - Train SBERT + ChromaDB $k$-NN clause classifier (v0).
  - Develop LLM extraction prompts, JSON schema, and span validator.
- **Week 3 (25–31 Oct):**
  - End-to-end integration v1: document $\rightarrow$ clauses $\rightarrow$ facts $\rightarrow$ flags $\rightarrow$ risk score.
  - Expand rule base to $\approx 25$ rules.
  - Build synthetic term-sheet generator (v0).
  - Create Streamlit UI skeleton with mock trace data.
  - Mid-project check-in with instructor.
- **Week 4 (1–7 Nov):**
  - Build full UI: upload flow, severity-coloured clause table, and rule-trace tree.
  - Add RAG-based plain-English summaries with number-consistency checker.
  - Complete gold labels for all 50 evaluation documents.
  - Build investor-lookup panel using SEC EDGAR / Form D APIs.
- **Week 5 (8–14 Nov):**
  - Run evaluation on dev set; perform error analysis.
  - Tune thresholds, prompts, and rules on dev data only.
  - Run held-out test set (30 documents) once.
  - Run LLM-only baseline for comparison.
  - Test robustness: scanned PDFs, missing clauses, non-term-sheet inputs.
  - Deploy demo (Docker or Streamlit Cloud).
  - MVP scope freeze.
- **Week 6 (15–21 Nov):**
  - Finalise report (limitations, AI-usage disclosure, individual contributions).
  - Generate all evaluation tables and plots via `make eval`.
  - Record demo video; rehearse viva (each member explains their module).
  - Tag release `v1.0`.

#### Main Risks & Mitigations
1. **Few real term sheets:** Mix of public and synthetic sheets, openly reported.
2. **LLM extraction errors:** Span validator and `"unknown"` fallback; regex backup for numeric terms.
3. **Evaluation circularity:** Annotators are separate from rule authors; strict dev/test split.
4. **Scope creep:** MVP freeze at end of Week 5; stretch items dropped first.
5. **Compute / API limits:** Small local models by default; aggressive caching of all LLM and EDGAR calls.

#### 5.2 Role Assignment (7 Roles)

| Role # | Role Title | Responsibilities and Deliverables | Pairing Partner |
| :---: | :--- | :--- | :---: |
| **1** | Team Lead, System Architect | Repo, CI, interface contracts (JSON schemas), end-to-end pipeline, deployment, report assembly, weekly stand-up | Member 6 |
| **2** | Data & Preprocessing | Corpus manifest, PDF/DOCX/OCR extraction, clause segmentation, synthetic generator, EDGAR/Form D client, Annotator A | Member 7 |
| **3** | Deep Learning: Embeddings & Search | SBERT pipeline, ChromaDB index, exemplar bank, $k$-NN/LR classifier, TF–IDF baseline, classification metrics | Member 4 |
| **4** | Deep Learning: Extraction & Summarisation | LLM attribute extraction, span validator, RAG summaries, number checker, LLM-only baseline | Member 3 |
| **5** | KR&R: Ontology & Rule Base | Ontology, YAML rules with sources, severity weights, norm calibration, rule unit tests | Member 6 |
| **6** | KR&R: Inference & Explainability | Forward-chaining engine, justification records, trace JSON/proof tree, risk scoring, grounding test | Member 5 |
| **7** | Full-Stack UI & Evaluation | Streamlit dashboard, trace visualisation, evaluation harness (`make eval`), plots, Annotator B, demo video | Member 2 |

#### Process and Accountability
- Progress log kept at `docs/LOG.md`.
- AI-usage log maintained at `docs/AI_USAGE.md` recording tool, purpose, and verification method for report disclosure.
- Peer code review in designated pairs ensures all members can explain the entire pipeline during the viva.

---

### References

- **[1]** S. Russell and P. Norvig. *Artificial Intelligence: A Modern Approach*, 4th ed. Pearson, 2020 (knowledge representation, first-order logic, inference by forward chaining).
- **[2]** National Venture Capital Association. *Model Legal Documents* (incl. Model Term Sheet and Enhanced Model Term Sheet). [https://nvca.org/resources/model-legal-documents/](https://nvca.org/resources/model-legal-documents/)
- **[3]** Y Combinator. *Documents: SAFE (Simple Agreement for Future Equity)*. [https://www.ycombinator.com/documents](https://www.ycombinator.com/documents)
- **[4]** S. N. Kaplan and P. Strömberg. *Financial contracting theory meets the real world: An empirical analysis of venture capital contracts*. Review of Economic Studies, 70(2):281–315, 2003.
- **[5]** N. Reimers and I. Gurevych. *Sentence-BERT: Sentence embeddings using Siamese BERT-networks*. In Proc. EMNLP-IJCNLP, pp. 3982–3992, 2019. [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)
- **[6]** P. Lewis et al. *Retrieval-augmented generation for knowledge-intensive NLP tasks*. In NeurIPS, 2020. [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
- **[7]** C. L. Forgy. *Rete: A fast algorithm for the many pattern/many object pattern match problem*. Artificial Intelligence, 19(1):17–37, 1982.
- **[8]** D. Tuggener, P. von Däniken, T. Peetz, and M. Cieliebak. *LEDGAR: A large-scale multi-label corpus for text classification of legal provisions in contracts*. In Proc. LREC, pp. 1235–1241, 2020. [https://aclanthology.org/2020.lrec-1.155](https://aclanthology.org/2020.lrec-1.155)
- **[9]** D. Hendrycks, C. Burns, A. Chen, and S. Ball. *CUAD: An expert-annotated NLP dataset for legal contract review*. In NeurIPS Datasets and Benchmarks, 2021. [https://arxiv.org/abs/2103.06268](https://arxiv.org/abs/2103.06268)
- **[10]** I. Chalkidis et al. *LEGAL-BERT: The muppets straight out of law school*. In Findings of EMNLP, 2020. [https://arxiv.org/abs/2010.02559](https://arxiv.org/abs/2010.02559)
- **[11]** N. Guha et al. *LegalBench: A collaboratively built benchmark for measuring legal reasoning in large language models*. In NeurIPS Datasets and Benchmarks, 2023. [https://arxiv.org/abs/2308.11462](https://arxiv.org/abs/2308.11462)
- **[12]** V. Magesh, F. Surani, M. Dahl, M. Suzgun, C. D. Manning, and D. E. Ho. *Hallucination-free? Assessing the reliability of leading AI legal research tools*. Journal of Empirical Legal Studies, 22(2), 2025. [https://arxiv.org/abs/2405.20362](https://arxiv.org/abs/2405.20362)
- **[13]** L. Pan, A. Albalak, X. Wang, and W. Y. Wang. *Logic-LM: Empowering large language models with symbolic solvers for faithful logical reasoning*. In Findings of EMNLP, 2023. [https://arxiv.org/abs/2305.12295](https://arxiv.org/abs/2305.12295)
- **[14]** U.S. Securities and Exchange Commission. *Accessing EDGAR Data (fair-access policy and APIs)*. [https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm](https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm)
- **[15]** U.S. SEC, DERA. *Form D Data Sets*. [https://www.sec.gov/dera/data/form-d](https://www.sec.gov/dera/data/form-d)
- **[16]** Harvey AI. [https://www.harvey.ai](https://www.harvey.ai)
- **[17]** Ironclad. [https://ironcladapp.com](https://ironcladapp.com)
