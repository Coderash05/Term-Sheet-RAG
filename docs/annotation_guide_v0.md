# TermSheet AI: Annotation Guide (v0)

## Overview
This guide is for Annotators (Members 2 and 7) to label our evaluation set of 50 term sheets[cite: 5]. You are creating the "gold standard" labels that our AI's extraction and reasoning modules will be tested against. To avoid circularity, annotators must not write the symbolic logic rules[cite: 5].

## 1. Clause Spans
*   For each document, identify the text boundaries for the core clause families defined in `clause_family_definitions_v0.md`.
*   Highlight the exact starting and ending text. If a clause is missing entirely, leave it unlabelled.

## 2. Attribute Extraction & The "Literal Span" Rule
*   You must extract the exact values defined in our JSON schema (e.g., liquidation multiple, participation, vesting months, board seats)[cite: 5].
*   **CRITICAL VALIDATOR RULE:** Every extracted numeric value *must* literally appear in the source text span you select[cite: 5]. 
*   If a value is not explicitly stated in the contract, label it as `null`. Do not guess or assume market standard if the text is silent.

## 3. Expected Risk Flags
*   Based on US market norms, label the expected risk flags for clauses that heavily favor investors.
*   Note the expected severity (e.g., HIGH). This will be used to verify that our forward-chaining rule engine calculates the correct deterministic risk score[cite: 4, 5].

## 4. Conflict Resolution
*   Both annotators will label the data independently.
*   Disagreements will be reconciled together so we can report our final Cohen's Kappa score (measuring human agreement)[cite: 5].
