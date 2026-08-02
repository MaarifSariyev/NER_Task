# Named Entity Recognition Pipeline

**Applied NLP / NER Technical Task — Project Documentation**

Author: Elnar Babayev

---

## 1. Project Overview

This project builds an English Named Entity Recognition (NER) system across a custom 8-label schema, following a four-stage methodology: (1) clean a small starter dataset, (2) expand and balance the training data, (3) train a first model and run quality assurance, and (4) diagnose weaknesses, relabel, and retrain.

The guiding philosophy throughout was **small but clean, honestly evaluated data over large but noisy data**. Every dataset went through structural validation, policy linting, and balance analysis before training. A key methodological decision was to build a real, hand-annotated benchmark to measure true generalization, rather than trusting metrics from synthetic-influenced test sets.

### The 8-Label Schema

| Label | Definition |
|---|---|
| PERSON | Names of real or fictional people |
| ORGANIZATION | Institutions, companies, agencies, unions, galleries, museums, teams |
| LOCATION | Countries, cities, regions, landmarks, address components |
| JOB | Occupations, professions, job titles, positions |
| PRODUCT | Products, software, hardware, AI products, named manufactured items |
| WORKOFART | Books, films, songs, paintings, and other named creative works |
| TIMEDATE | Dates, times, durations, relative temporal references |
| AMOUNT | Money, percentages, quantities, measurements |

---

## 2. Labeling Rules Per Label

Each label was governed by explicit annotation rules. These rules were encoded into an automated linter and applied consistently across all datasets and synthetic generation.

### General Rules (all labels)

- Offsets are the single source of truth. The surface value is always `text[start:end]`; no separate `value` field is stored, eliminating a whole class of drift bugs.
- The `end` offset is **exclusive** (Python slice convention), aligning with HuggingFace tokenizer offset mapping.
- Spans must not overlap (flat NER, single-label per token).
- Only the 8 schema labels are valid; the invalid `COMPANY` label is context-mapped to ORGANIZATION.

### PERSON

- Annotate only the person name; drop titles and honorifics (**Dr., Mr., President** → dropped).
- Keep the full name; never split (**Kim Jong-Un** stays as one entity).
- A job title next to a name is a **separate** JOB entity (**CEO Tim Cook** → CEO=JOB, Tim Cook=PERSON).

### JOB (most rule-intensive)

- Annotate the full multi-word title including seniority and specialization (**Senior Machine Learning Engineer**, **Chief Executive Officer**).
- Organization is a separate entity, never inside JOB (**Director at Microsoft** → Director=JOB, Microsoft=ORGANIZATION).
- No articles, prepositions, or honorifics inside the span (**the CEO** → CEO).
- Coordinated roles are separate JOB entities (**Doctors and nurses** → two JOB entities).
- Generic role words that describe a profession are JOB (**doctor, researcher, administrator**), but purely situational roles are NOT (**winner, victim, customer, employer** → not labeled).

### ORGANIZATION

- Keep the official name including a leading "The" when official (**The World Bank**).
- Departments, unions, galleries, and museums are organizations (**Public and Commercial Services Union**, **Boesky Gallery**, **Inhotim** museum).

### PRODUCT / WORKOFART / TIMEDATE / AMOUNT

- **PRODUCT**: keep the complete product name including brand and model (**iPhone 15 Pro Max**); AI products count (**ChatGPT, Gemini, Siri**). Brand splits off only when grammatically separate (**Dell's Alienware 18** → Dell=ORG, Alienware 18=PRODUCT).
- **WORKOFART**: keep the official title including a leading "The" (**The Origin of the World**); film franchise prefixes stay (**Avengers: Doomsday**).
- **TIMEDATE**: keep articles/demonstratives (**this month, the following day**); drop approximation modifiers (**about 3 hours** → 3 hours) and leading prepositions.
- **AMOUNT**: money, percent, quantity, and measurements; drop approximators (**nearly $9bn** → $9bn).

---

## 3. Datasets Used

Because no single public dataset covers all 8 labels — and none contains a JOB-title label — the training data was assembled from multiple sources, each converted to our schema and used for the labels it covers well:

| Dataset | Role / Labels Contributed | Notes |
|---|---|---|
| Starter (100 rows) | All 8 labels; hand-corrected gold | Cleaned in Stage 1; 763 spans, 43 rows fixed |
| MultiNERD | PERSON, ORGANIZATION, LOCATION, WORKOFART, TIMEDATE | Strong PER/LOC; weak TIMEDATE coverage; no JOB/PRODUCT/AMOUNT |
| OntoNotes 5.0 | AMOUNT (MONEY/PERCENT/QTY), PRODUCT, + others | Main source of AMOUNT and PRODUCT; no JOB |
| Synthetic JOB | JOB (+ context labels) | Generated to fill the JOB gap absent from all public sources |
| Synthetic multi-token | PRODUCT, JOB (multi-word) | Built to fix I-PRODUCT/I-JOB (multi-token spans) |
| Synthetic realistic | JOB, WORKOFART, PRODUCT | Longer real-style sentences to reduce template overfitting |
| Real benchmark | All 8 labels | Hand-annotated real news; split into train + held-out test |

**Label-to-source mapping** was handled by dedicated converters:

- **MultiNERD**: PER→PERSON, ORG→ORGANIZATION, LOC→LOCATION, MEDIA→WORKOFART, TIME→TIMEDATE (all other types dropped).
- **OntoNotes**: MONEY/PERCENT/QUANTITY→AMOUNT, PRODUCT→PRODUCT, plus PERSON/ORG/GPE/LOC/WORK_OF_ART/DATE/TIME mapped through; rows kept only if they contained AMOUNT or PRODUCT.

---

## 4. Core Analysis & Processing Scripts

The pipeline is built from a set of reusable Python scripts. The core ones essential to the methodology:

| Script | Purpose |
|---|---|
| `ner_lint.py` | Policy linter — validates every span against all labeling rules (offsets, boundaries, type-words, honorifics, label set, overlaps). Exits non-zero on structural errors. |
| `ner_analyze.py` | Comprehensive dataset analyzer — 8 sections: overview, structural integrity, policy checks, recall-gap scan, label balance, label×source, length/density, duplication, plus a PASS/FAIL verdict. |
| `recover_time_amount.py` | High-precision regex recovery of unlabeled TIMEDATE/AMOUNT (fixes partial-annotation gaps from MultiNERD/OntoNotes). |
| `balance_strict.py` | Aggressive quality gate + per-label quota with a hard cap — produces a small, balanced dataset; rare-labels-first selection stops PERSON/LOCATION from dominating. |
| `split_and_bio.py` | Train/test split (test = real rows only, all synthetic sources excluded) + char-offset → BIO token conversion for the model. |
| `train_ner.py` | Fine-tunes `xlm-roberta-base` with seqeval per-label metrics; logs all hyperparameters. |
| `eval_benchmark.py` | Evaluates the trained model on the real hand-annotated benchmark — the honest test. |

Supporting generators (`make_job_samples.py`, `make_multitoken_samples.py`, `make_realistic_jobs.py`) produced the synthetic data; `merge_dedup.py` combined sources and removed duplicates.

**Recommended pipeline order:**

```
convert sources → merge/dedup → recover TIMEDATE/AMOUNT → balance → split + BIO → train → evaluate on benchmark
```

---

## 5. Iterations: Problems Encountered and How They Were Solved

### Model 1 — Baseline

The first model reached **macro-F1 0.73** (accuracy 0.89) on the internal test set. Per-label analysis exposed severe weaknesses:

- **B-JOB F1 = 0.08** — synthetic→real transfer failure.
- **I-PRODUCT F1 = 0.00** — the model could not tag any multi-word product.
- **ORGANIZATION precision 0.62** — heavy over-prediction of ORG.

### Fix 1 — Multi-token synthetic samples

The root cause of the I-PRODUCT / I-JOB failure was too few multi-token examples. We generated synthetic samples deliberately rich in multi-word PRODUCT and JOB spans. Result on the internal test set: **I-PRODUCT 0.00 → 0.93, B-JOB 0.08 → 0.90, macro-F1 0.73 → 0.93**.

### Problem discovered — inflated metrics (test leakage)

Some scores were suspiciously perfect (PERSON 1.00, I-JOB 1.00). Investigation revealed that the train/test split only excluded one synthetic source; newer synthetic sources were leaking into the test set, so the model was being evaluated partly on template data it had memorized. The split script was fixed to exclude **ALL** synthetic sources from the test set.

### The decisive step — a real benchmark

Even with a clean split, the internal test set contained almost no real JOB examples. We hand-annotated a 44-sentence benchmark from real news text, resolving every ambiguous case by explicit rule decisions (galleries → ORGANIZATION, museums → ORGANIZATION, 'employers' → not JOB, ages → not TIMEDATE, film titles → WORKOFART). Evaluating Model 2 on this real benchmark told the true story:

| Metric | Synthetic-influenced test | Real benchmark |
|---|---|---|
| Overall F1 | 0.93 | 0.66 |
| JOB F1 | 0.90 | 0.22 |
| WORKOFART F1 | 0.94 | 0.47 |
| AMOUNT F1 | 0.92 | 0.56 |

This confirmed the classic risk of synthetic data: the model had memorized templates. Real news uses very different structures — JOB as a sentence subject ("Data scientists use..."), coordinated lists, and ORG+title+PERSON without an 'at' connector — none of which the template-based synthetic data contained.

### Fix 2 — Real data + real-structure synthetic

Two combined actions: (1) half the real benchmark rows were moved into training so the model finally saw real job/work structures; (2) a new synthetic generator (`make_realistic_jobs.py`) mimicked the real structures observed in the benchmark — subject+verb, definition, coordinated lists, 'the JOB of', ORG+title+PERSON, and mid-sentence titles.

---

## 6. Final Results

The final model was evaluated on the held-out real benchmark — the honest measure of real-world performance.

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| PERSON | 1.000 | 1.000 | 1.000 | 29 |
| ORGANIZATION | 0.720 | 0.783 | 0.750 | 23 |
| LOCATION | 0.750 | 0.900 | 0.818 | 10 |
| JOB | 0.867 | 0.684 | 0.765 | 19 |
| PRODUCT | 0.818 | 1.000 | 0.900 | 9 |
| WORKOFART | 0.923 | 1.000 | 0.960 | 12 |
| TIMEDATE | 0.667 | 0.857 | 0.750 | 14 |
| AMOUNT | 0.625 | 0.833 | 0.714 | 12 |
| **OVERALL** | **0.806** | **0.875** | **0.839** | **128** |

The trajectory of JOB F1 on the real benchmark tells the whole methodology story:

| Stage | JOB F1 (real benchmark) | Overall F1 |
|---|---|---|
| Model 2 (synthetic-only JOB) | 0.22 | 0.66 |
| Final (real + real-structure synthetic) | 0.765 | 0.839 |

Adding real data and real-structure synthetic samples raised JOB F1 from **0.22 to 0.765** and overall real-world F1 from **0.66 to 0.84** — a decisive validation that the diagnosis (structural mismatch, not just data volume) was correct.

---

## 7. Remaining Weaknesses & Honest Limitations

- **JOB recall (0.684)** is still the softest spot: real JOB data is scarce (only ~28 real examples total), so the model leans on synthetic structure. More real, lexically diverse job mentions would help most.
- **AMOUNT (0.71) and TIMEDATE (0.75)** show boundary errors — e.g. attaching a year to a work title, or splitting multi-token measurements.
- **ORGANIZATION precision (0.72)** — the model still over-predicts ORG on capitalized non-entities (nationalities, movement names).
- **Benchmark size (44 sentences)** is small; several labels have <15 support, so their F1 has wide confidence intervals. A larger real benchmark would give more reliable numbers.
- **Synthetic dependence**: JOB is ~98% synthetic in training. The realistic generator narrowed the gap, but this remains the primary structural risk.
- **Language**: the data and model are English-only. The choice of `xlm-roberta-base` leaves the door open for future Azerbaijani extension, but performance on Azerbaijani is untested and expected to be weak until Azerbaijani data is added.

---

## 8. Summary

Starting from a messy 100-row starter set, the project produced a balanced 8-label NER dataset and a fine-tuned model reaching **0.84 F1 on real, hand-annotated news text**. The central lesson — surfaced by building an honest benchmark — was that **synthetic data must match real-world structure, not just supply volume**. The iterative diagnose-fix-retrain loop, driven by per-label analysis and a real benchmark, turned a JOB F1 of 0.08 into 0.765 and a memorization-inflated 0.93 into a trustworthy 0.84.
