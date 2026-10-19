# RAG Assignment 1: Baseline vs Re-Ranking

**Document:** MCC Laws of Cricket (2017 Code), 168 pages

This assignment builds a two-stage Retrieval-Augmented Generation (RAG) pipeline and compares it against a baseline. The key question is: does adding a Cross-Encoder re-ranking step improve the quality of retrieved evidence before passing it to a language model?

---

## What is in this repository

| File | Description |
|---|---|
| `RAG_Baseline_vs_Reranking.ipynb` | Main notebook — runs both pipelines end to end |
| `report.md` | Written report covering architecture, results, and analysis |
| `requirements.txt` | Python packages needed to run the notebook |
| `results/` | Generated charts and CSV files (latency, rank shifts) |

---

## Models used

| Component | Model |
|---|---|
| Bi-Encoder (retrieval) | sentence-transformers/all-MiniLM-L6-v2 |
| Cross-Encoder (re-ranking) | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM (answer generation) | google/gemma-3-1b-it |

---

## How to install

```bash
pip install -r requirements.txt
```

The notebook will download model weights automatically on first run. An internet connection is required.

---

## How to run

Open the notebook and run all cells from top to bottom:

```bash
jupyter notebook RAG_Baseline_vs_Reranking.ipynb
```

Or run it non-interactively (recommended for a clean full run):

```bash
jupyter nbconvert --to notebook --execute --inplace RAG_Baseline_vs_Reranking.ipynb
```

**Note:** Running on CPU takes approximately 5–10 minutes. Generation alone takes about 20 seconds per query.

---

## Key results

- Recall@10: 10/11 = 0.91
- Hit@3 baseline (Pipeline A): 0.73
- Hit@3 re-ranked (Pipeline B): 0.82
- MRR@3 baseline: 0.56
- MRR@3 re-ranked: 0.82
- Re-ranking overhead: 125 ms per query (0.6% of total time)

Two genuine retrieval rescues were found (Q09, Q11). One regression was found (Q04).
