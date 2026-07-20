# Multi-head RoBERTa — results

Fine-tuned **multi-head RoBERTa**: one shared `roberta-base` encoder with one token-classification
head per dataset, each over that dataset's **original** label set (no label unification). Recipe:
batch 2, lr 2e-5, 10 epochs, linear schedule, warmup 0, seed 42, max-length 512, bf16. Cross-dataset
batching reproduces the BiLSTM multi-head scheme (per-epoch dataset sampling **with replacement**,
proportional to each dataset's batch count). Union-leakage removal applied (train sentences whose
tokens appear in any of the four valid sets dropped). Scores are **span-F1 on each dataset's
original-label dev set**. Source: `models/multihead_roberta/dev_metrics.json`.

Only the **Shared: Both** (fully-shared encoder) variant maps cleanly to a monolithic transformer, so
the emb-only / LSTM-only columns of the paper's BiLSTM table have no RoBERTa analog.

## Multi-head span-F1 vs. the BiLSTM multi-head (paper Table `tab:multi_head_matrix`)

BiLSTM columns are reproduced from the paper for comparison; the RoBERTa column is this run.

| Dataset  | BiLSTM Shared: LSTM | BiLSTM Shared: EMB | BiLSTM Shared: Both | BiLSTM Reference | **RoBERTa multi-head (Shared: Both)** |
|----------|:---:|:---:|:---:|:---:|:---:|
| DNRTI    | 0.43 | 0.52 | 0.52 | 0.45 | **0.61** |
| ATTACKER | 0.01 | 0.19 | 0.21 | 0.04 | **0.42** |
| APTNER   | 0.36 | 0.38 | 0.37 | 0.35 | **0.44** |
| CYNER    | 0.34 | 0.41 | 0.39 | 0.40 | **0.80** |

The fully-shared RoBERTa multi-head improves span-F1 over the best BiLSTM multi-head variant on every
dataset, with the largest gains on ATTACKER (0.21 → 0.42) and CYNER (0.41 → 0.80).

## Full metric breakdown — RoBERTa multi-head (Shared: Both)

Strict span-F1 with its precision/recall, plus the unlabelled and loose (partial-overlap, same-label)
span-F1 variants — analogous to the paper's cross-dataset metric tables.

| Dataset  | Precision | Recall | Span-F1 | Unlabelled F1 | Loose F1 |
|----------|:---:|:---:|:---:|:---:|:---:|
| DNRTI    | 0.63 | 0.59 | 0.61 | 0.68 | 0.69 |
| ATTACKER | 0.44 | 0.41 | 0.42 | 0.54 | 0.54 |
| APTNER   | 0.38 | 0.51 | 0.44 | 0.49 | 0.52 |
| CYNER    | 0.80 | 0.79 | 0.80 | 0.84 | 0.84 |

APTNER is the only dataset where recall (0.51) exceeds precision (0.38); the others are roughly
balanced. CYNER is strongest across every metric.
