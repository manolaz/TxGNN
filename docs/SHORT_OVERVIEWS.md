# TxGNN: Short Overview

A quick-read summary of what this repository and its underlying paper are about. For depth, see [CONTEXTS.md](CONTEXTS.md) (problem/motivation), [NETWORK_DESIGN.md](NETWORK_DESIGN.md) (architecture/math), [ALGORITHMS.md](ALGORITHMS.md) (algorithms + practical gotchas), [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) (code layout), and [IMPROVEMENTS_PLAN.md](IMPROVEMENTS_PLAN.md) (proposed future work).

## What the paper is

*"Zero-shot Prediction of Therapeutic Use with Geometric Deep Learning and Clinician Centered Design"* (Huang, Chandak, Wang, et al., Harvard/Stanford, medRxiv 2023). It introduces **TxGNN**, a graph neural network for **drug repurposing**: given a disease, predict which existing drugs could plausibly treat it (`indication`), should be avoided (`contraindication`), or could be used off-label (`off-label use`). The headline claim is **zero-shot generalization** — useful predictions for diseases that have few or no known treatments in the training data, which is the population that matters most for repurposing but that ordinary supervised models handle worst.

## What the repository is

The official `txgnn` Python package plus reproduction scripts ([reproduce/](../reproduce/)), a demo notebook ([TxGNN_Demo.ipynb](../TxGNN_Demo.ipynb)), and this `docs/` folder of deep-dive documentation (written for this workspace, not part of the original paper release). It operates on a precompiled biomedical knowledge graph (~17,080 diseases, ~7,957 drugs, plus proteins/phenotypes/exposures, connected by dozens of relation types) hosted on Harvard Dataverse.

## How the model works (one paragraph)

A **heterogeneous RGCN** (optionally attention-based) encodes every node type (drug, disease, gene/protein, effect/phenotype, exposure) into a shared embedding space, message-passing separately per relation type. Training is two-phase: **pretrain** on link prediction across *all* KG relations (so the encoder learns broad biology, not just therapeutic labels), then **fine-tune** on just the six drug-disease edge types. For diseases with weak KG connectivity, a **prototype/metric-learning module** finds the Top-K most similar diseases (via neighborhood-overlap profiles, PPI random walks, or BERT semantic similarity) and fuses their embeddings into the query disease's representation before scoring. Scoring itself uses a **DistMult** bilinear decoder per relation. Post-hoc, **GraphMask** learns sparse edge gates (via Lagrangian-constrained optimization) to explain which KG edges drove a given prediction, without retraining the model.

## How it's evaluated

Beyond standard edge-level AUROC/AUPRC, the repo implements **disease-centric splits** (`complex_disease`, disease-area ontology splits, `few_edeges_to_kg`/`few_edeges_to_indications`) that hold out *entire diseases* rather than random edges, specifically to test extrapolation rather than interpolation. **`TxEval`** then evaluates per-disease ranked drug lists with Recall@k, Enrichment vs. random, MRR, and AP — because a clinician only ever looks at the top of a ranked list, not a pooled classification score.

## What we've found while documenting this codebase

- The core "pretrain broadly, fine-tune narrowly, borrow from similar diseases when data is sparse" idea is implemented faithfully and is well-motivated (see [CONTEXTS.md](CONTEXTS.md) §1–3), but several pieces are genuinely **open empirical questions rather than settled facts** — see [CONTEXTS.md](CONTEXTS.md) §4's five research questions (does the prototype module actually help on truly cold diseases? which similarity measure transfers signal? how much does the split protocol itself inflate the zero-shot claim? is GraphMask's explanation faithful *and* sparse? does optimizing AUROC/AUPRC actually produce good top-K precision?).
- On that last point: the repo's own baseline numbers ([IMPROVEMENTS_PLAN.md](IMPROVEMENTS_PLAN.md)) show a real gap — **AUPRC ≈ 0.87** but **Precision@10 ≈ 0.14** and **Precision@100 ≈ 0.025** — meaning strong pooled metrics do not automatically translate into a clinically useful top-of-list ranking. [IMPROVEMENTS_PLAN.md](IMPROVEMENTS_PLAN.md) proposes decoder, negative-sampling, loss, and feature-initialization changes to close that gap.
- The shipped code has some **rough edges worth knowing before relying on it as-is** (cataloged in [ALGORITHMS.md §15](ALGORITHMS.md#15-practical-experience--known-gotchas-when-actually-running-these-algorithms)): `sim_measure='bert'` hardcodes the original authors' cluster paths and won't run for anyone else without source edits; the demo notebook has a couple of copy-paste bugs (duplicate `epochs_per_layer` kwarg, a `RxEval`/`TxEval` typo); `retrieve_save_gates()` and `retrieve_gates_scores_penalties()` have a signature mismatch; disease-area splits require PyTorch Geometric while the other splits don't; and `finetune()` loads the entire KG onto the GPU with no minibatch fallback, unlike `pretrain()`.
- Node features are **structure-only** (Xavier-random, no chemistry/sequence/text) — the model currently learns everything from graph topology alone, which is both a simplicity strength and the main lever [IMPROVEMENTS_PLAN.md](IMPROVEMENTS_PLAN.md) proposes to pull (ChemBERTa/PubMedBERT/ESM-2 initializations).

## The knowledge graph at a glance

| Aspect | Detail |
| :--- | :--- |
| Node types | `drug`, `disease`, `gene/protein`, `effect/phenotype`, `exposure` |
| Core therapeutic relations | `indication`, `contraindication`, `off-label use` (plus auto-generated `rev_*` reverses for bidirectional message passing) |
| Other relation types | dozens more (e.g. `protein_protein`, `disease_disease`, `disease_phenotype_positive`, `rev_disease_protein`) that supply the broad biological structure used during pretraining and disease-similarity profiling |
| Scale | ~17,080 diseases, ~7,957 drugs, plus proteins/phenotypes/exposures, millions of edges total |
| Source files | `kg.csv` / `node.csv` / `edges.csv`, auto-downloaded from Harvard Dataverse by `TxData.__init__` ([TxData.py](../txgnn/TxData.py)) |
| Disease taxonomy | Human Disease Ontology ([HumanDO.obo](../txgnn/data_splits/HumanDO.obo)) + MONDO cross-references, used to build the nine clinical disease-area splits |

## Quickstart (the standard 5-step workflow)

```python
from txgnn import TxData, TxGNN, TxEval

data = TxData(data_folder_path='./data')
data.prepare_split(split='complex_disease', seed=42)          # 1. load KG + build a split

model = TxGNN(data=data, device='cuda:0')
model.model_initialize(n_hid=100, n_inp=100, n_out=100, proto=True)  # 2. instantiate

model.pretrain(n_epoch=2)                                      # 3. pretrain on full KG (optional but recommended)
model.finetune(n_epoch=500, valid_per_n=20)                     # 4. fine-tune on drug-disease edges

evaluator = TxEval(model=model)
result = evaluator.eval_disease_centric(disease_idxs='test_set')  # 5. evaluate per-disease rankings
```
See [README.md](../README.md) for the full API (saving/loading, GraphMask training/explanation retrieval) and [ALGORITHMS.md §15](ALGORITHMS.md#15-practical-experience--known-gotchas-when-actually-running-these-algorithms) before running this for real (the demo's `n_epoch=30` is a smoke test, not a usable model — use `n_epoch=500`).

## Available data splits (`TxData.prepare_split(split=...)`)

| Split | What it tests |
| :--- | :--- |
| `random` | Baseline; mostly interpolation |
| `complex_disease` | The paper's headline zero-shot split — entire diseases held out |
| `complex_disease_cv` | 20-fold cross-validated version of the above (`seed` must be 1–20) |
| `disease_eval` | Case-study: mask one specific disease (`disease_eval_idx`) |
| `few_edeges_to_kg` / `few_edeges_to_indications` | Diseases with sparse KG connectivity / sparse known drugs, respectively |
| `cardiovascular`, `autoimmune`, `diabetes`, ... (9 disease areas) | Ontology-driven clinical-area splits (require PyTorch Geometric) |
| `full_graph` | No held-out test set — maximizes training data for a deployed model |

## Repository map

```
txgnn/            # the installable package
  TxData.py       # KG download, preprocessing, split construction
  model.py        # HeteroRGCN / AttHeteroRGCN encoder, DistMultPredictor
  TxGNN.py        # pretrain/finetune/train_graphmask orchestration, save/load
  TxEval.py       # disease-centric evaluation entry point
  utils.py        # negative samplers, metrics, similarity profiling, KG utilities
  graphmask/      # Lagrangian optimization, hard/soft concrete gates
  data_splits/    # Disease Ontology parsing + DataSplitter for disease-area splits
reproduce/        # CLI training script + result-gathering notebook
TxGNN_Demo.ipynb  # end-to-end walkthrough notebook
docs/             # this documentation set (not part of the original paper release)
```

## Key hyperparameters at a glance

| Name | Default | Meaning |
| :--- | :--- | :--- |
| `n_hid` / `n_inp` / `n_out` | 128 | Embedding dimensions (hidden/input/output) |
| `proto` | `True` | Enable the prototype/metric-learning module |
| `proto_num` | 5 | Number of similar diseases (K) retrieved per query |
| `sim_measure` | `'all_nodes_profile'` | Disease similarity notion (see [ALGORITHMS.md §6](ALGORITHMS.md#6-disease-similarity-profiling-algorithms-utilspy)) |
| `agg_measure` | `'rarity'` | How prototype and raw embeddings are fused |
| `attention` | `False` | Use `AttHeteroRGCNLayer` instead of the plain RGCN layer |
| `allowance` | 0.005 | GraphMask's max tolerated prediction divergence |

## License / provenance

MIT License, © 2022 Machine Learning for Medicine and Science @ Harvard (see [LICENSE](../LICENSE)). Official repo: `github.com/mims-harvard/TxGNN`.
