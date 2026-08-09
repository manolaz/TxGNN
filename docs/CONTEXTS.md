# TxGNN: Research Problem Statement & Context

This document frames the scientific and engineering problem that motivates every design decision.Where those documents answer "how is it built?", this document answers **"why does it need to exist, and what problem is it solving?"**

---

## 1. Background: The Clinical and Scientific Context

Drug development is slow (often 10+ years) and expensive, while a large share of diseases — particularly rare and orphan diseases — have **no approved therapies at all**. Meanwhile, thousands of approved drugs already exist with well-characterized safety profiles. **Drug repurposing** — finding new therapeutic uses for existing drugs — is one of the fastest and cheapest paths to a treatment, because it can skip much of the safety-validation pipeline required for a novel compound.

The scientific opportunity is that modern biomedical knowledge graphs (KGs) now encode, at scale, the relationships that a clinician or pharmacologist would otherwise need years to piece together manually: drug-target interactions, protein-protein interactions, disease-phenotype associations, drug-drug interactions, and the sparse set of already-known drug-disease indications/contraindications. TxGNN operates on such a KG — in the released benchmark, roughly 17,080 diseases and 7,957 drugs/therapeutic candidates connected through millions of biological and clinical edges (see [README.md](../README.md)).

The catch: the diseases that most need new treatment options — rare diseases, newly characterized conditions, diseases with poor molecular understanding — are precisely the diseases with the **fewest** existing KG edges to learn from. A purely supervised model trained to reproduce known drug-disease edges will, by construction, perform best on well-studied diseases that already have many treatments and worst on exactly the diseases where a repurposing prediction would matter most.

## 2. Research Problem Statement

> **Given a biomedical knowledge graph with sparse and unevenly distributed therapeutic labels, learn a model that predicts drug-disease therapeutic relationships (indication, contraindication, off-label use) that generalizes to diseases with few or zero observed treatments in training, while remaining interpretable enough for a clinician to trust and act on its output.**

This decomposes into four coupled sub-problems, each of which drives a major component of the system:

| Sub-problem                                                                         | Why it's hard                                                                                                                                                                                                                           | Where it's addressed                                                                                                                                                                                                                        |
| :---------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **(P1) Representation learning over a heterogeneous, multi-relational graph** | Drugs, diseases, proteins, phenotypes, and exposures are structurally and semantically different entity types connected by dozens of relation types; a homogeneous GNN cannot respect this.                                             | Heterogeneous RGCN / attention encoder —[NETWORK_DESIGN.md §3](NETWORK_DESIGN.md#3-heterogeneous-rgcn-encoder)                                                                                                                             |
| **(P2) Zero-/few-shot generalization to under-characterized diseases**        | Message passing alone produces poor embeddings for diseases with little or no direct graph connectivity ("cold" nodes); yet these are the target population.                                                                            | Prototype/metric-learning module —[NETWORK_DESIGN.md §4](NETWORK_DESIGN.md#4-prototype-learning-module), [ALGORITHMS.md §6](ALGORITHMS.md#6-disease-similarity-profiling-algorithms-utilspy)                                               |
| **(P3) Honest, non-inflated evaluation of generalization**                    | Naively splitting edges at random mostly measures interpolation on well-known diseases, not the extrapolation claim being made; sparse positive labels and easy negatives can make offline metrics look better than real-world utility. | Disease-centric splitting & ranking evaluation —[ALGORITHMS.md §5, §7, §8](ALGORITHMS.md#5-data-splitting-algorithms-create_fold-utilspy)                                                                                                |
| **(P4) Explainability for clinical trust and validation**                     | A ranked drug list without justification is not actionable for a clinician or a bench scientist deciding what to validate experimentally; the reasoning must be traceable to specific KG evidence.                                      | GraphMask edge-masking explainer —[NETWORK_DESIGN.md §6](NETWORK_DESIGN.md#6-graphmask-explainability-module), [ALGORITHMS.md §9](ALGORITHMS.md#9-graphmask-training-algorithm-txgnnpy-train_graphmask-graphmasklagrangian_optimizationpy) |

## 3. Why Prior Approaches Fall Short (Motivation)

Each prior-approach failure below is framed as **what it does → the specific problem that creates → why TxGNN's design avoids it**, so the motivation is a falsifiable comparison rather than a rhetorical list.

### 3.1. Pure collaborative-filtering / matrix-factorization repurposing models

**What they do.** Treat drug repurposing as filling in a sparse drug × disease interaction matrix (analogous to a recommender system), learning latent drug and disease factors purely from the pattern of known interactions.

**The problem this creates.** These models have no representation of *why* a drug and disease might interact beyond co-occurrence in the label matrix — there is no surrounding biological graph (proteins, pathways, phenotypes) to fall back on. A disease or drug with few or zero labeled interactions therefore has an under-determined or entirely unlearnable latent factor: the cold-start problem is unsolvable *by construction*, not just in practice, because the model's only input signal is the very labels that are missing.

**Why TxGNN avoids it.** The heterogeneous KG encoder (P1) means a disease's representation is shaped by its non-therapeutic edges (disease-protein, disease-phenotype, disease-disease) even when it has zero drug edges — there is always *some* signal to encode, unlike a matrix-factorization model where a cold row/column is a literal blank.

### 3.2. Standard homogeneous or single-relation GNN link predictors

**What they do.** Either (a) collapse the KG into a single homogeneous graph (all edges treated identically) so one GNN can be applied directly, or (b) train a completely separate model per relation type to respect relation semantics.

**The problem this creates.** Option (a) destroys the very distinction that makes the KG informative — a "this edge is a protein-protein interaction" message and a "this edge is a contraindication" message get mixed into the same aggregation, so the model cannot learn relation-specific reasoning. Option (b) solves that but at the cost of losing shared structure: a relation with very few examples (e.g., `off-label use`) cannot borrow statistical strength from a relation with many examples (e.g., `indication`), because they never share parameters or gradients.

**Why TxGNN avoids it.** The RGCN's per-relation weight matrices $W_r$ ([NETWORK_DESIGN.md §3](NETWORK_DESIGN.md#3-heterogeneous-rgcn-encoder)) keep relation semantics distinct while still sharing the same underlying node embeddings across all relations — a single encoder pass produces one embedding per node that every relation type (rare or common) draws from, which is precisely why broad pretraining (§2, P1) can benefit the sparser therapeutic relations at fine-tuning time.

### 3.3. Vanilla DistMult/TransE-style KG embedding models trained only on target relations

**What they do.** Apply a standard KG-embedding decoder (DistMult, TransE, etc.) but train it end-to-end only on the therapeutic edges of interest, without a separate broad-pretraining phase or an explicit mechanism for transferring information between similar entities.

**The problem this creates.** Without either (i) pretraining on the surrounding KG or (ii) an explicit similarity-based transfer mechanism, the embeddings for diseases/drugs with many labeled edges overfit to their observed neighborhood, while the embeddings for diseases with zero labeled edges never receive a meaningful gradient signal at all — there is no architectural pathway for a well-characterized disease's learned representation to "help" a poorly-characterized one.

**Why TxGNN avoids it.** This is exactly the gap closed by combining pretraining (P1) with the explicit prototype/metric-learning module (P2): the prototype module is a *designed* borrowing mechanism (Top-K similarity retrieval + fusion, [NETWORK_DESIGN.md §4](NETWORK_DESIGN.md#4-prototype-learning-module)) rather than hoping generalization emerges implicitly from training on the target relations alone.

### 3.4. Black-box GNN predictors without post-hoc explanation

**What they do.** Output a probability/score for a candidate drug-disease pair with no accompanying justification — the model is treated purely as an oracle.

**The problem this creates.** A ranked list of repurposing candidates with no supporting evidence cannot be triaged responsibly: a clinician or bench scientist deciding which of, say, the top 20 candidates to pursue for expensive experimental validation has no way to distinguish a prediction backed by a plausible shared-pathway mechanism from one that is a statistical artifact of the training data. This isn't a convenience gap — it can materially change which predictions are *actionable* versus merely *numerically high-scoring*.

**Why TxGNN avoids it.** GraphMask (P4) is bolted on as a dedicated post-hoc step precisely because explanation is treated as a first-class deliverable, not an afterthought — it operates on the *already-trained, frozen* predictor (see [ALGORITHMS.md §9](ALGORITHMS.md#9-graphmask-training-algorithm-txgnnpy-train_graphmask-graphmasklagrangian_optimizationpy)) specifically so that adding explainability never requires retraining or risks changing the underlying prediction.

TxGNN's architecture is therefore a direct, one-to-one response to these four gaps: pretraining across the full heterogeneous KG (P1) supplies structural signal even to sparsely-labeled entities; the prototype module (P2) explicitly transfers signal from similar diseases; disease-centric splits and ranking metrics (P3) make the "does it generalize?" claim falsifiable rather than assumed; and GraphMask (P4) turns predictions into inspectable evidence.

### 3.5. A deeper engineering problem: constructing an honest disease-area test set

Beyond the four modeling-level gaps above, building the ontology-driven disease-area splits (P3) surfaces its own non-trivial *engineering* problem, solved by `DataSplitter` in [datasplit.py](../txgnn/data_splits/datasplit.py):

**Problem.** Given a clinical disease-area root (e.g., DOID `1287` = "cardiovascular system disease"), it's not enough to just hold out that disease's direct treatment edges — a *fair* test set needs (a) every descendant disease in the ontology subtree, correctly cross-referenced from Disease Ontology (DO) IDs to the KG's internal MONDO-based node indices, and (b) a comparably-sized set of *non-therapeutic* test edges sampled from the same local neighborhood, so the test set isn't trivially distinguishable from the rest of the graph by degree or locality alone.

**How it's solved.**
- `load_do()` parses [HumanDO.obo](../txgnn/data_splits/HumanDO.obo) once and recursively expands `doid2children` over up to 20 levels of ontology depth, so `get_nodes_for_doid` can resolve an entire clinical area (not just one exact DOID) to KG node indices — cross-referencing through [mondo_references.csv](../txgnn/data_splits/mondo_references.csv) and a BERT-grouped-disease mapping ([kg_grouped_diseases_bert_map.csv](../txgnn/data_splits/kg_grouped_diseases_bert_map.csv)) to handle diseases that were merged/grouped during KG construction.
- `get_edge_group` / `get_one_hop_edge_group` use PyTorch Geometric's `k_hop_subgraph` utility to sample the *local neighborhood* around the disease-area nodes (1-hop for the mechanistic-masking variant, 2-hop for the general test-edge variant) and draw the non-therapeutic test edges from *within that neighborhood* rather than uniformly from the whole KG — this is what makes the held-out set represent "this disease area's local biology," not an arbitrary random sample that happens to touch the disease.
- The two modes (`one_hop` with `mask_ratio`, vs. the default `test_size`-based 2-hop sampling) directly implement the two distinct realism scenarios from [ALGORITHMS.md §4](ALGORITHMS.md#4-knowledge-graph-preprocessing-preprocess_kg-utilspy): fully masking a fraction of the immediate biological neighborhood (simulating "this disease's molecular mechanism is itself poorly understood") versus only holding out a modest fraction of a wider 2-hop region (simulating "the biology is known, but treatments haven't been established yet").

**Trade-off / known constraint.** This path requires PyTorch Geometric (`torch_geometric.utils.k_hop_subgraph`) purely for this legacy disease-area construction step — noted explicitly in [README.md](../README.md)'s installation instructions ("if you want to use disease-area split, you should also install PyG") — meaning the disease-area splits carry an extra dependency that the `complex_disease` / `random` splits (implemented in pure pandas/numpy, [ALGORITHMS.md §5](ALGORITHMS.md#5-data-splitting-algorithms-create_fold-utilspy)) do not.

## 4. Research Questions

Each question below follows the same structure: **why it matters**, **how it is operationalized in code** (so it is falsifiable, not rhetorical), and **what a positive vs. negative answer would imply** for the project.

### RQ1 — Generalization: Does broad pretraining + prototype learning actually beat direct supervision on unseen diseases?

**Question.** Can a heterogeneous GNN pretrained on the full biological KG and fine-tuned with prototype-based metric learning outperform a model trained only on direct therapeutic labels, specifically on diseases held out *entirely* from training (zero-shot)?

**Why it matters.** This is the load-bearing claim of the entire project (§1–§2). If pretraining + prototypes don't measurably help on genuinely unseen diseases, the added architectural complexity (two-stage training, similarity profiling, fusion gating) isn't justified over a simpler direct-supervision baseline.

**How it's operationalized.**
- The `--model` flag in [reproduce/train.py](../reproduce/train.py) directly encodes the ablation: `GNN` (`proto=False`, no prototype module, effectively "direct supervision only"), `TxGAT` (adds attention), and `TxGNN` (full pipeline: pretrain → finetune with `proto=True`). Running all three on the same `complex_disease` split isolates the contribution of pretraining and prototype learning from architecture alone.
- Comparison is read off **macro AUROC/AUPRC** on the `complex_disease` / disease-area test splits (§7 in [ALGORITHMS.md](ALGORITHMS.md)), specifically on the subset of diseases with zero or near-zero training-set treatment edges (the `few_edeges_to_kg` / `few_edeges_to_indications` splits, §5).
- The no-KG ablation path in `TxGNN.model_initialize` (`if self.no_kg and proto: proto = False`, see [TxGNN.py](../txgnn/TxGNN.py)) provides a further control: it forces `proto=False` whenever the KG itself is stripped down, confirming that prototype learning is meaningless without the surrounding graph it draws similarity from — i.e., the two components (broad KG + prototypes) are coupled, not independent levers.

**What the answer implies.** A *positive* result (TxGNN > GNN-only on cold diseases, with the gap widening as diseases get colder) validates the core architectural thesis and justifies its training-time cost (pretrain + finetune vs. a single training phase). A *negative* or *shrinking* result would suggest the prototype module either isn't retrieving useful neighbors for those diseases (→ investigate RQ2) or that the gains are concentrated in diseases that weren't truly cold to begin with (→ investigate RQ3, since split leakage would produce exactly this false-positive signal).

### RQ2 — Transfer mechanism: Which disease-similarity notion actually carries useful signal, and when?

**Question.** Which notion of "disease similarity" — direct neighborhood overlap (`all_nodes_profile`), narrower protein overlap (`protein_profile`), multi-hop PPI random walk (`protein_random_walk`), semantic/BERT similarity (`bert`), or a concatenation (`profile+bert`) — best transfers useful signal to an under-characterized disease? Does the best choice change depending on *how* cold the disease is?

**Why it matters.** RQ1 asks *whether* transfer helps; RQ2 asks *how* it works, which determines whether the mechanism is trustworthy or a lucky artifact. A disease with zero KG edges of any kind can only be reached via `bert` (semantic similarity needs no graph connectivity at all), while a disease with some protein annotations but no treatments might be best served by `protein_random_walk`. If the "best" similarity measure is arbitrary or unstable across splits, that undermines confidence that the model is learning a real biological signal rather than exploiting a spurious correlation in one particular benchmark split.

**How it's operationalized.**
- `sim_measure` and `agg_measure` are first-class hyperparameters of `TxGNN.model_initialize` (see [README.md](../README.md) and [NETWORK_DESIGN.md §4](NETWORK_DESIGN.md#4-prototype-learning-module)), so a grid over `{all_nodes_profile, protein_profile, protein_random_walk, bert, profile+bert} × {avg, rarity, learn, 100proto}` can be run on identical splits and identical random seeds, isolating the similarity-measure choice from everything else.
- Stratify results by disease "coldness" using the two purpose-built splits in [ALGORITHMS.md §5](ALGORITHMS.md#5-data-splitting-algorithms-create_fold-utilspy): `few_edeges_to_kg` (thresholds on raw KG degree — tests whether *topological* similarity measures still work when a disease has almost no graph footprint at all) vs. `few_edeges_to_indications` (thresholds on known-drug count only, KG-rich otherwise — tests whether topological measures suffice when biology is well characterized but treatments aren't).
- `TxGNN.retrieve_sim_diseases()` (see [TxGNN.py](../txgnn/TxGNN.py)) exposes the actual top-k retrieved prototype diseases for a given relation, making it possible to manually audit *which* diseases are being borrowed from and sanity-check clinical plausibility (e.g., does a rare cardiomyopathy actually retrieve other cardiomyopathies, or unrelated diseases that merely share a generic protein?).

**What the answer implies.** If the best-performing similarity measure is consistent across both coldness regimes, that's evidence of a general, robust transfer mechanism. If `few_edeges_to_kg` diseases only benefit from `bert` (semantic, graph-independent) while `few_edeges_to_indications` diseases benefit from `protein_random_walk` (topological), that's a mechanistically interpretable result: it says structural similarity requires *some* KG footprint to work, and semantic similarity is the necessary fallback for truly disconnected diseases — directly informing which measure to default to for a new deployment KG with different sparsity characteristics.

### RQ3 — Evaluation validity: How much does the split protocol itself inflate or deflate the zero-shot claim?

**Question.** How much does apparent performance change between a random edge split and a disease-centric split? Does that gap itself quantify the degree to which naive evaluation protocols overstate real-world zero-shot capability?

**Why it matters.** This question is *prior to* RQ1 and RQ2 in the sense that it validates the ruler being used to measure them. A model could appear to "generalize" purely because random edge splitting leaks 90% of a disease's other treatments into the training set (§5 in [ALGORITHMS.md](ALGORITHMS.md)) — in which case RQ1's "zero-shot win" would be an artifact of measurement, not a real capability.

**How it's operationalized.**
- [reproduce/train.py](../reproduce/train.py)'s `--split` argument directly supports running the *identical* model/training code over `random`, `complex_disease`, and the disease-area splits (`cardiovascular`, `autoimmune`, etc.) with the same seed — so the *only* variable that changes is the partitioning strategy. The delta in macro AUROC/AUPRC between `random` and `complex_disease` is a direct, reproducible measurement of "how much of the apparent skill was interpolation vs. extrapolation."
- `complex_disease_cv` (20-fold cross-validation, [ALGORITHMS.md §5](ALGORITHMS.md#5-data-splitting-algorithms-create_fold-utilspy)) turns this from a single point estimate into a distribution, so the evaluation-validity gap itself can be reported with a confidence interval rather than a single (possibly lucky or unlucky) split.
- `TxEval.eval_disease_centric` further decomposes any aggregate delta into *which specific diseases* lose the most performance under stricter splitting (via the per-disease `AUROC`/`Recall@k` breakdown in `disease_centric_evaluation`, [ALGORITHMS.md §8](ALGORITHMS.md#8-disease-centric-ranking-evaluation-txevaleval_disease_centric--disease_centric_evaluation-txevalpy-utilspy)), rather than only a single pooled number.

**What the answer implies.** A large random-vs-disease-centric gap is not necessarily bad news — it is the expected, honest signature of a genuinely hard zero-shot problem, and it recalibrates expectations for what "good" performance looks like on this task. A *small* gap would be more surprising and would need investigation: either the model doesn't rely much on disease-specific interpolation shortcuts (encouraging), or the "unseen" diseases in `complex_disease` still leak through non-therapeutic edges (e.g., a disease-disease edge to a well-treated disease), meaning the split isn't as strict as it appears (§5's note on the `disease_eval` one-off case study existing specifically to sanity-check this).

### RQ4 — Explanation faithfulness: Can GraphMask compress a prediction into a small, honest evidence subgraph?

**Question.** Can a learned edge-masking explanation (GraphMask) recover a sparse subgraph that is simultaneously (a) **faithful** — reproducing the original prediction with minimal divergence — and (b) **small** enough to be human-interpretable, without retraining the underlying predictor?

**Why it matters.** An explanation that is faithful but not sparse (e.g., "keep 95% of edges") is not actionable — a clinician cannot review a nearly-whole knowledge graph. An explanation that is sparse but unfaithful (predictions change substantially once most edges are masked) is worse than no explanation, because it would misrepresent what the model actually relies on. Both failure modes are plausible a priori, which is exactly why this is framed as an open question rather than an assumed property of GraphMask.

**How it's operationalized.**
- `evaluate_graphmask` (see [utils.py](../txgnn/utils.py), used inside `TxGNN.train_graphmask`) reports, every `valid_per_n` epochs, both halves of the trade-off simultaneously: **mean divergence** (MSE between sigmoid outputs of the masked vs. original model) and **`%masked_L1` / `%masked_L2`** (fraction of edges gated to zero per RGCN layer) — see [ALGORITHMS.md §9](ALGORITHMS.md#9-graphmask-training-algorithm-txgnnpy-train_graphmask-graphmasklagrangian_optimizationpy). Faithfulness and sparsity are logged as two separate numbers precisely so one cannot be optimized while silently sacrificing the other.
- The Lagrangian formulation (`LagrangianOptimization.update`, [graphmask/lagrangian_optimization.py](../txgnn/graphmask/lagrangian_optimization.py)) makes the faithfulness side of the trade-off a hard constraint (`allowance`, default `0.005`) rather than a soft, hand-tuned penalty weight — so "faithful enough" is a fixed, auditable threshold, and the only free variable left to observe is how much sparsity the optimizer can achieve *subject to* staying under that threshold.
- At `mode='testing'`, `evaluate_graphmask` additionally reports **AUROC/AUPRC of the masked vs. original predictions on the held-out test set** — so faithfulness is checked not just as raw divergence on validation but as whether the *downstream ranking quality* (the metric that actually matters per RQ5) survives the masking.

**What the answer implies.** If GraphMask reliably achieves high sparsity (most edges masked) while keeping test AUROC/AUPRC within a small tolerance of the unmasked model, the explanations can be trusted as a faithful summary of model reasoning and used for downstream biological hypothesis generation. If sparsity can only be pushed to a modest level before divergence exceeds `allowance`, that itself is informative — it would indicate the prediction genuinely depends on a broad, diffuse pattern of evidence rather than a few salient edges, which is a meaningful (if less convenient) finding about how the underlying model actually reasons.

### RQ5 — Top-K precision: Does optimizing pooled AUROC/AUPRC actually produce a clinically useful ranked list?

**Question.** Since a clinician or wet-lab scientist can only act on a handful of top-ranked candidates per disease, does optimizing for pooled AUROC/AUPRC during training actually translate into good Precision@K / MRR at the *top* of a disease's ranked candidate list — or is a ranking-aware training objective needed instead?

**Why it matters.** AUROC/AUPRC are computed over the *entire* score distribution and are dominated by how well the model separates positives from the (very large) pool of easy negatives; they do not specifically reward getting the top 10–100 candidates right, which is the only part of the ranking a human will ever actually look at. It is entirely possible for a model to have strong AUPRC while still burying most true positives outside the top 100 for individual diseases — the two objectives are correlated but not identical.

**How it's operationalized.**
- `disease_centric_evaluation`'s `Recall@k` / `Enrichment@k` / `MRR@k` / `AP@k` outputs (`k ∈ {1%, 5%, 10%, 10, 50, 100}`, [ALGORITHMS.md §8](ALGORITHMS.md#8-disease-centric-ranking-evaluation-txevaleval_disease_centric--disease_centric_evaluation-txevalpy-utilspy)) are computed *independently* of AUROC/AUPRC in the same evaluation pass, so the two families of metrics can be directly compared per disease and in aggregate — e.g., checking whether diseases with high per-disease AUROC also have high `Recall@10`, or whether that correlation breaks down for diseases with very few positives (where AUROC is a noisy statistic to begin with).
- The current baseline numbers cited in [IMPROVEMENTS_PLAN.md](IMPROVEMENTS_PLAN.md) — **AUPRC ≈ 0.872** but **Precision@10 ≈ 0.138** and **Precision@100 ≈ 0.025** — are themselves the empirical evidence motivating this research question: a high pooled AUPRC coexists with modest top-of-list precision, suggesting the answer to RQ5 (as currently measured) leans toward "no, pooled metrics are not a sufficient proxy for top-K clinical usefulness."
- [IMPROVEMENTS_PLAN.md](IMPROVEMENTS_PLAN.md) Strategies 1–3 (alternative decoders, harder negative sampling, ranking-aware losses like BPR/InfoNCE/asymmetric focal loss) are the proposed interventions to test the causal side of this question: if replacing BCE with a ranking-aware loss measurably improves Precision@10/@100 *without* degrading AUROC/AUPRC, that confirms the training objective — not just the architecture — was the bottleneck.

**What the answer implies.** If pooled and top-K metrics track each other closely, the existing BCE-based training objective is adequate and effort should focus elsewhere (e.g., RQ1/RQ2). If they diverge, as the current baseline numbers suggest, it justifies prioritizing [IMPROVEMENTS_PLAN.md](IMPROVEMENTS_PLAN.md)'s ranking-loss and hard-negative-sampling strategies over further architectural changes, since the gap is in *what is optimized* rather than *what the model is capable of representing*.

## 5. Scope and Non-Goals

**In scope:**

- Link prediction for three therapeutic relation types (indication, contraindication, off-label use) between drug and disease nodes in a fixed, pre-constructed KG.
- Zero-/few-shot generalization across diseases (not drugs — the current splits partition by disease, not by drug).
- Post-hoc structural explainability (which KG edges/paths support a prediction), not causal or mechanistic explanation (why the underlying biology causes the effect).

**Explicitly out of scope (see [IMPROVEMENTS_PLAN.md](IMPROVEMENTS_PLAN.md) for proposed future work):**

- Learning or refining the knowledge graph itself (entity resolution, edge confidence estimation, KG completion beyond the therapeutic relations).
- Incorporating raw multimodal biological features (SMILES/ChemBERTa for drugs, protein sequence embeddings, clinical-text embeddings) — the baseline uses structure-only, randomly-initialized node features (see [NETWORK_DESIGN.md §2](NETWORK_DESIGN.md#2-graph-representation--node-initialization)).
- Clinical trial design, dosing, or safety/toxicity prediction — TxGNN predicts *candidate* therapeutic relationships to be triaged for further validation, not a deployment-ready clinical decision.

## 6. Success Criteria

Given the problem statement above, a candidate improvement or new experiment should be evaluated against the metrics and splits already implemented (see [ALGORITHMS.md §5, §7, §8](ALGORITHMS.md#5-data-splitting-algorithms-create_fold-utilspy)):

- **Macro AUROC/AUPRC** on `complex_disease` / disease-area splits, not just micro-averaged or random-split numbers (avoids the P3 evaluation-validity failure mode).
- **Disease-centric ranking metrics** (Recall@k, Enrichment vs. random, MRR, AP) — because the deployment use case is "rank candidates for *this* disease," not "classify this edge in isolation."
- **Zero-shot-specific slices** (`disease_eval`, `few_edeges_to_kg`, `few_edeges_to_indications`) reported *separately* from aggregate metrics, since aggregate numbers can be dominated by well-characterized diseases and mask poor cold-start performance.
- **GraphMask sparsity/faithfulness trade-off** (% edges masked vs. divergence from original prediction) when explainability changes are made, not just downstream predictive accuracy.

---

## Summary

TxGNN's central research bet is that **broad, multi-relational biological structure plus an explicit similarity-based transfer mechanism can substitute for the direct therapeutic labels that under-characterized diseases lack** — and that this bet can only be validated with evaluation protocols (disease-centric splits, ranking metrics, zero-shot slices) designed specifically to detect whether that transfer actually works, rather than protocols that reward memorizing well-studied diseases. Every algorithmic choice cataloged in [ALGORITHMS.md](ALGORITHMS.md) exists to either enable that transfer (pretraining, prototype learning), validate it honestly (splits, disease-centric evaluation), or make its output actionable (GraphMask explanations).
