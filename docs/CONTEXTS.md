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

- **Pure collaborative-filtering / matrix-factorization repurposing models** (drug × disease interaction matrices) ignore the surrounding biological KG entirely, so they have no signal at all for a disease or drug with few labeled interactions — the cold-start problem is unsolvable by construction.
- **Standard homogeneous or single-relation GNN link predictors** either collapse all relation types into one (losing the semantic distinction between "this edge is a protein interaction" vs. "this edge is a contraindication") or require one model per relation type (losing shared structure across relations, and unable to reuse rare-relation signal from abundant ones).
- **Vanilla DistMult/TransE-style KG embedding models trained only on the target relations** overfit to the diseases/drugs that already have many labeled therapeutic edges, and provide no mechanism to transfer knowledge to a disease with zero such edges — there is no "borrowing" mechanism from similar, well-characterized diseases.
- **Black-box GNN predictors without post-hoc explanation** cannot support the actual clinical/scientific workflow, where a predicted repurposing candidate must be triaged and prioritized for expensive wet-lab or clinical validation based on *why* the model believes it — not just a probability score.

TxGNN's architecture is a direct response to each of these gaps: pretraining across the full heterogeneous KG (P1) supplies structural signal even to sparsely-labeled entities; the prototype module (P2) explicitly transfers signal from similar diseases; disease-centric splits and ranking metrics (P3) make the "does it generalize?" claim falsifiable rather than assumed; and GraphMask (P4) turns predictions into inspectable evidence.

## 4. Research Questions

1. **RQ1 (Generalization):** Can a heterogeneous GNN pretrained on broad biological structure and fine-tuned with prototype-based metric learning outperform models trained only on direct therapeutic labels, specifically on diseases held out entirely from training (zero-shot)?
2. **RQ2 (Mechanism of transfer):** Which notion of "disease similarity" (direct neighborhood overlap, multi-hop PPI random walk, semantic/BERT similarity, or combinations) best transfers useful signal to an under-characterized disease, and does this vary by how "cold" the disease is (§5 in [ALGORITHMS.md](ALGORITHMS.md)'s `few_edeges_to_kg` vs. `few_edeges_to_indications` split)?
3. **RQ3 (Evaluation validity):** How much does apparent performance change between a random edge split and a disease-centric split, and does that gap itself quantify the degree to which prior evaluation protocols overstated real-world zero-shot capability?
4. **RQ4 (Explainability faithfulness):** Can a learned edge-masking explanation (GraphMask) recover a sparse subgraph that is simultaneously (a) faithful — reproducing the original prediction with minimal divergence — and (b) small enough to be human-interpretable, without retraining the underlying predictor?
5. **RQ5 (Precision at the top of the ranking):** Since a clinician or wet-lab scientist can only act on a handful of top-ranked candidates per disease, does optimizing for pooled AUROC/AUPRC actually translate into good Precision@K / MRR at the top of a disease's ranked candidate list, or is a ranking-aware objective needed? (This is the open question that motivates [IMPROVEMENTS_PLAN.md](IMPROVEMENTS_PLAN.md).)

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
