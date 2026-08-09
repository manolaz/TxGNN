# TxGNN: Implemented Algorithms

This document catalogs the concrete algorithms implemented in the `txgnn` package — training procedures, sampling strategies, data-splitting routines, similarity computations, and explainability optimization. Each algorithm is presented as **Problem → Solution (pseudocode) → Design Rationale / Trade-offs**, so the *why* behind each implementation choice is explicit, not just the *what*. For the underlying architecture and math notation (RGCN, DistMult, prototype fusion, GraphMask formulation), see [NETWORK_DESIGN.md](NETWORK_DESIGN.md). For module/system layout, see [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md).

---

## 1. Pretraining Algorithm (`TxGNN.pretrain`, [TxGNN.py](../txgnn/TxGNN.py))

**Problem.** A knowledge graph with 17k+ diseases and 8k+ drugs has very few labeled therapeutic edges (indication/contraindication/off-label) relative to the number of entities. Training the encoder from scratch directly on these sparse labels would overfit and would not exploit the rich biological structure (protein-protein interactions, disease-phenotype associations, drug-drug interactions, etc.) that surrounds the therapeutic edges. There is no way to supervise a GNN encoder on relations for which there are no labels if it only ever sees the therapeutic subgraph.

**Solution.** Self-supervised link-prediction pretraining over **all** relation types in the KG using minibatch neighbor sampling, so every edge type (not just drug-disease) contributes gradient signal to the shared encoder:
```
Input: heterogeneous graph G, epochs, lr, batch_size
1. Build an edge dataloader over all canonical edge types with a
   2-layer full-neighborhood block sampler (dgl.dataloading.MultiLayerFullNeighborSampler(2))
2. Attach a Minibatch_NegSampler (see §3.1) that corrupts the destination
   node of each positive edge ('fix_dst' scheme)
3. optimizer = AdamW(model.parameters(), lr)
4. for epoch in 1..n_epoch:
     for (input_nodes, pos_graph, neg_graph, blocks) in dataloader:
        pos_score, neg_score = model.forward_minibatch(pos_graph, neg_graph, blocks, G, pretrain_mode=True)
        loss = BCE(sigmoid([pos_score; neg_score]), [1..1, 0..0])
        loss.backward(); optimizer.step()
        every train_print_per_n steps: log micro/macro AUROC & AUPRC
5. return best_model = deepcopy(model)   # no validation gating during pretrain
```

**Design rationale / trade-offs.**
- *Why minibatching, not full-batch?* The full KG (all node/edge types) is too large to fit link-prediction computation for every relation in one GPU pass; block sampling with a 2-layer neighborhood keeps memory bounded regardless of KG size.
- *Why "reconstruct every relation" instead of only the target relations?* This is a form of **multi-task self-supervised pretraining** — it forces the encoder to learn embeddings that are informative across biology broadly, which is what later makes prototype-based zero-shot transfer to rare diseases possible (a disease with no drug edges still has protein/phenotype edges that shaped its embedding).
- *Why no validation-based model selection here?* Pretraining optimizes a proxy task (generic link prediction), not the actual clinical objective; the real model selection happens during fine-tuning where macro AUROC on the *actual* drug-disease task is tracked. Selecting a "best" pretrain checkpoint by a proxy metric could bias away from what fine-tuning ultimately needs.
- *Alternative considered:* training only on drug-disease edges from scratch — rejected because it starves the model of structural information for diseases with few or no known treatments, precisely the diseases the project targets.

## 2. Fine-tuning Algorithm (`TxGNN.finetune`)

**Problem.** After pretraining gives generic embeddings, the model must be specialized to predict the three clinically meaningful relations (indication, contraindication, off-label use) without forgetting/overwriting the broader structural knowledge, and while remaining usable for diseases with very sparse or zero direct treatment edges (the zero-shot setting).

**Solution.** Full-batch optimization restricted to the six drug-disease edge types (`indication`, `contraindication`, `off-label use`, and their reverses), with prototype/metric learning active:
```
Input: pretrained model, epochs, lr
1. Reinitialize decoder relation weights: xavier_uniform(model.w_rels)
2. neg_sampler = Full_Graph_NegSampler(G, k=1, method='fix_dst')
3. optimizer = AdamW(...); scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8)
4. for epoch in 1..n_epoch:
     neg_graph = neg_sampler(G)
     pos_score, neg_score = model(G, neg_graph, pretrain_mode=False, mode='train')  # only dd_etypes
     loss = BCE(sigmoid([pos_score; neg_score]), labels)
     loss.backward(); optimizer.step(); scheduler.step(loss)
     every valid_per_n epochs:
        metrics, val_loss = evaluate_fb(model, g_valid_pos, g_valid_neg, G, dd_etypes)
        if macro_auroc > best_val_acc: best_model = deepcopy(model)
5. Evaluate best_model once on the held-out test edges (evaluate_fb)
```

**Design rationale / trade-offs.**
- *Why reinitialize `w_rels` (the DistMult relation matrices)?* The pretraining phase trained relation embeddings for dozens of relation types simultaneously; the therapeutic relations' decoder weights would otherwise carry pretraining-scale noise unrelated to the fine-tuning distribution. Reinitializing them (while keeping the encoder weights) cleanly separates "what a node is" (encoder, retained) from "how a relation scores a pair" (decoder, relearned).
- *Why full-batch instead of minibatch here?* The drug-disease subgraph is orders of magnitude smaller than the full KG, so it fits in memory; full-batch training also gives a stable, exact gradient for the smaller, noisier, class-imbalanced therapeutic labels rather than a noisy minibatch estimate.
- *Why select on macro AUROC instead of micro AUROC or loss?* The three relations have very different edge counts (`indication` vastly outnumbers `off-label use`). Micro-averaging (or raw loss) would let the model "win" simply by mastering the dominant relation. Macro AUROC forces balanced competence across all three clinically distinct tasks, which matches the actual downstream use case (a clinician cares equally about spurious contraindications and useful off-label suggestions).
- *Why `ReduceLROnPlateau`?* Full-batch training over hundreds of epochs on a smaller, harder-to-fit label distribution benefits from automatic LR decay when loss plateaus, avoiding manual LR-schedule tuning per experiment.

## 3. Negative Sampling Algorithms ([utils.py](../txgnn/utils.py))

**Problem.** Link prediction needs contrastive negative examples, but the KG only records positive (observed) edges. Naively sampling a uniformly random non-edge as "negative" produces trivially easy negatives (e.g., a random protein paired with a random drug), which teaches the model little and inflates offline metrics without improving real discriminative power.

**Solution.** `construct_negative_graph_each_etype` implements several corruption schemes, selectable per experiment, that vary in how "hard" or representative the generated negative is:

| Method | Strategy | Problem it targets |
| :--- | :--- | :--- |
| `corrupt_dst` / `corrupt_src` / `corrupt_both` | Uniform random replacement of destination / source / both endpoints. | Simplest baseline; fast but produces mostly "easy" negatives. |
| `multinomial_src` / `multinomial_dst` | Sample replacement endpoint proportional to node degree$^{0.75}$ (frequency-based negative sampling, as in word2vec). | Prevents the model from trivially learning "low-degree node ⇒ negative"; forces it to discriminate among plausible, popular candidates. |
| `inverse_src` / `inverse_dst` | Sample proportional to **negative** degree$^{0.75}$, biasing toward low-degree nodes. | Used to probe/stress-test behavior on rare, low-degree entities specifically. |
| `fix_src` / `fix_dst` | Uniformly sample among nodes with nonzero degree for that relation (used by default in pretrain/finetune). | Guarantees negatives are drawn only from nodes that could plausibly participate in that relation type (a protein can't be a "negative drug"), avoiding degenerate/impossible negatives that waste training signal. |

### 3.1. `Minibatch_NegSampler`
Used during minibatch pretraining; precomputes per-etype sampling weights once (`in_degrees ** 0.75` or a 0/1 mask) so weights aren't recomputed per batch (an efficiency choice), then for each minibatch of positive edges repeats the source `k` times and draws destinations via `torch.multinomial`.

### 3.2. `Full_Graph_NegSampler`
Used during full-batch finetuning/evaluation; recomputes one negative graph per epoch across the whole KG, honoring the same weighting schemes. Recomputing every epoch (rather than reusing a fixed negative set) prevents the model from memorizing a static set of negatives — a form of implicit data augmentation.

**Design rationale.** Supporting 8 interchangeable strategies (rather than hardcoding one) lets the same training/evaluation code be reused for ablation studies that measure how sensitive the reported AUROC/AUPRC is to negative-sampling difficulty — a known confound in KG link-prediction benchmarks.

## 4. Knowledge-Graph Preprocessing (`preprocess_kg`, [utils.py](../txgnn/utils.py))

**Problem.** The raw KG is undirected/mixed-format and uses inconsistent, non-contiguous entity identifiers; some relation types are symmetric (e.g., protein-protein interaction) and would otherwise be double-counted as two directed edges `(u,v)` and `(v,u)`, inflating degree statistics and biasing negative sampling and similarity profiles that rely on degree/neighbor counts.

**Solution.**
```
1. If split targets a clinical disease-area (e.g. cardiovascular, diabetes):
     - Use DataSplitter + the Human Disease Ontology (HumanDO.obo) to find all
       descendant diseases of the target MONDO/DO id
     - Carve out a "test_kg" of held-out edges for those diseases (optionally
       one-hop masked), tag remaining edges as 'train'
2. Deduplicate undirected relations: for symmetric relation types (e.g.
   'protein_protein'), collapse (u,v) and (v,u) into one canonical direction
   using a sorted-id string key so the same edge isn't stored twice
3. Assign contiguous integer indices (x_idx / y_idx) per node type
4. Persist kg_directed.csv for reuse by later split generation
```

**Design rationale / trade-offs.**
- *Why ontology-driven disease-area splits (step 1)?* A single disease ID does not capture clinically related diseases (e.g., "cardiovascular disease" spans dozens of specific diagnoses). Walking the Human Disease Ontology's descendant tree from a MONDO/DO root id automatically gathers the full clinically-relevant disease set, avoiding hand-curated (and incomplete) disease lists.
- *Why an optional "one-hop mask" (`mask_ratio`)?* Even after holding out a disease's *treatment* edges, the disease could still leak information through its remaining KG neighborhood (proteins, phenotypes). Masking a fraction of one-hop edges too simulates the realistic zero-shot scenario where a rare disease's *molecular mechanism* is also poorly characterized, not just its treatments.
- *Why sorted-string dedup instead of a directed-graph library utility?* It is a simple, dependency-free way to get a canonical, order-independent key for an edge, robust to whatever ID types (numeric or string-composite) appear in the source CSV.
- *Caching (`kg_directed.csv`)* avoids recomputation of this (relatively expensive, `tqdm`-wrapped) relation-by-relation and node-type-by-node-type pass on every experiment run.

## 5. Data-Splitting Algorithms (`create_fold`, [utils.py](../txgnn/utils.py))

**Problem.** A single "random split" evaluation is insufficient for a drug-repurposing model whose stated goal is generalizing to diseases with little or no existing treatment — random edge splits mostly test *interpolation* (filling in missing edges for diseases already well-represented in training), not the *zero-/few-shot extrapolation* that is the actual scientific claim.

**Solution.** TxGNN supports multiple train/valid/test partitioning strategies selected by `split`, each simulating a distinct real-world evaluation scenario:

| Split | Problem it simulates | Mechanism |
| :--- | :--- | :--- |
| **`random`** | Baseline / sanity check. | Stratified random split per relation type (so rare relations still appear in valid/test). |
| **`complex_disease`** | Generalizing to an *entirely unseen disease*, not just an unseen edge of a known disease. | Diseases (not edges) are shuffled and partitioned into train/valid/test buckets; all drug-disease edges for a disease follow its bucket. |
| **`disease_eval`** | Deployment/case-study scenario: "What would the model predict for disease X if we had never seen any of its treatments?" | A single target disease (or list) is entirely held out for testing. |
| **`few_edeges_to_kg`** | Real rare diseases are poorly characterized *biologically*, not just clinically. | Diseases with ≤3 total KG connections are routed to the test set. |
| **`few_edeges_to_indications`** | Real rare diseases have few *known drugs*, even if biologically well-studied. | Thresholds on number of known drug indications (≤3) rather than raw KG degree. |
| **`complex_disease_cv`** | A single train/test split can overstate or understate performance due to which diseases happened to land in the test bucket. | 20-fold cross-validation variant of `complex_disease` using `sklearn.model_selection.KFold` over the shuffled disease list, for statistically robust reporting. |
| **`full_graph`** | Production deployment: maximize data used for the final shipped model. | 95/5 train/valid split with no true test set (early-stopping only). |

**Design rationale.** Disease-centric partitioning (splitting by *disease*, not by *edge*) is the central methodological choice that makes TxGNN's "zero-shot" claim testable: if edges were split randomly, a disease could have 90% of its indications in train and 10% in test, which is a much easier interpolation task than predicting for a disease with *zero* treatment edges seen during training.

After partitioning, `reverse_rel_generation` symmetrizes the KG by duplicating each split's edges with `rev_<relation>` in the opposite direction — necessary because the RGCN encoder (§3 in [NETWORK_DESIGN.md](NETWORK_DESIGN.md)) requires explicit reverse edge types to pass messages bidirectionally; without this step, information would only flow one way along every relation.

## 6. Disease Similarity Profiling Algorithms ([utils.py](../txgnn/utils.py))

**Problem.** The GNN encoder alone cannot produce a meaningful embedding for a disease that has few or no KG edges (a "cold" node) — message passing has nothing to aggregate. Yet these are exactly the diseases the model must serve. The prototype/metric-learning module (`DistMultPredictor`) needs a way to find diseases that are structurally or semantically similar to a poorly-connected query disease, so it can *borrow* signal from their embeddings.

**Solution — multiple interchangeable similarity notions, each solving a different sub-case of "few-shot disease":**

### 6.1. `obtain_disease_profile`
```
Input: disease node d, list of relation types R, list of neighbor node types T
For each (relation r, node type t) in zip(R, T):
    neighbors = successors(d, etype=r)
    profile_r = one-hot bit-vector over all nodes of type t, set to 1 at `neighbors`
Return concatenation of all profile_r vectors  # a multi-hot "neighborhood fingerprint"
```
*Problem solved:* diseases sharing many of the same proteins/phenotypes/other diseases are likely biologically related even without a direct disease-disease edge; this profile makes that overlap comparable via cosine similarity (`sim_matrix`).
*Trade-off:* still requires *some* neighbors to exist — a disease with literally zero annotated neighbors of any of the profiled types produces an all-zero vector and cannot be meaningfully compared. This is addressed by falling back to the random-walk profile (§6.2) or accepting degraded prototype quality for such extreme cases.

### 6.2. `obtain_protein_random_walk_profile`
```
Input: disease d, num_walks, path_length, walk_mode ('bit' | 'prob')
Repeat num_walks times:
    current = random neighbor of d via 'rev_disease_protein'  (a protein associated with d)
    walk = [current]
    Repeat path_length times:
        current = random neighbor of current via 'protein_protein'
        append current to walk (stop early if no neighbors)
    Accumulate all visited protein ids across walks
If walk_mode == 'bit':  profile[visited] = 1               # binary reachability signature
If walk_mode == 'prob': profile[node] = visit_count / total  # visitation-frequency signature
```
*Problem solved:* two diseases can implicate different proteins that nonetheless sit in the same functional module of the protein-protein interaction (PPI) network (e.g., interacting partners, shared pathway). A direct one-hop protein-overlap profile (§6.1's `protein_profile` variant) misses this; a random walk propagates multiple hops out into the PPI graph, capturing "functional neighborhood" rather than only "identical protein" overlap.
*Trade-off:* stochastic — profiles are not deterministic across recomputation (no fixed seed), and the walk can terminate early at dead ends (nodes with no PPI neighbors), so `num_walks`/`path_length` must be large enough to give a stable estimate; this is a variance/cost trade-off (more walks = more stable profile, more compute).

### 6.3. `sim_matrix`
Standard batched cosine similarity: L2-normalizes both embedding matrices (with an epsilon floor for numerical stability, avoiding divide-by-zero for the all-zero profile edge case above) and computes the pairwise dot product.

### 6.4. Prototype retrieval
At inference/training, for each query disease the model takes `torch.topk(sim, k+1)` (excluding self-similarity during training, since a disease is trivially most similar to itself and including it would let the model "cheat" by copying its own embedding) or `torch.topk(sim, k)` during evaluation on unseen diseases, then L1-normalizes the top-k similarity scores to use as convex-combination weights over the neighbors' GNN embeddings (see `agg_measure` fusion strategies in [NETWORK_DESIGN.md](NETWORK_DESIGN.md#43-prototype-fusion-strategies)).
*Design rationale for excluding self at train time only:* at evaluation time on genuinely unseen diseases, the query disease is (by construction) not itself in the similarity key-set, so no self-exclusion is needed or possible — the `k` vs `k+1` branching in the code (`src_h.shape[0] == src_h_keys.shape[0]`) detects which regime applies automatically.

## 7. Evaluation Metrics Algorithm (`get_all_metrics_fb`, [utils.py](../txgnn/utils.py))

**Problem.** A single pooled AUROC/AUPRC across all edges can hide poor performance on rarer, clinically important relations (`off-label use` has far fewer examples than `indication`), and can hide poor performance on individual diseases with unusual prediction distributions.

**Solution.**
```
Input: positive scores, negative scores per edge type, aggregate labels/scores
For each drug-disease edge type e in {indication, contraindication, off-label use, and reverses}:
    y_e = [1]*len(pos_e) + [0]*len(neg_e)
    pred_e = concat(pos_e, neg_e)
    auroc[e] = roc_auc_score(y_e, pred_e); auprc[e] = average_precision_score(y_e, pred_e)
micro_auroc/micro_auprc = roc_auc_score / average_precision_score over ALL edges pooled together
macro_auroc/macro_auprc = mean of per-relation AUROC/AUPRC
Return (auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc)
```
**Design rationale.** Reporting both micro (favors the numerically dominant relation) and macro (equal-weight per relation) side by side lets a practitioner detect exactly this kind of imbalance-driven blind spot — a model could show a strong micro AUROC purely from acing `indication` while being near-random on `off-label use`, which macro-averaging exposes and which drives fine-tuning's model-selection criterion (§2).

## 8. Disease-Centric Ranking Evaluation (`TxEval.eval_disease_centric` → `disease_centric_evaluation`, [TxEval.py](../txgnn/TxEval.py), [utils.py](../txgnn/utils.py))

**Problem.** Edge-level AUROC/AUPRC answers "can the model separate true from false drug-disease pairs on average?" — but a clinician's real question is disease-specific and ranking-based: "for *this* disease, does the model surface the true treatments near the *top* of its ranked drug list?" A model can have decent pooled AUROC while still burying true positives below hundreds of false candidates for individual diseases, which would make it clinically useless despite a good aggregate score.

**Solution.** For each target disease and relation, `disease_centric_evaluation` scores **every drug node** against that disease, ranks them, and computes a battery of ranking-quality metrics rather than a single classification score:
```
1. Determine target disease indices:
   - explicit list, OR
   - 'test_set': all diseases present in the held-out test split for the relation
2. For each disease:
   a. Build one evaluation edge per drug node (disease -> all drugs) and score with the model
   b. Label each drug: 1 if a true held-out treatment, -1 if seen during training (excluded
      from ranking metrics so the model isn't penalized/rewarded for already-known answers),
      0 otherwise
   c. Rank all candidate drugs by predicted score (argsort descending)
   d. Compute, at multiple cutoffs k in {1%, 5%, 10% of candidate pool, and fixed 10/50/100}:
        - Recall@k: fraction of true positives captured in the top-k
        - Recall_Random@k: expected recall of a random ranking (simulated 500x, or
          analytically k/N), used as a calibration baseline
        - Enrichment@k = Recall@k / Recall_Random@k  (how much better than chance)
        - MRR@k (mean reciprocal rank) and AP@k (average precision) restricted to top-k
   e. Compute per-disease AUROC/AUPRC and confusion-matrix-derived metrics
      (accuracy, sensitivity, specificity, PPV, NPV, FPR, FNR, FDR) at a 0.5 threshold
3. Aggregate (mean/std) each metric across all evaluated diseases; optionally show
   distribution plots (Seaborn) and persist raw per-disease results (pickle) for
   downstream analysis (see reproduce/ notebooks)
```
**Design rationale / trade-offs.**
- *Why exclude training-set positives (`label = -1`) from ranking metrics instead of just leaving them in the candidate pool as negatives?* If a drug already known to treat the disease (seen in training) were scored as a false positive when it ranks highly at test time, the metric would unfairly punish the model for correctly recalling *known* facts. Marking it `-1` and excluding it isolates the metric to genuinely novel predictions.
- *Why simulate a random-ranking baseline with 500 trials instead of a closed-form `k/N`?* The closed-form estimate is available (`else` branch, `Recall_Random[i] = k/num_items`) and used in "downstream" full-catalog mode; the Monte-Carlo simulation (`simulate_random=True`) is used for the "subset" mode where `N` is a small, disease-specific candidate pool for which the exact combinatorial expectation is less convenient to reason about directly, and simulation is a simple, verifiably-correct fallback.
- *Why report both fixed cutoffs (10/50/100) and percentage cutoffs (1%/5%/10%)?* Percentage cutoffs adapt to how many drugs are annotated for a given relation (fair across relations with very different candidate-pool sizes); fixed cutoffs let stakeholders reason in absolute terms ("if a clinician only looks at the top 10 suggestions...").
- *Why also report classification metrics (sensitivity/specificity/etc.) alongside ranking metrics?* Ranking metrics answer "is the right answer near the top?" while confusion-matrix metrics answer "at a fixed decision threshold, how many errors of each type occur?" — the two views are complementary for both list-based (recommend top-k drugs) and binary-decision (approve/reject a specific drug) use cases.

## 9. GraphMask Training Algorithm ([TxGNN.py](../txgnn/TxGNN.py) `train_graphmask`, [graphmask/lagrangian_optimization.py](../txgnn/graphmask/lagrangian_optimization.py))

**Problem.** A trained GNN's predictions are a black box — knowing *that* the model predicts drug X for disease Y does not tell a clinician or researcher *which* biological pathway or evidence (which edges in the KG) the prediction actually relies on. Post-hoc explanation must find a *minimal* subgraph that is still *sufficient* to reproduce the prediction, but manually tuning a fixed sparsity-vs-faithfulness penalty weight per relation/dataset is brittle and requires expensive re-tuning whenever the data changes.

**Solution.** GraphMask learns per-layer edge gates that erase as many messages as possible while keeping predictions close to the unmasked model, solved via **constrained (Lagrangian) optimization** rather than a fixed-weight penalty term, so the sparsity/faithfulness trade-off is found automatically during training.

### 9.1. Lagrangian dual ascent/descent (`LagrangianOptimization.update`)
```
State: alpha (dual variable, initialized 0.55), its own RMSprop optimizer
Given f = sparsity penalty term, g = constraint violation = relu(divergence - allowance):
    loss = f + softplus(alpha) * g
    zero_grad(model_optimizer); zero_grad(alpha_optimizer)
    loss.backward()
    model_optimizer.step()          # minimize loss w.r.t. gate/model params
    alpha.grad *= -1                # flip sign: ASCEND on alpha (dual problem)
    alpha_optimizer.step()
    clip alpha to [min_alpha=-2, max_alpha=30]
```
This performs simultaneous **primal descent** (shrink prediction divergence & mask more edges) and **dual ascent** (increase the penalty multiplier whenever the divergence constraint `g > 0` is violated), which automatically balances sparsity vs. faithfulness without manual penalty-weight tuning.
*Why `softplus(alpha)` instead of `alpha` directly?* Guarantees the penalty multiplier stays non-negative (a valid Lagrange multiplier for an inequality constraint) while remaining smoothly differentiable, instead of clamping a possibly-negative raw value.
*Why flip the gradient sign on `alpha`?* The dual variable must *ascend* on the Lagrangian (increase the penalty when the constraint is violated) while the primal (model/gate) parameters *descend*; PyTorch optimizers only minimize, so negating the gradient converts the built-in minimizer into a maximizer for `alpha` specifically.
*Why clip alpha to [-2, 30]?* Prevents the dual variable from diverging to extreme values that would either zero out the sparsity penalty entirely (alpha too negative) or make training numerically unstable / dominate the loss (alpha too large).

### 9.2. Training loop sketch (`train_graphmask`)
```
Freeze all base GNN and decoder weights (disable_all_gradients)
Initialize gate networks (MultipleInputsLayernormLinear -> Squeezer -> SoftConcrete) per layer
lagrangian = LagrangianOptimization(gate_optimizer, ...)
for epoch in 1..n_epoch:
    original_scores  = model.graphmask_forward(G, pos, neg, graphmask_mode=False)   # baseline, no masking
    masked_scores    = model.graphmask_forward(G, pos, neg, graphmask_mode=True)    # gates active
    divergence = MSE(sigmoid(original_scores), sigmoid(masked_scores))
    penalty    = sparsity penalty (expected number of open gates, layer-wise)
    g = relu(divergence - allowance)      # constraint: divergence must stay below `allowance`
    f = penalty * penalty_scaling
    lagrangian.update(f, g)
    periodically: evaluate_graphmask(...) reports %masked edges per layer and AUROC/AUPRC
                  of masked vs. original predictions on validation/test
```
*Why freeze the base GNN and decoder (`disable_all_gradients`)?* The goal is to explain an *already-trained* model's behavior, not to retrain it; if encoder weights could still move, the "explanation" gates would be optimizing a moving target, and any accuracy gained could come from silently re-training the predictor rather than from finding a faithful, minimal explanation of the original model.
*Why train layer-by-layer / with reversed-layer ordering (see [NETWORK_DESIGN.md §6.2](NETWORK_DESIGN.md#62-lagrangian-optimization))?* Masking decisions at a later GNN layer depend on representations already shaped by earlier layers' masking; optimizing from the last layer backward gives each layer's gate network a more stable target to fit against, improving convergence.

The gate distribution itself (`SoftConcrete`, in [graphmask/sigmoid_penalty.py](../txgnn/graphmask/sigmoid_penalty.py) and [graphmask/hard_concrete.py](../txgnn/graphmask/hard_concrete.py)) is a stretched/rectified concrete (Gumbel-Softmax-like) relaxation of a Bernoulli gate, enabling gradient-based optimization of a discrete "keep/drop this edge" decision — see [NETWORK_DESIGN.md §6](NETWORK_DESIGN.md#6-graphmask-explainability-module) for the formal gate equations.
*Why a continuous relaxation instead of directly optimizing binary gates?* A hard 0/1 gate is non-differentiable, so gradient descent cannot directly learn which edges to drop; the (Hard/Soft) Concrete relaxation gives a differentiable surrogate that can still saturate to near-0/near-1 at convergence, combining trainability with an (approximately) discrete final explanation.

## 10. Model Persistence Algorithms (`TxGNN.save_model` / `load_pretrained`)

**Problem.** Re-deriving disease similarity profiles (§6), re-splitting data, and retraining from scratch every time a model needs to be reused (for demoing, further evaluation, or GraphMask training) is expensive and defeats the purpose of a pretrain/fine-tune pipeline that can take hours.

**Solution.** Serializes/restores: encoder + decoder `state_dict`, the relation-to-index mapping (`rel2idx`), and configuration hyperparameters (`n_hid`, `n_inp`, `n_out`, `proto`, `sim_measure`, etc.) so a checkpoint can be reloaded and immediately used for inference/evaluation/explanation without repeating the pipeline.
*Design rationale.* Persisting the configuration alongside the weights (rather than just the raw `state_dict`) avoids a class of latent bugs where a checkpoint is loaded into a model instantiated with mismatched hyperparameters (e.g., different `n_hid`), which would fail silently or produce garbage predictions rather than a clear error.

## 11. Graph Construction Algorithms (`create_dgl_graph`, `initialize_node_embedding`, [utils.py](../txgnn/utils.py))

**Problem.** DGL's heterogeneous graph API requires edges as per-relation `(src_idx, dst_idx)` integer tensors and an explicit node count per type; the raw split CSVs are relation-mixed pandas DataFrames with arbitrary (sometimes non-contiguous) node indices, and every node needs an initial feature vector since the KG carries no raw node features (drugs/diseases/proteins have no shared numeric attributes across types).

**Solution.**
```
create_dgl_graph(df_train, df):
    for each (src_type, relation, dst_type) triple in df_train:
        collect (x_idx, y_idx) arrays as the edge list for that canonical etype
    num_nodes_dict[node_type] = 1 + max index seen for that type across BOTH df_train and full df
        # uses the full df's max, not just df_train's, so held-out test-only node ids still get
        # allocated a slot in the graph (needed for zero-shot disease nodes with no train edges)
    build dgl.heterograph(...); assign an integer id per edge for bookkeeping

initialize_node_embedding(g, n_inp):
    for each node type: allocate an nn.Parameter of shape (num_nodes, n_inp),
    Xavier-uniform initialize it, store as g.nodes[ntype].data['inp'], requires_grad=False
```
**Design rationale / trade-offs.**
- *Why size node-count arrays using the full `df`, not just `df_train`?* A disease held out entirely for zero-shot testing (§5) must still exist as a valid node in the graph (with an embedding slot) even though it has no training edges — otherwise the model could not produce *any* embedding for it at evaluation time. Sizing off the training data alone would cause an index-out-of-range error the moment a test-only disease is referenced.
- *Why Xavier-uniform, non-learnable (`requires_grad=False`) initial embeddings rather than learned input features?* There are no raw domain features shared across all node types (a protein's "features" are not comparable to a drug's), so a random-but-well-scaled initialization is the standard KG-embedding approach; freezing it forces all task-relevant signal to be learned through the RGCN's relation-specific weight matrices rather than through directly memorizing an arbitrary input vector, which would not generalize to new nodes with a similar structural role but a different random initialization.

## 12. Robustness / Ablation Perturbation Algorithms ([utils.py](../txgnn/utils.py): `remove_random_edges`, `add_random_edges`, `randomize_edges`, `remove_relation_type`)

**Problem.** Claims about a KG-based model (e.g., "the model relies on real biological structure, not just node identity" or "the model is robust to a noisy/incomplete KG") are not verifiable from accuracy numbers on the untouched benchmark alone — such claims need controlled perturbation experiments that measure how performance degrades under KG corruption or ablation of a specific relation type.

**Solution.** A family of deterministic (fixed `np.random.seed(42)`), reproducible graph-corruption utilities:
```
remove_random_edges(g, K%):     # per relation type, delete K% of existing edges uniformly at random
add_random_edges(g, K%):        # per relation type, add K% new random (src,dst) pairs, no dedup check
randomize_edges(g):             # per relation type, replace ALL existing edges with an equal number
                                 # of random ones — a full structure-destroying control
remove_relation_type(g, rel):   # delete every edge of `rel` AND its `rev_` counterpart entirely
```
**Design rationale / trade-offs.**
- *Why a fixed seed (42) inside each function rather than a caller-supplied seed?* Guarantees the exact same corrupted graph is produced across repeated experiment runs and across different callers, which is required for a fair, reproducible ablation comparison (e.g., "compare model A and model B on the *same* 10%-edge-removed graph").
- *Why per-relation-type percentages instead of a single global percentage of all edges?* Relation types differ by orders of magnitude in edge count (protein-protein interactions vastly outnumber drug-disease edges); a global percentage would let corruption concentrate almost entirely in the largest relation types and barely touch the smaller (often more clinically relevant) ones. Per-type percentages ensure every relation is perturbed proportionally.
- *`randomize_edges` vs. `remove_relation_type`:* the former is a structure-preserving-count-but-destroying-topology control (tests whether the model relies on genuine graph structure vs. just degree/count statistics), while the latter is a total-information-removal control for one specific relation (tests how much a particular biological relation type contributes to overall predictive power, e.g., "how much does removing all protein-protein interactions hurt indication prediction?").
- *Known limitation:* `add_random_edges`/`randomize_edges` do not check for accidental duplication of an existing true edge, which is an acceptable approximation at the perturbation percentages typically used (small K relative to graph size) but would not be appropriate for exact counterfactual analysis.

## 13. Disease-Area Split Filtering (`process_disease_area_split`, [utils.py](../txgnn/utils.py))

**Problem.** After generating a clinical disease-area test set (§4/§5) from a curated disease list (e.g., [data/disease_files/cardiovascular.csv](../data/disease_files/cardiovascular.csv)), disease identifiers in that curated file may use different ID conventions or merged/composite IDs than the KG's internal `x_idx`/`y_idx`, and some listed diseases may map to no node actually present in the trained graph (e.g., after other filtering steps).

**Solution.**
```
1. Build id -> idx maps from the KG dataframe for 'disease' nodes (both x and y columns)
2. Also register merged/composite ids (values containing '_') under each individual sub-id,
   so a disease recorded as e.g. "1234_5678" is matched if the disease list references either half
3. Map each disease-list entry to a KG node index via map_node_id_2_idx (returns 'null' if absent)
4. Drop any disease-relation test rows (rev_indication/contraindication/off-label) whose
   disease index isn't in the successfully-mapped disease-area list
```
**Design rationale.** Silently dropping unmappable rows (rather than raising) is a pragmatic choice appropriate for a curated-but-imperfect clinical ontology mapping; the alternative (hard failure) would make disease-area evaluation unusable whenever the disease ontology and the KG's ID scheme disagree on edge cases, which is common in biomedical data integration.

## 14. Two-Hop Neighborhood Retrieval (`find_two_hops`, [utils.py](../txgnn/utils.py))

**Problem.** For qualitative inspection/explanation of a specific prediction (e.g., in the [reproduce/](../reproduce/) analysis notebooks), it's useful to pull the local neighborhood around a node without loading/traversing the full DGL graph object — a lightweight, DataFrame-native lookup is more convenient for ad hoc exploratory analysis.

**Solution.**
```
1. one_hop = all KG rows where the target (idx, type) appears as either endpoint
2. neighbors = the set of (idx, type) pairs seen as the *other* endpoint of those rows
3. two_hop = all KG rows where either endpoint matches one of those 1-hop neighbors
4. Return the deduplicated union of one_hop and two_hop rows
```
**Design rationale.** Operating directly on the flat, human-readable edge DataFrame (rather than the compiled DGL graph) trades computational efficiency for interpretability and ease of use in notebooks — appropriate since this is an exploratory/explanatory tool, not a training-time hot path.

---

## Summary Table

| Algorithm | Location | Problem Addressed | Purpose |
| :--- | :--- | :--- | :--- |
| Minibatch link-prediction pretraining | `TxGNN.pretrain` | Sparse therapeutic labels can't teach broad biological structure alone | Learn general KG structure across all relations |
| Full-batch therapeutic fine-tuning | `TxGNN.finetune` | Generic embeddings must specialize without discarding structural priors | Specialize embeddings for indication/contraindication/off-label prediction |
| Negative sampling (8 variants) | `utils.construct_negative_graph_each_etype` | Uniform-random negatives are too easy / uninformative | Generate contrastive negative triples of varying difficulty |
| KG preprocessing & relation dedup | `utils.preprocess_kg` | Inconsistent IDs and double-counted symmetric edges | Normalize raw KG into directed, indexed edges |
| Disease-area / random / complex-disease / few-edge splits | `utils.create_fold` | Random splits only test interpolation, not zero-shot generalization | Construct train/valid/test partitions for different generalization regimes |
| Disease profile & random-walk similarity | `utils.obtain_disease_profile`, `utils.obtain_protein_random_walk_profile` | GNN can't embed diseases with few/no KG edges | Feature disease similarity for prototype learning |
| Micro/macro AUROC & AUPRC | `utils.get_all_metrics_fb` | Pooled metrics hide poor performance on rare relations | Relation-balanced performance evaluation |
| Disease-centric ranking evaluation | `TxEval.eval_disease_centric` / `disease_centric_evaluation` | Edge-level metrics don't reflect per-disease clinical usefulness | Clinically meaningful per-disease drug ranking assessment |
| Lagrangian GraphMask optimization | `graphmask/lagrangian_optimization.py`, `TxGNN.train_graphmask` | Manual sparsity/faithfulness penalty tuning is brittle | Learn sparse, faithful edge explanations under a divergence constraint |
| Graph construction & node initialization | `utils.create_dgl_graph`, `utils.initialize_node_embedding` | No shared raw features across heterogeneous node types; test-only nodes must still exist | Build a valid DGL heterograph with initial embeddings for all nodes |
| Robustness/ablation edge perturbation | `utils.remove_random_edges`, `add_random_edges`, `randomize_edges`, `remove_relation_type` | Accuracy alone can't verify reliance on real biological structure | Reproducible KG corruption for robustness and ablation studies |
| Disease-area split filtering | `utils.process_disease_area_split` | Curated disease lists may not cleanly map to internal KG indices | Align ontology-derived disease lists with graph node indices |
| Two-hop neighborhood retrieval | `utils.find_two_hops` | Full DGL traversal is overkill for ad hoc, exploratory node inspection | Lightweight DataFrame-native local subgraph extraction |
