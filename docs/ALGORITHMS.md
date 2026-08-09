# TxGNN: Implemented Algorithms

This document catalogs the concrete algorithms implemented in the `txgnn` package — training procedures, sampling strategies, data-splitting routines, similarity computations, and explainability optimization — as pseudocode. For the underlying architecture and math notation (RGCN, DistMult, prototype fusion, GraphMask formulation), see [NETWORK_DESIGN.md](NETWORK_DESIGN.md). For module/system layout, see [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md).

---

## 1. Pretraining Algorithm (`TxGNN.pretrain`, [TxGNN.py](../txgnn/TxGNN.py))

Link-prediction pretraining over **all** relation types in the KG using minibatch neighbor sampling.

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
Purpose: learn general-purpose entity embeddings by reconstructing every relation type (protein-protein, drug-drug, disease-phenotype, etc.), giving the encoder a broad structural prior before the therapeutic task is introduced.

## 2. Fine-tuning Algorithm (`TxGNN.finetune`)

Full-batch optimization restricted to the six drug-disease edge types (`indication`, `contraindication`, `off-label use`, and their reverses), with prototype/metric learning active.

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
Model selection uses **macro AUROC on validation** (averaged per relation) rather than micro AUROC, to avoid bias toward the majority relation.

## 3. Negative Sampling Algorithms ([utils.py](../txgnn/utils.py))

Both encoders need corrupted (negative) triples for the BCE contrastive objective. `construct_negative_graph_each_etype` implements several corruption schemes selectable per experiment:

| Method | Strategy |
| :--- | :--- |
| `corrupt_dst` / `corrupt_src` / `corrupt_both` | Uniform random replacement of destination / source / both endpoints. |
| `multinomial_src` / `multinomial_dst` | Sample replacement endpoint proportional to node degree$^{0.75}$ (frequency-based negative sampling, as in word2vec). |
| `inverse_src` / `inverse_dst` | Sample proportional to **negative** degree$^{0.75}$, biasing toward low-degree nodes. |
| `fix_src` / `fix_dst` | Uniformly sample among nodes with nonzero degree for that relation (used by default in pretrain/finetune). |

### 3.1. `Minibatch_NegSampler`
Used during minibatch pretraining; precomputes per-etype sampling weights once (`in_degrees ** 0.75` or a 0/1 mask), then for each minibatch of positive edges repeats the source `k` times and draws destinations via `torch.multinomial`.

### 3.2. `Full_Graph_NegSampler`
Used during full-batch finetuning/evaluation; recomputes one negative graph per epoch across the whole KG, honoring the same weighting schemes.

## 4. Knowledge-Graph Preprocessing (`preprocess_kg`, [utils.py](../txgnn/utils.py))

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

## 5. Data-Splitting Algorithms (`create_fold`, [utils.py](../txgnn/utils.py))

TxGNN supports multiple train/valid/test partitioning strategies selected by `split`:

- **`random`**: stratified random split per relation type (so rare relations still appear in valid/test).
- **`complex_disease`**: splits are made **disease-centric** — diseases (not edges) are shuffled and partitioned into train/valid/test buckets; all drug-disease edges for a disease follow its bucket. This tests generalization to *entire unseen diseases* rather than held-out edges of already-seen diseases.
- **`disease_eval`**: a single target disease (or list) is entirely held out for testing; used for zero-shot case studies.
- **`few_edeges_to_kg`**: diseases with ≤3 total KG connections are routed to the test set — simulates poorly characterized/rare diseases.
- **`few_edeges_to_indications`**: same idea but thresholds on number of known drug indications rather than raw KG degree.
- **`complex_disease_cv`**: 20-fold cross-validation variant of `complex_disease` using `sklearn.model_selection.KFold` over the shuffled disease list.
- **`full_graph`**: 95/5 train/valid split with no true test set, for deployment-oriented "use all data" training.

After partitioning, `reverse_rel_generation` symmetrizes the KG by duplicating each split's edges with `rev_<relation>` in the opposite direction, so the heterogeneous graph has explicit reverse edge types for message passing.

## 6. Disease Similarity Profiling Algorithms ([utils.py](../txgnn/utils.py))

Used by the prototype/metric-learning module (`DistMultPredictor`) to find diseases structurally or semantically similar to a poorly-connected query disease.

### 6.1. `obtain_disease_profile`
```
Input: disease node d, list of relation types R, list of neighbor node types T
For each (relation r, node type t) in zip(R, T):
    neighbors = successors(d, etype=r)
    profile_r = one-hot bit-vector over all nodes of type t, set to 1 at `neighbors`
Return concatenation of all profile_r vectors  # a multi-hot "neighborhood fingerprint"
```
Similarity between two diseases is then cosine similarity (`sim_matrix`) between their concatenated profile vectors.

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
This captures multi-hop functional proximity through the protein-protein interaction subgraph, useful when two diseases share no direct protein annotations but touch the same PPI neighborhood.

### 6.3. `sim_matrix`
Standard batched cosine similarity: L2-normalizes both embedding matrices (with an epsilon floor for numerical stability) and computes the pairwise dot product.

### 6.4. Prototype retrieval
At inference/training, for each query disease the model takes `torch.topk(sim, k+1)` (excluding self-similarity during training, since a disease is trivially most similar to itself) or `torch.topk(sim, k)` during evaluation on unseen diseases, then L1-normalizes the top-k similarity scores to use as convex-combination weights over the neighbors' GNN embeddings (see `agg_measure` fusion strategies in [NETWORK_DESIGN.md](NETWORK_DESIGN.md#43-prototype-fusion-strategies)).

## 7. Evaluation Metrics Algorithm (`get_all_metrics_fb`, [utils.py](../txgnn/utils.py))

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
Macro-averaging prevents the numerically dominant relation (typically `indication`) from masking poor performance on rarer relations like `off-label use`.

## 8. Disease-Centric Evaluation (`TxEval.eval_disease_centric`, [TxEval.py](../txgnn/TxEval.py))

```
1. Determine target disease indices:
   - explicit list, OR
   - 'test_set': all diseases present in the held-out test split for the relation
2. For each disease: rank all candidate drugs by predicted score for the
   chosen relation, compare against ground-truth indications/contraindications
3. Optionally simulate a random baseline ranking for calibration comparison
4. Aggregate ranking-quality metrics across diseases and optionally persist
   the raw results (pickle) for downstream plotting (see reproduce/ notebooks)
```
This complements the edge-level AUROC/AUPRC by measuring how useful the model's ranked drug list is *per disease* — the practical, clinician-facing use case.

## 9. GraphMask Training Algorithm ([TxGNN.py](../txgnn/TxGNN.py) `train_graphmask`, [graphmask/lagrangian_optimization.py](../txgnn/graphmask/lagrangian_optimization.py))

GraphMask learns per-layer edge gates that erase as many messages as possible while keeping predictions close to the unmasked model, solved via **constrained (Lagrangian) optimization** rather than a fixed-weight penalty term.

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
The gate distribution itself (`SoftConcrete`, in [graphmask/sigmoid_penalty.py](../txgnn/graphmask/sigmoid_penalty.py) and [graphmask/hard_concrete.py](../txgnn/graphmask/hard_concrete.py)) is a stretched/rectified concrete (Gumbel-Softmax-like) relaxation of a Bernoulli gate, enabling gradient-based optimization of a discrete "keep/drop this edge" decision — see [NETWORK_DESIGN.md §6](NETWORK_DESIGN.md#6-graphmask-explainability-module) for the formal gate equations.

## 10. Model Persistence Algorithms (`TxGNN.save_model` / `load_pretrained`)

Serializes/restores: encoder + decoder `state_dict`, the relation-to-index mapping (`rel2idx`), and configuration hyperparameters (`n_hid`, `n_inp`, `n_out`, `proto`, `sim_measure`, etc.) so a checkpoint can be reloaded without re-deriving disease similarity profiles from scratch.

---

## Summary Table

| Algorithm | Location | Purpose |
| :--- | :--- | :--- |
| Minibatch link-prediction pretraining | `TxGNN.pretrain` | Learn general KG structure across all relations |
| Full-batch therapeutic fine-tuning | `TxGNN.finetune` | Specialize embeddings for indication/contraindication/off-label prediction |
| Negative sampling (8 variants) | `utils.construct_negative_graph_each_etype` | Generate contrastive negative triples |
| KG preprocessing & relation dedup | `utils.preprocess_kg` | Normalize raw KG into directed, indexed edges |
| Disease-area / random / complex-disease / few-edge splits | `utils.create_fold` | Construct train/valid/test partitions for different generalization regimes |
| Disease profile & random-walk similarity | `utils.obtain_disease_profile`, `utils.obtain_protein_random_walk_profile` | Feature disease similarity for prototype learning |
| Micro/macro AUROC & AUPRC | `utils.get_all_metrics_fb` | Relation-balanced performance evaluation |
| Disease-centric ranking evaluation | `TxEval.eval_disease_centric` | Clinically meaningful per-disease drug ranking assessment |
| Lagrangian GraphMask optimization | `graphmask/lagrangian_optimization.py`, `TxGNN.train_graphmask` | Learn sparse, faithful edge explanations under a divergence constraint |
