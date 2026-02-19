# TODO

## Cognitive Layer Roadmap

1. Add `v2/__init__.py` exports once integration phase begins.
2. Define a stable type-ID contract from `ObjKNN` into `ShortTermMemory`.
3. Add optional adjacency normalization (`D^-1 A`, symmetric, or row-softmax) in `TypeLevelGNN`.
4. Add train/eval flow for `TypeLevelGNN` with explicit loss API.
5. Add snapshot persistence for `LongTermMemory` (save/load `.npz`).
6. Add configurable edge update rule variants in `SleepConsolidation`:
   - directed transitions
   - bidirectional co-activation
   - temporal windowing
7. Add STM eviction policies beyond FIFO (confidence-aware, recency+confidence hybrid).
8. Add sanity-check tests for shape validation and boundary type IDs across all modules.
9. Add integration harness that runs: perception -> STM append -> sleep consolidate -> GNN predict.
10. Plan PyTorch migration points:
    - `TypeLevelGNN` parameter storage and forward pass
    - optional autograd-backed optimization loop
    - device management (CPU/GPU) and dtype policy


# OPEN QUESTIONS

1. Type identity stability:
   - Are `ObjKNN` cluster IDs stable across retraining sessions?
   - If not, do we need a remapping/alignment stage before writing into LTM?

2. Feature provenance:
   - Which exact feature vector should define type centroids in STM records?
   - Raw perception features, reduced embeddings, or post-filter vectors?

3. Sleep schedule:
   - Should consolidation trigger by time interval, STM size threshold, or explicit event?

4. Edge semantics:
   - Should edges represent transition probability, co-occurrence strength, or causal hypotheses?
   - Directed only, or both directed and undirected views?

5. Confidence usage:
   - Is confidence calibrated enough to use directly as consolidation weight?
   - Do we need clipping or temperature scaling before updates?

6. Forgetting behavior:
   - How aggressively should LTM edges and counts decay over long horizons?
   - Should centroids also decay or remain cumulative?

7. GNN objective:
   - Predict next-step type features, denoise current state, or produce planning embeddings?

8. Meta-cognition scope:
   - Should anomaly score influence STM filtering, sleep triggers, or human-readable alerts first?

9. Evaluation:
   - What metrics define success for preliminary cognition modules before full pipeline integration?
   - Candidate metrics: prediction MSE, type-transition perplexity, anomaly precision/recall.

