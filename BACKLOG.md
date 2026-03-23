# State-First PGN Architecture Backlog

Purpose:

- accept raw `PGN` as a first-class ingestion format
- stop treating chess notation as the primary training surface
- train the `WavePDE` backbone on structured board-state and transition supervision
- keep transcript generation only as an optional compatibility surface

---

## Current Architectural Direction

```text
[PGN files]
     |
     v
+---------------------------+
| PGN ingestion             |
| per-ply move parsing      |
+---------------------------+
     |
     v
+---------------------------+
| board-state serialization |
| state_t / state_t+1       |
+---------------------------+
     |
     v
+---------------------------+
| InputAdapter              |
| state-token embeddings    |
+---------------------------+
     |
     v
+---------------------------+
| WavePDECore               |
| reusable backbone         |
+---------------------------+
     |
     +----------------------------+
     |                            |
     v                            v
+--------------------+   +----------------------+
| transition/policy  |   | probe bundle         |
| successor targets  |   | facts / legality /   |
| legal-move scoring |   | tactical abstractions|
+--------------------+   +----------------------+
```

Shipped first slice:

```text
[PGN files]
     |
     v
+---------------------------+
| PGN -> parquet builder    |
| state_tokens             |
| next_state_tokens        |
+---------------------------+
     |
     v
+---------------------------+
| state-transition corpus   |
+---------------------------+
     |
     v
+---------------------------+
| WavePDE chess model       |
| paired token supervision  |
| state_t -> state_t+1      |
+---------------------------+
```

---

## Epics

### EPIC A. PGN Ingestion Boundary

- [x] A1. Discover raw `.pgn` files from a file or directory source
  - shipped: recursive PGN discovery and validation

- [x] A2. Parse PGN into per-ply state-transition examples
  - shipped:
    - mainline move parsing
    - SAN move recording
    - per-ply `state_t` and `state_t+1` extraction

- [x] A3. Write derived state-transition parquet
  - shipped:
    - `state_tokens`
    - `next_state_tokens`
    - `move_san`
    - `ply`
    - `transcript`

### EPIC B. Board-State Serialization Contract

- [x] B1. Add a fixed-length board-state token serialization
  - shipped:
    - 64 square occupancy slots
    - side to move
    - castling rights
    - en passant file
    - halfmove/fullmove buckets

- [x] B2. Add transcript -> board-state extraction
  - shipped:
    - transcript-derived board-state tokens for the current position

- [x] B3. Expand the state payload beyond coarse board facts
  - shipped:
    - white/black attack maps
    - in-check and pinned-piece summaries
    - king-pressure summaries
    - mobility and attacked-piece-count summaries
  - residual:
    - repetition and irreversible-move context remain future extensions

### EPIC C. State-First Training Path

- [x] C1. Add a state-transition parquet corpus
  - shipped:
    - `StateTransitionParquetCorpus`
    - file rotation and batch sampling

- [x] C2. Add paired proposer supervision for `state_t -> state_t+1`
  - shipped:
    - paired token loss on aligned state sequences
    - reusable through the existing `train!` entrypoint

- [x] C3. Add a task-specific state-transition entrypoint
  - shipped: `scripts/train_chess_state_transition.jl`

- [x] C4. Add an evaluation harness for state-transition checkpoints
  - shipped:
    - held-out paired state-transition loss
    - exact slot and exact sequence accuracy
    - board-fact recovery metrics for predicted successor states

### EPIC D. Policy / Action Modeling

- [x] D1. Add a legal-move policy target
  - shipped:
    - transcript-driven legal candidate generation
    - explicit state-context candidate normalization
    - one-hot policy labels over the legal set
    - strict missing-target validation for training-time use

- [x] D2. Split successor-state prediction from action selection
  - shipped:
    - `policy_condition_mode=:state_only` keeps direct `state_t -> state_t+1`
    - `policy_condition_mode=:state_action` trains `(state_t, move) -> state_t+1`
    - masked paired loss keeps action-conditioning tokens out of successor targets

- [x] D3. Compare next-state-only vs policy-conditioned successor training
  - shipped:
    - `compare_state_transition_training_modes(...)`
    - `scripts/compare_chess_state_policy_modes.jl`
    - smoke-tested comparison between `:state_only` and `:state_action`

### EPIC E. Probe Bundle For Abstraction Capture

- [x] E1. Expand the checker/probe bundle beyond the current board-fact set
  - shipped:
    - white/black attack maps
    - in-check indicators
    - pinned-piece counts
    - king-pressure summaries
    - mobility summaries
    - attacked-piece-count summaries

- [x] E2. Decide whether transition supervision should keep sharing the checker head
  - shipped decision:
    - the transcript-side transition-consistency path remains a compatibility-layer use of the shared checker head
    - the primary state-first path no longer depends on checker-head sharing for transition learning
    - future transition expansion should happen on the state/policy path rather than by growing the shared transcript checker surface

- [x] E3. Add legality-aware abstraction metrics beyond the shipped probe/eval surface
  - shipped:
    - probe metrics by concept family through `board_probe_metrics`
    - state-transition exact slot and exact sequence accuracy through the evaluation harness
    - board-fact recovery metrics for predicted successor states
    - slot-family breakdowns across coarse state, attack maps, and pressure/count fields
    - move legality after successor decoding

### EPIC F. Dual-Surface Compatibility

- [x] F1. Keep transcript LM as an optional auxiliary objective, not the primary objective
  - shipped:
    - `DualSurfaceStateModel` keeps state-transition loss primary and transcript loss auxiliary

- [x] F2. Add transcript reconstruction or move-text generation from state-derived latents
  - shipped:
    - auxiliary `move_san` text prediction from state-derived latents in the dual-surface path

- [x] F3. Compare transcript-first, state-first, and hybrid training regimes
  - shipped:
    - `compare_surface_training_modes(...)`
    - `scripts/compare_chess_surface_modes.jl`
    - smoke-tested three-way comparison harness

### EPIC G. Runtime Experiments

- [x] G1. Run state-transition training on real PGN-derived data beyond toy tests
  - shipped:
    - real-PGN Hikaru archive runtime slice with `72` training games and `16` held-out games
    - best state-first checkpoint: `tmp/state_first_runtime/checkpoints/state_first_checkpoint.jls`

- [x] G2. Compare PGN-derived state training against the current transcript-token baseline
  - shipped:
    - matched runtime comparison between state-first and transcript-token baselines on the Hikaru-derived split
    - state-first held-out loss is decisively lower on its evaluation surface than the transcript baseline is on the tokenized split

- [x] G3. Re-run symbolic transfer experiments from a meaningful state-first checkpoint
  - shipped:
    - symbolic transfer rerun from the real-PGN state-first checkpoint
    - fine-tuned transfer beat both scratch and frozen-core on the smoke comparison

---

## Priority Order

### P0

- Completed.

### P1

- Completed.

### P2

- Completed.

### P3

- Move to larger-scale state-first experiments and next-phase architecture work tracked in `docs/plans/modular-refactor-next-steps.md`.

---

## Notes

- The modular backlog that produced the reusable `WavePDECore`, adapters, heads, checker path, symbolic bridge tasks, and transfer harness is complete and no longer the active source of open work.
- The active bottleneck is representational: transcript notation is still too surface-level for maximal abstraction capture.
- The state-first PGN backlog is closed. Follow-on work now lives in `docs/plans/modular-refactor-next-steps.md` as the next-phase experimental plan rather than as open backlog debt.
