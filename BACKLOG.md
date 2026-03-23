# Modular Refactor Backlog

Purpose:

- separate the reusable `WavePDE` backbone from chess-specific surface logic
- add proposer/checker structure without throwing away the current chess-trained core
- prepare the repo for later domain transfer into logic/language reasoning tasks

---

## Target Architecture

```text
[Domain Tokens]
      |
      v
+----------------------+
| InputAdapter         |
| chess / logic / lang |
+----------------------+
      |
      v
+----------------------+
| WavePDECore          |
| reusable backbone    |
+----------------------+
      |
      +-----------------------+
      |                       |
      v                       v
+------------------+   +------------------+
| ProposerHead     |   | CheckerHead      |
| domain-specific  |   | domain-specific  |
+------------------+   +------------------+
```

Near-term chess version:

```text
[Chess tokens]
      |
      v
+------------------+
| ChessInputAdapter|
+------------------+
      |
      v
+------------------+
| WavePDECore      |
+------------------+
      |
      +----------------------+
      |                      |
      v                      v
+------------------+   +------------------+
| ChessMoveHead    |   | ChessCheckerHead |
+------------------+   +------------------+
```

---

## Epics

### EPIC A. Split Current Chess Model Into Modules

- [x] A1. Extract `ChessInputAdapter`
  - current: `TokenEmbedding`
  - target: chess-specific input adapter module
  - result: surface encoding isolated from the reusable core

- [x] A2. Extract `WavePDECore`
  - current: `WavePDEChessLM` owns blocks directly
  - target: reusable module containing:
    - WavePDE block stack
    - final normalization
    - core config
  - result: backbone can be reused across domains

- [x] A3. Extract `ChessMoveHead`
  - current: tied logits are embedded in the main model
  - target: standalone proposer head
  - result: output projection becomes swappable

- [x] A4. Compose top-level chess model from modules
  - target:
    - `ChessInputAdapter`
    - `WavePDECore`
    - `ChessMoveHead`
  - result: main model becomes composition, not one monolith

### EPIC B. Introduce Explicit Head Interfaces

- [x] B1. Add `AbstractProposerHead`-style interface
  - contract: hidden state -> logits or candidate scores

- [x] B2. Add `AbstractCheckerHead`-style interface
  - contract: hidden state -> consistency outputs
  - first version can be a scalar or small predicate bundle

- [x] B3. Add a multi-head composition model
  - target:
    - `ChessInputAdapter`
    - `WavePDECore`
    - `ChessMoveHead`
    - `ChessCheckerHead`

- [x] B4. Keep proposer and checker independent
  - reason: later domain transfer should replace heads without rewriting the core

### EPIC C. Make The Core Domain-Agnostic

- [x] C1. Rename chess-specific core assumptions at the reusable boundary
  - shipped:
    - reusable `AbstractInputAdapter` / `input_adapter_output` contract
    - adapter-facing error text and interface path no longer hardcode chess-only assumptions

- [x] C2. Define generic core tensor contract
  - input: `(d_model, seq_len, batch)`
  - output: `(d_model, seq_len, batch)`

- [x] C3. Separate config types
  - `WavePDECoreConfig`
  - `ChessAdapterConfig`
  - `ChessHeadConfig`

- [x] C4. Remove vocab/output assumptions from the core
  - vocab belongs to adapters/heads, not the backbone

### EPIC D. Add Checker Infrastructure

- [x] D1. Add first `ChessCheckerHead`
  - shipped: pooled checker head with scalar/vector score output
  - remaining richer probe outputs belong in later checker/probe work

- [x] D2. Add checker loss plumbing
  - shipped:
    - proposer-only training still works for `ChessModel`
    - `ChessMultiHeadModel` uses composite proposer + checker loss when checker targets are present
    - parquet loader accepts optional checker supervision columns

- [x] D3. Add proposer + checker inference path
  - shipped:
    - proposer generates top-k candidates
    - checker reranks candidates on appended-token contexts through the existing checker head

- [x] D4. Add checker metrics
  - shipped:
    - generic checker prediction error metrics
    - rerank-vs-proposer comparison metrics
    - board-fact classification metrics with accuracy, exact-match, and Brier score
    - candidate-legality metrics for board-derived legality labels

### EPIC E. Add Latent-State Supervision

- [x] E1. Add board-derived target extraction
  - shipped:
    - transcript normalization plus 28-token chess transcript encoding/decoding
    - side to move, in-check, castling-rights, material-bucket, and game-phase targets
    - legality labels for sampled SAN candidates

- [x] E2. Add probe heads
  - shipped:
    - vector-valued `ChessCheckerHead` now serves as the first board-fact probe bundle

- [x] E3. Integrate transition-consistency targets into training
  - shipped:
    - candidate-move transition target extraction for board facts at `t+1`
    - appended-candidate transition contexts in training batches
    - optional transition checker loss for transcript-derived training

- [x] E4. Extend dataset pipeline for auxiliary labels
  - shipped:
    - `ChessParquetCorpus(...; board_target_mode=:transcript_board_facts)` derives tokenized transcripts and board-fact checker targets directly from parquet transcript columns

### EPIC F. Prepare For Domain Transfer

- [x] F1. Introduce `GenericInputAdapter` interface
  - later implementations:
    - `ChessInputAdapter`
    - `LogicInputAdapter`
    - `LanguageInputAdapter`

- [x] F2. Introduce `GenericProposerHead` interface
  - later implementations:
    - `ChessMoveHead`
    - `LogicStepHead`
    - `ThoughtTokenHead`

- [x] F3. Introduce `GenericCheckerHead` interface
  - later implementations:
    - `ChessCheckerHead`
    - `LogicConsistencyHead`
    - `LanguageArgumentChecker`

- [x] F4. Add freeze/unfreeze policies
  - shipped:
    - train adapters only
    - train heads only
    - full fine-tune

### EPIC G. Build Bridge Tasks Before Language

- [x] G1. Add synthetic symbolic tasks
  - shipped:
    - propositional logic
    - entailment
    - contradiction detection
    - simple rule chaining

- [x] G2. Reuse the same `WavePDECore` across those tasks
  - shipped:
    - symbolic bridge training path reuses the existing modular model stack and `WavePDECore`

- [x] G3. Compare transfer settings
  - shipped:
    - scratch full training
    - chess-core init with frozen core
    - chess-core init with fine-tuning

### EPIC H. Repository Restructure

- [x] H1. Split file layout
  - target:
    - `src/Core/`
    - `src/Adapters/`
    - `src/Heads/`
    - `src/Models/`
    - `src/Training/`

- [x] H2. Move current monolith into modules
  - current:
    - `src/WavePDEChess.jl`
  - target:
    - `src/Core/WavePDECore.jl`
    - `src/Adapters/ChessInputAdapter.jl`
    - `src/Heads/ChessMoveHead.jl`
    - `src/Models/ChessModel.jl`

- [x] H3. Add task-specific entrypoints
  - shipped: `scripts/train_chess_lm.jl`
  - shipped: `scripts/train_chess_checker.jl`

- [x] H5. Add reasoning-oriented training entrypoint
  - shipped: `scripts/train_chess_reasoning.jl`

- [x] H4. Add architecture docs
  - current modular chess model
  - future transferable reasoning architecture

---

## Priority Order

### P0

- Completed foundational modular split and multi-head composition.

### P1

- Completed checker supervision, board-target extraction, transcript-derived auxiliary labels, and training entrypoints.

### P2

- Completed transition-consistency supervision, symbolic bridge tasks, core reuse, and transfer-comparison scaffolding.

### P3

- Solver-intent decision and longer-horizon transfer work after P2 is landed.

---

## Notes

- The reusable asset is the `WavePDE` backbone, not the chess token shell.
- The first modular target is not a full reasoning model. It is a modular chess model with:
  - isolated input adapter
  - isolated reusable core
  - isolated proposer head
  - newly added checker head
- Later domain transfer should replace adapters and heads first, not the backbone.
