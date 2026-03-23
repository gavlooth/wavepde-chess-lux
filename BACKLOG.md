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

- [ ] A1. Extract `ChessInputAdapter`
  - current: `TokenEmbedding`
  - target: chess-specific input adapter module
  - result: surface encoding isolated from the reusable core

- [ ] A2. Extract `WavePDECore`
  - current: `WavePDEChessLM` owns blocks directly
  - target: reusable module containing:
    - WavePDE block stack
    - final normalization
    - core config
  - result: backbone can be reused across domains

- [ ] A3. Extract `ChessMoveHead`
  - current: tied logits are embedded in the main model
  - target: standalone proposer head
  - result: output projection becomes swappable

- [ ] A4. Compose top-level chess model from modules
  - target:
    - `ChessInputAdapter`
    - `WavePDECore`
    - `ChessMoveHead`
  - result: main model becomes composition, not one monolith

### EPIC B. Introduce Explicit Head Interfaces

- [ ] B1. Add `AbstractProposerHead`-style interface
  - contract: hidden state -> logits or candidate scores

- [ ] B2. Add `AbstractCheckerHead`-style interface
  - contract: hidden state -> consistency outputs
  - first version can be a scalar or small predicate bundle

- [ ] B3. Add a multi-head composition model
  - target:
    - `ChessInputAdapter`
    - `WavePDECore`
    - `ChessMoveHead`
    - `ChessCheckerHead`

- [ ] B4. Keep proposer and checker independent
  - reason: later domain transfer should replace heads without rewriting the core

### EPIC C. Make The Core Domain-Agnostic

- [ ] C1. Rename chess-specific core assumptions
  - avoid names that imply move-token semantics inside the backbone

- [ ] C2. Define generic core tensor contract
  - input: `(d_model, seq_len, batch)`
  - output: `(d_model, seq_len, batch)`

- [ ] C3. Separate config types
  - `WavePDECoreConfig`
  - `ChessAdapterConfig`
  - `ChessHeadConfig`

- [ ] C4. Remove vocab/output assumptions from the core
  - vocab belongs to adapters/heads, not the backbone

### EPIC D. Add Checker Infrastructure

- [ ] D1. Add first `ChessCheckerHead`
  - initial candidate outputs:
    - legality proxy
    - side-to-move proxy
    - in-check proxy
    - castling-rights proxy
    - scalar consistency score

- [ ] D2. Add checker loss plumbing
  - total loss =
    - proposer loss
    - checker loss
    - optional weighted auxiliary losses

- [ ] D3. Add proposer + checker inference path
  - proposer generates top-k candidates
  - checker reranks or rescales them

- [ ] D4. Add checker metrics
  - legality accuracy
  - calibration
  - rerank win rate vs proposer-only baseline

### EPIC E. Add Latent-State Supervision

- [ ] E1. Add board-derived target extraction
  - targets:
    - side to move
    - in check
    - castling rights
    - legality for sampled candidates
    - material bucket
    - game phase

- [ ] E2. Add probe heads
  - hidden state -> board facts

- [ ] E3. Add transition-consistency targets
  - latent state at `t` + candidate move -> board facts at `t+1`

- [ ] E4. Extend dataset pipeline for auxiliary labels

### EPIC F. Prepare For Domain Transfer

- [ ] F1. Introduce `GenericInputAdapter` interface
  - later implementations:
    - `ChessInputAdapter`
    - `LogicInputAdapter`
    - `LanguageInputAdapter`

- [ ] F2. Introduce `GenericProposerHead` interface
  - later implementations:
    - `ChessMoveHead`
    - `LogicStepHead`
    - `ThoughtTokenHead`

- [ ] F3. Introduce `GenericCheckerHead` interface
  - later implementations:
    - `ChessCheckerHead`
    - `LogicConsistencyHead`
    - `LanguageArgumentChecker`

- [ ] F4. Add freeze/unfreeze policies
  - train adapters only
  - train heads only
  - full fine-tune

### EPIC G. Build Bridge Tasks Before Language

- [ ] G1. Add synthetic symbolic tasks
  - propositional logic
  - entailment
  - contradiction detection
  - simple rule chaining

- [ ] G2. Reuse the same `WavePDECore` across those tasks
  - swap only adapters and heads

- [ ] G3. Compare transfer settings
  - from scratch
  - chess-core init
  - chess-core frozen
  - chess-core fine-tuned

### EPIC H. Repository Restructure

- [ ] H1. Split file layout
  - target:
    - `src/Core/`
    - `src/Adapters/`
    - `src/Heads/`
    - `src/Models/`
    - `src/Training/`

- [ ] H2. Move current monolith into modules
  - current:
    - `src/WavePDEChess.jl`
  - target:
    - `src/Core/WavePDECore.jl`
    - `src/Adapters/ChessInputAdapter.jl`
    - `src/Heads/ChessMoveHead.jl`
    - `src/Models/ChessModel.jl`

- [ ] H3. Add task-specific entrypoints
  - `scripts/train_chess_lm.jl`
  - `scripts/train_chess_checker.jl`
  - `scripts/train_chess_reasoning.jl`

- [ ] H4. Add architecture docs
  - current modular chess model
  - future transferable reasoning architecture

---

## Priority Order

### P0

- [ ] A1. Extract `ChessInputAdapter`
- [ ] A2. Extract `WavePDECore`
- [ ] A3. Extract `ChessMoveHead`
- [ ] A4. Compose modular chess model
- [ ] B1. Add proposer head interface
- [ ] B2. Add checker head interface
- [ ] B3. Add multi-head composition model
- [ ] H1. Split file layout
- [ ] H2. Move current monolith into modules

### P1

- [ ] D1. Add first `ChessCheckerHead`
- [ ] D2. Add checker loss plumbing
- [ ] E1. Add board-derived target extraction
- [ ] E2. Add probe heads
- [ ] H3. Add training entrypoints

### P2

- [ ] D3. Add proposer + checker inference path
- [ ] D4. Add checker metrics
- [ ] E3. Add transition-consistency targets
- [ ] E4. Extend dataset pipeline
- [ ] F1. Add generic input adapter interface
- [ ] F2. Add generic proposer head interface
- [ ] F3. Add generic checker head interface

### P3

- [ ] F4. Add freeze/unfreeze policies
- [ ] G1. Add symbolic bridge tasks
- [ ] G2. Reuse same `WavePDECore` on bridge tasks
- [ ] G3. Run transfer comparisons
- [ ] H4. Add architecture docs

---

## Notes

- The reusable asset is the `WavePDE` backbone, not the chess token shell.
- The first modular target is not a full reasoning model. It is a modular chess model with:
  - isolated input adapter
  - isolated reusable core
  - isolated proposer head
  - newly added checker head
- Later domain transfer should replace adapters and heads first, not the backbone.
