# WavePDE Chess Lux

Julia/Lux implementation of a chess language model that replaces the Mamba
backbone with Wave-PDE blocks inspired by the Wave-PDE Nets paper.

Paper reference in the local workspace:

- `2510.04304v1.pdf`

## Contents

- `src/WavePDEChess.jl`: top-level module facade and compatibility layer
- `src/Core/`: reusable `WavePDECore` backbone
- `src/Adapters/`: chess-specific input adapter
- `src/Heads/`: proposer/checker head interfaces and chess heads
- `src/Models/`: proposer-only and multi-head chess model composition
- `src/Training/`: DuckDB-backed parquet loader and training loop
- `scripts/train_chess_wavepde.jl`: entrypoint for training
- `test/runtests.jl`: smoke tests for the model and a one-step training run

## Current Modular Layout

Near-term chess model:

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

## Relation To The Paper

Paper-faithful choices:

- `c(x)` and `γ(x)` are produced by learned 1x1 projections and constrained with `softplus`
- the Laplacian is computed spectrally with FFTs
- the solver now uses a velocity-Verlet style split-damping update instead of the simpler undamped-half-step variant

Intentional divergences:

- the paper treats the PDE over a generic spatial domain; this code maps that domain to token positions, so WavePDE mixes across the sequence axis
- `dt` is learned per layer instead of remaining a fixed hyperparameter
- the paper studies Wave-PDE as a general architecture family, while this repo uses it as the primary language-model backbone for chess move sequences

Practical improvement over the earlier repo version:

- the mixer now uses a split damping factor `exp(-γ dt / 2)` inside the leapfrog update, matching the paper's symplectic integration story more closely
- the code now warns when the learned `dt` and wave speed imply a rough `dt * max(c) >= 1` stability violation
- the model constructors and mixer input path now fail loudly on invalid dimensions instead of assuming everything is well-formed

## Training

Point `CHESS_DATA_DIR` at a directory containing parquet files with a `tokenized`
column of integer move sequences.

```bash
CHESS_DATA_DIR=/path/to/parquet julia --project=. scripts/train_chess_wavepde.jl
```

Useful environment variables:

- `WAVEPDE_MAX_ITERS`
- `WAVEPDE_BATCH_SIZE`
- `WAVEPDE_LR`
- `WAVEPDE_MIN_TOKENS`
- `WAVEPDE_CHECKPOINT`

The default model preset matches the chess Mamba 11M configuration for sequence
length, width, and depth, while swapping the mixer for a spectral Wave-PDE
layer stack.
