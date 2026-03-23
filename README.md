# WavePDE Chess Lux

Julia/Lux implementation of a chess language model that replaces the Mamba
backbone with Wave-PDE blocks inspired by the Wave-PDE Nets paper.

## Contents

- `src/WavePDEChess.jl`: model, DuckDB-backed parquet loader, and training loop
- `scripts/train_chess_wavepde.jl`: entrypoint for training
- `test/runtests.jl`: smoke tests for the model and a one-step training run

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
