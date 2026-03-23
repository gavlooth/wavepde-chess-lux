# Modular Refactor Next Steps

## Scope

This file records known deferred work after the shipped P0 modular split.

## Next Steps

1. Implement checker loss plumbing in `src/Training/Training.jl`.
   - Add a composite loss path for proposer + checker outputs.
   - Keep proposer-only training working as a simpler path.

2. Add checker-aware inference in `src/Models/ChessMultiHeadModel.jl` or a dedicated inference module.
   - proposer generates candidate logits or top-k moves
   - checker rescales or reranks candidates

3. Separate config surfaces more fully.
   - keep `WavePDECoreConfig`, `ChessAdapterConfig`, `ChessMoveHeadConfig`, and `ChessCheckerHeadConfig`
   - reduce dependence on the compatibility wrapper `WavePDEConfig` over time

4. Decide and document solver intent.
   - either keep the current split-damping integrator as an intentional variant
   - or revert to the paper-stated additive damping update if strict paper reproduction is the goal

5. Add task-specific training entrypoints.
   - `train_chess_lm.jl`
   - `train_chess_checker.jl`
   - later, combined proposer/checker or reasoning entrypoints
