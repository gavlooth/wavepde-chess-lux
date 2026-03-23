# Modular Refactor Next Steps

## Scope

This file records post-backlog follow-up work after the shipped modular split, board-target supervision path, symbolic bridge surface, and transfer-comparison harness.

## Next Steps

1. Run empirical symbolic-transfer experiments with a meaningful chess checkpoint.
   - compare scratch vs transplanted-core settings beyond single-step smoke runs
   - record loss curves and checkpoint recommendations

2. Decide whether transition supervision should stay in the current checker head or split into a dedicated transition probe.
   - current shipped path shares the checker head
   - a separate probe bundle may be cleaner if transition targets grow

3. Separate config surfaces more fully.
   - keep `WavePDECoreConfig`, `ChessAdapterConfig`, `ChessMoveHeadConfig`, and `ChessCheckerHeadConfig`
   - reduce dependence on the compatibility wrapper `WavePDEConfig` over time

4. Decide and document solver intent.
   - either keep the current split-damping integrator as an intentional variant
   - or revert to the paper-stated additive damping update if strict paper reproduction is the goal
