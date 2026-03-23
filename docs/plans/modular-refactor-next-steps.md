# State-First Post-Backlog Next Steps

## Scope

The modular split and the first state-first PGN backlog are complete. The active follow-up work is now the post-backlog scaling and regime-improvement phase:

- raw `PGN` accepted at the ingestion boundary
- board-state / successor-state supervision used as the primary runtime surface
- transcript tokens retained only as an optional compatibility layer
- policy-conditioned and dual-surface comparison paths shipped in smoke form
- real-PGN runtime checkpoints and transfer smoke results available

## Next Steps

1. Scale the state-first runtime beyond the Hikaru sample.
   - move from the current `72/16` game slice to a materially larger PGN corpus
   - keep held-out state-transition evaluation in place while increasing corpus diversity
   - decide whether the current `210`-token serialization still scales cleanly

2. Improve successor legality and reachability.
   - current state-first evals produce valid boards but `successor_reachable_rate` remains poor
   - use the shipped `:state_action` path to test whether move-conditioned supervision materially improves reachable successor decoding
   - decide whether decoding should stay purely generative or become legality-constrained

3. Turn the richer probe surface into larger real-data supervision.
   - attack structure, mobility, king-pressure, and pinned-piece summaries are shipped
   - move from smoke/probe metrics into larger trained runs
   - add repetition / irreversible-move context if the current serialization saturates too early

4. Re-run hybrid and transfer comparisons from stronger checkpoints.
   - dual-surface and symbolic-transfer comparison harnesses are now shipped
   - rerun them from a checkpoint stronger than the current short Hikaru sample
   - determine whether the state-first core gives a real transfer advantage beyond smoke scale
