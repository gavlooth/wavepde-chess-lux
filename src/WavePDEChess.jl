module WavePDEChess

"""
WavePDEChess

Chess language models built around reusable Wave-PDE residual cores.

Paper reference:
    `2510.04304v1.pdf` in the local workspace (`Wave-PDE Nets`).

Paper-faithful parts in this implementation:
- wave speed `c(x)` and damping `γ(x)` are produced from learned 1x1 projections
  and constrained with `softplus`
- the Laplacian is applied spectrally with FFTs
- the solver uses a symplectic velocity-Verlet style update with split damping

Intentional divergences from the paper:
- the "spatial" domain is the token axis, so the PDE mixes sequence positions
  rather than feature channels
- `dt` is a learned scalar per layer instead of a fixed hyperparameter
- Wave-PDE is used as a reusable backbone inside modular chess models
"""

using DBInterface
using DuckDB
using FFTW
using Lux
using Optimisers
using PyCall
using Random
using Serialization
using Statistics: mean
using Zygote

export WavePDECoreConfig,
    ChessAdapterConfig,
    ChessMoveHeadConfig,
    ChessCheckerHeadConfig,
    ChessModelConfig,
    ChessMultiHeadModelConfig,
    DualSurfaceStateModelConfig,
    DualSurfaceTrainingConfig,
    WavePDECore,
    AbstractInputAdapter,
    ChessInputAdapter,
    AbstractProposerHead,
    AbstractCheckerHead,
    ChessMoveHead,
    ChessCheckerHead,
    ChessModel,
    ChessMultiHeadModel,
    DualSurfaceStateModel,
    WavePDEConfig,
    WavePDEChessLM,
    TrainingConfig,
    ChessParquetCorpus,
    DualSurfaceParquetCorpus,
    StateTransitionParquetCorpus,
    chess_mamba_11m_config,
    autoregressive_cross_entropy,
    CHESS_BOARD_TARGET_NAMES,
    BOARD_STATE_VOCAB_SIZE,
    BOARD_STATE_SEQUENCE_LENGTH,
    board_fact_metrics,
    board_probe_metrics,
    board_state_tokens,
    board_probe_targets_from_payload,
    board_probe_targets_from_tokens,
    board_probe_targets_from_transcript,
    board_state_tokens_from_transcript,
    candidate_legality_targets,
    candidate_legality_metrics,
    checker_prediction_metrics,
    discover_pgn_files,
    legal_san_candidates_from_transcript,
    decode_chess_tokens,
    ensure_pgn_state_parquet,
    encode_chess_candidate_san,
    encode_chess_transcript,
    append_chess_candidate_san,
    extract_board_targets_from_tokens,
    extract_board_targets_from_transcript,
    parse_pgn_state_examples,
    parse_transcript_parquet_state_examples,
    PolicyTargetBundle,
    policy_legal_candidates,
    policy_labels,
    policy_target_bundle,
    policy_target_index,
    load_state_transition_checkpoint,
    compare_state_transition_training_modes,
    evaluate_state_transition_corpus,
    evaluate_state_transition_checkpoint,
    state_transition_eval_data_dir,
    state_slot_family_metrics,
    successor_legality_metrics,
    rerank_comparison_metrics,
    checker_scalarize,
    input_adapter_output,
    transition_board_targets,
    transcript_state_examples,
    parse_transcript_language_examples,
    ensure_transcript_state_parquet,
    ensure_transcript_language_parquet,
    write_transcript_language_parquet,
    write_transcript_state_parquet,
    write_pgn_state_parquet,
    proposer_topk,
    rerank_next_token_candidates,
    proposer_output,
    checker_output,
    parameter_count,
    sample_batch,
    sample_dual_surface_batch,
    dual_surface_loss,
    train_dual_surface!,
    compare_surface_training_modes,
    train!

include("Core/WavePDECore.jl")
include("Heads/Interfaces.jl")
include("Adapters/ChessInputAdapter.jl")
include("Heads/ChessMoveHead.jl")
include("Heads/ChessCheckerHead.jl")
include("Models/ChessModel.jl")
include("Models/ChessMultiHeadModel.jl")
include("Training/ChessTargets.jl")
include("Training/PGNStateData.jl")
include("Training/PolicyTargets.jl")
include("Training/StateTransitionEval.jl")
include("Training/TranscriptStateData.jl")
include("Training/Training.jl")
include("Training/DualSurfaceTraining.jl")
include("Training/CheckerMetrics.jl")
using .CheckerMetrics:
    board_fact_metrics,
    board_probe_metrics,
    candidate_legality_metrics,
    checker_prediction_metrics,
    state_slot_family_metrics,
    rerank_comparison_metrics

Base.@kwdef struct WavePDEConfig
    vocab_size::Int = 28
    d_model::Int = 288
    n_layer::Int = 20
    max_seq_len::Int = 1536
    solver_steps::Int = 4
    dt_init::Float32 = 0.05f0
    norm_eps::Float32 = 1f-5
    pad_token::Int = 0
end

function ChessModelConfig(config::WavePDEConfig)
    adapter = ChessAdapterConfig(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        pad_token=config.pad_token,
    )
    core = WavePDECoreConfig(
        d_model=config.d_model,
        n_layer=config.n_layer,
        solver_steps=config.solver_steps,
        dt_init=config.dt_init,
        norm_eps=config.norm_eps,
    )
    proposer = ChessMoveHeadConfig(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        tie_embeddings=true,
        bias=false,
    )
    return ChessModelConfig(
        adapter=adapter,
        core=core,
        proposer=proposer,
        max_seq_len=config.max_seq_len,
    )
end

function chess_mamba_11m_config(; vocab_size::Int=28, solver_steps::Int=4, dt_init::Float32=0.05f0)
    return ChessModelConfig(
        adapter=ChessAdapterConfig(vocab_size=vocab_size, d_model=288, pad_token=0),
        core=WavePDECoreConfig(
            d_model=288,
            n_layer=20,
            solver_steps=solver_steps,
            dt_init=dt_init,
            norm_eps=1f-5,
        ),
        proposer=ChessMoveHeadConfig(vocab_size=vocab_size, d_model=288, tie_embeddings=true, bias=false),
        max_seq_len=1536,
    )
end

ChessModel(config::WavePDEConfig) = ChessModel(ChessModelConfig(config))

const WavePDEChessLM = ChessModel

parameter_count(::Nothing) = 0
parameter_count(x::Number) = 1
parameter_count(x::AbstractArray) = length(x)
parameter_count(x::Tuple) = isempty(x) ? 0 : sum(parameter_count, x)
parameter_count(x::NamedTuple) = isempty(values(x)) ? 0 : sum(parameter_count, values(x))

end
