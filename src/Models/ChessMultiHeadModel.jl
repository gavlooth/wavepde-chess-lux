Base.@kwdef struct ChessMultiHeadModelConfig
    adapter::ChessAdapterConfig = ChessAdapterConfig()
    core::WavePDECoreConfig = WavePDECoreConfig()
    proposer::ChessMoveHeadConfig = ChessMoveHeadConfig()
    checker::ChessCheckerHeadConfig = ChessCheckerHeadConfig()
    max_seq_len::Int = 1536
end

struct ChessMultiHeadModel{A, C, P, K} <: Lux.AbstractLuxLayer
    config::ChessMultiHeadModelConfig
    adapter::A
    core::C
    proposer::P
    checker::K
end

function validate_chess_multihead_config(config::ChessMultiHeadModelConfig)
    config.max_seq_len > 0 || throw(ArgumentError("max_seq_len must be positive"))
    config.adapter.d_model == config.core.d_model || throw(ArgumentError("adapter d_model must match core d_model"))
    config.proposer.d_model == config.core.d_model || throw(ArgumentError("proposer d_model must match core d_model"))
    config.checker.d_model == config.core.d_model || throw(ArgumentError("checker d_model must match core d_model"))
    config.proposer.vocab_size == config.adapter.vocab_size || throw(ArgumentError("proposer vocab_size must match adapter vocab_size"))
    return nothing
end

function ChessMultiHeadModel(config::ChessMultiHeadModelConfig)
    validate_chess_multihead_config(config)
    return ChessMultiHeadModel(
        config,
        ChessInputAdapter(config.adapter),
        WavePDECore(config.core),
        ChessMoveHead(config.proposer),
        ChessCheckerHead(config.checker),
    )
end

Lux.initialparameters(rng::AbstractRNG, model::ChessMultiHeadModel) = (
    adapter=Lux.initialparameters(rng, model.adapter),
    core=Lux.initialparameters(rng, model.core),
    proposer=Lux.initialparameters(rng, model.proposer),
    checker=Lux.initialparameters(rng, model.checker),
)

Lux.initialstates(rng::AbstractRNG, model::ChessMultiHeadModel) = (
    adapter=Lux.initialstates(rng, model.adapter),
    core=Lux.initialstates(rng, model.core),
    proposer=Lux.initialstates(rng, model.proposer),
    checker=Lux.initialstates(rng, model.checker),
)

function (model::ChessMultiHeadModel)(tokens::AbstractArray{<:Integer}, ps, st)
    size(tokens, 1) > model.config.max_seq_len && throw(ArgumentError(
        "Sequence length $(size(tokens, 1)) exceeds configured max_seq_len $(model.config.max_seq_len).",
    ))
    hidden, st_adapter = model.adapter(tokens, ps.adapter, st.adapter)
    hidden, st_core = model.core(hidden, ps.core, st.core)
    logits, st_proposer = proposer_output(model.proposer, hidden, ps.adapter, ps.proposer, st.proposer)
    checker_scores, st_checker = checker_output(model.checker, hidden, ps.checker, st.checker)
    return (
        proposer=logits,
        checker=checker_scores,
    ), (
        adapter=st_adapter,
        core=st_core,
        proposer=st_proposer,
        checker=st_checker,
    )
end
