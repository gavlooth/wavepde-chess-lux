Base.@kwdef struct ChessModelConfig
    adapter::ChessAdapterConfig = ChessAdapterConfig()
    core::SequenceCoreConfig = WavePDECoreConfig()
    proposer::ChessMoveHeadConfig = ChessMoveHeadConfig()
    max_seq_len::Int = 1536
end

struct ChessModel{A, C, P} <: Lux.AbstractLuxLayer
    config::ChessModelConfig
    adapter::A
    core::C
    proposer::P
end

function validate_chess_model_config(config::ChessModelConfig)
    config.max_seq_len > 0 || throw(ArgumentError("max_seq_len must be positive"))
    config.adapter.d_model == core_d_model(config.core) || throw(ArgumentError("adapter d_model must match core d_model"))
    config.proposer.d_model == core_d_model(config.core) || throw(ArgumentError("proposer d_model must match core d_model"))
    config.proposer.vocab_size == config.adapter.vocab_size || throw(ArgumentError("proposer vocab_size must match adapter vocab_size"))
    return nothing
end

function ChessModel(config::ChessModelConfig)
    validate_chess_model_config(config)
    return ChessModel(
        config,
        ChessInputAdapter(config.adapter),
        build_sequence_core(config.core),
        ChessMoveHead(config.proposer),
    )
end

Lux.initialparameters(rng::AbstractRNG, model::ChessModel) = (
    adapter=Lux.initialparameters(rng, model.adapter),
    core=Lux.initialparameters(rng, model.core),
    proposer=Lux.initialparameters(rng, model.proposer),
)

Lux.initialstates(rng::AbstractRNG, model::ChessModel) = (
    adapter=Lux.initialstates(rng, model.adapter),
    core=Lux.initialstates(rng, model.core),
    proposer=Lux.initialstates(rng, model.proposer),
)

function encode_hidden(model::ChessModel, tokens::AbstractArray{<:Integer}, ps, st)
    size(tokens, 1) > model.config.max_seq_len && throw(ArgumentError(
        "Sequence length $(size(tokens, 1)) exceeds configured max_seq_len $(model.config.max_seq_len).",
    ))
    hidden, st_adapter = model.adapter(tokens, ps.adapter, st.adapter)
    hidden, st_core = model.core(hidden, ps.core, st.core)
    return hidden, (adapter=st_adapter, core=st_core)
end

function (model::ChessModel)(tokens::AbstractArray{<:Integer}, ps, st)
    hidden, core_state = encode_hidden(model, tokens, ps, st)
    logits, st_proposer = proposer_output(model.proposer, hidden, ps.adapter, ps.proposer, st.proposer)
    return logits, (adapter=core_state.adapter, core=core_state.core, proposer=st_proposer)
end
