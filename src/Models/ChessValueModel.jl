Base.@kwdef struct BoardValueModelConfig
    adapter::ChessAdapterConfig = ChessAdapterConfig(vocab_size=58, d_model=288, pad_token=0)
    core::SequenceCoreConfig = WavePDECoreConfig()
    value_head::ChessCheckerHeadConfig = ChessCheckerHeadConfig(d_model=288, output_dim=1, pooling=:mean)
    max_seq_len::Int = 210
end

struct BoardValueModel{A, C, V} <: Lux.AbstractLuxLayer
    config::BoardValueModelConfig
    adapter::A
    core::C
    value_head::V
end

function validate_board_value_model_config(config::BoardValueModelConfig)
    config.max_seq_len > 0 || throw(ArgumentError("max_seq_len must be positive"))
    config.adapter.d_model == core_d_model(config.core) || throw(ArgumentError("adapter d_model must match core d_model"))
    config.value_head.d_model == core_d_model(config.core) || throw(ArgumentError("value_head d_model must match core d_model"))
    config.value_head.output_dim == 1 || throw(ArgumentError("value_head output_dim must be 1 for scalar board-value prediction"))
    return nothing
end

function BoardValueModel(config::BoardValueModelConfig)
    validate_board_value_model_config(config)
    return BoardValueModel(
        config,
        ChessInputAdapter(config.adapter),
        build_sequence_core(config.core),
        ChessCheckerHead(config.value_head),
    )
end

Lux.initialparameters(rng::AbstractRNG, model::BoardValueModel) = (
    adapter=Lux.initialparameters(rng, model.adapter),
    core=Lux.initialparameters(rng, model.core),
    value_head=Lux.initialparameters(rng, model.value_head),
)

Lux.initialstates(rng::AbstractRNG, model::BoardValueModel) = (
    adapter=Lux.initialstates(rng, model.adapter),
    core=Lux.initialstates(rng, model.core),
    value_head=Lux.initialstates(rng, model.value_head),
)

function encode_hidden(model::BoardValueModel, tokens::AbstractArray{<:Integer}, ps, st)
    size(tokens, 1) > model.config.max_seq_len && throw(ArgumentError(
        "Sequence length $(size(tokens, 1)) exceeds configured max_seq_len $(model.config.max_seq_len).",
    ))
    hidden, st_adapter = model.adapter(tokens, ps.adapter, st.adapter)
    hidden, st_core = model.core(hidden, ps.core, st.core)
    return hidden, (adapter=st_adapter, core=st_core)
end

function (model::BoardValueModel)(tokens::AbstractArray{<:Integer}, ps, st)
    hidden, core_state = encode_hidden(model, tokens, ps, st)
    values, st_value = checker_output(model.value_head, hidden, ps.value_head, st.value_head)
    return values, (adapter=core_state.adapter, core=core_state.core, value_head=st_value)
end
