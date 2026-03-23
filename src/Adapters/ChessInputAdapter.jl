Base.@kwdef struct ChessAdapterConfig
    vocab_size::Int = 28
    d_model::Int = 288
    pad_token::Int = 0
end

struct ChessInputAdapter <: AbstractInputAdapter
    config::ChessAdapterConfig
end

function Lux.initialparameters(rng::AbstractRNG, adapter::ChessInputAdapter)
    adapter.config.vocab_size > 0 || throw(ArgumentError("vocab_size must be positive"))
    adapter.config.d_model > 0 || throw(ArgumentError("d_model must be positive"))
    return (weight=glorot_uniform(rng, adapter.config.d_model, adapter.config.vocab_size),)
end

Lux.initialstates(::AbstractRNG, ::ChessInputAdapter) = NamedTuple()

function input_adapter_output(adapter::ChessInputAdapter, tokens::AbstractArray{<:Integer}, ps, st)
    ndims(tokens) == 2 || throw(ArgumentError(
        "Input adapter expects tokens with shape (seq_len, batch), got $(size(tokens)).",
    ))
    if any(tokens .< 0) || any(tokens .>= adapter.config.vocab_size)
        throw(ArgumentError("Token ids must be in [0, $(adapter.config.vocab_size - 1)]."))
    end
    token_idx = Int.(tokens) .+ 1
    embeddings = reshape(ps.weight[:, vec(token_idx)], adapter.config.d_model, size(tokens)...)
    return embeddings, st
end
