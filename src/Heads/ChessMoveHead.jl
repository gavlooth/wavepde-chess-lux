Base.@kwdef struct ChessMoveHeadConfig
    vocab_size::Int = 28
    d_model::Int = 288
    tie_embeddings::Bool = true
    bias::Bool = false
end

struct ChessMoveHead <: AbstractProposerHead
    config::ChessMoveHeadConfig
end

function Lux.initialparameters(rng::AbstractRNG, head::ChessMoveHead)
    head.config.vocab_size > 0 || throw(ArgumentError("vocab_size must be positive"))
    head.config.d_model > 0 || throw(ArgumentError("d_model must be positive"))
    if head.config.tie_embeddings
        return NamedTuple()
    end
    return (
        weight=glorot_uniform(rng, head.config.vocab_size, head.config.d_model),
        bias=zeros(Float32, head.config.vocab_size),
    )
end

Lux.initialstates(::AbstractRNG, ::ChessMoveHead) = NamedTuple()

function tied_logits(hidden::AbstractArray{T, 3}, embedding_weight::AbstractMatrix{T}) where {T<:AbstractFloat}
    feature_dim, seq_len, batch_size = size(hidden)
    flattened_hidden = reshape(hidden, feature_dim, :)
    flattened_logits = transpose(embedding_weight) * flattened_hidden
    return reshape(flattened_logits, size(embedding_weight, 2), seq_len, batch_size)
end

function untied_logits(hidden::AbstractArray{T, 3}, weight::AbstractMatrix{T}, bias::AbstractVector{T}) where {T<:AbstractFloat}
    _, seq_len, batch_size = size(hidden)
    flattened_hidden = reshape(hidden, size(hidden, 1), :)
    flattened_logits = weight * flattened_hidden .+ reshape(bias, :, 1)
    return reshape(flattened_logits, size(weight, 1), seq_len, batch_size)
end

function proposer_output(head::ChessMoveHead, hidden::AbstractArray{T, 3}, adapter_ps, ps, st) where {T<:AbstractFloat}
    logits = if head.config.tie_embeddings
        tied_logits(hidden, adapter_ps.weight)
    else
        untied_logits(hidden, ps.weight, ps.bias)
    end
    return logits, st
end
