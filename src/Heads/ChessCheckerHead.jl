Base.@kwdef struct ChessCheckerHeadConfig
    d_model::Int = 288
    output_dim::Int = 1
    pooling::Symbol = :mean
end

struct ChessCheckerHead <: AbstractCheckerHead
    config::ChessCheckerHeadConfig
end

function Lux.initialparameters(rng::AbstractRNG, head::ChessCheckerHead)
    head.config.d_model > 0 || throw(ArgumentError("d_model must be positive"))
    head.config.output_dim > 0 || throw(ArgumentError("output_dim must be positive"))
    return (
        weight=glorot_uniform(rng, head.config.output_dim, head.config.d_model),
        bias=zeros(Float32, head.config.output_dim),
    )
end

Lux.initialstates(::AbstractRNG, ::ChessCheckerHead) = NamedTuple()

function pool_hidden(hidden::AbstractArray{T, 3}, pooling::Symbol) where {T<:AbstractFloat}
    if pooling == :mean
        return dropdims(mean(hidden; dims=2), dims=2)
    elseif pooling == :last
        return hidden[:, end, :]
    else
        throw(ArgumentError("Unsupported checker pooling mode: $(pooling)"))
    end
end

function checker_output(head::ChessCheckerHead, hidden::AbstractArray{T, 3}, ps, st) where {T<:AbstractFloat}
    summary = pool_hidden(hidden, head.config.pooling)
    scores = ps.weight * summary .+ reshape(ps.bias, :, 1)
    return scores, st
end
