module DispersivePDE

using FFTW
using Lux
using Optimisers
using Random
using Statistics: mean
using Zygote

export DispersionConfig,
    DispersiveBlock,
    DispersiveStack,
    frequency_grid,
    make_identity_dataset,
    train!

Base.@kwdef struct DispersionConfig
    channels::Int
    n_layer::Int = 4
    length::Int
    dt_init::Float32 = 0.05f0
    dt_floor::Float32 = 1f-4
    alpha_init::Float32 = 0.01f0
    alpha_floor::Float32 = 1f-4
    beta_init::Float32 = 0.01f0
    phase_limit::Float32 = Float32(pi)
    decay_limit::Float32 = 2.0f0
    residual_init_scale::Float32 = 0.1f0
    norm_eps::Float32 = 1f-5
    stability_eps::Float32 = 1f-6
    warn_on_clamp::Bool = true
end

@inline softplus_scalar(x::T) where {T<:Real} = log1p(exp(-abs(x))) + max(x, zero(T))
@inline inverse_softplus(x::T) where {T<:Real} = x + log(-expm1(-x))
@inline sigmoid_scalar(x::T) where {T<:Real} = inv(one(x) + exp(-x))

function logit_scalar(p::T) where {T<:Real}
    zero(T) < p < one(T) || throw(ArgumentError("logit input must lie in (0, 1), got $(p)."))
    return log(p / (one(T) - p))
end

function frequency_grid(seq_len::Int)
    seq_len > 0 || throw(ArgumentError("sequence length must be positive"))
    half = fld(seq_len, 2)
    raw_modes = if iseven(seq_len)
        vcat(collect(0:(half - 1)), collect(-half:-1))
    else
        vcat(collect(0:half), collect(-half:-1))
    end
    return (2f0 * Float32(pi) / Float32(seq_len)) .* Float32.(raw_modes)
end

function glorot_uniform(rng::AbstractRNG, out_dim::Int, in_dim::Int)
    limit = sqrt(6.0f0 / Float32(in_dim + out_dim))
    return rand(rng, Float32, out_dim, in_dim) .* (2.0f0 * limit) .- limit
end

function linear1x1(x::AbstractArray, weight::AbstractMatrix, bias::AbstractVector)
    size(x, 1) == size(weight, 2) || throw(ArgumentError(
        "expected feature dimension $(size(weight, 2)), got $(size(x, 1))",
    ))
    trailing_dims = size(x)[2:end]
    y = reshape(weight * reshape(x, size(x, 1), :), size(weight, 1), trailing_dims...)
    bias_shape = (length(bias), ntuple(_ -> 1, ndims(x) - 1)...)
    return y .+ reshape(bias, bias_shape...)
end

struct RMSNormLayer{T} <: Lux.AbstractLuxLayer
    channels::Int
    eps::T
end

Lux.initialparameters(::AbstractRNG, layer::RMSNormLayer) = (scale=ones(Float32, layer.channels),)
Lux.initialstates(::AbstractRNG, ::RMSNormLayer) = NamedTuple()

function (layer::RMSNormLayer)(x::AbstractArray{T, 3}, ps, st) where {T<:AbstractFloat}
    size(x, 1) == layer.channels || throw(ArgumentError(
        "RMSNormLayer expected $(layer.channels) channels, got $(size(x, 1))",
    ))
    rms = sqrt.(mean(abs2, x; dims=1) .+ layer.eps)
    scale = reshape(ps.scale, :, 1, 1)
    return x .* (scale ./ rms), st
end

struct DispersiveBlock{N} <: Lux.AbstractLuxLayer
    channels::Int
    length::Int
    dt_init::Float32
    dt_floor::Float32
    alpha_init::Float32
    alpha_floor::Float32
    beta_init::Float32
    phase_limit::Float32
    decay_limit::Float32
    residual_init_scale::Float32
    stability_eps::Float32
    warn_on_clamp::Bool
    norm::N
    k::Vector{Float32}
end

function DispersiveBlock(cfg::DispersionConfig)
    cfg.channels > 0 || throw(ArgumentError("channels must be positive"))
    cfg.n_layer >= 0 || throw(ArgumentError("n_layer must be non-negative"))
    cfg.length > 0 || throw(ArgumentError("length must be positive"))
    cfg.dt_init > 0 || throw(ArgumentError("dt_init must be positive"))
    cfg.dt_floor >= 0 || throw(ArgumentError("dt_floor must be non-negative"))
    cfg.alpha_floor >= 0 || throw(ArgumentError("alpha_floor must be non-negative"))
    cfg.phase_limit > 0 || throw(ArgumentError("phase_limit must be positive"))
    cfg.decay_limit > 0 || throw(ArgumentError("decay_limit must be positive"))
    0f0 < cfg.residual_init_scale < 1f0 || throw(ArgumentError(
        "residual_init_scale must be in (0, 1).",
    ))
    cfg.norm_eps > 0 || throw(ArgumentError("norm_eps must be positive"))
    cfg.stability_eps > 0 || throw(ArgumentError("stability_eps must be positive"))
    norm = RMSNormLayer(cfg.channels, cfg.norm_eps)
    return DispersiveBlock(
        cfg.channels,
        cfg.length,
        cfg.dt_init,
        cfg.dt_floor,
        cfg.alpha_init,
        cfg.alpha_floor,
        cfg.beta_init,
        cfg.phase_limit,
        cfg.decay_limit,
        cfg.residual_init_scale,
        cfg.stability_eps,
        cfg.warn_on_clamp,
        norm,
        frequency_grid(cfg.length),
    )
end

function Lux.initialparameters(rng::AbstractRNG, layer::DispersiveBlock)
    return (
        norm=Lux.initialparameters(rng, layer.norm),
        in_weight=glorot_uniform(rng, layer.channels, layer.channels),
        in_bias=zeros(Float32, layer.channels),
        out_weight=layer.residual_init_scale .* glorot_uniform(rng, layer.channels, layer.channels),
        out_bias=zeros(Float32, layer.channels),
        alpha_raw=fill(Float32(inverse_softplus(max(layer.alpha_init - layer.alpha_floor, 1f-6))), layer.channels),
        beta_raw=fill(layer.beta_init, layer.channels),
        log_dt=Float32[inverse_softplus(max(layer.dt_init - layer.dt_floor, 1f-6))],
        residual_gate_raw=Float32[logit_scalar(layer.residual_init_scale)],
    )
end

Lux.initialstates(rng::AbstractRNG, layer::DispersiveBlock) = (norm=Lux.initialstates(rng, layer.norm),)

function airy_operator_control(
    alpha::AbstractVector{T},
    beta::AbstractVector{T},
    dt_raw::T,
    k::AbstractVector{<:Real},
    decay_limit::T,
    phase_limit::T,
    eps::T,
) where {T<:AbstractFloat}
    max_alpha = max(maximum(alpha), eps)
    max_abs_beta = max(maximum(abs, beta), eps)
    max_abs_k = max(maximum(abs, k), eps)
    max_decay = dt_raw * max_alpha * (max_abs_k^2)
    max_phase = dt_raw * max_abs_beta * (max_abs_k^3)
    dt_decay = decay_limit / (max_alpha * (max_abs_k^2) + eps)
    dt_phase = phase_limit / (max_abs_beta * (max_abs_k^3) + eps)
    dt_eff = max(eps, min(dt_raw, dt_decay, dt_phase))
    return dt_eff, (
        max_decay=max_decay,
        max_phase=max_phase,
        clamped_decay=max_decay > decay_limit,
        clamped_phase=max_phase > phase_limit,
    )
end

function maybe_warn_dispersive_stability(
    layer::DispersiveBlock,
    dt_raw::T,
    dt_eff::T,
    stats,
) where {T<:AbstractFloat}
    layer.warn_on_clamp || return nothing
    (stats.clamped_decay || stats.clamped_phase) || return nothing
    @warn(
        "DispersiveBlock stability control clamped dt from $(dt_raw) to $(dt_eff): " *
        "max_decay=$(stats.max_decay) (limit $(layer.decay_limit)), " *
        "max_phase=$(stats.max_phase) (limit $(layer.phase_limit))"
    )
    return nothing
end

function airy_symbol(
    layer::DispersiveBlock,
    alpha::AbstractVector{<:AbstractFloat},
    beta::AbstractVector{<:AbstractFloat},
    k_grid::AbstractVector{<:Real},
    dt::T,
    ::Type{T},
) where {T<:AbstractFloat}
    alpha_term = reshape(T.(alpha), :, 1, 1)
    beta_term = reshape(T.(beta), :, 1, 1)
    k = reshape(T.(k_grid), 1, :, 1)
    imag_unit = complex(zero(T), one(T))
    phase = dt .* (-alpha_term .* (k .^ 2) .+ imag_unit .* beta_term .* (k .^ 3))
    return exp.(phase)
end

function (layer::DispersiveBlock)(x::AbstractArray{T, 3}, ps, st) where {T<:AbstractFloat}
    size(x, 1) == layer.channels || throw(ArgumentError(
        "DispersiveBlock expected $(layer.channels) channels, got $(size(x, 1))",
    ))
    size(x, 2) > 0 || throw(ArgumentError("DispersiveBlock requires seq_len > 0"))
    size(x, 3) > 0 || throw(ArgumentError("DispersiveBlock requires a non-empty batch"))

    h, st_norm = layer.norm(x, ps.norm, st.norm)
    mixed = linear1x1(h, ps.in_weight, ps.in_bias)
    alpha = softplus_scalar.(ps.alpha_raw) .+ layer.alpha_floor
    beta = ps.beta_raw
    dt_raw = softplus_scalar(ps.log_dt[1]) + layer.dt_floor
    k_grid = size(x, 2) == layer.length ? layer.k : frequency_grid(size(x, 2))
    dt_eff, stats = airy_operator_control(
        alpha,
        beta,
        dt_raw,
        k_grid,
        layer.decay_limit,
        layer.phase_limit,
        layer.stability_eps,
    )
    Zygote.ignore() do
        maybe_warn_dispersive_stability(layer, dt_raw, dt_eff, stats)
    end

    x_fft = FFTW.fft(complex.(mixed), 2) .* airy_symbol(layer, alpha, beta, k_grid, T(dt_eff), T)
    x_ifft = T.(real.(FFTW.ifft(x_fft, 2)))
    projected = linear1x1(x_ifft, ps.out_weight, ps.out_bias)
    residual_gate = T(sigmoid_scalar(ps.residual_gate_raw[1]))
    return x .+ residual_gate .* projected, (norm=st_norm,)
end

struct DispersiveStack{L, N} <: Lux.AbstractLuxLayer
    layers::L
    norm::N
    channels::Int
end

function DispersiveStack(n_layers::Int, cfg::DispersionConfig)
    n_layers >= 0 || throw(ArgumentError("n_layers must be non-negative"))
    blocks = ntuple(_ -> DispersiveBlock(cfg), n_layers)
    norm = RMSNormLayer(cfg.channels, cfg.norm_eps)
    return DispersiveStack(blocks, norm, cfg.channels)
end

DispersiveStack(cfg::DispersionConfig) = DispersiveStack(cfg.n_layer, cfg)

function Lux.initialparameters(rng::AbstractRNG, model::DispersiveStack)
    return (
        layers=Tuple(Lux.initialparameters(rng, layer) for layer in model.layers),
        norm=Lux.initialparameters(rng, model.norm),
        head_weight=glorot_uniform(rng, model.channels, model.channels),
        head_bias=zeros(Float32, model.channels),
    )
end

function Lux.initialstates(rng::AbstractRNG, model::DispersiveStack)
    return (
        layers=Tuple(Lux.initialstates(rng, layer) for layer in model.layers),
        norm=Lux.initialstates(rng, model.norm),
    )
end

apply_layers(::Tuple{}, x, ::Tuple{}, ::Tuple{}) = x, ()

function apply_layers(layers::Tuple, x, ps_layers::Tuple, st_layers::Tuple)
    head_layer = first(layers)
    head_ps = first(ps_layers)
    head_st = first(st_layers)
    x_next, st_next = head_layer(x, head_ps, head_st)
    x_tail, st_tail = apply_layers(Base.tail(layers), x_next, Base.tail(ps_layers), Base.tail(st_layers))
    return x_tail, (st_next, st_tail...)
end

function (model::DispersiveStack)(x::AbstractArray{T, 3}, ps, st) where {T<:AbstractFloat}
    hidden, layer_states = apply_layers(model.layers, x, ps.layers, st.layers)
    hidden, norm_state = model.norm(hidden, ps.norm, st.norm)
    return linear1x1(hidden, ps.head_weight, ps.head_bias), (layers=layer_states, norm=norm_state)
end

function make_identity_dataset(
    channels::Int,
    length::Int,
    batch_size::Int,
    num_batches::Int;
    rng::AbstractRNG=Random.default_rng(),
)
    channels > 0 || throw(ArgumentError("channels must be positive"))
    length > 0 || throw(ArgumentError("length must be positive"))
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    num_batches > 0 || throw(ArgumentError("num_batches must be positive"))

    data = Vector{Tuple{Array{Float32, 3}, Array{Float32, 3}}}(undef, num_batches)
    for idx in 1:num_batches
        x = randn(rng, Float32, channels, length, batch_size)
        data[idx] = (x, copy(x))
    end
    return data
end

function mse_loss(model::DispersiveStack, ps, st, x::AbstractArray{T, 3}, y::AbstractArray{T, 3}) where {T<:AbstractFloat}
    ŷ, st_next = Lux.apply(model, x, ps, st)
    return mean(abs2, ŷ .- y), st_next
end

function train!(
    model::DispersiveStack,
    data;
    epochs::Int=10,
    lr::Float32=1f-3,
    rng::AbstractRNG=Random.default_rng(),
)
    epochs > 0 || throw(ArgumentError("epochs must be positive"))
    lr > 0 || throw(ArgumentError("lr must be positive"))

    ps, st = Lux.setup(rng, model)
    optimizer = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(optimizer, ps)
    losses = Float32[]

    for _ in 1:epochs
        epoch_loss = 0.0f0
        for (x, y) in data
            grads = Zygote.gradient(ps) do params
                loss, _ = mse_loss(model, params, st, x, y)
                return loss
            end[1]
            batch_loss, st = mse_loss(model, ps, st, x, y)
            epoch_loss += Float32(batch_loss)
            opt_state, ps = Optimisers.update(opt_state, ps, grads)
        end
        push!(losses, epoch_loss)
    end

    return (parameters=ps, state=st, losses=losses)
end

end
