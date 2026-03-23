Base.@kwdef struct WavePDECoreConfig
    d_model::Int = 288
    n_layer::Int = 20
    solver_steps::Int = 4
    dt_init::Float32 = 0.05f0
    norm_eps::Float32 = 1f-5
    cfl_safety_factor::Float32 = 0.95f0
    cfl_eps::Float32 = 1f-6
end

@inline softplus_scalar(x::T) where {T<:Real} = log1p(exp(-abs(x))) + max(x, zero(T))
@inline inverse_softplus(x::T) where {T<:Real} = x + log(-expm1(-x))

function glorot_uniform(rng::AbstractRNG, out_dim::Int, in_dim::Int)
    limit = sqrt(6.0f0 / Float32(in_dim + out_dim))
    return rand(rng, Float32, out_dim, in_dim) .* (2.0f0 * limit) .- limit
end

function linear1x1(x::AbstractArray, weight::AbstractMatrix, bias::AbstractVector)
    feature_dim = size(x, 1)
    @assert feature_dim == size(weight, 2) "Input feature size does not match weight."
    trailing_dims = size(x)[2:end]
    y = reshape(weight * reshape(x, feature_dim, :), size(weight, 1), trailing_dims...)
    bias_shape = (length(bias), ntuple(_ -> 1, ndims(x) - 1)...)
    return y .+ reshape(bias, bias_shape...)
end

function spectral_modes(seq_len::Int, ::Type{T}) where {T<:AbstractFloat}
    half = fld(seq_len, 2)
    raw_modes = if iseven(seq_len)
        vcat(collect(0:(half - 1)), collect(-half:-1))
    else
        vcat(collect(0:half), collect(-half:-1))
    end
    scaled = (2T(pi) / T(seq_len)) .* T.(raw_modes)
    return reshape(scaled, 1, seq_len, 1)
end

function spectral_laplacian(u::AbstractArray{T, 3}) where {T<:AbstractFloat}
    k = spectral_modes(size(u, 2), T)
    if u isa CUDA.AbstractGPUArray
        k = CUDA.CuArray(k)
    end
    lap_hat = FFTW.fft(complex.(u), 2) .* (-k .^ 2)
    return T.(real.(FFTW.ifft(lap_hat, 2)))
end

function split_damped_leapfrog_step(
    u::AbstractArray{T, 3},
    v::AbstractArray{T, 3},
    c_sq::AbstractArray{T, 3},
    damping_split::AbstractArray{T, 3},
    dt::T,
) where {T<:AbstractFloat}
    v_a = damping_split .* v
    v_b = v_a .+ (dt / 2) .* (c_sq .* spectral_laplacian(u))
    u_next = u .+ dt .* v_b
    v_c = v_b .+ (dt / 2) .* (c_sq .* spectral_laplacian(u_next))
    v_next = damping_split .* v_c
    return u_next, v_next
end

struct RMSNormLayer{T} <: Lux.AbstractLuxLayer
    d_model::Int
    eps::T
end

Lux.initialparameters(::AbstractRNG, layer::RMSNormLayer) = (scale=ones(Float32, layer.d_model),)
Lux.initialstates(::AbstractRNG, ::RMSNormLayer) = NamedTuple()

function (layer::RMSNormLayer)(x::AbstractArray, ps, st)
    rms = sqrt.(mean(abs2, x; dims=1) .+ layer.eps)
    scale = reshape(ps.scale, :, ntuple(_ -> 1, ndims(x) - 1)...)
    return x .* (scale ./ rms), st
end

struct WavePDESpectralMixer{T} <: Lux.AbstractLuxLayer
    d_model::Int
    solver_steps::Int
    dt_init::T
    dt_floor::T
    cfl_safety_factor::T
    cfl_eps::T
end

function validate_wavepde_input(layer::WavePDESpectralMixer, x::AbstractArray)
    ndims(x) == 3 || throw(ArgumentError(
        "WavePDESpectralMixer expects a 3D tensor (d_model, seq_len, batch), got shape $(size(x)).",
    ))
    size(x, 1) == layer.d_model || throw(ArgumentError(
        "WavePDESpectralMixer expected feature dimension $(layer.d_model), got $(size(x, 1)).",
    ))
    size(x, 2) > 0 || throw(ArgumentError("WavePDESpectralMixer requires seq_len > 0."))
    size(x, 3) > 0 || throw(ArgumentError("WavePDESpectralMixer requires batch size > 0."))
    return nothing
end

function Lux.initialparameters(rng::AbstractRNG, layer::WavePDESpectralMixer)
    layer.d_model > 0 || throw(ArgumentError("d_model must be positive"))
    layer.solver_steps > 0 || throw(ArgumentError("solver_steps must be positive"))
    layer.dt_init > 0 || throw(ArgumentError("dt_init must be positive"))
    layer.dt_floor >= 0 || throw(ArgumentError("dt_floor must be non-negative"))
    zero(layer.cfl_safety_factor) < layer.cfl_safety_factor < one(layer.cfl_safety_factor) || throw(ArgumentError("cfl_safety_factor must be in (0, 1)."))
    layer.cfl_eps > 0 || throw(ArgumentError("cfl_eps must be positive"))
    return (
        c_weight=glorot_uniform(rng, layer.d_model, layer.d_model),
        c_bias=zeros(Float32, layer.d_model),
        gamma_weight=glorot_uniform(rng, layer.d_model, layer.d_model),
        gamma_bias=fill(-2.0f0, layer.d_model),
        log_dt=Float32(inverse_softplus(layer.dt_init)),
    )
end

Lux.initialstates(::AbstractRNG, ::WavePDESpectralMixer) = NamedTuple()

function wavepde_cfl_control(
    c::AbstractArray{T},
    dt_raw::T,
    cfl_safety_factor::T,
    cfl_eps::T,
) where {T<:AbstractFloat}
    max_c = max(maximum(c), cfl_eps)
    max_dt = cfl_safety_factor / (max_c + cfl_eps)
    dt = min(dt_raw, max_dt)
    raw_cfl = dt_raw * max_c
    clamped = dt < dt_raw
    return dt, raw_cfl, clamped
end

function maybe_warn_wavepde_stability(dt_raw::T, raw_cfl::T, clamped::Bool, cfl_safety_factor::T) where {T<:AbstractFloat}
    clamped || return nothing
    @warn "WavePDESpectralMixer CFL control clamped dt: raw dt * max(c) = $raw_cfl exceeded safety factor $cfl_safety_factor"
    return nothing
end

function (layer::WavePDESpectralMixer)(x::AbstractArray{T, 3}, ps, st) where {T<:AbstractFloat}
    validate_wavepde_input(layer, x)
    c = softplus_scalar.(linear1x1(x, ps.c_weight, ps.c_bias)) .+ layer.dt_floor
    gamma = softplus_scalar.(linear1x1(x, ps.gamma_weight, ps.gamma_bias))
    dt_raw = softplus_scalar(ps.log_dt) + layer.dt_floor
    dt, raw_cfl, clamped = wavepde_cfl_control(c, dt_raw, layer.cfl_safety_factor, layer.cfl_eps)

    u = x
    v = zero(x)
    c_sq = c .* c
    damping_split = exp.(-gamma .* (dt / 2))
    Zygote.ignore() do
        maybe_warn_wavepde_stability(dt_raw, raw_cfl, clamped, layer.cfl_safety_factor)
    end

    for _ in 1:layer.solver_steps
        u, v = split_damped_leapfrog_step(u, v, c_sq, damping_split, dt)
    end

    return u, st
end

struct WavePDEBlock{N, M} <: Lux.AbstractLuxLayer
    norm::N
    mixer::M
end

function WavePDEBlock(
    d_model::Int,
    solver_steps::Int,
    dt_init::Float32,
    norm_eps::Float32,
    cfl_safety_factor::Float32,
    cfl_eps::Float32,
)
    d_model > 0 || throw(ArgumentError("d_model must be positive"))
    solver_steps > 0 || throw(ArgumentError("solver_steps must be positive"))
    dt_init > 0 || throw(ArgumentError("dt_init must be positive"))
    norm_eps > 0 || throw(ArgumentError("norm_eps must be positive"))
    0f0 < cfl_safety_factor < 1f0 || throw(ArgumentError("cfl_safety_factor must be in (0, 1)."))
    cfl_eps > 0f0 || throw(ArgumentError("cfl_eps must be positive"))
    norm = RMSNormLayer(d_model, norm_eps)
    mixer = WavePDESpectralMixer(d_model, solver_steps, dt_init, 1f-4, cfl_safety_factor, cfl_eps)
    return WavePDEBlock(norm, mixer)
end

Lux.initialparameters(rng::AbstractRNG, layer::WavePDEBlock) = (
    norm=Lux.initialparameters(rng, layer.norm),
    mixer=Lux.initialparameters(rng, layer.mixer),
)

Lux.initialstates(rng::AbstractRNG, layer::WavePDEBlock) = (
    norm=Lux.initialstates(rng, layer.norm),
    mixer=Lux.initialstates(rng, layer.mixer),
)

function (layer::WavePDEBlock)(x::AbstractArray{T, 3}, ps, st) where {T<:AbstractFloat}
    h, st_norm = layer.norm(x, ps.norm, st.norm)
    h, st_mixer = layer.mixer(h, ps.mixer, st.mixer)
    return x .+ h, (norm=st_norm, mixer=st_mixer)
end

struct WavePDECore{B, N} <: Lux.AbstractLuxLayer
    config::WavePDECoreConfig
    blocks::B
    norm::N
end

function WavePDECore(config::WavePDECoreConfig)
    config.d_model > 0 || throw(ArgumentError("d_model must be positive"))
    config.n_layer >= 0 || throw(ArgumentError("n_layer must be non-negative"))
    config.solver_steps > 0 || throw(ArgumentError("solver_steps must be positive"))
    config.dt_init > 0 || throw(ArgumentError("dt_init must be positive"))
    config.norm_eps > 0 || throw(ArgumentError("norm_eps must be positive"))
    0f0 < config.cfl_safety_factor < 1f0 || throw(ArgumentError("cfl_safety_factor must be in (0, 1)."))
    config.cfl_eps > 0f0 || throw(ArgumentError("cfl_eps must be positive"))
    blocks = ntuple(
        _ -> WavePDEBlock(
            config.d_model,
            config.solver_steps,
            config.dt_init,
            config.norm_eps,
            config.cfl_safety_factor,
            config.cfl_eps,
        ),
        config.n_layer,
    )
    norm = RMSNormLayer(config.d_model, config.norm_eps)
    return WavePDECore(config, blocks, norm)
end

Lux.initialparameters(rng::AbstractRNG, core::WavePDECore) = (
    blocks=Tuple(Lux.initialparameters(rng, block) for block in core.blocks),
    norm=Lux.initialparameters(rng, core.norm),
)

Lux.initialstates(rng::AbstractRNG, core::WavePDECore) = (
    blocks=Tuple(Lux.initialstates(rng, block) for block in core.blocks),
    norm=Lux.initialstates(rng, core.norm),
)

function apply_blocks(::Tuple{}, x, ::Tuple{}, ::Tuple{})
    return x, ()
end

function apply_blocks(blocks::Tuple, x, ps_blocks::Tuple, st_blocks::Tuple)
    head_block = first(blocks)
    head_ps = first(ps_blocks)
    head_st = first(st_blocks)
    x_next, st_head = head_block(x, head_ps, head_st)
    x_final, st_tail = apply_blocks(Base.tail(blocks), x_next, Base.tail(ps_blocks), Base.tail(st_blocks))
    return x_final, (st_head, st_tail...)
end

function (core::WavePDECore)(hidden::AbstractArray{T, 3}, ps, st) where {T<:AbstractFloat}
    hidden, st_blocks = apply_blocks(core.blocks, hidden, ps.blocks, st.blocks)
    hidden, st_norm = core.norm(hidden, ps.norm, st.norm)
    return hidden, (blocks=st_blocks, norm=st_norm)
end
