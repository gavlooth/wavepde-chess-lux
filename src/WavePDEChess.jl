module WavePDEChess

"""
WavePDEChess

Chess language model built around Wave-PDE residual blocks.

Paper reference:
    `2510.04304v1.pdf` in the repository root (`Wave-PDE Nets`).

Paper-faithful parts in this implementation:
- wave speed `c(x)` and damping `γ(x)` are produced from learned 1x1 projections
  and constrained with `softplus`
- the Laplacian is applied spectrally with FFTs
- the solver uses a symplectic velocity-Verlet style update with split damping

Intentional divergences from the paper:
- the "spatial" domain is the token axis, so the PDE mixes sequence positions
  rather than feature channels
- `dt` is a learned scalar per layer instead of a fixed hyperparameter
- Wave-PDE is the main backbone, not one branch inside a larger hybrid block
"""

using DBInterface
using DuckDB
using FFTW
using Lux
using Optimisers
using Random
using Serialization
using Statistics: mean
using Zygote

export WavePDEConfig,
    WavePDEChessLM,
    TrainingConfig,
    ChessParquetCorpus,
    chess_mamba_11m_config,
    autoregressive_cross_entropy,
    parameter_count,
    sample_batch,
    train!

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
    u_hat = FFTW.fft(complex.(u), 2)
    lap_hat = (-k .^ 2) .* u_hat
    return T.(real.(FFTW.ifft(lap_hat, 2)))
end

function split_damped_leapfrog_step(
    u::AbstractArray{T, 3},
    v::AbstractArray{T, 3},
    c_sq::AbstractArray{T, 3},
    damping_split::AbstractArray{T, 3},
    dt::T,
) where {T<:AbstractFloat}
    # The paper emphasizes a symplectic velocity-Verlet update; we keep that
    # structure but use the chess model's sequence-axis spectral operator.
    v_a = damping_split .* v
    v_b = v_a .+ (dt / 2) .* (c_sq .* spectral_laplacian(u))
    u_next = u .+ dt .* v_b
    v_c = v_b .+ (dt / 2) .* (c_sq .* spectral_laplacian(u_next))
    v_next = damping_split .* v_c
    return u_next, v_next
end

struct TokenEmbedding <: Lux.AbstractLuxLayer
    vocab_size::Int
    d_model::Int
end

Lux.initialparameters(rng::AbstractRNG, layer::TokenEmbedding) = (
    weight=glorot_uniform(rng, layer.d_model, layer.vocab_size),
)

Lux.initialstates(::AbstractRNG, ::TokenEmbedding) = NamedTuple()

function (layer::TokenEmbedding)(tokens::AbstractArray{<:Integer}, ps, st)
    if any(tokens .< 0) || any(tokens .>= layer.vocab_size)
        throw(ArgumentError("Token ids must be in [0, $(layer.vocab_size - 1)]."))
    end
    token_idx = Int.(tokens) .+ 1
    embeddings = reshape(ps.weight[:, vec(token_idx)], layer.d_model, size(tokens)...)
    return embeddings, st
end

struct RMSNormLayer{T} <: Lux.AbstractLuxLayer
    d_model::Int
    eps::T
end

Lux.initialparameters(::AbstractRNG, layer::RMSNormLayer) = (
    scale=ones(Float32, layer.d_model),
)

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
    return (
        c_weight=glorot_uniform(rng, layer.d_model, layer.d_model),
        c_bias=zeros(Float32, layer.d_model),
        gamma_weight=glorot_uniform(rng, layer.d_model, layer.d_model),
        gamma_bias=fill(-2.0f0, layer.d_model),
        log_dt=Float32(inverse_softplus(layer.dt_init)),
    )
end

Lux.initialstates(::AbstractRNG, ::WavePDESpectralMixer) = NamedTuple()

function (layer::WavePDESpectralMixer)(x::AbstractArray{T, 3}, ps, st) where {T<:AbstractFloat}
    validate_wavepde_input(layer, x)

    # Paper-faithful parameterization: c(x), γ(x) from 1x1 projections + softplus.
    # Chess-specific divergence: the PDE runs over sequence positions, not channels.
    c = softplus_scalar.(linear1x1(x, ps.c_weight, ps.c_bias)) .+ layer.dt_floor
    gamma = softplus_scalar.(linear1x1(x, ps.gamma_weight, ps.gamma_bias))
    dt = softplus_scalar(ps.log_dt) + layer.dt_floor

    u = x
    v = zero(x)
    c_sq = c .* c
    damping_split = exp.(-gamma .* (dt / 2))

    # This mirrors the paper's stability intent, but is only a heuristic guard
    # here because c and dt are both learned and c varies per token.
    max_wave_cfl = dt * maximum(c)
    max_wave_cfl < one(T) || @warn "WavePDESpectralMixer stability heuristic exceeded: dt * max(c) = $max_wave_cfl >= 1"

    for _ in 1:layer.solver_steps
        u, v = split_damped_leapfrog_step(u, v, c_sq, damping_split, dt)
    end

    return u, st
end

struct WavePDEBlock{N, M} <: Lux.AbstractLuxLayer
    norm::N
    mixer::M
end

function WavePDEBlock(d_model::Int, solver_steps::Int, dt_init::Float32, norm_eps::Float32)
    d_model > 0 || throw(ArgumentError("d_model must be positive"))
    solver_steps > 0 || throw(ArgumentError("solver_steps must be positive"))
    dt_init > 0 || throw(ArgumentError("dt_init must be positive"))
    norm_eps > 0 || throw(ArgumentError("norm_eps must be positive"))
    norm = RMSNormLayer(d_model, norm_eps)
    mixer = WavePDESpectralMixer(d_model, solver_steps, dt_init, 1f-4)
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

struct WavePDEChessLM{E, B, N} <: Lux.AbstractLuxLayer
    config::WavePDEConfig
    embedding::E
    blocks::B
    norm::N
end

function WavePDEChessLM(config::WavePDEConfig)
    config.vocab_size > 0 || throw(ArgumentError("vocab_size must be positive"))
    config.d_model > 0 || throw(ArgumentError("d_model must be positive"))
    config.n_layer >= 0 || throw(ArgumentError("n_layer must be non-negative"))
    config.max_seq_len > 0 || throw(ArgumentError("max_seq_len must be positive"))
    config.solver_steps > 0 || throw(ArgumentError("solver_steps must be positive"))
    config.dt_init > 0 || throw(ArgumentError("dt_init must be positive"))
    config.norm_eps > 0 || throw(ArgumentError("norm_eps must be positive"))
    embedding = TokenEmbedding(config.vocab_size, config.d_model)
    blocks = ntuple(_ -> WavePDEBlock(config.d_model, config.solver_steps, config.dt_init, config.norm_eps), config.n_layer)
    norm = RMSNormLayer(config.d_model, config.norm_eps)
    return WavePDEChessLM(config, embedding, blocks, norm)
end

Lux.initialparameters(rng::AbstractRNG, model::WavePDEChessLM) = (
    embedding=Lux.initialparameters(rng, model.embedding),
    blocks=Tuple(Lux.initialparameters(rng, block) for block in model.blocks),
    norm=Lux.initialparameters(rng, model.norm),
)

Lux.initialstates(rng::AbstractRNG, model::WavePDEChessLM) = (
    embedding=Lux.initialstates(rng, model.embedding),
    blocks=Tuple(Lux.initialstates(rng, block) for block in model.blocks),
    norm=Lux.initialstates(rng, model.norm),
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

function tied_logits(hidden::AbstractArray{T, 3}, embedding_weight::AbstractMatrix{T}) where {T<:AbstractFloat}
    feature_dim, seq_len, batch_size = size(hidden)
    flattened_hidden = reshape(hidden, feature_dim, :)
    flattened_logits = transpose(embedding_weight) * flattened_hidden
    return reshape(flattened_logits, size(embedding_weight, 2), seq_len, batch_size)
end

function (model::WavePDEChessLM)(tokens::AbstractArray{<:Integer}, ps, st)
    if size(tokens, 1) > model.config.max_seq_len
        throw(ArgumentError("Sequence length $(size(tokens, 1)) exceeds configured max_seq_len $(model.config.max_seq_len)."))
    end

    hidden, st_embedding = model.embedding(tokens, ps.embedding, st.embedding)
    hidden, st_blocks = apply_blocks(model.blocks, hidden, ps.blocks, st.blocks)
    hidden, st_norm = model.norm(hidden, ps.norm, st.norm)
    logits = tied_logits(hidden, ps.embedding.weight)

    return logits, (embedding=st_embedding, blocks=st_blocks, norm=st_norm)
end

function chess_mamba_11m_config(; vocab_size::Int=28, solver_steps::Int=4, dt_init::Float32=0.05f0)
    return WavePDEConfig(
        vocab_size=vocab_size,
        d_model=288,
        n_layer=20,
        max_seq_len=1536,
        solver_steps=solver_steps,
        dt_init=dt_init,
        norm_eps=1f-5,
        pad_token=0,
    )
end

parameter_count(x::Number) = 1
parameter_count(x::AbstractArray) = length(x)
parameter_count(x::Tuple) = sum(parameter_count, x)
parameter_count(x::NamedTuple) = sum(parameter_count, values(x))

Base.@kwdef struct TrainingConfig
    data_dir::String = joinpath(
        pwd(),
        "chess-mamba-vs-xformer",
        "chess-mamba-vs-xformer",
        "data",
    )
    batch_size::Int = 12
    learning_rate::Float32 = 6.0f-4
    max_iters::Int = 100
    log_interval::Int = 10
    min_tokens::Int = 8
    train_file_update_interval::Int = 10
    checkpoint_path::String = joinpath(pwd(), "checkpoints", "wavepde_chess_checkpoint.jls")
    seed::Int = 1337
end

mutable struct ChessParquetCorpus
    db::DuckDB.DB
    conn
    files::Vector{String}
    active_file::String
    active_games::Vector{Vector{Int32}}
    min_tokens::Int
end

function discover_parquet_files(data_dir::AbstractString)
    files = String[]
    for (root, _, names) in walkdir(data_dir)
        for name in names
            endswith(name, ".parquet") || continue
            push!(files, joinpath(root, name))
        end
    end
    sort!(files)
    return files
end

function is_lfs_pointer(path::AbstractString)
    open(path, "r") do io
        prefix = read(io, min(filesize(path), 16))
        marker = collect(codeunits("version"))
        return length(prefix) >= length(marker) && prefix[1:length(marker)] == marker
    end
end

function assert_real_parquet(path::AbstractString)
    if is_lfs_pointer(path)
        throw(ArgumentError("$(path) is a Git LFS pointer, not a real parquet file. Download the dataset blobs or point TrainingConfig.data_dir at actual parquet files."))
    end
    return nothing
end

sql_escape(path::AbstractString) = replace(path, "'" => "''")

function load_tokenized_games(conn, path::AbstractString; min_tokens::Int=8)
    assert_real_parquet(path)
    rows = DBInterface.execute(conn, "SELECT tokenized FROM read_parquet('$(sql_escape(path))')")
    games = Vector{Vector{Int32}}()
    for row in rows
        sequence = Int32[x for x in row[1] if !ismissing(x)]
        length(sequence) >= min_tokens || continue
        push!(games, sequence)
    end
    return games
end

function ChessParquetCorpus(data_dir::AbstractString; min_tokens::Int=8)
    files = discover_parquet_files(data_dir)
    isempty(files) && throw(ArgumentError("No parquet files found under $(data_dir)."))

    db = DuckDB.DB()
    conn = DBInterface.connect(db)
    active_file = first(files)
    active_games = load_tokenized_games(conn, active_file; min_tokens=min_tokens)
    isempty(active_games) && throw(ArgumentError("No usable tokenized games found in $(active_file)."))

    return ChessParquetCorpus(db, conn, files, active_file, active_games, min_tokens)
end

function reload_file!(corpus::ChessParquetCorpus, path::AbstractString)
    corpus.active_file = path
    corpus.active_games = load_tokenized_games(corpus.conn, path; min_tokens=corpus.min_tokens)
    isempty(corpus.active_games) && throw(ArgumentError("No usable tokenized games found in $(path)."))
    return corpus
end

function maybe_rotate_file!(corpus::ChessParquetCorpus, rng::AbstractRNG, step::Int; every::Int)
    if every > 0 && step > 1 && step % every == 0 && length(corpus.files) > 1
        next_file = corpus.files[rand(rng, eachindex(corpus.files))]
        next_file == corpus.active_file || reload_file!(corpus, next_file)
    end
    return corpus
end

function sample_batch(corpus::ChessParquetCorpus, rng::AbstractRNG; batch_size::Int, max_seq_len::Int)
    sampled_games = Vector{Vector{Int32}}(undef, batch_size)
    max_batch_len = 0

    for i in 1:batch_size
        game = corpus.active_games[rand(rng, eachindex(corpus.active_games))]
        sampled_games[i] = game
        max_batch_len = max(max_batch_len, min(length(game), max_seq_len))
    end

    sequences = zeros(Int32, max_batch_len, batch_size)
    for (i, game) in enumerate(sampled_games)
        game_len = min(length(game), max_batch_len)
        sequences[1:game_len, i] .= game[1:game_len]
    end

    return sequences
end

function autoregressive_cross_entropy(logits::AbstractArray{T, 3}, targets::AbstractMatrix{<:Integer}) where {T<:AbstractFloat}
    _, seq_len, batch_size = size(logits)
    @assert seq_len == size(targets, 1) "Logit and target sequence lengths differ."
    @assert batch_size == size(targets, 2) "Logit and target batch sizes differ."

    total = zero(T)
    count = seq_len * batch_size

    for b in 1:batch_size
        for t in 1:seq_len
            target_idx = Int(targets[t, b]) + 1
            column = view(logits, :, t, b)
            max_logit = maximum(column)
            log_denom = max_logit + log(sum(exp.(column .- max_logit)))
            total += log_denom - column[target_idx]
        end
    end

    return total / T(count)
end

function autoregressive_loss(model::WavePDEChessLM, ps, st, batch::AbstractMatrix{<:Integer})
    @assert size(batch, 1) >= 2 "Batch sequence length must be at least 2 for next-token prediction."
    inputs = batch[1:(end - 1), :]
    targets = batch[2:end, :]
    logits, _ = Lux.apply(model, inputs, ps, st)
    return autoregressive_cross_entropy(logits, targets)
end

function save_checkpoint(path::AbstractString, payload)
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, payload)
    end
    return path
end

function train!(model::WavePDEChessLM, corpus::ChessParquetCorpus, cfg::TrainingConfig)
    rng = MersenneTwister(cfg.seed)
    ps, st = Lux.setup(rng, model)
    optimizer = Optimisers.Adam(cfg.learning_rate)
    opt_state = Optimisers.setup(optimizer, ps)
    losses = Float32[]

    for step in 1:cfg.max_iters
        maybe_rotate_file!(corpus, rng, step; every=cfg.train_file_update_interval)
        batch = sample_batch(corpus, rng; batch_size=cfg.batch_size, max_seq_len=model.config.max_seq_len)

        loss = autoregressive_loss(model, ps, st, batch)
        grads = Zygote.gradient(p -> autoregressive_loss(model, p, st, batch), ps)[1]
        opt_state, ps = Optimisers.update(opt_state, ps, grads)

        push!(losses, Float32(loss))
        if step % cfg.log_interval == 0 || step == 1 || step == cfg.max_iters
            println("step=$(step) loss=$(round(loss; digits=4)) seq_len=$(size(batch, 1)) file=$(basename(corpus.active_file))")
        end
    end

    checkpoint = (
        model_config=model.config,
        training_config=cfg,
        parameters=ps,
        state=st,
        losses=losses,
    )
    save_checkpoint(cfg.checkpoint_path, checkpoint)

    return checkpoint
end

end
