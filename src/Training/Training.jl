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

function extract_proposer_logits(output)
    if output isa NamedTuple
        haskey(output, :proposer) || throw(ArgumentError("Model output NamedTuple must contain :proposer logits."))
        return output.proposer
    end
    return output
end

function autoregressive_loss(model, ps, st, batch::AbstractMatrix{<:Integer})
    @assert size(batch, 1) >= 2 "Batch sequence length must be at least 2 for next-token prediction."
    inputs = batch[1:(end - 1), :]
    targets = batch[2:end, :]
    output, _ = Lux.apply(model, inputs, ps, st)
    logits = extract_proposer_logits(output)
    return autoregressive_cross_entropy(logits, targets)
end

function save_checkpoint(path::AbstractString, payload)
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, payload)
    end
    return path
end

function train!(model, corpus::ChessParquetCorpus, cfg::TrainingConfig)
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
