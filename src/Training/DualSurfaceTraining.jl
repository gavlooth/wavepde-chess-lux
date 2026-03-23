Base.@kwdef struct DualSurfaceStateModelConfig
    adapter::ChessAdapterConfig = ChessAdapterConfig(vocab_size=BOARD_STATE_VOCAB_SIZE, d_model=288, pad_token=0)
    core::WavePDECoreConfig = WavePDECoreConfig()
    state_head::ChessMoveHeadConfig = ChessMoveHeadConfig(vocab_size=BOARD_STATE_VOCAB_SIZE, d_model=288, tie_embeddings=true, bias=false)
    transcript_head::ChessMoveHeadConfig = ChessMoveHeadConfig(vocab_size=length(CHESS_TRANSCRIPT_STOI), d_model=288, tie_embeddings=false, bias=true)
    max_seq_len::Int = BOARD_STATE_SEQUENCE_LENGTH
end

struct DualSurfaceStateModel{A, C, S, T} <: Lux.AbstractLuxLayer
    config::DualSurfaceStateModelConfig
    adapter::A
    core::C
    state_head::S
    transcript_head::T
end

mutable struct DualSurfaceParquetCorpus
    db::DuckDB.DB
    conn
    files::Vector{String}
    active_file::String
    active_states::Vector{Vector{Int32}}
    active_next_states::Vector{Vector{Int32}}
    active_move_san::Vector{String}
    min_tokens::Int
end

Base.@kwdef struct DualSurfaceTrainingConfig
    data_dir::String = joinpath(pwd(), "tmp", "transcript_state_dataset")
    batch_size::Int = 12
    learning_rate::Float32 = 6.0f-4
    max_iters::Int = 100
    log_interval::Int = 10
    min_tokens::Int = BOARD_STATE_SEQUENCE_LENGTH
    train_file_update_interval::Int = 10
    state_loss_weight::Float32 = 1.0f0
    transcript_loss_weight::Float32 = 0.2f0
    training_policy::Symbol = :full
    checkpoint_path::String = joinpath(pwd(), "checkpoints", "wavepde_dual_surface_checkpoint.jls")
    seed::Int = 1337
end

function validate_dual_surface_model_config(config::DualSurfaceStateModelConfig)
    config.max_seq_len > 0 || throw(ArgumentError("max_seq_len must be positive"))
    config.adapter.d_model == config.core.d_model || throw(ArgumentError("adapter d_model must match core d_model"))
    config.state_head.d_model == config.core.d_model || throw(ArgumentError("state_head d_model must match core d_model"))
    config.transcript_head.d_model == config.core.d_model || throw(ArgumentError("transcript_head d_model must match core d_model"))
    config.state_head.vocab_size == config.adapter.vocab_size || throw(ArgumentError("state_head vocab_size must match adapter vocab_size"))
    config.transcript_head.tie_embeddings && throw(ArgumentError("transcript_head cannot tie embeddings to the state adapter because the vocabularies differ"))
    return nothing
end

function DualSurfaceStateModel(config::DualSurfaceStateModelConfig)
    validate_dual_surface_model_config(config)
    return DualSurfaceStateModel(
        config,
        ChessInputAdapter(config.adapter),
        WavePDECore(config.core),
        ChessMoveHead(config.state_head),
        ChessMoveHead(config.transcript_head),
    )
end

Lux.initialparameters(rng::AbstractRNG, model::DualSurfaceStateModel) = (
    adapter=Lux.initialparameters(rng, model.adapter),
    core=Lux.initialparameters(rng, model.core),
    state_head=Lux.initialparameters(rng, model.state_head),
    transcript_head=Lux.initialparameters(rng, model.transcript_head),
)

Lux.initialstates(rng::AbstractRNG, model::DualSurfaceStateModel) = (
    adapter=Lux.initialstates(rng, model.adapter),
    core=Lux.initialstates(rng, model.core),
    state_head=Lux.initialstates(rng, model.state_head),
    transcript_head=Lux.initialstates(rng, model.transcript_head),
)

function (model::DualSurfaceStateModel)(tokens::AbstractArray{<:Integer}, ps, st)
    size(tokens, 1) > model.config.max_seq_len && throw(ArgumentError(
        "Sequence length $(size(tokens, 1)) exceeds configured max_seq_len $(model.config.max_seq_len).",
    ))
    hidden, st_adapter = model.adapter(tokens, ps.adapter, st.adapter)
    hidden, st_core = model.core(hidden, ps.core, st.core)
    state_logits, st_state = proposer_output(model.state_head, hidden, ps.adapter, ps.state_head, st.state_head)
    transcript_logits, st_transcript = proposer_output(model.transcript_head, hidden, ps.adapter, ps.transcript_head, st.transcript_head)
    return (
        state=state_logits,
        transcript=transcript_logits,
    ), (
        adapter=st_adapter,
        core=st_core,
        state_head=st_state,
        transcript_head=st_transcript,
    )
end

function trainable_parameter_groups(::DualSurfaceStateModel, policy::Symbol)
    validate_training_policy(policy)
    if policy == :full
        return (:adapter, :core, :state_head, :transcript_head)
    elseif policy == :adapters_only
        return (:adapter,)
    else
        return (:state_head, :transcript_head)
    end
end

function load_dual_surface_examples(conn, path::AbstractString; min_tokens::Int=BOARD_STATE_SEQUENCE_LENGTH)
    assert_real_parquet(path)
    columns = parquet_column_names(conn, path)
    state_column = find_state_tokens_column(columns)
    next_state_column = find_next_state_tokens_column(columns)
    move_column = "move_san" in columns ? "move_san" : nothing
    state_column === nothing && throw(ArgumentError("Dual-surface parquet $(path) must include a state_tokens column."))
    next_state_column === nothing && throw(ArgumentError("Dual-surface parquet $(path) must include a next_state_tokens column."))
    move_column === nothing && throw(ArgumentError("Dual-surface parquet $(path) must include a move_san column."))

    rows = DBInterface.execute(
        conn,
        "SELECT $(state_column) AS state_tokens, $(next_state_column) AS next_state_tokens, $(move_column) AS move_san FROM read_parquet('$(sql_escape(path))')",
    )

    states = Vector{Vector{Int32}}()
    next_states = Vector{Vector{Int32}}()
    moves = String[]
    for row in rows
        state_tokens = parse_int_sequence(row[1])
        next_state_tokens = parse_int_sequence(row[2])
        move_san = row[3]
        move_san === missing && continue
        length(state_tokens) >= min_tokens || continue
        length(next_state_tokens) >= min_tokens || continue
        length(state_tokens) == length(next_state_tokens) || throw(ArgumentError(
            "Dual-surface state length mismatch in $(path): got $(length(state_tokens)) and $(length(next_state_tokens)).",
        ))
        push!(states, state_tokens)
        push!(next_states, next_state_tokens)
        push!(moves, String(move_san))
    end

    isempty(states) && throw(ArgumentError("No usable dual-surface examples found in $(path)."))
    return states, next_states, moves
end

function DualSurfaceParquetCorpus(data_dir::AbstractString; min_tokens::Int=BOARD_STATE_SEQUENCE_LENGTH)
    files = discover_parquet_files(data_dir)
    isempty(files) && throw(ArgumentError("No parquet files found under $(data_dir)."))

    db = DuckDB.DB()
    conn = DBInterface.connect(db)
    active_file = first(files)
    active_states, active_next_states, active_move_san = load_dual_surface_examples(conn, active_file; min_tokens=min_tokens)

    return DualSurfaceParquetCorpus(
        db,
        conn,
        files,
        active_file,
        active_states,
        active_next_states,
        active_move_san,
        min_tokens,
    )
end

function reload_file!(corpus::DualSurfaceParquetCorpus, path::AbstractString)
    corpus.active_file = path
    corpus.active_states, corpus.active_next_states, corpus.active_move_san = load_dual_surface_examples(
        corpus.conn,
        path;
        min_tokens=corpus.min_tokens,
    )
    isempty(corpus.active_states) && throw(ArgumentError("No usable dual-surface examples found in $(path)."))
    return corpus
end

function maybe_rotate_file!(corpus::DualSurfaceParquetCorpus, rng::AbstractRNG, step::Int; every::Int)
    if every > 0 && step > 1 && step % every == 0 && length(corpus.files) > 1
        next_file = corpus.files[rand(rng, eachindex(corpus.files))]
        next_file == corpus.active_file || reload_file!(corpus, next_file)
    end
    return corpus
end

function parse_transcript_language_examples(source::AbstractString; min_tokens::Int=8)
    paths = if isfile(source)
        [source]
    elseif isdir(source)
        discover_parquet_files(source)
    else
        throw(ArgumentError("Transcript language source $(source) does not exist."))
    end

    db = DuckDB.DB()
    conn = DBInterface.connect(db)
    examples = NamedTuple[]
    for path in paths
        assert_real_parquet(path)
        columns = parquet_column_names(conn, path)
        transcript_column = find_transcript_column(columns)
        transcript_column === nothing && continue
        rows = DBInterface.execute(
            conn,
            "SELECT $(transcript_column) AS transcript FROM read_parquet('$(sql_escape(path))')",
        )
        for row in rows
            transcript = row[1]
            transcript === missing && continue
            tokenized = encode_chess_transcript(String(transcript))
            length(tokenized) >= min_tokens || continue
            push!(examples, (tokenized=tokenized, transcript=String(transcript)))
        end
    end
    isempty(examples) && throw(ArgumentError("No transcript-language examples found in $(source)."))
    return examples
end

function write_transcript_language_parquet(
    source::AbstractString,
    output_dir::AbstractString;
    file_name::AbstractString="transcript_language.parquet",
    min_tokens::Int=8,
)
    examples = parse_transcript_language_examples(source; min_tokens=min_tokens)
    mkpath(output_dir)
    parquet_path = joinpath(output_dir, file_name)

    db = DuckDB.DB()
    conn = DBInterface.connect(db)
    DBInterface.execute(conn, """
        CREATE TABLE transcript_language (
            tokenized INTEGER[],
            transcript VARCHAR
        )
    """)

    batch_size = 256
    for offset in 1:batch_size:length(examples)
        batch = examples[offset:min(offset + batch_size - 1, length(examples))]
        values = String[]
        for example in batch
            push!(
                values,
                "($(state_sql_list(example.tokenized)), '$(state_sql_escape(example.transcript))')",
            )
        end
        DBInterface.execute(conn, "INSERT INTO transcript_language VALUES " * join(values, ", "))
    end

    isfile(parquet_path) && rm(parquet_path; force=true)
    DBInterface.execute(conn, "COPY transcript_language TO '$(state_sql_escape(parquet_path))' (FORMAT PARQUET)")
    return parquet_path
end

function ensure_transcript_language_parquet(
    source::AbstractString,
    output_dir::AbstractString;
    file_name::AbstractString="transcript_language.parquet",
    min_tokens::Int=8,
)
    parquet_path = joinpath(output_dir, file_name)
    isfile(parquet_path) || write_transcript_language_parquet(
        source,
        output_dir;
        file_name=file_name,
        min_tokens=min_tokens,
    )
    return parquet_path
end

function sample_dual_surface_batch(
    corpus::DualSurfaceParquetCorpus,
    rng::AbstractRNG;
    batch_size::Int,
    max_seq_len::Int,
)
    sampled_states = Vector{Vector{Int32}}(undef, batch_size)
    sampled_next_states = Vector{Vector{Int32}}(undef, batch_size)
    sampled_moves = Vector{String}(undef, batch_size)
    max_batch_len = 0

    for i in 1:batch_size
        idx = rand(rng, eachindex(corpus.active_states))
        sampled_states[i] = corpus.active_states[idx]
        sampled_next_states[i] = corpus.active_next_states[idx]
        sampled_moves[i] = corpus.active_move_san[idx]
        max_batch_len = max(max_batch_len, min(length(sampled_states[i]), max_seq_len))
    end

    tokens = zeros(Int32, max_batch_len, batch_size)
    target_tokens = zeros(Int32, max_batch_len, batch_size)
    transcript_targets = zeros(Int32, max_batch_len, batch_size)
    transcript_mask = falses(max_batch_len, batch_size)

    for i in 1:batch_size
        token_count = min(length(sampled_states[i]), max_batch_len)
        tokens[1:token_count, i] .= sampled_states[i][1:token_count]
        target_tokens[1:token_count, i] .= sampled_next_states[i][1:token_count]

        move_tokens = encode_chess_candidate_san(sampled_moves[i])
        move_count = min(length(move_tokens), max_batch_len)
        transcript_targets[1:move_count, i] .= move_tokens[1:move_count]
        transcript_mask[1:move_count, i] .= true
    end

    return (
        tokens=tokens,
        target_tokens=target_tokens,
        transcript_targets=transcript_targets,
        transcript_mask=transcript_mask,
    )
end

function masked_cross_entropy(
    logits::AbstractArray{T, 3},
    targets::AbstractMatrix{<:Integer},
    mask::BitMatrix,
) where {T<:AbstractFloat}
    _, seq_len, batch_size = size(logits)
    seq_len == size(targets, 1) || throw(ArgumentError("Masked cross entropy sequence lengths differ."))
    batch_size == size(targets, 2) || throw(ArgumentError("Masked cross entropy batch sizes differ."))
    size(mask) == size(targets) || throw(ArgumentError("Masked cross entropy mask shape must match targets."))

    total = zero(T)
    count = 0
    for b in 1:batch_size
        for t in 1:seq_len
            mask[t, b] || continue
            target_idx = Int(targets[t, b]) + 1
            column = view(logits, :, t, b)
            max_logit = maximum(column)
            log_denom = max_logit + log(sum(exp.(column .- max_logit)))
            total += log_denom - column[target_idx]
            count += 1
        end
    end
    count > 0 || throw(ArgumentError("Masked cross entropy requires at least one active target token."))
    return total / T(count)
end

function dual_surface_loss(
    model::DualSurfaceStateModel,
    ps,
    st,
    batch;
    state_loss_weight::Real=1.0,
    transcript_loss_weight::Real=0.2,
)
    outputs, _ = Lux.apply(model, batch.tokens, ps, st)
    state_loss = autoregressive_cross_entropy(outputs.state, batch.target_tokens)
    transcript_loss = masked_cross_entropy(outputs.transcript, batch.transcript_targets, batch.transcript_mask)
    total_loss = Float32(state_loss_weight) * state_loss + Float32(transcript_loss_weight) * transcript_loss
    return total_loss, (
        state_loss=Float32(state_loss),
        transcript_loss=Float32(transcript_loss),
    )
end

function train_dual_surface!(model::DualSurfaceStateModel, corpus::DualSurfaceParquetCorpus, cfg::DualSurfaceTrainingConfig)
    rng = MersenneTwister(cfg.seed)
    training_policy = validate_training_policy(cfg.training_policy)
    ps, st = Lux.setup(rng, model)
    optimizer = Optimisers.Adam(cfg.learning_rate)
    opt_state = Optimisers.setup(optimizer, ps)
    losses = Float32[]
    state_losses = Float32[]
    transcript_losses = Float32[]

    for step in 1:cfg.max_iters
        maybe_rotate_file!(corpus, rng, step; every=cfg.train_file_update_interval)
        batch = sample_dual_surface_batch(
            corpus,
            rng;
            batch_size=cfg.batch_size,
            max_seq_len=model.config.max_seq_len,
        )

        total_loss, parts = dual_surface_loss(
            model,
            ps,
            st,
            batch;
            state_loss_weight=cfg.state_loss_weight,
            transcript_loss_weight=cfg.transcript_loss_weight,
        )
        grads = Zygote.gradient(
            p -> first(dual_surface_loss(
                model,
                p,
                st,
                batch;
                state_loss_weight=cfg.state_loss_weight,
                transcript_loss_weight=cfg.transcript_loss_weight,
            )),
            ps,
        )[1]
        grads = apply_training_policy(model, grads, training_policy)
        opt_state, ps = Optimisers.update(opt_state, ps, grads)

        push!(losses, Float32(total_loss))
        push!(state_losses, parts.state_loss)
        push!(transcript_losses, parts.transcript_loss)
        if step % cfg.log_interval == 0 || step == 1 || step == cfg.max_iters
            println(
                "step=$(step) total_loss=$(round(total_loss; digits=4)) state_loss=$(round(parts.state_loss; digits=4)) transcript_loss=$(round(parts.transcript_loss; digits=4)) seq_len=$(size(batch.tokens, 1)) file=$(basename(corpus.active_file))",
            )
        end
    end

    checkpoint = (
        model_config=model.config,
        training_config=cfg,
        parameters=ps,
        state=st,
        losses=losses,
        state_losses=state_losses,
        transcript_losses=transcript_losses,
    )
    save_checkpoint(cfg.checkpoint_path, checkpoint)
    return checkpoint
end

function compare_surface_training_modes(
    transcript_source::AbstractString,
    state_data_dir::AbstractString;
    d_model::Int=16,
    n_layer::Int=2,
    solver_steps::Int=1,
    dt_init::Float32=0.05f0,
    norm_eps::Float32=1f-5,
    batch_size::Int=2,
    learning_rate::Float32=1.0f-3,
    max_iters::Int=1,
    seed::Int=1337,
    output_dir::AbstractString=mktempdir(),
)
    transcript_data_dir = joinpath(output_dir, "transcript_lm")
    ensure_transcript_language_parquet(
        transcript_source,
        transcript_data_dir;
        min_tokens=8,
    )

    transcript_config = ChessModelConfig(
        adapter=ChessAdapterConfig(vocab_size=length(CHESS_TRANSCRIPT_STOI), d_model=d_model, pad_token=0),
        core=WavePDECoreConfig(d_model=d_model, n_layer=n_layer, solver_steps=solver_steps, dt_init=dt_init, norm_eps=norm_eps),
        proposer=ChessMoveHeadConfig(vocab_size=length(CHESS_TRANSCRIPT_STOI), d_model=d_model, tie_embeddings=true, bias=false),
        max_seq_len=96,
    )
    transcript_corpus = ChessParquetCorpus(transcript_data_dir; min_tokens=8)
    transcript_cfg = TrainingConfig(
        data_dir=transcript_data_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_iters=max_iters,
        log_interval=1,
        min_tokens=8,
        train_file_update_interval=1,
        checkpoint_path=joinpath(output_dir, "transcript_only.jls"),
        seed=seed,
    )
    transcript_checkpoint = train!(ChessModel(transcript_config), transcript_corpus, transcript_cfg)

    state_config = ChessModelConfig(
        adapter=ChessAdapterConfig(vocab_size=BOARD_STATE_VOCAB_SIZE, d_model=d_model, pad_token=0),
        core=WavePDECoreConfig(d_model=d_model, n_layer=n_layer, solver_steps=solver_steps, dt_init=dt_init, norm_eps=norm_eps),
        proposer=ChessMoveHeadConfig(vocab_size=BOARD_STATE_VOCAB_SIZE, d_model=d_model, tie_embeddings=true, bias=false),
        max_seq_len=BOARD_STATE_SEQUENCE_LENGTH,
    )
    state_corpus = StateTransitionParquetCorpus(state_data_dir; min_tokens=BOARD_STATE_SEQUENCE_LENGTH)
    state_cfg = TrainingConfig(
        data_dir=state_data_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_iters=max_iters,
        log_interval=1,
        min_tokens=BOARD_STATE_SEQUENCE_LENGTH,
        train_file_update_interval=1,
        checkpoint_path=joinpath(output_dir, "state_only.jls"),
        seed=seed,
    )
    state_checkpoint = train!(ChessModel(state_config), state_corpus, state_cfg)

    hybrid_config = DualSurfaceStateModelConfig(
        adapter=ChessAdapterConfig(vocab_size=BOARD_STATE_VOCAB_SIZE, d_model=d_model, pad_token=0),
        core=WavePDECoreConfig(d_model=d_model, n_layer=n_layer, solver_steps=solver_steps, dt_init=dt_init, norm_eps=norm_eps),
        state_head=ChessMoveHeadConfig(vocab_size=BOARD_STATE_VOCAB_SIZE, d_model=d_model, tie_embeddings=true, bias=false),
        transcript_head=ChessMoveHeadConfig(vocab_size=length(CHESS_TRANSCRIPT_STOI), d_model=d_model, tie_embeddings=false, bias=true),
        max_seq_len=BOARD_STATE_SEQUENCE_LENGTH,
    )
    hybrid_corpus = DualSurfaceParquetCorpus(state_data_dir; min_tokens=BOARD_STATE_SEQUENCE_LENGTH)
    hybrid_cfg = DualSurfaceTrainingConfig(
        data_dir=state_data_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_iters=max_iters,
        log_interval=1,
        min_tokens=BOARD_STATE_SEQUENCE_LENGTH,
        train_file_update_interval=1,
        checkpoint_path=joinpath(output_dir, "hybrid.jls"),
        seed=seed,
    )
    hybrid_checkpoint = train_dual_surface!(DualSurfaceStateModel(hybrid_config), hybrid_corpus, hybrid_cfg)

    return (
        transcript_first=(
            checkpoint_path=transcript_cfg.checkpoint_path,
            final_loss=last(transcript_checkpoint.losses),
        ),
        state_first=(
            checkpoint_path=state_cfg.checkpoint_path,
            final_loss=last(state_checkpoint.losses),
        ),
        hybrid=(
            checkpoint_path=hybrid_cfg.checkpoint_path,
            final_loss=last(hybrid_checkpoint.losses),
            final_state_loss=last(hybrid_checkpoint.state_losses),
            final_transcript_loss=last(hybrid_checkpoint.transcript_losses),
        ),
    )
end
