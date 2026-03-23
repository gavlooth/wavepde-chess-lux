function board_value_requested_device()
    raw = lowercase(strip(get(ENV, "WAVEPDE_DEVICE", get(ENV, "WAVEPDE_VALUE_DEVICE", "auto"))))
    return Symbol(raw)
end

function board_value_resolve_device()
    requested = board_value_requested_device()
    if requested == :cpu
        return :cpu
    end
    if requested != :auto && requested != :gpu && requested != :cuda
        throw(ArgumentError("Unsupported WAVEPDE_DEVICE=$(requested). Use auto, cpu, gpu, or cuda."))
    end
    if CUDA.functional()
        return :gpu
    end
    if requested != :auto
        throw(
            ArgumentError("WAVEPDE_DEVICE=$(requested) requires CUDA, but CUDA.functional() is false."),
        )
    end
    return :cpu
end

move_to_device(x::NamedTuple, device::Symbol) =
    NamedTuple{keys(x)}(map(value -> move_to_device(value, device), values(x)))
move_to_device(x::Tuple, device::Symbol) = Tuple(move_to_device(value, device) for value in x)
move_to_device(x::AbstractArray, device::Symbol) =
    (device == :gpu ? CUDA.cu(x) : Array(x))
move_to_device(x::Nothing, device::Symbol) = nothing
move_to_device(x::Number, device::Symbol) = x
move_to_device(x::AbstractString, device::Symbol) = x
move_to_device(x::Bool, device::Symbol) = x
move_to_device(x, device::Symbol) = x

Base.@kwdef struct BoardValueTrainingConfig
    data_dir::String = joinpath(pwd(), "tmp", "hf", "stockfish")
    batch_size::Int = 32
    learning_rate::Float32 = 6.0f-4
    max_iters::Int = 100
    log_interval::Int = 10
    min_tokens::Int = BOARD_STATE_SEQUENCE_LENGTH
    train_file_update_interval::Int = 10
    training_policy::Symbol = :full
    cp_scale::Float32 = 400f0
    chunk_rows::Int = 20_000
    checkpoint_path::String = joinpath(pwd(), "checkpoints", "wavepde_chess_value_checkpoint.jls")
    seed::Int = 1337
end

mutable struct BoardValueParquetCorpus
    db::DuckDB.DB
    conn
    files::Vector{String}
    file_row_counts::Dict{String, Int}
    active_file::String
    active_offset::Int
    active_states::Vector{Vector{Int32}}
    active_values::Vector{Float32}
    active_depths::Vector{Int}
    min_tokens::Int
    cp_scale::Float32
    chunk_rows::Int
end

function BoardValueParquetCorpus(
    data_dir::AbstractString;
    min_tokens::Int=BOARD_STATE_SEQUENCE_LENGTH,
    cp_scale::Float32=400f0,
    chunk_rows::Int=20_000,
)
    files = discover_parquet_files(data_dir)
    isempty(files) && throw(ArgumentError("No parquet files found under $(data_dir)."))
    chunk_rows > 0 || throw(ArgumentError("chunk_rows must be positive, got $(chunk_rows)."))

    db = DuckDB.DB()
    conn = DBInterface.connect(db)
    file_row_counts = Dict(path => count_parquet_rows(conn, path) for path in files)
    active_file = first(files)
    active_offset = 0
    active_states, active_values, active_depths = load_board_value_examples(
        conn,
        active_file;
        min_tokens=min_tokens,
        cp_scale=cp_scale,
        limit=chunk_rows,
        offset=active_offset,
    )

    return BoardValueParquetCorpus(
        db,
        conn,
        files,
        file_row_counts,
        active_file,
        active_offset,
        active_states,
        active_values,
        active_depths,
        min_tokens,
        cp_scale,
        chunk_rows,
    )
end

function reload_file!(corpus::BoardValueParquetCorpus, path::AbstractString; offset::Int=0)
    corpus.active_file = path
    corpus.active_offset = max(offset, 0)
    corpus.active_states, corpus.active_values, corpus.active_depths = load_board_value_examples(
        corpus.conn,
        path;
        min_tokens=corpus.min_tokens,
        cp_scale=corpus.cp_scale,
        limit=corpus.chunk_rows,
        offset=corpus.active_offset,
    )
    return corpus
end

function maybe_rotate_file!(corpus::BoardValueParquetCorpus, rng::AbstractRNG, step::Int; every::Int)
    if every > 0 && (step == 1 || (step > 1 && step % every == 0))
        next_file = corpus.files[rand(rng, eachindex(corpus.files))]
        row_count = get(corpus.file_row_counts, next_file, 0)
        max_offset = max(row_count - corpus.chunk_rows, 0)
        next_offset = max_offset == 0 ? 0 : rand(rng, 0:max_offset)
        if next_file != corpus.active_file || next_offset != corpus.active_offset
            reload_file!(corpus, next_file; offset=next_offset)
        end
    end
    return corpus
end

function sample_board_value_batch(
    corpus::BoardValueParquetCorpus,
    rng::AbstractRNG;
    batch_size::Int,
    max_seq_len::Int,
)
    sampled_states = Vector{Vector{Int32}}(undef, batch_size)
    sampled_values = zeros(Float32, 1, batch_size)
    max_batch_len = 0

    for i in 1:batch_size
        idx = rand(rng, eachindex(corpus.active_states))
        state_tokens = corpus.active_states[idx]
        sampled_states[i] = state_tokens
        sampled_values[1, i] = corpus.active_values[idx]
        max_batch_len = max(max_batch_len, min(length(state_tokens), max_seq_len))
    end

    tokens = zeros(Int32, max_batch_len, batch_size)
    for i in 1:batch_size
        token_count = min(length(sampled_states[i]), max_batch_len)
        tokens[1:token_count, i] .= sampled_states[i][1:token_count]
    end

    return (
        tokens=tokens,
        value_targets=sampled_values,
    )
end

function trainable_parameter_groups(::BoardValueModel, policy::Symbol)
    validate_training_policy(policy)
    if policy == :full
        return (:adapter, :core, :value_head)
    elseif policy == :adapters_only
        return (:adapter,)
    else
        return (:value_head,)
    end
end

training_value_targets(batch::NamedTuple) = get(batch, :value_targets, nothing)

function board_value_loss(model::BoardValueModel, ps, st, batch)
    targets = training_value_targets(batch)
    targets === nothing && throw(ArgumentError("Board value training batch must include value_targets."))
    predictions, _ = Lux.apply(model, batch.tokens, ps, st)
    size(predictions) == size(targets) || throw(ArgumentError(
        "Board value prediction and target shapes differ: got $(size(predictions)) vs $(size(targets)).",
    ))
    return checker_regression_loss(predictions, targets)
end

function train_value!(model::BoardValueModel, corpus::BoardValueParquetCorpus, cfg::BoardValueTrainingConfig)
    rng = MersenneTwister(cfg.seed)
    training_policy = validate_training_policy(cfg.training_policy)
    ps, st = Lux.setup(rng, model)
    device = board_value_resolve_device()
    ps, st = move_to_device((ps, st), device)
    optimizer = Optimisers.Adam(cfg.learning_rate)
    opt_state = Optimisers.setup(optimizer, ps)
    losses = Float32[]

    if device == :gpu
        println("training_device=gpu")
    else
        println("training_device=cpu")
    end

    for step in 1:cfg.max_iters
        maybe_rotate_file!(corpus, rng, step; every=cfg.train_file_update_interval)
        batch = move_to_device(
            sample_board_value_batch(corpus, rng; batch_size=cfg.batch_size, max_seq_len=model.config.max_seq_len),
            device,
        )

        loss = board_value_loss(model, ps, st, batch)
        grads = Zygote.gradient(p -> board_value_loss(model, p, st, batch), ps)[1]
        grads = apply_training_policy(model, grads, training_policy)
        opt_state, ps = Optimisers.update(opt_state, ps, grads)

        push!(losses, Float32(loss))
        if step % cfg.log_interval == 0 || step == 1 || step == cfg.max_iters
            println(
                "step=$(step) loss=$(round(loss; digits=4)) seq_len=$(size(batch.tokens, 1)) file=$(basename(corpus.active_file)) offset=$(corpus.active_offset)",
            )
        end
    end

    checkpoint = (
        model_config=model.config,
        training_config=cfg,
        parameters=move_to_device(ps, :cpu),
        state=move_to_device(st, :cpu),
        losses=losses,
    )
    save_checkpoint(cfg.checkpoint_path, checkpoint)
    return checkpoint
end

function load_board_value_checkpoint(checkpoint_path::AbstractString)
    payload = open(checkpoint_path, "r") do io
        deserialize(io)
    end
    haskey(payload, :model_config) || throw(ArgumentError("Checkpoint $(checkpoint_path) is missing a model_config field."))
    model = BoardValueModel(payload.model_config)
    return (
        payload=payload,
        model=model,
        parameters=payload.parameters,
        state=payload.state,
    )
end

function board_value_metrics(predictions::AbstractMatrix{<:Real}, targets::AbstractMatrix{<:Real})
    size(predictions) == size(targets) || throw(ArgumentError(
        "board_value_metrics expects matching shapes, got $(size(predictions)) and $(size(targets)).",
    ))
    regression = checker_prediction_metrics(predictions, targets)
    pred_vec = vec(Float32.(predictions))
    target_vec = vec(Float32.(targets))
    count_examples = length(pred_vec)
    direction_matches = sum((pred_vec .>= 0f0) .== (target_vec .>= 0f0))
    return (
        mse=regression.mse,
        mae=regression.mae,
        rmse=regression.rmse,
        max_abs_error=regression.max_abs_error,
        direction_accuracy=direction_matches / count_examples,
        mean_prediction=mean(pred_vec),
        mean_target=mean(target_vec),
        predicted_positive_rate=sum(pred_vec .>= 0f0) / count_examples,
        target_positive_rate=sum(target_vec .>= 0f0) / count_examples,
    )
end

function evaluate_board_value_corpus(
    model::BoardValueModel,
    ps,
    st,
    corpus::BoardValueParquetCorpus;
    batch_size::Int=32,
    max_examples::Int=0,
)
    total_predictions = Float32[]
    total_targets = Float32[]
    examples_seen = 0

    for file_path in corpus.files
        row_count = get(corpus.file_row_counts, file_path, 0)
        for offset in 0:corpus.chunk_rows:max(row_count - 1, 0)
            reload_file!(corpus, file_path; offset=offset)
            for start_idx in 1:batch_size:length(corpus.active_states)
                subset = start_idx:min(start_idx + batch_size - 1, length(corpus.active_states))
                max_len = maximum(length(state) for state in corpus.active_states[subset])
                tokens = zeros(Int32, max_len, length(subset))
                targets = zeros(Float32, 1, length(subset))
                for (batch_idx, example_idx) in enumerate(subset)
                    state_tokens = corpus.active_states[example_idx]
                    tokens[1:length(state_tokens), batch_idx] .= state_tokens
                    targets[1, batch_idx] = corpus.active_values[example_idx]
                end
                predictions, _ = Lux.apply(model, tokens, ps, st)
                append!(total_predictions, vec(Float32.(predictions)))
                append!(total_targets, vec(Float32.(targets)))
                examples_seen += length(subset)
                if max_examples > 0 && examples_seen >= max_examples
                    break
                end
            end
            if max_examples > 0 && examples_seen >= max_examples
                break
            end
        end
        if max_examples > 0 && examples_seen >= max_examples
            break
        end
    end

    isempty(total_predictions) && throw(ArgumentError("No board-value examples were evaluated."))
    prediction_matrix = reshape(total_predictions, 1, :)
    target_matrix = reshape(total_targets, 1, :)
    metrics = board_value_metrics(prediction_matrix, target_matrix)
    return merge(
        metrics,
        (
            num_examples=length(total_predictions),
        ),
    )
end

function evaluate_board_value_checkpoint(
    checkpoint_path::AbstractString,
    data_dir::AbstractString;
    batch_size::Int=32,
    max_examples::Int=0,
    chunk_rows::Int=20_000,
)
    checkpoint = load_board_value_checkpoint(checkpoint_path)
    cp_scale = hasproperty(checkpoint.payload.training_config, :cp_scale) ? checkpoint.payload.training_config.cp_scale : 400f0
    corpus = BoardValueParquetCorpus(
        data_dir;
        min_tokens=BOARD_STATE_SEQUENCE_LENGTH,
        cp_scale=Float32(cp_scale),
        chunk_rows=chunk_rows,
    )
    evaluation = evaluate_board_value_corpus(
        checkpoint.model,
        checkpoint.parameters,
        checkpoint.state,
        corpus;
        batch_size=batch_size,
        max_examples=max_examples,
    )
    return merge(
        (
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            cp_scale=Float32(cp_scale),
        ),
        evaluation,
    )
end
