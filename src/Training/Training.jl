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
    checker_loss_weight::Float32 = 1.0f0
    transition_loss_weight::Float32 = 0.0f0
    transition_candidates_per_example::Int = 1
    training_policy::Symbol = :full
    board_target_mode::Symbol = :none
    policy_condition_mode::Symbol = :state_only
    checkpoint_path::String = joinpath(pwd(), "checkpoints", "wavepde_chess_checkpoint.jls")
    seed::Int = 1337
end

mutable struct ChessParquetCorpus
    db::DuckDB.DB
    conn
    files::Vector{String}
    active_file::String
    active_games::Vector{Vector{Int32}}
    active_checker_targets::Union{Nothing, Vector{Vector{Float32}}}
    checker_target_dim::Int
    min_tokens::Int
    board_target_mode::Symbol
end

mutable struct StateTransitionParquetCorpus
    db::DuckDB.DB
    conn
    files::Vector{String}
    active_file::String
    active_states::Vector{Vector{Int32}}
    active_next_states::Vector{Vector{Int32}}
    active_moves::Union{Nothing, Vector{String}}
    min_tokens::Int
end

const CHECKER_COLUMN_CANDIDATES = (
    "checker_targets",
    "checker_target",
    "checker_labels",
    "checker_score",
)

const TRAINING_POLICIES = (:full, :adapters_only, :heads_only)
const BOARD_TARGET_MODES = (:none, :transcript_board_facts)
const POLICY_CONDITION_MODES = (:state_only, :state_action)
const TRANSCRIPT_COLUMN_CANDIDATES = ("transcript",)
const MOVE_COLUMN_CANDIDATES = ("move_san",)

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

function parquet_column_names(conn, path::AbstractString)
    rows = DBInterface.execute(conn, "DESCRIBE SELECT * FROM read_parquet('$(sql_escape(path))')")
    return String[row[1] for row in rows]
end

function find_checker_column(columns::AbstractVector{<:AbstractString})
    for candidate in CHECKER_COLUMN_CANDIDATES
        candidate in columns && return candidate
    end
    return nothing
end

function find_transcript_column(columns::AbstractVector{<:AbstractString})
    for candidate in TRANSCRIPT_COLUMN_CANDIDATES
        candidate in columns && return candidate
    end
    return nothing
end

function find_state_tokens_column(columns::AbstractVector{<:AbstractString})
    for candidate in STATE_TOKENS_COLUMN_CANDIDATES
        candidate in columns && return candidate
    end
    return nothing
end

function find_next_state_tokens_column(columns::AbstractVector{<:AbstractString})
    for candidate in NEXT_STATE_TOKENS_COLUMN_CANDIDATES
        candidate in columns && return candidate
    end
    return nothing
end

function find_move_column(columns::AbstractVector{<:AbstractString})
    for candidate in MOVE_COLUMN_CANDIDATES
        candidate in columns && return candidate
    end
    return nothing
end

function parse_int_sequence(values)
    return Int32[x for x in values if !ismissing(x)]
end

function parse_float_sequence(values)
    return Float32[x for x in values if !ismissing(x)]
end

function validate_positive_count(name::AbstractString, value::Integer)
    value > 0 || throw(ArgumentError("$(name) must be positive, got $(value)."))
    return value
end

function validate_board_target_mode(mode::Symbol)
    mode in BOARD_TARGET_MODES || throw(ArgumentError(
        "Unsupported board_target_mode $(mode). Supported modes are $(BOARD_TARGET_MODES).",
    ))
    return mode
end

function validate_policy_condition_mode(mode::Symbol)
    mode in POLICY_CONDITION_MODES || throw(ArgumentError(
        "Unsupported policy_condition_mode $(mode). Supported modes are $(POLICY_CONDITION_MODES).",
    ))
    return mode
end

function load_transcript_board_examples(conn, path::AbstractString, transcript_column::AbstractString; min_tokens::Int=8)
    rows = DBInterface.execute(
        conn,
        "SELECT $(transcript_column) AS transcript FROM read_parquet('$(sql_escape(path))')",
    )
    games = Vector{Vector{Int32}}()
    checker_targets = Vector{Vector{Float32}}()
    checker_target_dim = length(CHESS_BOARD_TARGET_NAMES)

    for row in rows
        transcript = row[1]
        transcript === missing && continue
        sequence = encode_chess_transcript(String(transcript))
        length(sequence) >= min_tokens || continue
        target = extract_board_targets_from_transcript(String(transcript))
        length(target) == checker_target_dim || throw(ArgumentError(
            "Transcript-derived checker target length mismatch in $(path): expected $(checker_target_dim), got $(length(target)).",
        ))
        push!(games, sequence)
        push!(checker_targets, target)
    end

    isempty(games) && throw(ArgumentError(
        "No usable transcript-derived games found in $(path).",
    ))
    return games, checker_targets, checker_target_dim
end

function build_transition_examples(
    token_sequences::AbstractVector{<:AbstractVector{<:Integer}},
    rng::AbstractRNG;
    max_seq_len::Int,
    candidates_per_example::Int=1,
)
    candidates_per_example = validate_positive_count("candidates_per_example", candidates_per_example)
    max_seq_len = validate_positive_count("max_seq_len", max_seq_len)
    transition_tokens = Vector{Vector{Int32}}()
    transition_targets = Vector{Vector{Float32}}()
    transition_target_dim = 0

    for tokens in token_sequences
        context_len = min(length(tokens), max_seq_len)
        context_len >= max_seq_len && continue
        context_tokens = view(tokens, 1:context_len)
        transcript = decode_chess_tokens(context_tokens)
        legal_moves = legal_san_candidates_from_transcript(transcript)
        isempty(legal_moves) && continue

        selected_candidates = if candidates_per_example >= length(legal_moves)
            shuffle(rng, legal_moves)
        else
            legal_moves[randperm(rng, length(legal_moves))[1:candidates_per_example]]
        end

        transition = transition_board_targets(transcript, selected_candidates)
        transition_target_dim == 0 && (transition_target_dim = size(transition.targets, 1))

        for (candidate_idx, candidate) in enumerate(selected_candidates)
            candidate_tokens = encode_chess_candidate_san(candidate)
            context_len + length(candidate_tokens) > max_seq_len && continue
            augmented_tokens = append_chess_candidate_san(context_tokens, candidate)
            push!(transition_tokens, augmented_tokens)
            push!(transition_targets, vec(transition.targets[:, candidate_idx]))
        end
    end

    isempty(transition_tokens) && return nothing, nothing, 0
    return transition_tokens, transition_targets, transition_target_dim
end

function load_training_examples(conn, path::AbstractString; min_tokens::Int=8, board_target_mode::Symbol=:none)
    assert_real_parquet(path)
    board_target_mode = validate_board_target_mode(board_target_mode)
    columns = parquet_column_names(conn, path)
    checker_column = find_checker_column(columns)
    transcript_column = find_transcript_column(columns)

    if board_target_mode == :transcript_board_facts
        transcript_column === nothing && throw(ArgumentError(
            "board_target_mode=:transcript_board_facts requires a transcript column in $(path).",
        ))
        return load_transcript_board_examples(conn, path, transcript_column; min_tokens=min_tokens)
    end

    if checker_column === nothing
        rows = DBInterface.execute(conn, "SELECT tokenized FROM read_parquet('$(sql_escape(path))')")
        games = Vector{Vector{Int32}}()
        for row in rows
            sequence = parse_int_sequence(row[1])
            length(sequence) >= min_tokens || continue
            push!(games, sequence)
        end
        return games, nothing, 0
    end

    rows = DBInterface.execute(
        conn,
        "SELECT tokenized, $(checker_column) AS checker_targets FROM read_parquet('$(sql_escape(path))')",
    )
    games = Vector{Vector{Int32}}()
    checker_targets = Vector{Vector{Float32}}()
    checker_target_dim = 0
    for row in rows
        sequence = parse_int_sequence(row[1])
        length(sequence) >= min_tokens || continue
        checker_values = row[2]
        checker_values === missing && continue
        target = parse_float_sequence(checker_values)
        isempty(target) && continue
        if checker_target_dim == 0
            checker_target_dim = length(target)
        else
            length(target) == checker_target_dim || throw(ArgumentError(
                "Checker target length mismatch in $(path): expected $(checker_target_dim), got $(length(target)).",
            ))
        end
        push!(games, sequence)
        push!(checker_targets, target)
    end
    isempty(games) && throw(ArgumentError("No usable tokenized games found in $(path)."))
    return games, checker_targets, checker_target_dim
end

function load_tokenized_games(conn, path::AbstractString; min_tokens::Int=8, board_target_mode::Symbol=:none)
    games, _, _ = load_training_examples(conn, path; min_tokens=min_tokens, board_target_mode=board_target_mode)
    return games
end

function load_state_transition_examples(conn, path::AbstractString; min_tokens::Int=BOARD_STATE_SEQUENCE_LENGTH)
    assert_real_parquet(path)
    columns = parquet_column_names(conn, path)
    state_column = find_state_tokens_column(columns)
    next_state_column = find_next_state_tokens_column(columns)
    move_column = find_move_column(columns)

    state_column === nothing && throw(ArgumentError("State-transition parquet $(path) must include a state_tokens column."))
    next_state_column === nothing && throw(ArgumentError("State-transition parquet $(path) must include a next_state_tokens column."))

    select_columns = ["$(state_column) AS state_tokens", "$(next_state_column) AS next_state_tokens"]
    move_column === nothing || push!(select_columns, "$(move_column) AS move_san")
    rows = DBInterface.execute(
        conn,
        "SELECT $(join(select_columns, ", ")) FROM read_parquet('$(sql_escape(path))')",
    )
    states = Vector{Vector{Int32}}()
    next_states = Vector{Vector{Int32}}()
    moves = move_column === nothing ? nothing : String[]
    for row in rows
        state_tokens = parse_int_sequence(row[1])
        next_state_tokens = parse_int_sequence(row[2])
        length(state_tokens) >= min_tokens || continue
        length(next_state_tokens) >= min_tokens || continue
        length(state_tokens) == length(next_state_tokens) || throw(ArgumentError(
            "State-transition length mismatch in $(path): got $(length(state_tokens)) and $(length(next_state_tokens)).",
        ))
        push!(states, state_tokens)
        push!(next_states, next_state_tokens)
        if moves !== nothing
            move_san = row[3]
            move_text = move_san === missing ? "" : String(move_san)
            push!(moves, move_text)
        end
    end

    isempty(states) && throw(ArgumentError("No usable state-transition examples found in $(path)."))
    return states, next_states, moves
end

function ChessParquetCorpus(data_dir::AbstractString; min_tokens::Int=8, board_target_mode::Symbol=:none)
    files = discover_parquet_files(data_dir)
    isempty(files) && throw(ArgumentError("No parquet files found under $(data_dir)."))

    db = DuckDB.DB()
    conn = DBInterface.connect(db)
    active_file = first(files)
    board_target_mode = validate_board_target_mode(board_target_mode)
    active_games, active_checker_targets, checker_target_dim = load_training_examples(
        conn,
        active_file;
        min_tokens=min_tokens,
        board_target_mode=board_target_mode,
    )
    isempty(active_games) && throw(ArgumentError("No usable tokenized games found in $(active_file)."))

    return ChessParquetCorpus(
        db,
        conn,
        files,
        active_file,
        active_games,
        active_checker_targets,
        checker_target_dim,
        min_tokens,
        board_target_mode,
    )
end

function StateTransitionParquetCorpus(data_dir::AbstractString; min_tokens::Int=BOARD_STATE_SEQUENCE_LENGTH)
    files = discover_parquet_files(data_dir)
    isempty(files) && throw(ArgumentError("No parquet files found under $(data_dir)."))

    db = DuckDB.DB()
    conn = DBInterface.connect(db)
    active_file = first(files)
    active_states, active_next_states, active_moves = load_state_transition_examples(
        conn,
        active_file;
        min_tokens=min_tokens,
    )

    return StateTransitionParquetCorpus(
        db,
        conn,
        files,
        active_file,
        active_states,
        active_next_states,
        active_moves,
        min_tokens,
    )
end

function reload_file!(corpus::ChessParquetCorpus, path::AbstractString)
    corpus.active_file = path
    corpus.active_games, corpus.active_checker_targets, corpus.checker_target_dim = load_training_examples(
        corpus.conn,
        path;
        min_tokens=corpus.min_tokens,
        board_target_mode=corpus.board_target_mode,
    )
    isempty(corpus.active_games) && throw(ArgumentError("No usable tokenized games found in $(path)."))
    return corpus
end

function reload_file!(corpus::StateTransitionParquetCorpus, path::AbstractString)
    corpus.active_file = path
    corpus.active_states, corpus.active_next_states, corpus.active_moves = load_state_transition_examples(
        corpus.conn,
        path;
        min_tokens=corpus.min_tokens,
    )
    isempty(corpus.active_states) && throw(ArgumentError("No usable state-transition examples found in $(path)."))
    return corpus
end

function maybe_rotate_file!(corpus::ChessParquetCorpus, rng::AbstractRNG, step::Int; every::Int)
    if every > 0 && step > 1 && step % every == 0 && length(corpus.files) > 1
        next_file = corpus.files[rand(rng, eachindex(corpus.files))]
        next_file == corpus.active_file || reload_file!(corpus, next_file)
    end
    return corpus
end

function maybe_rotate_file!(corpus::StateTransitionParquetCorpus, rng::AbstractRNG, step::Int; every::Int)
    if every > 0 && step > 1 && step % every == 0 && length(corpus.files) > 1
        next_file = corpus.files[rand(rng, eachindex(corpus.files))]
        next_file == corpus.active_file || reload_file!(corpus, next_file)
    end
    return corpus
end

function sample_training_batch(
    corpus::ChessParquetCorpus,
    rng::AbstractRNG;
    batch_size::Int,
    max_seq_len::Int,
    transition_candidates_per_example::Int=0,
    policy_condition_mode::Symbol=:state_only,
)
    sampled_games = Vector{Vector{Int32}}(undef, batch_size)
    sampled_checker_targets = corpus.active_checker_targets === nothing ? nothing : Vector{Vector{Float32}}(undef, batch_size)
    max_batch_len = 0

    for i in 1:batch_size
        idx = rand(rng, eachindex(corpus.active_games))
        game = corpus.active_games[idx]
        sampled_games[i] = game
        sampled_checker_targets === nothing || (sampled_checker_targets[i] = corpus.active_checker_targets[idx])
        max_batch_len = max(max_batch_len, min(length(game), max_seq_len))
    end

    tokens = zeros(Int32, max_batch_len, batch_size)
    for (i, game) in enumerate(sampled_games)
        game_len = min(length(game), max_batch_len)
        tokens[1:game_len, i] .= game[1:game_len]
    end

    checker_targets = nothing
    if sampled_checker_targets !== nothing
        checker_targets = zeros(Float32, corpus.checker_target_dim, batch_size)
        for i in 1:batch_size
            target = sampled_checker_targets[i]
            length(target) == corpus.checker_target_dim || throw(ArgumentError(
                "Checker target length mismatch in sampled batch: expected $(corpus.checker_target_dim), got $(length(target)).",
            ))
            checker_targets[:, i] .= target
        end
    end

    transition_tokens = nothing
    transition_targets = nothing
    if corpus.board_target_mode == :transcript_board_facts && transition_candidates_per_example > 0
        transition_token_vectors, transition_target_vectors, transition_target_dim = build_transition_examples(
            sampled_games,
            rng;
            max_seq_len=max_seq_len,
            candidates_per_example=transition_candidates_per_example,
        )
        if transition_token_vectors !== nothing
            max_transition_len = maximum(length.(transition_token_vectors))
            transition_tokens = zeros(Int32, max_transition_len, length(transition_token_vectors))
            for (i, sequence) in enumerate(transition_token_vectors)
                transition_tokens[1:length(sequence), i] .= sequence
            end
            transition_targets = zeros(Float32, transition_target_dim, length(transition_target_vectors))
            for (i, target) in enumerate(transition_target_vectors)
                transition_targets[:, i] .= target
            end
        end
    end

    return (
        tokens=tokens,
        checker_targets=checker_targets,
        transition_tokens=transition_tokens,
        transition_targets=transition_targets,
    )
end

function sample_batch(corpus::ChessParquetCorpus, rng::AbstractRNG; batch_size::Int, max_seq_len::Int)
    return sample_training_batch(corpus, rng; batch_size=batch_size, max_seq_len=max_seq_len).tokens
end

function sample_training_batch(
    corpus::StateTransitionParquetCorpus,
    rng::AbstractRNG;
    batch_size::Int,
    max_seq_len::Int,
    transition_candidates_per_example::Int=0,
    policy_condition_mode::Symbol=:state_only,
)
    policy_condition_mode = validate_policy_condition_mode(policy_condition_mode)
    sampled_states = Vector{Vector{Int32}}(undef, batch_size)
    sampled_next_states = Vector{Vector{Int32}}(undef, batch_size)
    sampled_moves = policy_condition_mode == :state_action ? Vector{String}(undef, batch_size) : String[]
    max_batch_len = 0

    for i in 1:batch_size
        idx = rand(rng, eachindex(corpus.active_states))
        state_tokens = corpus.active_states[idx]
        next_state_tokens = corpus.active_next_states[idx]
        sampled_states[i] = state_tokens
        sampled_next_states[i] = next_state_tokens
        if policy_condition_mode == :state_action
            corpus.active_moves === nothing && throw(ArgumentError(
                "policy_condition_mode=:state_action requires parquet rows with move_san values.",
            ))
            candidate = corpus.active_moves[idx]
            isempty(candidate) && throw(ArgumentError(
                "policy_condition_mode=:state_action requires non-empty move_san values in $(corpus.active_file).",
            ))
            conditioned_tokens = append_policy_action_tokens(state_tokens, candidate)
            length(conditioned_tokens) <= max_seq_len || throw(ArgumentError(
                "Conditioned state-action sequence length $(length(conditioned_tokens)) exceeds max_seq_len $(max_seq_len). Increase model.config.max_seq_len or reduce action-token length.",
            ))
            sampled_moves[i] = candidate
            max_batch_len = max(max_batch_len, length(conditioned_tokens))
        else
            max_batch_len = max(max_batch_len, min(length(state_tokens), max_seq_len))
        end
    end

    tokens = zeros(Int32, max_batch_len, batch_size)
    target_tokens = zeros(Int32, max_batch_len, batch_size)
    target_mask = falses(max_batch_len, batch_size)
    for i in 1:batch_size
        if policy_condition_mode == :state_action
            conditioned_tokens = append_policy_action_tokens(sampled_states[i], sampled_moves[i])
            tokens[1:length(conditioned_tokens), i] .= conditioned_tokens
            target_length = length(sampled_next_states[i])
            target_tokens[1:target_length, i] .= sampled_next_states[i]
            target_mask[1:target_length, i] .= true
        else
            token_count = min(length(sampled_states[i]), max_batch_len)
            tokens[1:token_count, i] .= sampled_states[i][1:token_count]
            target_tokens[1:token_count, i] .= sampled_next_states[i][1:token_count]
            target_mask[1:token_count, i] .= true
        end
    end

    return (
        tokens=tokens,
        target_tokens=target_tokens,
        target_mask=target_mask,
    )
end

function sample_batch(corpus::StateTransitionParquetCorpus, rng::AbstractRNG; batch_size::Int, max_seq_len::Int)
    return sample_training_batch(corpus, rng; batch_size=batch_size, max_seq_len=max_seq_len).tokens
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

function masked_cross_entropy(
    logits::AbstractArray{T, 3},
    targets::AbstractMatrix{<:Integer},
    target_mask::AbstractMatrix{Bool},
) where {T<:AbstractFloat}
    _, seq_len, batch_size = size(logits)
    @assert seq_len == size(targets, 1) "Logit and target sequence lengths differ."
    @assert batch_size == size(targets, 2) "Logit and target batch sizes differ."
    @assert size(target_mask) == size(targets) "Target mask shape $(size(target_mask)) must match targets $(size(targets))."

    total = zero(T)
    count = 0

    for b in 1:batch_size
        for t in 1:seq_len
            target_mask[t, b] || continue
            target_idx = Int(targets[t, b]) + 1
            column = view(logits, :, t, b)
            max_logit = maximum(column)
            log_denom = max_logit + log(sum(exp.(column .- max_logit)))
            total += log_denom - column[target_idx]
            count += 1
        end
    end

    count > 0 || throw(ArgumentError("Masked cross-entropy received an empty target mask."))
    return total / T(count)
end

function extract_proposer_logits(output)
    if output isa NamedTuple
        haskey(output, :proposer) || throw(ArgumentError("Model output NamedTuple must contain :proposer logits."))
        return output.proposer
    end
    return output
end

function extract_checker_scores(output)
    if output isa NamedTuple && haskey(output, :checker)
        return output.checker
    end
    return nothing
end

training_tokens(batch::AbstractMatrix{<:Integer}) = batch
training_tokens(batch::NamedTuple) = batch.tokens

training_checker_targets(::AbstractMatrix) = nothing
training_checker_targets(batch::NamedTuple) = get(batch, :checker_targets, nothing)

training_transition_tokens(::AbstractMatrix) = nothing
training_transition_tokens(batch::NamedTuple) = get(batch, :transition_tokens, nothing)

training_transition_targets(::AbstractMatrix) = nothing
training_transition_targets(batch::NamedTuple) = get(batch, :transition_targets, nothing)

training_target_tokens(::AbstractMatrix) = nothing
training_target_tokens(batch::NamedTuple) = get(batch, :target_tokens, nothing)

training_target_mask(::AbstractMatrix) = nothing
training_target_mask(batch::NamedTuple) = get(batch, :target_mask, nothing)

function validate_training_policy(policy::Symbol)
    policy in TRAINING_POLICIES || throw(ArgumentError(
        "Unsupported training policy $(policy). Supported policies are $(TRAINING_POLICIES).",
    ))
    return policy
end

function trainable_parameter_groups(::ChessModel, policy::Symbol)
    validate_training_policy(policy)
    if policy == :full
        return (:adapter, :core, :proposer)
    elseif policy == :adapters_only
        return (:adapter,)
    else
        return (:proposer,)
    end
end

function trainable_parameter_groups(::ChessMultiHeadModel, policy::Symbol)
    validate_training_policy(policy)
    if policy == :full
        return (:adapter, :core, :proposer, :checker)
    elseif policy == :adapters_only
        return (:adapter,)
    else
        return (:proposer, :checker)
    end
end

function zero_like_tree(x::AbstractArray)
    y = similar(x)
    fill!(y, zero(eltype(x)))
    return y
end

zero_like_tree(x::Number) = zero(x)
zero_like_tree(x::Nothing) = nothing
zero_like_tree(x::Tuple) = tuple((zero_like_tree(v) for v in x)...)

function zero_like_tree(x::NamedTuple)
    return NamedTuple{keys(x)}(tuple((zero_like_tree(v) for v in values(x))...))
end

zero_like_tree(x) = zero(x)

function mask_frozen_parameter_groups(grads::NamedTuple, trainable_groups::Tuple{Vararg{Symbol}})
    return (; (
        group => (group in trainable_groups ? value : zero_like_tree(value))
        for (group, value) in pairs(grads)
    )...)
end

function apply_training_policy(model, grads, policy::Symbol)
    return mask_frozen_parameter_groups(grads, trainable_parameter_groups(model, policy))
end

function next_token_prediction_loss(model, ps, st, batch_tokens::AbstractMatrix{<:Integer})
    @assert size(batch_tokens, 1) >= 2 "Batch sequence length must be at least 2 for next-token prediction."
    inputs = batch_tokens[1:(end - 1), :]
    targets = batch_tokens[2:end, :]
    output, _ = Lux.apply(model, inputs, ps, st)
    logits = extract_proposer_logits(output)
    return autoregressive_cross_entropy(logits, targets), output
end

function paired_token_prediction_loss(
    model,
    ps,
    st,
    input_tokens::AbstractMatrix{<:Integer},
    target_tokens::AbstractMatrix{<:Integer},
    target_mask::Union{Nothing, AbstractMatrix{Bool}}=nothing,
)
    size(input_tokens) == size(target_tokens) || throw(ArgumentError(
        "Paired token prediction expects matching input and target shapes, got $(size(input_tokens)) vs $(size(target_tokens)).",
    ))
    output, _ = Lux.apply(model, input_tokens, ps, st)
    logits = extract_proposer_logits(output)
    size(logits, 2) == size(target_tokens, 1) || throw(ArgumentError(
        "Model output sequence length $(size(logits, 2)) does not match paired target length $(size(target_tokens, 1)).",
    ))
    size(logits, 3) == size(target_tokens, 2) || throw(ArgumentError(
        "Model output batch size $(size(logits, 3)) does not match paired target batch size $(size(target_tokens, 2)).",
    ))
    loss = target_mask === nothing ? autoregressive_cross_entropy(logits, target_tokens) : masked_cross_entropy(logits, target_tokens, target_mask)
    return loss, output
end

function proposer_prediction_loss(model, ps, st, batch)
    target_tokens = training_target_tokens(batch)
    target_mask = training_target_mask(batch)
    if target_tokens === nothing
        return next_token_prediction_loss(model, ps, st, training_tokens(batch))
    end
    return paired_token_prediction_loss(model, ps, st, training_tokens(batch), target_tokens, target_mask)
end

function checker_regression_loss(predictions::AbstractArray{T, 2}, targets::AbstractArray{<:Real, 2}) where {T<:AbstractFloat}
    size(predictions) == size(targets) || throw(ArgumentError(
        "Checker prediction and target shapes differ: got $(size(predictions)) vs $(size(targets)).",
    ))

    total = zero(T)
    count = length(predictions)
    for i in eachindex(predictions, targets)
        diff = predictions[i] - T(targets[i])
        total += diff * diff
    end
    return total / T(count)
end

function autoregressive_loss(model::ChessModel, ps, st, batch; checker_loss_weight::Real=1.0, transition_loss_weight::Real=0.0)
    token_loss, _ = proposer_prediction_loss(model, ps, st, batch)
    return token_loss
end

function autoregressive_loss(
    model::ChessMultiHeadModel,
    ps,
    st,
    batch;
    checker_loss_weight::Real=1.0,
    transition_loss_weight::Real=0.0,
)
    token_batch = training_tokens(batch)
    token_loss, output = proposer_prediction_loss(model, ps, st, batch)
    total_loss = token_loss

    checker_targets = training_checker_targets(batch)
    if checker_targets !== nothing
        checker_scores = extract_checker_scores(output)
        checker_scores === nothing && throw(ArgumentError("Multi-head model output must include :checker scores when checker supervision is provided."))
        checker_loss = checker_regression_loss(checker_scores, checker_targets)
        total_loss += Float32(checker_loss_weight) * checker_loss
    end

    transition_targets = training_transition_targets(batch)
    transition_tokens = training_transition_tokens(batch)
    if transition_loss_weight > 0 && transition_tokens !== nothing && transition_targets !== nothing
        transition_outputs, _ = Lux.apply(model, transition_tokens, ps, st)
        transition_scores = extract_checker_scores(transition_outputs)
        transition_scores === nothing && throw(ArgumentError("Multi-head model output must include :checker scores when transition supervision is provided."))
        transition_loss = checker_regression_loss(transition_scores, transition_targets)
        total_loss += Float32(transition_loss_weight) * transition_loss
    end

    return total_loss
end

function save_checkpoint(path::AbstractString, payload)
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, payload)
    end
    return path
end

function train!(model, corpus, cfg::TrainingConfig)
    rng = MersenneTwister(cfg.seed)
    training_policy = validate_training_policy(cfg.training_policy)
    policy_condition_mode = validate_policy_condition_mode(cfg.policy_condition_mode)
    ps, st = Lux.setup(rng, model)
    optimizer = Optimisers.Adam(cfg.learning_rate)
    opt_state = Optimisers.setup(optimizer, ps)
    losses = Float32[]

    for step in 1:cfg.max_iters
        maybe_rotate_file!(corpus, rng, step; every=cfg.train_file_update_interval)
        batch = sample_training_batch(
            corpus,
            rng;
            batch_size=cfg.batch_size,
            max_seq_len=model.config.max_seq_len,
            transition_candidates_per_example=cfg.transition_candidates_per_example,
            policy_condition_mode=policy_condition_mode,
        )

        loss = autoregressive_loss(
            model,
            ps,
            st,
            batch;
            checker_loss_weight=cfg.checker_loss_weight,
            transition_loss_weight=cfg.transition_loss_weight,
        )
        grads = Zygote.gradient(
            p -> autoregressive_loss(
                model,
                p,
                st,
                batch;
                checker_loss_weight=cfg.checker_loss_weight,
                transition_loss_weight=cfg.transition_loss_weight,
            ),
            ps,
        )[1]
        grads = apply_training_policy(model, grads, training_policy)
        opt_state, ps = Optimisers.update(opt_state, ps, grads)

        push!(losses, Float32(loss))
        if step % cfg.log_interval == 0 || step == 1 || step == cfg.max_iters
            println("step=$(step) loss=$(round(loss; digits=4)) seq_len=$(size(batch.tokens, 1)) file=$(basename(corpus.active_file))")
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
