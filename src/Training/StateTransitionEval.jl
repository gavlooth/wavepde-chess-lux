using Serialization
using PyCall

py"""
import chess

_WAVEPDE_PIECE_TOKEN_TO_PIECE = {
    0: None,
    1: chess.Piece(chess.PAWN, chess.WHITE),
    2: chess.Piece(chess.KNIGHT, chess.WHITE),
    3: chess.Piece(chess.BISHOP, chess.WHITE),
    4: chess.Piece(chess.ROOK, chess.WHITE),
    5: chess.Piece(chess.QUEEN, chess.WHITE),
    6: chess.Piece(chess.KING, chess.WHITE),
    7: chess.Piece(chess.PAWN, chess.BLACK),
    8: chess.Piece(chess.KNIGHT, chess.BLACK),
    9: chess.Piece(chess.BISHOP, chess.BLACK),
    10: chess.Piece(chess.ROOK, chess.BLACK),
    11: chess.Piece(chess.QUEEN, chess.BLACK),
    12: chess.Piece(chess.KING, chess.BLACK),
}

_WAVEPDE_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

def _wavepde_board_target_vector(board):
    material = 0
    for piece, value in _WAVEPDE_PIECE_VALUES.items():
        material += len(board.pieces(piece, chess.WHITE)) * value
        material -= len(board.pieces(piece, chess.BLACK)) * value

    fullmove = board.fullmove_number
    if fullmove <= 10:
        phase = (1.0, 0.0, 0.0)
    elif fullmove <= 30:
        phase = (0.0, 1.0, 0.0)
    else:
        phase = (0.0, 0.0, 1.0)

    return [
        1.0 if board.turn == chess.WHITE else 0.0,
        1.0 if board.is_check() else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
        1.0 if material > 0 else 0.0,
        1.0 if material == 0 else 0.0,
        1.0 if material < 0 else 0.0,
        phase[0],
        phase[1],
        phase[2],
    ]

def wavepde_board_from_state_tokens(tokens):
    board = chess.Board()
    board.clear_board()

    for square_index in range(64):
        token = int(tokens[square_index])
        piece = _WAVEPDE_PIECE_TOKEN_TO_PIECE.get(token)
        if piece is not None:
            board.set_piece_at(square_index, piece)

    board.turn = chess.WHITE if int(tokens[64]) == 13 else chess.BLACK

    castling_rights = ""
    if int(tokens[65]) == 16:
        castling_rights += "K"
    if int(tokens[66]) == 16:
        castling_rights += "Q"
    if int(tokens[67]) == 16:
        castling_rights += "k"
    if int(tokens[68]) == 16:
        castling_rights += "q"
    board.set_castling_fen(castling_rights if castling_rights else "-")

    ep_token = int(tokens[69])
    if ep_token == 17:
        board.ep_square = None
    else:
        ep_file = ep_token - 18
        ep_rank = 5 if board.turn == chess.WHITE else 2
        board.ep_square = chess.square(ep_file, ep_rank)

    board.halfmove_clock = max(int(tokens[70]) - 26, 0)
    board.fullmove_number = max(int(tokens[71]) - 26 + 1, 1)
    return board

def wavepde_board_target_vector_from_state_tokens(tokens):
    try:
        return _wavepde_board_target_vector(wavepde_board_from_state_tokens(tokens))
    except Exception:
        return [0.0] * 12

def wavepde_state_is_valid(tokens):
    try:
        return bool(wavepde_board_from_state_tokens(tokens).is_valid())
    except Exception:
        return False

def wavepde_state_is_reachable_from_source(source_tokens, target_tokens):
    try:
        source = wavepde_board_from_state_tokens(source_tokens)
        target = wavepde_board_from_state_tokens(target_tokens)
        if not target.is_valid():
            return False

        target_fen = target.fen()
        for move in source.legal_moves:
            candidate = source.copy(stack=False)
            candidate.push(move)
            if candidate.fen() == target_fen:
                return True
        return False
    except Exception:
        return False
"""

function load_state_transition_checkpoint(checkpoint_path::AbstractString)
    payload = open(checkpoint_path, "r") do io
        deserialize(io)
    end
    haskey(payload, :model_config) || throw(ArgumentError("Checkpoint $(checkpoint_path) is missing a model_config field."))
    model = WavePDEChess.WavePDEChessLM(payload.model_config)
    ps = payload.parameters
    st = payload.state
    return (payload=payload, model=model, parameters=ps, state=st)
end

function state_transition_eval_data_dir()
    if haskey(ENV, "CHESS_PGN_SOURCE")
        source = ENV["CHESS_PGN_SOURCE"]
        output_dir = get(
            ENV,
            "CHESS_EVAL_DIR",
            joinpath(@__DIR__, "..", "..", "tmp", "pgn_state_eval_dataset"),
        )
        file_name = get(ENV, "WAVEPDE_PGN_PARQUET_NAME", "pgn_state_transitions.parquet")
        max_games = parse(Int, get(ENV, "WAVEPDE_PGN_MAX_GAMES", "0"))
        max_plies = parse(Int, get(ENV, "WAVEPDE_PGN_MAX_PLIES", "0"))
        WavePDEChess.ensure_pgn_state_parquet(
            source,
            output_dir;
            file_name=file_name,
            max_games=max_games,
            max_plies=max_plies,
        )
        return output_dir
    end

    return get(
        ENV,
        "CHESS_EVAL_DIR",
        get(
            ENV,
            "CHESS_DATA_DIR",
            joinpath(@__DIR__, "..", "..", "tmp", "pgn_state_dataset"),
        ),
    )
end

function pack_state_transition_batch(
    states::AbstractVector{<:AbstractVector{<:Integer}},
    next_states::AbstractVector{<:AbstractVector{<:Integer}},
    start_idx::Int,
    batch_size::Int,
    max_seq_len::Int,
)
    subset = start_idx:min(start_idx + batch_size - 1, length(states))
    batch_states = states[subset]
    batch_next_states = next_states[subset]
    max_len = maximum(min(length(state), max_seq_len) for state in batch_states)

    inputs = zeros(Int32, max_len, length(batch_states))
    targets = zeros(Int32, max_len, length(batch_states))
    for (batch_idx, state) in enumerate(batch_states)
        token_count = min(length(state), max_len)
        inputs[1:token_count, batch_idx] .= state[1:token_count]
        targets[1:token_count, batch_idx] .= batch_next_states[batch_idx][1:token_count]
    end

    return inputs, targets
end

function pack_state_transition_batch(
    states::AbstractVector{<:AbstractVector{<:Integer}},
    next_states::AbstractVector{<:AbstractVector{<:Integer}},
    moves::Union{Nothing, AbstractVector{<:AbstractString}},
    start_idx::Int,
    batch_size::Int,
    max_seq_len::Int;
    policy_condition_mode::Symbol=:state_only,
)
    subset = start_idx:min(start_idx + batch_size - 1, length(states))
    batch_states = states[subset]
    batch_next_states = next_states[subset]
    state_seq_len = length(first(batch_states))

    if policy_condition_mode == :state_only
        inputs, targets = pack_state_transition_batch(states, next_states, start_idx, batch_size, max_seq_len)
        target_mask = trues(size(targets))
        source_states = zeros(Int32, state_seq_len, length(batch_states))
        for (batch_idx, state) in enumerate(batch_states)
            source_states[:, batch_idx] .= Int32.(state[1:state_seq_len])
        end
        return inputs, targets, target_mask, source_states
    elseif policy_condition_mode == :state_action
        moves === nothing && throw(ArgumentError(
            "policy_condition_mode=:state_action requires move_san values for evaluation batches.",
        ))
        batch_moves = moves[subset]
        conditioned_sequences = [WavePDEChess.append_policy_action_tokens(state, move) for (state, move) in zip(batch_states, batch_moves)]
        max_len = maximum(length(sequence) for sequence in conditioned_sequences)
        max_len <= max_seq_len || throw(ArgumentError(
            "Conditioned evaluation sequence length $(max_len) exceeds max_seq_len $(max_seq_len).",
        ))

        inputs = zeros(Int32, max_len, length(batch_states))
        targets = zeros(Int32, max_len, length(batch_states))
        target_mask = falses(max_len, length(batch_states))
        source_states = zeros(Int32, state_seq_len, length(batch_states))

        for batch_idx in eachindex(batch_states)
            conditioned = conditioned_sequences[batch_idx]
            next_state = batch_next_states[batch_idx]
            inputs[1:length(conditioned), batch_idx] .= conditioned
            targets[1:length(next_state), batch_idx] .= Int32.(next_state)
            target_mask[1:length(next_state), batch_idx] .= true
            source_states[:, batch_idx] .= Int32.(batch_states[batch_idx][1:state_seq_len])
        end

        return inputs, targets, target_mask, source_states
    end

    throw(ArgumentError("Unsupported policy_condition_mode $(policy_condition_mode)."))
end

function argmax_tokens(logits::AbstractArray{<:Real, 3})
    vocab_size, seq_len, batch_size = size(logits)
    tokens = zeros(Int32, seq_len, batch_size)
    for batch_idx in 1:batch_size
        for seq_idx in 1:seq_len
            tokens[seq_idx, batch_idx] = Int32(argmax(view(logits, :, seq_idx, batch_idx)) - 1)
        end
    end
    return tokens
end

function state_fact_matrix(state_tokens::AbstractMatrix{<:Integer})
    num_facts = length(WavePDEChess.CHESS_BOARD_TARGET_NAMES)
    facts = zeros(Float32, num_facts, size(state_tokens, 2))
    for batch_idx in 1:size(state_tokens, 2)
        facts[:, batch_idx] .= Float32.(pycall(py"wavepde_board_target_vector_from_state_tokens", PyAny, collect(Int.(view(state_tokens, :, batch_idx)))))
    end
    return facts
end

function state_board_validity_flags(state_tokens::AbstractMatrix{<:Integer})
    flags = Vector{Bool}(undef, size(state_tokens, 2))
    for batch_idx in 1:size(state_tokens, 2)
        flags[batch_idx] = Bool(pycall(py"wavepde_state_is_valid", PyAny, collect(Int.(view(state_tokens, :, batch_idx)))))
    end
    return flags
end

function successor_legality_metrics(source_states::AbstractMatrix{<:Integer}, predicted_states::AbstractMatrix{<:Integer})
    size(source_states) == size(predicted_states) || throw(ArgumentError(
        "successor_legality_metrics expects source and predicted states with the same shape, got $(size(source_states)) and $(size(predicted_states)).",
    ))
    num_examples = size(source_states, 2)
    num_examples > 0 || throw(ArgumentError("successor_legality_metrics requires a non-empty batch."))

    valid_board_count = 0
    reachable_count = 0
    for batch_idx in 1:num_examples
        source_tokens = collect(Int.(view(source_states, :, batch_idx)))
        predicted_tokens = collect(Int.(view(predicted_states, :, batch_idx)))
        valid_board_count += Bool(pycall(py"wavepde_state_is_valid", PyAny, predicted_tokens))
        reachable_count += Bool(pycall(py"wavepde_state_is_reachable_from_source", PyAny, source_tokens, predicted_tokens))
    end

    total = Float64(num_examples)
    return (
        valid_board_rate=valid_board_count / total,
        reachable_rate=reachable_count / total,
        unreachable_rate=(num_examples - reachable_count) / total,
    )
end

function evaluate_state_transition_batches(model, ps, st, states, next_states; batch_size::Int)
    max_seq_len = size(first(states), 1)
    total_loss = 0.0
    total_tokens = 0
    total_correct_tokens = 0
    total_tokens_seen = 0
    total_exact_matches = 0
    total_examples = 0
    source_state_cols = Matrix{Int32}(undef, size(first(states), 1), 0)
    target_state_cols = Matrix{Int32}(undef, size(first(next_states), 1), 0)
    predicted_state_cols = Matrix{Int32}(undef, size(first(next_states), 1), 0)
    predicted_fact_cols = Matrix{Float32}(undef, length(WavePDEChess.CHESS_BOARD_TARGET_NAMES), 0)
    target_fact_cols = Matrix{Float32}(undef, length(WavePDEChess.CHESS_BOARD_TARGET_NAMES), 0)

    for start_idx in 1:batch_size:length(states)
        inputs, targets = pack_state_transition_batch(states, next_states, start_idx, batch_size, max_seq_len)
        loss, output = WavePDEChess.paired_token_prediction_loss(model, ps, st, inputs, targets)
        logits = WavePDEChess.extract_proposer_logits(output)
        predicted_tokens = argmax_tokens(logits)

        total_loss += Float64(loss) * length(targets)
        total_tokens += length(targets)
        total_correct_tokens += count(identity, predicted_tokens .== targets)
        total_tokens_seen += length(targets)
        total_exact_matches += count(
            batch_idx -> all(view(predicted_tokens, :, batch_idx) .== view(targets, :, batch_idx)),
            1:size(targets, 2),
        )
        total_examples += size(targets, 2)
        source_state_cols = hcat(source_state_cols, inputs)
        target_state_cols = hcat(target_state_cols, targets)
        predicted_state_cols = hcat(predicted_state_cols, predicted_tokens)

        predicted_fact_cols = hcat(predicted_fact_cols, state_fact_matrix(predicted_tokens))
        target_fact_cols = hcat(target_fact_cols, state_fact_matrix(targets))
    end

    slot_accuracy = total_correct_tokens / total_tokens_seen
    sequence_match_rate = total_exact_matches / total_examples
    fact_metrics = WavePDEChess.board_fact_metrics(predicted_fact_cols, target_fact_cols)
    slot_family_metrics = WavePDEChess.state_slot_family_metrics(predicted_state_cols, target_state_cols)
    legality_metrics = successor_legality_metrics(source_state_cols, predicted_state_cols)
    return (
        token_loss=total_loss / total_tokens,
        exact_slot_accuracy=slot_accuracy,
        exact_sequence_match_rate=sequence_match_rate,
        board_fact_metrics=fact_metrics,
        state_slot_family_metrics=slot_family_metrics,
        successor_legality_metrics=legality_metrics,
        num_examples=total_examples,
        num_tokens=total_tokens_seen,
    )
end

function evaluate_state_transition_corpus(model, ps, st, corpus; batch_size::Int=8, policy_condition_mode::Symbol=:state_only)
    total_loss = 0.0
    total_tokens = 0
    total_correct_tokens = 0
    total_tokens_seen = 0
    total_exact_matches = 0
    total_examples = 0
    source_state_cols = Matrix{Int32}(undef, size(first(corpus.active_states), 1), 0)
    target_state_cols = Matrix{Int32}(undef, size(first(corpus.active_next_states), 1), 0)
    predicted_state_cols = Matrix{Int32}(undef, size(first(corpus.active_next_states), 1), 0)
    predicted_fact_cols = Matrix{Float32}(undef, length(WavePDEChess.CHESS_BOARD_TARGET_NAMES), 0)
    target_fact_cols = Matrix{Float32}(undef, length(WavePDEChess.CHESS_BOARD_TARGET_NAMES), 0)

    for file_path in corpus.files
        reload_file!(corpus, file_path)
        file_states = corpus.active_states
        file_next_states = corpus.active_next_states
        file_moves = corpus.active_moves
        file_max_seq_len = if policy_condition_mode == :state_action
            WavePDEChess.BOARD_STATE_SEQUENCE_LENGTH + 1 + WavePDEChess.MAX_POLICY_ACTION_TOKENS
        else
            size(first(file_states), 1)
        end

        for start_idx in 1:batch_size:length(file_states)
            inputs, targets, target_mask, source_states = pack_state_transition_batch(
                file_states,
                file_next_states,
                file_moves,
                start_idx,
                batch_size,
                file_max_seq_len;
                policy_condition_mode=policy_condition_mode,
            )
            loss, output = WavePDEChess.paired_token_prediction_loss(model, ps, st, inputs, targets, target_mask)
            logits = WavePDEChess.extract_proposer_logits(output)
            predicted_tokens = argmax_tokens(logits)
            masked_correct = count(((predicted_tokens .== targets) .& target_mask))
            masked_token_count = count(target_mask)

            total_loss += Float64(loss) * masked_token_count
            total_tokens += masked_token_count
            total_correct_tokens += masked_correct
            total_tokens_seen += masked_token_count
            total_exact_matches += count(
                batch_idx -> all(view(predicted_tokens, 1:size(source_states, 1), batch_idx) .== view(targets, 1:size(source_states, 1), batch_idx)),
                1:size(targets, 2),
            )
            total_examples += size(targets, 2)
            source_state_cols = hcat(source_state_cols, source_states)
            target_state_cols = hcat(target_state_cols, targets[1:size(source_states, 1), :])
            predicted_state_cols = hcat(predicted_state_cols, predicted_tokens[1:size(source_states, 1), :])

            predicted_fact_cols = hcat(predicted_fact_cols, state_fact_matrix(predicted_tokens[1:size(source_states, 1), :]))
            target_fact_cols = hcat(target_fact_cols, state_fact_matrix(targets[1:size(source_states, 1), :]))
        end
    end

    fact_metrics = WavePDEChess.board_fact_metrics(predicted_fact_cols, target_fact_cols)
    slot_family_metrics = WavePDEChess.state_slot_family_metrics(predicted_state_cols, target_state_cols)
    legality_metrics = successor_legality_metrics(source_state_cols, predicted_state_cols)
    return (
        token_loss=total_loss / total_tokens,
        exact_slot_accuracy=total_correct_tokens / total_tokens_seen,
        exact_sequence_match_rate=total_exact_matches / total_examples,
        board_fact_metrics=fact_metrics,
        state_slot_family_metrics=slot_family_metrics,
        successor_legality_metrics=legality_metrics,
        num_examples=total_examples,
        num_tokens=total_tokens_seen,
    )
end

function evaluate_state_transition_checkpoint(
    checkpoint_path::AbstractString,
    data_dir::AbstractString;
    batch_size::Int=8,
    policy_condition_mode::Symbol=:state_only,
)
    checkpoint = load_state_transition_checkpoint(checkpoint_path)
    corpus = WavePDEChess.StateTransitionParquetCorpus(data_dir; min_tokens=WavePDEChess.BOARD_STATE_SEQUENCE_LENGTH)
    evaluation = evaluate_state_transition_corpus(
        checkpoint.model,
        checkpoint.parameters,
        checkpoint.state,
        corpus;
        batch_size=batch_size,
        policy_condition_mode=policy_condition_mode,
    )
    return merge(
        (
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
        ),
        evaluation,
    )
end

function compare_state_transition_training_modes(
    train_data_dir::AbstractString,
    eval_data_dir::AbstractString;
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
    mkpath(output_dir)

    state_only_config = WavePDEChess.ChessModelConfig(
        adapter=WavePDEChess.ChessAdapterConfig(vocab_size=WavePDEChess.BOARD_STATE_VOCAB_SIZE, d_model=d_model, pad_token=0),
        core=WavePDEChess.WavePDECoreConfig(d_model=d_model, n_layer=n_layer, solver_steps=solver_steps, dt_init=dt_init, norm_eps=norm_eps),
        proposer=WavePDEChess.ChessMoveHeadConfig(vocab_size=WavePDEChess.BOARD_STATE_VOCAB_SIZE, d_model=d_model, tie_embeddings=true, bias=false),
        max_seq_len=WavePDEChess.BOARD_STATE_SEQUENCE_LENGTH,
    )
    state_only_corpus = WavePDEChess.StateTransitionParquetCorpus(train_data_dir; min_tokens=WavePDEChess.BOARD_STATE_SEQUENCE_LENGTH)
    state_only_train_cfg = WavePDEChess.TrainingConfig(
        data_dir=train_data_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_iters=max_iters,
        log_interval=1,
        min_tokens=WavePDEChess.BOARD_STATE_SEQUENCE_LENGTH,
        train_file_update_interval=1,
        policy_condition_mode=:state_only,
        checkpoint_path=joinpath(output_dir, "state_only_transition.jls"),
        seed=seed,
    )
    state_only_checkpoint = WavePDEChess.train!(WavePDEChess.ChessModel(state_only_config), state_only_corpus, state_only_train_cfg)
    state_only_eval = evaluate_state_transition_checkpoint(
        state_only_train_cfg.checkpoint_path,
        eval_data_dir;
        batch_size=batch_size,
        policy_condition_mode=:state_only,
    )

    state_action_config = WavePDEChess.ChessModelConfig(
        adapter=WavePDEChess.ChessAdapterConfig(vocab_size=WavePDEChess.STATE_ACTION_VOCAB_SIZE, d_model=d_model, pad_token=0),
        core=WavePDEChess.WavePDECoreConfig(d_model=d_model, n_layer=n_layer, solver_steps=solver_steps, dt_init=dt_init, norm_eps=norm_eps),
        proposer=WavePDEChess.ChessMoveHeadConfig(vocab_size=WavePDEChess.STATE_ACTION_VOCAB_SIZE, d_model=d_model, tie_embeddings=true, bias=false),
        max_seq_len=WavePDEChess.BOARD_STATE_SEQUENCE_LENGTH + 1 + WavePDEChess.MAX_POLICY_ACTION_TOKENS,
    )
    state_action_corpus = WavePDEChess.StateTransitionParquetCorpus(train_data_dir; min_tokens=WavePDEChess.BOARD_STATE_SEQUENCE_LENGTH)
    state_action_train_cfg = WavePDEChess.TrainingConfig(
        data_dir=train_data_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_iters=max_iters,
        log_interval=1,
        min_tokens=WavePDEChess.BOARD_STATE_SEQUENCE_LENGTH,
        train_file_update_interval=1,
        policy_condition_mode=:state_action,
        checkpoint_path=joinpath(output_dir, "state_action_transition.jls"),
        seed=seed + 1,
    )
    state_action_checkpoint = WavePDEChess.train!(WavePDEChess.ChessModel(state_action_config), state_action_corpus, state_action_train_cfg)
    state_action_eval = evaluate_state_transition_checkpoint(
        state_action_train_cfg.checkpoint_path,
        eval_data_dir;
        batch_size=batch_size,
        policy_condition_mode=:state_action,
    )

    return (
        train_data_dir=abspath(train_data_dir),
        eval_data_dir=abspath(eval_data_dir),
        state_only=(
            checkpoint_path=state_only_train_cfg.checkpoint_path,
            final_loss=last(state_only_checkpoint.losses),
            eval=state_only_eval,
        ),
        state_action=(
            checkpoint_path=state_action_train_cfg.checkpoint_path,
            final_loss=last(state_action_checkpoint.losses),
            eval=state_action_eval,
        ),
    )
end
