const BOARD_STATE_EMPTY_TOKEN = 0
const BOARD_STATE_WHITE_PAWN_TOKEN = 1
const BOARD_STATE_WHITE_KNIGHT_TOKEN = 2
const BOARD_STATE_WHITE_BISHOP_TOKEN = 3
const BOARD_STATE_WHITE_ROOK_TOKEN = 4
const BOARD_STATE_WHITE_QUEEN_TOKEN = 5
const BOARD_STATE_WHITE_KING_TOKEN = 6
const BOARD_STATE_BLACK_PAWN_TOKEN = 7
const BOARD_STATE_BLACK_KNIGHT_TOKEN = 8
const BOARD_STATE_BLACK_BISHOP_TOKEN = 9
const BOARD_STATE_BLACK_ROOK_TOKEN = 10
const BOARD_STATE_BLACK_QUEEN_TOKEN = 11
const BOARD_STATE_BLACK_KING_TOKEN = 12
const BOARD_STATE_SIDE_WHITE_TOKEN = 13
const BOARD_STATE_SIDE_BLACK_TOKEN = 14
const BOARD_STATE_FLAG_FALSE_TOKEN = 15
const BOARD_STATE_FLAG_TRUE_TOKEN = 16
const BOARD_STATE_EP_NONE_TOKEN = 17
const BOARD_STATE_EP_FILE_BASE = 18
const BOARD_STATE_HALFMOVE_BASE = 26
const BOARD_STATE_FULLMOVE_BASE = 42
const BOARD_STATE_COUNT_BASE = 26
const BOARD_STATE_COUNT_BUCKETS = 16
const BOARD_STATE_VOCAB_SIZE = 58
const BOARD_STATE_COARSE_LENGTH = 72
const BOARD_STATE_SEQUENCE_LENGTH = 210
const BOARD_PROBE_FIELD_ORDER = (
    (:attacked_white, 64),
    (:attacked_black, 64),
    (:in_check, 2),
    (:pinned_count, 2),
    (:king_pressure, 2),
    (:mobility, 2),
    (:attacked_piece_count, 2),
)
const BOARD_PROBE_TARGET_LENGTH = 138

const BOARD_STATE_PIECE_SYMBOL_TOKENS = Dict(
    "." => BOARD_STATE_EMPTY_TOKEN,
    "P" => BOARD_STATE_WHITE_PAWN_TOKEN,
    "N" => BOARD_STATE_WHITE_KNIGHT_TOKEN,
    "B" => BOARD_STATE_WHITE_BISHOP_TOKEN,
    "R" => BOARD_STATE_WHITE_ROOK_TOKEN,
    "Q" => BOARD_STATE_WHITE_QUEEN_TOKEN,
    "K" => BOARD_STATE_WHITE_KING_TOKEN,
    "p" => BOARD_STATE_BLACK_PAWN_TOKEN,
    "n" => BOARD_STATE_BLACK_KNIGHT_TOKEN,
    "b" => BOARD_STATE_BLACK_BISHOP_TOKEN,
    "r" => BOARD_STATE_BLACK_ROOK_TOKEN,
    "q" => BOARD_STATE_BLACK_QUEEN_TOKEN,
    "k" => BOARD_STATE_BLACK_KING_TOKEN,
)

const STATE_TOKENS_COLUMN_CANDIDATES = ("state_tokens",)
const NEXT_STATE_TOKENS_COLUMN_CANDIDATES = ("next_state_tokens",)

py"""
import chess
import chess.pgn

def _wavepde_board_state_payload(board):
    def attack_map(color):
        return [1 if board.is_attacked_by(color, square) else 0 for square in chess.SQUARES]

    def piece_squares(color):
        return [square for square in chess.SQUARES if board.piece_at(square) is not None and board.piece_at(square).color == color]

    def pinned_count(color):
        return sum(1 for square in piece_squares(color) if board.is_pinned(color, square))

    def king_pressure(color):
        king_square = board.king(color)
        if king_square is None:
            return 0
        return len(board.attackers(not color, king_square))

    def mobility(color):
        temp = board.copy(stack=False)
        temp.turn = color
        return sum(1 for _ in temp.legal_moves)

    def attacked_piece_count(color):
        opponent = not color
        return sum(1 for square in piece_squares(color) if board.is_attacked_by(opponent, square))

    return {
        "squares": [board.piece_at(square).symbol() if board.piece_at(square) is not None else "." for square in chess.SQUARES],
        "white_attacks": attack_map(chess.WHITE),
        "black_attacks": attack_map(chess.BLACK),
        "white_to_move": bool(board.turn == chess.WHITE),
        "white_can_castle_kingside": bool(board.has_kingside_castling_rights(chess.WHITE)),
        "white_can_castle_queenside": bool(board.has_queenside_castling_rights(chess.WHITE)),
        "black_can_castle_kingside": bool(board.has_kingside_castling_rights(chess.BLACK)),
        "black_can_castle_queenside": bool(board.has_queenside_castling_rights(chess.BLACK)),
        "ep_file": -1 if board.ep_square is None else chess.square_file(board.ep_square),
        "halfmove_clock": int(board.halfmove_clock),
        "fullmove_number": int(board.fullmove_number),
        "white_in_check": bool(board.is_attacked_by(chess.BLACK, board.king(chess.WHITE)) if board.king(chess.WHITE) is not None else False),
        "black_in_check": bool(board.is_attacked_by(chess.WHITE, board.king(chess.BLACK)) if board.king(chess.BLACK) is not None else False),
        "white_pinned_count": int(pinned_count(chess.WHITE)),
        "black_pinned_count": int(pinned_count(chess.BLACK)),
        "white_king_pressure": int(king_pressure(chess.WHITE)),
        "black_king_pressure": int(king_pressure(chess.BLACK)),
        "white_mobility": int(mobility(chess.WHITE)),
        "black_mobility": int(mobility(chess.BLACK)),
        "white_attacked_piece_count": int(attacked_piece_count(chess.WHITE)),
        "black_attacked_piece_count": int(attacked_piece_count(chess.BLACK)),
    }

def wavepde_parse_pgn_state_examples(paths, max_games=0, max_plies=0):
    examples = []
    game_count = 0

    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break

                game_count += 1
                board = game.board()
                transcript_moves = []

                for ply_index, move in enumerate(game.mainline_moves(), start=1):
                    state_payload = _wavepde_board_state_payload(board)
                    san = board.san(move)
                    board.push(move)
                    next_state_payload = _wavepde_board_state_payload(board)
                    transcript_moves.append(san)
                    examples.append({
                        "state": state_payload,
                        "next_state": next_state_payload,
                        "move_san": san,
                        "ply": ply_index,
                        "transcript": " ".join(transcript_moves),
                    })
                    if max_plies > 0 and ply_index >= max_plies:
                        break

                if max_games > 0 and game_count >= max_games:
                    return examples

    return examples

def wavepde_board_state_from_transcript(moves):
    board = chess.Board()
    for san in moves:
        board.push_san(_wavepde_restore_castling(san))
    return _wavepde_board_state_payload(board)
"""

function board_state_bucket(value::Integer, bucket_count::Integer)
    return min(max(Int(value), 0), bucket_count - 1)
end

board_bool_token(value::Bool) = Int32(value ? BOARD_STATE_FLAG_TRUE_TOKEN : BOARD_STATE_FLAG_FALSE_TOKEN)

board_count_token(value::Integer; base::Integer=BOARD_STATE_COUNT_BASE) = Int32(Int(base) + board_state_bucket(value, BOARD_STATE_COUNT_BUCKETS))

function board_state_tokens(payload)
    white_attacks = payload["white_attacks"]
    black_attacks = payload["black_attacks"]
    length(white_attacks) == 64 || throw(ArgumentError("Board-state payload must contain 64 white attack entries, got $(length(white_attacks))."))
    length(black_attacks) == 64 || throw(ArgumentError("Board-state payload must contain 64 black attack entries, got $(length(black_attacks))."))

    tokens = Vector{Int32}(undef, BOARD_STATE_SEQUENCE_LENGTH)
    for index in 1:64
        symbol = String(payload["squares"][index])
        haskey(BOARD_STATE_PIECE_SYMBOL_TOKENS, symbol) || throw(ArgumentError("Unsupported board-state piece symbol $(repr(symbol))."))
        tokens[index] = Int32(BOARD_STATE_PIECE_SYMBOL_TOKENS[symbol])
    end

    tokens[65] = Int32(payload["white_to_move"] ? BOARD_STATE_SIDE_WHITE_TOKEN : BOARD_STATE_SIDE_BLACK_TOKEN)
    tokens[66] = board_bool_token(Bool(payload["white_can_castle_kingside"]))
    tokens[67] = board_bool_token(Bool(payload["white_can_castle_queenside"]))
    tokens[68] = board_bool_token(Bool(payload["black_can_castle_kingside"]))
    tokens[69] = board_bool_token(Bool(payload["black_can_castle_queenside"]))
    ep_file = Int(payload["ep_file"])
    tokens[70] = Int32(ep_file < 0 ? BOARD_STATE_EP_NONE_TOKEN : BOARD_STATE_EP_FILE_BASE + ep_file)
    tokens[71] = board_count_token(payload["halfmove_clock"]; base=BOARD_STATE_HALFMOVE_BASE)
    tokens[72] = board_count_token(Int(payload["fullmove_number"]) - 1; base=BOARD_STATE_FULLMOVE_BASE)

    for index in 1:64
        tokens[72 + index] = board_bool_token(Int(white_attacks[index]) != 0)
        tokens[136 + index] = board_bool_token(Int(black_attacks[index]) != 0)
    end

    tokens[201] = board_bool_token(Bool(payload["white_in_check"]))
    tokens[202] = board_bool_token(Bool(payload["black_in_check"]))
    tokens[203] = board_count_token(payload["white_pinned_count"])
    tokens[204] = board_count_token(payload["black_pinned_count"])
    tokens[205] = board_count_token(payload["white_king_pressure"])
    tokens[206] = board_count_token(payload["black_king_pressure"])
    tokens[207] = board_count_token(payload["white_mobility"])
    tokens[208] = board_count_token(payload["black_mobility"])
    tokens[209] = board_count_token(payload["white_attacked_piece_count"])
    tokens[210] = board_count_token(payload["black_attacked_piece_count"])
    return tokens
end

function board_probe_targets_from_payload(payload)
    return (
        attacked_white=Float32[Float32(x) for x in payload["white_attacks"]],
        attacked_black=Float32[Float32(x) for x in payload["black_attacks"]],
        in_check=Float32[
            payload["white_in_check"] ? 1.0f0 : 0.0f0,
            payload["black_in_check"] ? 1.0f0 : 0.0f0,
        ],
        pinned_count=Float32[
            Float32(payload["white_pinned_count"]),
            Float32(payload["black_pinned_count"]),
        ],
        king_pressure=Float32[
            Float32(payload["white_king_pressure"]),
            Float32(payload["black_king_pressure"]),
        ],
        mobility=Float32[
            Float32(payload["white_mobility"]),
            Float32(payload["black_mobility"]),
        ],
        attacked_piece_count=Float32[
            Float32(payload["white_attacked_piece_count"]),
            Float32(payload["black_attacked_piece_count"]),
        ],
    )
end

function flatten_board_probe_targets(targets::NamedTuple)
    flat = Vector{Float32}(undef, BOARD_PROBE_TARGET_LENGTH)
    offset = 1
    for (field, width) in BOARD_PROBE_FIELD_ORDER
        values = getfield(targets, field)
        length(values) == width || throw(ArgumentError(
            "Board probe field $(field) expected length $(width), got $(length(values)).",
        ))
        flat[offset:(offset + width - 1)] .= Float32.(values)
        offset += width
    end
    return flat
end

function split_board_probe_targets(values::AbstractVector{<:Real})
    length(values) == BOARD_PROBE_TARGET_LENGTH || throw(ArgumentError(
        "Expected $(BOARD_PROBE_TARGET_LENGTH) board probe values, got $(length(values)).",
    ))
    offset = 1
    parts = Pair{Symbol, Vector{Float32}}[]
    for (field, width) in BOARD_PROBE_FIELD_ORDER
        push!(parts, field => Float32.(collect(values[offset:(offset + width - 1)])))
        offset += width
    end
    return (; parts...)
end

function split_board_probe_targets(values::AbstractMatrix{<:Real})
    size(values, 1) == BOARD_PROBE_TARGET_LENGTH || throw(ArgumentError(
        "Expected board probe matrix with $(BOARD_PROBE_TARGET_LENGTH) rows, got $(size(values, 1)).",
    ))
    offset = 1
    parts = Pair{Symbol, Matrix{Float32}}[]
    for (field, width) in BOARD_PROBE_FIELD_ORDER
        push!(parts, field => Float32.(values[offset:(offset + width - 1), :]))
        offset += width
    end
    return (; parts...)
end

function board_probe_targets_from_transcript(transcript::AbstractString)
    moves = san_moves_from_transcript(transcript)
    state_payload = pycall(py"wavepde_board_state_from_transcript", PyAny, moves)
    return board_probe_targets_from_payload(state_payload)
end

function board_state_tokens_from_transcript(transcript::AbstractString)
    moves = san_moves_from_transcript(transcript)
    state_payload = pycall(py"wavepde_board_state_from_transcript", PyAny, moves)
    return board_state_tokens(state_payload)
end

function board_probe_targets_from_tokens(tokens::AbstractVector{<:Integer})
    return board_probe_targets_from_transcript(decode_chess_tokens(tokens))
end

function discover_pgn_files(source::AbstractString)
    if isfile(source)
        endswith(lowercase(source), ".pgn") || throw(ArgumentError("Expected a .pgn file, got $(source)."))
        return [source]
    elseif isdir(source)
        files = String[]
        for (root, _, names) in walkdir(source)
            for name in names
                endswith(lowercase(name), ".pgn") || continue
                push!(files, joinpath(root, name))
            end
        end
        sort!(files)
        isempty(files) && throw(ArgumentError("No PGN files found under $(source)."))
        return files
    end
    throw(ArgumentError("PGN source $(source) does not exist."))
end

function parse_pgn_state_examples(source::AbstractString; max_games::Int=0, max_plies::Int=0)
    files = discover_pgn_files(source)
    raw_examples = pycall(py"wavepde_parse_pgn_state_examples", PyAny, files, max_games, max_plies)
    examples = NamedTuple[]
    for raw in raw_examples
        probe_targets = board_probe_targets_from_payload(raw["state"])
        push!(examples, (
            state_tokens=board_state_tokens(raw["state"]),
            next_state_tokens=board_state_tokens(raw["next_state"]),
            move_san=String(raw["move_san"]),
            ply=Int(raw["ply"]),
            transcript=String(raw["transcript"]),
            attacked_white=probe_targets.attacked_white,
            attacked_black=probe_targets.attacked_black,
            in_check=probe_targets.in_check,
            pinned_count=probe_targets.pinned_count,
            king_pressure=probe_targets.king_pressure,
            mobility=probe_targets.mobility,
            attacked_piece_count=probe_targets.attacked_piece_count,
        ))
    end
    isempty(examples) && throw(ArgumentError("No PGN state-transition examples were extracted from $(source)."))
    return examples
end

state_sql_list(values::AbstractVector{<:Real}) = "[" * join(Int.(values), ", ") * "]"
state_sql_escape(value::AbstractString) = replace(value, "'" => "''")

function write_pgn_state_parquet(
    source::AbstractString,
    output_dir::AbstractString;
    file_name::AbstractString="pgn_state_transitions.parquet",
    max_games::Int=0,
    max_plies::Int=0,
)
    examples = parse_pgn_state_examples(source; max_games=max_games, max_plies=max_plies)
    mkpath(output_dir)
    parquet_path = joinpath(output_dir, file_name)

    db = DuckDB.DB()
    conn = DBInterface.connect(db)
    DBInterface.execute(conn, """
        CREATE TABLE state_transitions (
            state_tokens INTEGER[],
            next_state_tokens INTEGER[],
            move_san VARCHAR,
            ply INTEGER,
            transcript VARCHAR,
            attacked_white INTEGER[],
            attacked_black INTEGER[],
            in_check INTEGER[],
            pinned_count INTEGER[],
            king_pressure INTEGER[],
            mobility INTEGER[],
            attacked_piece_count INTEGER[]
        )
    """)

    batch_size = 128
    for offset in 1:batch_size:length(examples)
        batch = examples[offset:min(offset + batch_size - 1, length(examples))]
        values = String[]
        for example in batch
            push!(
                values,
                "($(state_sql_list(example.state_tokens)), $(state_sql_list(example.next_state_tokens)), '$(state_sql_escape(example.move_san))', $(example.ply), '$(state_sql_escape(example.transcript))', $(state_sql_list(example.attacked_white)), $(state_sql_list(example.attacked_black)), $(state_sql_list(example.in_check)), $(state_sql_list(example.pinned_count)), $(state_sql_list(example.king_pressure)), $(state_sql_list(example.mobility)), $(state_sql_list(example.attacked_piece_count)))",
            )
        end
        DBInterface.execute(conn, "INSERT INTO state_transitions VALUES " * join(values, ", "))
    end

    isfile(parquet_path) && rm(parquet_path; force=true)
    DBInterface.execute(conn, "COPY state_transitions TO '$(state_sql_escape(parquet_path))' (FORMAT PARQUET)")
    return parquet_path
end

function ensure_pgn_state_parquet(
    source::AbstractString,
    output_dir::AbstractString;
    file_name::AbstractString="pgn_state_transitions.parquet",
    max_games::Int=0,
    max_plies::Int=0,
)
    parquet_path = joinpath(output_dir, file_name)
    isfile(parquet_path) || write_pgn_state_parquet(
        source,
        output_dir;
        file_name=file_name,
        max_games=max_games,
        max_plies=max_plies,
    )
    return parquet_path
end
