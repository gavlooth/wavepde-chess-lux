const CHESS_TRANSCRIPT_STOI = Dict(
    ' ' => 0,
    '.' => 1,
    'a' => 2,
    'b' => 3,
    'c' => 4,
    'd' => 5,
    'e' => 6,
    'f' => 7,
    'g' => 8,
    'h' => 9,
    '1' => 10,
    '2' => 11,
    '3' => 12,
    '4' => 13,
    '5' => 14,
    '6' => 15,
    '7' => 16,
    '8' => 17,
    'B' => 18,
    'N' => 19,
    'R' => 20,
    'Q' => 21,
    'K' => 22,
    'O' => 23,
    'x' => 24,
    '+' => 25,
    '#' => 26,
    '=' => 27,
)

const CHESS_TRANSCRIPT_ITOS = Dict(value => key for (key, value) in CHESS_TRANSCRIPT_STOI)
const CHESS_BOARD_TARGET_NAMES = (
    :side_to_move_white,
    :in_check,
    :white_can_castle_kingside,
    :white_can_castle_queenside,
    :black_can_castle_kingside,
    :black_can_castle_queenside,
    :white_material_advantage,
    :material_balanced,
    :black_material_advantage,
    :opening_phase,
    :middlegame_phase,
    :endgame_phase,
)

py"""
import chess

_WAVEPDE_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

def _wavepde_restore_castling(token: str) -> str:
    if token == "OOO":
        return "O-O-O"
    if token == "OO":
        return "O-O"
    return token

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

def wavepde_extract_board_targets(moves):
    board = chess.Board()
    for san in moves:
        board.push_san(_wavepde_restore_castling(san))
    return _wavepde_board_target_vector(board)

def wavepde_candidate_legality(transcript_moves, candidate_moves):
    board = chess.Board()
    for san in transcript_moves:
        board.push_san(_wavepde_restore_castling(san))

    labels = []
    for candidate in candidate_moves:
        try:
            board.parse_san(_wavepde_restore_castling(candidate))
            labels.append(1.0)
        except Exception:
            labels.append(0.0)
    return labels

def wavepde_legal_san_moves(transcript_moves, limit=0):
    board = chess.Board()
    for san in transcript_moves:
        board.push_san(_wavepde_restore_castling(san))

    moves = [board.san(move) for move in sorted(board.legal_moves, key=lambda move: move.uci())]
    if limit and limit > 0:
        return moves[:limit]
    return moves

def wavepde_transition_board_targets(transcript_moves, candidate_moves):
    board = chess.Board()
    for san in transcript_moves:
        board.push_san(_wavepde_restore_castling(san))

    legality = []
    targets = []
    empty_target = [0.0] * len(_wavepde_board_target_vector(board))
    for candidate in candidate_moves:
        try:
            move = board.parse_san(_wavepde_restore_castling(candidate))
            next_board = board.copy(stack=False)
            next_board.push(move)
            legality.append(1.0)
            targets.append(_wavepde_board_target_vector(next_board))
        except Exception:
            legality.append(0.0)
            targets.append(list(empty_target))
    return {"legality": legality, "targets": targets}
"""

const _PY_EXTRACT_BOARD_TARGETS = py"wavepde_extract_board_targets"
const _PY_CANDIDATE_LEGALITY = py"wavepde_candidate_legality"
const _PY_LEGAL_SAN_MOVES = py"wavepde_legal_san_moves"
const _PY_TRANSITION_BOARD_TARGETS = py"wavepde_transition_board_targets"

function normalize_chess_transcript(transcript::AbstractString)
    game_cleaned = occursin("\n\n", transcript) ? split(transcript, "\n\n"; limit=2)[2] : String(transcript)
    tokens = split(strip(game_cleaned))
    normalized = String[]
    for token in tokens
        token in ("1-0", "0-1", "1/2-1/2", "*") && continue
        if occursin('.', token)
            push!(normalized, "." * split(token, "."; limit=2)[2])
        else
            push!(normalized, token)
        end
    end
    return join(normalized, " ")
end

function encode_chess_transcript(transcript::AbstractString)
    normalized = replace(normalize_chess_transcript(transcript), "-" => "")
    encoded = Int32[]
    for c in normalized
        haskey(CHESS_TRANSCRIPT_STOI, c) || throw(ArgumentError("Unsupported chess transcript character $(repr(c))."))
        push!(encoded, Int32(CHESS_TRANSCRIPT_STOI[c]))
    end
    return encoded
end

function decode_chess_tokens(tokens::AbstractVector{<:Integer})
    chars = Char[]
    for token in tokens
        token_id = Int(token)
        haskey(CHESS_TRANSCRIPT_ITOS, token_id) || throw(ArgumentError(
            "decode_chess_tokens only supports the 28-token transcript vocabulary; got token id $(token_id).",
        ))
        push!(chars, CHESS_TRANSCRIPT_ITOS[token_id])
    end
    decoded = String(chars)
    return replace(replace(decoded, "OOO" => "O-O-O"), "OO" => "O-O")
end

function san_moves_from_transcript(transcript::AbstractString)
    normalized = normalize_chess_transcript(transcript)
    stripped = replace(normalized, "-" => "")
    moves = String[]
    for token in split(stripped)
        move = startswith(token, ".") ? token[2:end] : token
        isempty(move) || push!(moves, move)
    end
    return moves
end

function extract_board_targets_from_transcript(transcript::AbstractString)
    targets = pycall(_PY_EXTRACT_BOARD_TARGETS, PyAny, san_moves_from_transcript(transcript))
    return Float32[Float32(value) for value in targets]
end

function extract_board_targets_from_tokens(tokens::AbstractVector{<:Integer})
    return extract_board_targets_from_transcript(decode_chess_tokens(tokens))
end

function candidate_legality_targets(transcript::AbstractString, candidate_moves::AbstractVector{<:AbstractString})
    labels = pycall(_PY_CANDIDATE_LEGALITY, PyAny, san_moves_from_transcript(transcript), collect(candidate_moves))
    return Float32[Float32(value) for value in labels]
end

function legal_san_candidates_from_transcript(transcript::AbstractString; limit::Integer=0)
    moves = pycall(_PY_LEGAL_SAN_MOVES, PyAny, san_moves_from_transcript(transcript), Int(limit))
    return String[m for m in moves]
end

function encode_chess_candidate_san(candidate::AbstractString)
    encoded = Int32[CHESS_TRANSCRIPT_STOI[' ']]
    for c in replace(String(candidate), "-" => "")
        haskey(CHESS_TRANSCRIPT_STOI, c) || throw(ArgumentError("Unsupported chess candidate character $(repr(c))."))
        push!(encoded, Int32(CHESS_TRANSCRIPT_STOI[c]))
    end
    return encoded
end

function append_chess_candidate_san(tokens::AbstractVector{<:Integer}, candidate::AbstractString)
    return vcat(Int32.(tokens), encode_chess_candidate_san(candidate))
end

function transition_board_targets(transcript::AbstractString, candidate_moves::AbstractVector{<:AbstractString})
    result = pycall(_PY_TRANSITION_BOARD_TARGETS, PyAny, san_moves_from_transcript(transcript), collect(candidate_moves))
    legality = Float32[Float32(value) for value in result["legality"]]
    target_rows = result["targets"]
    if target_rows isa AbstractMatrix
        size(target_rows) == (length(candidate_moves), length(CHESS_BOARD_TARGET_NAMES)) || throw(ArgumentError(
            "Transition board target matrix mismatch: expected $(length(candidate_moves))x$(length(CHESS_BOARD_TARGET_NAMES)), got $(size(target_rows)).",
        ))
        return (legality=legality, targets=permutedims(Float32.(target_rows)))
    end

    targets = zeros(Float32, length(CHESS_BOARD_TARGET_NAMES), length(candidate_moves))
    for candidate_idx in eachindex(candidate_moves)
        row = target_rows[candidate_idx]
        length(row) == length(CHESS_BOARD_TARGET_NAMES) || throw(ArgumentError(
            "Transition board target length mismatch: expected $(length(CHESS_BOARD_TARGET_NAMES)), got $(length(row)).",
        ))
        for target_idx in eachindex(row)
            targets[target_idx, candidate_idx] = Float32(row[target_idx])
        end
    end
    return (legality=legality, targets=targets)
end
