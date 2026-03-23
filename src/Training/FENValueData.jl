const FEN_VALUE_FEN_COLUMN_CANDIDATES = ("fen",)
const FEN_VALUE_CP_COLUMN_CANDIDATES = ("cp",)
const FEN_VALUE_MATE_COLUMN_CANDIDATES = ("mate",)
const FEN_VALUE_DEPTH_COLUMN_CANDIDATES = ("depth",)

py"""
def wavepde_board_state_payload_from_fen(fen):
    board = chess.Board(fen)
    return _wavepde_board_state_payload(board)
"""

function find_fen_value_fen_column(columns::AbstractVector{<:AbstractString})
    for candidate in FEN_VALUE_FEN_COLUMN_CANDIDATES
        candidate in columns && return candidate
    end
    return nothing
end

function find_fen_value_cp_column(columns::AbstractVector{<:AbstractString})
    for candidate in FEN_VALUE_CP_COLUMN_CANDIDATES
        candidate in columns && return candidate
    end
    return nothing
end

function find_fen_value_mate_column(columns::AbstractVector{<:AbstractString})
    for candidate in FEN_VALUE_MATE_COLUMN_CANDIDATES
        candidate in columns && return candidate
    end
    return nothing
end

function find_fen_value_depth_column(columns::AbstractVector{<:AbstractString})
    for candidate in FEN_VALUE_DEPTH_COLUMN_CANDIDATES
        candidate in columns && return candidate
    end
    return nothing
end

function board_state_tokens_from_fen(fen::AbstractString)
    payload = pycall(py"wavepde_board_state_payload_from_fen", PyAny, String(fen))
    return board_state_tokens(payload)
end

function sanitize_float_cell(value)
    if value === missing
        return nothing
    end
    parsed = Float64(value)
    return isfinite(parsed) ? Float32(parsed) : nothing
end

function sanitize_int_cell(value)
    if value === missing
        return nothing
    end
    parsed = Int(round(Float64(value)))
    return parsed
end

function value_target_from_engine_eval(
    cp_value,
    mate_value;
    cp_scale::Real=400,
)
    mate = sanitize_int_cell(mate_value)
    if mate !== nothing && mate != 0
        return Float32(sign(mate))
    end

    cp = sanitize_float_cell(cp_value)
    cp === nothing && return nothing
    return Float32(tanh(cp / Float32(cp_scale)))
end

function count_parquet_rows(conn, path::AbstractString)
    rows = DBInterface.execute(conn, "SELECT COUNT(*) FROM read_parquet('$(sql_escape(path))')")
    for row in rows
        return Int(row[1])
    end
    return 0
end

function load_board_value_examples(
    conn,
    path::AbstractString;
    min_tokens::Int=BOARD_STATE_SEQUENCE_LENGTH,
    cp_scale::Float32=400f0,
    limit::Int=50_000,
    offset::Int=0,
)
    assert_real_parquet(path)
    columns = parquet_column_names(conn, path)
    fen_column = find_fen_value_fen_column(columns)
    cp_column = find_fen_value_cp_column(columns)
    mate_column = find_fen_value_mate_column(columns)
    depth_column = find_fen_value_depth_column(columns)

    fen_column === nothing && throw(ArgumentError("FEN value parquet $(path) must include a fen column."))
    cp_column === nothing && mate_column === nothing && throw(ArgumentError(
        "FEN value parquet $(path) must include at least one of cp or mate columns.",
    ))

    select_columns = ["$(fen_column) AS fen"]
    cp_column === nothing || push!(select_columns, "$(cp_column) AS cp")
    mate_column === nothing || push!(select_columns, "$(mate_column) AS mate")
    depth_column === nothing || push!(select_columns, "$(depth_column) AS depth")
    query = "SELECT $(join(select_columns, ", ")) FROM read_parquet('$(sql_escape(path))') LIMIT $(Int(limit)) OFFSET $(Int(offset))"
    rows = DBInterface.execute(conn, query)

    states = Vector{Vector{Int32}}()
    values = Float32[]
    depths = Int[]
    for row in rows
        fen_value = row[1]
        fen_value === missing && continue
        cp_idx = cp_column === nothing ? nothing : 2
        mate_idx = mate_column === nothing ? nothing : (cp_column === nothing ? 2 : 3)
        depth_idx = depth_column === nothing ? nothing :
            (1 + (cp_column === nothing ? 0 : 1) + (mate_column === nothing ? 0 : 1) + 1)

        cp_value = cp_idx === nothing ? missing : row[cp_idx]
        mate_value = mate_idx === nothing ? missing : row[mate_idx]
        target = value_target_from_engine_eval(cp_value, mate_value; cp_scale=cp_scale)
        target === nothing && continue

        state_tokens = board_state_tokens_from_fen(String(fen_value))
        length(state_tokens) >= min_tokens || continue
        push!(states, state_tokens)
        push!(values, target)
        push!(depths, depth_idx === nothing || row[depth_idx] === missing ? 0 : Int(round(Float64(row[depth_idx]))))
    end

    isempty(states) && throw(ArgumentError("No usable FEN value examples found in $(path) for offset $(offset)."))
    return states, values, depths
end
