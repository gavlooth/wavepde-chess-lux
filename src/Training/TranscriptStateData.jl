const TRANSCRIPT_STATE_SOURCE_COLUMN_CANDIDATES = ("transcript",)

py"""
def wavepde_transcript_state_examples(transcript_moves):
    board = chess.Board()
    examples = []
    running = []
    for ply_index, san in enumerate(transcript_moves, start=1):
        state_payload = _wavepde_board_state_payload(board)
        board.push_san(_wavepde_restore_castling(san))
        next_state_payload = _wavepde_board_state_payload(board)
        running.append(san)
        examples.append({
            "state": state_payload,
            "next_state": next_state_payload,
            "move_san": san,
            "ply": ply_index,
            "transcript": " ".join(running),
        })
    return examples
"""

function find_transcript_state_source_column(columns::AbstractVector{<:AbstractString})
    for candidate in TRANSCRIPT_STATE_SOURCE_COLUMN_CANDIDATES
        candidate in columns && return candidate
    end
    return nothing
end

function transcript_state_examples(transcript::AbstractString)
    moves = san_moves_from_transcript(transcript)
    raw_examples = pycall(py"wavepde_transcript_state_examples", PyAny, moves)
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
    return examples
end

function parse_transcript_parquet_state_examples(source::AbstractString)
    paths = if isfile(source)
        [source]
    elseif isdir(source)
        discover_parquet_files(source)
    else
        throw(ArgumentError("Transcript parquet source $(source) does not exist."))
    end

    db = DuckDB.DB()
    conn = DBInterface.connect(db)
    examples = NamedTuple[]
    for path in paths
        assert_real_parquet(path)
        columns = parquet_column_names(conn, path)
        transcript_column = find_transcript_state_source_column(columns)
        transcript_column === nothing && continue
        rows = DBInterface.execute(
            conn,
            "SELECT $(transcript_column) AS transcript FROM read_parquet('$(sql_escape(path))')",
        )
        for row in rows
            transcript = row[1]
            transcript === missing && continue
            append!(examples, transcript_state_examples(String(transcript)))
        end
    end
    isempty(examples) && throw(ArgumentError("No transcript-derived state examples found in $(source)."))
    return examples
end

function write_transcript_state_parquet(
    source::AbstractString,
    output_dir::AbstractString;
    file_name::AbstractString="transcript_state_transitions.parquet",
)
    examples = parse_transcript_parquet_state_examples(source)
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

    batch_size = 256
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

function ensure_transcript_state_parquet(
    source::AbstractString,
    output_dir::AbstractString;
    file_name::AbstractString="transcript_state_transitions.parquet",
)
    parquet_path = joinpath(output_dir, file_name)
    isfile(parquet_path) || write_transcript_state_parquet(source, output_dir; file_name=file_name)
    return parquet_path
end
