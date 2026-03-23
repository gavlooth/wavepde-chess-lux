include(joinpath(@__DIR__, "..", "src", "WavePDEChess.jl"))
using .WavePDEChess

function env_int(name::String, default::Int)
    return parse(Int, get(ENV, name, string(default)))
end

function run_build_chess_pgn_parquet()
    source = get(
        ENV,
        "CHESS_PGN_SOURCE",
        joinpath(@__DIR__, "..", "tmp_download"),
    )
    output_dir = get(
        ENV,
        "CHESS_DATA_DIR",
        joinpath(@__DIR__, "..", "tmp", "pgn_state_dataset"),
    )
    file_name = get(ENV, "WAVEPDE_PGN_PARQUET_NAME", "pgn_state_transitions.parquet")
    max_games = env_int("WAVEPDE_PGN_MAX_GAMES", 0)
    max_plies = env_int("WAVEPDE_PGN_MAX_PLIES", 0)

    println("entrypoint=build_chess_pgn_parquet")
    println("pgn_source=$(source)")
    println("output_dir=$(output_dir)")
    println("file_name=$(file_name)")
    println("max_games=$(max_games) max_plies=$(max_plies)")

    parquet_path = write_pgn_state_parquet(
        source,
        output_dir;
        file_name=file_name,
        max_games=max_games,
        max_plies=max_plies,
    )
    println("parquet=$(parquet_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_build_chess_pgn_parquet()
end
