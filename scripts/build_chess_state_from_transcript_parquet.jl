include(joinpath(@__DIR__, "..", "src", "WavePDEChess.jl"))
using .WavePDEChess

function run_build_chess_state_from_transcript_parquet()
    source = get(
        ENV,
        "CHESS_TRANSCRIPT_SOURCE",
        get(ENV, "CHESS_DATA_DIR", joinpath(@__DIR__, "..", "tmp_download")),
    )
    output_dir = get(
        ENV,
        "CHESS_STATE_DATA_DIR",
        joinpath(@__DIR__, "..", "tmp", "transcript_state_dataset"),
    )
    file_name = get(ENV, "WAVEPDE_TRANSCRIPT_STATE_PARQUET_NAME", "transcript_state_transitions.parquet")

    println("entrypoint=build_chess_state_from_transcript_parquet")
    println("source=$(source)")
    println("output_dir=$(output_dir)")
    println("file_name=$(file_name)")

    parquet_path = write_transcript_state_parquet(source, output_dir; file_name=file_name)
    println("parquet=$(parquet_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_build_chess_state_from_transcript_parquet()
end
