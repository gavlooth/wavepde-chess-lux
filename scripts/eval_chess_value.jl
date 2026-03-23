include(joinpath(@__DIR__, "..", "src", "WavePDEChess.jl"))
using .WavePDEChess

function env_int(name::String, default::Int)
    return parse(Int, get(ENV, name, string(default)))
end

function run_eval_chess_value()
    checkpoint_path = get(
        ENV,
        "WAVEPDE_CHECKPOINT",
        joinpath(@__DIR__, "..", "checkpoints", "wavepde_chess_value_checkpoint.jls"),
    )
    data_dir = get(
        ENV,
        "CHESS_EVAL_DIR",
        get(ENV, "CHESS_DATA_DIR", joinpath(@__DIR__, "..", "tmp_download", "hf", "stockfish")),
    )
    batch_size = env_int("WAVEPDE_BATCH_SIZE", 32)
    max_examples = env_int("WAVEPDE_MAX_EVAL_EXAMPLES", 0)
    chunk_rows = env_int("WAVEPDE_CHUNK_ROWS", 20_000)

    result = evaluate_board_value_checkpoint(
        checkpoint_path,
        data_dir;
        batch_size=batch_size,
        max_examples=max_examples,
        chunk_rows=chunk_rows,
    )

    println("entrypoint=eval_chess_value")
    println("checkpoint=$(result.checkpoint_path)")
    println("data_dir=$(result.data_dir)")
    println("cp_scale=$(result.cp_scale)")
    println("num_examples=$(result.num_examples)")
    println("mse=$(result.mse)")
    println("mae=$(result.mae)")
    println("rmse=$(result.rmse)")
    println("max_abs_error=$(result.max_abs_error)")
    println("direction_accuracy=$(result.direction_accuracy)")
    println("mean_prediction=$(result.mean_prediction)")
    println("mean_target=$(result.mean_target)")
    println("predicted_positive_rate=$(result.predicted_positive_rate)")
    println("target_positive_rate=$(result.target_positive_rate)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_eval_chess_value()
end
