include(joinpath(@__DIR__, "train_chess_checker.jl"))

function run_chess_reasoning_training()
    ENV["WAVEPDE_BOARD_TARGET_MODE"] = get(ENV, "WAVEPDE_BOARD_TARGET_MODE", "transcript_board_facts")
    ENV["WAVEPDE_CHECKER_OUTPUT_DIM"] = get(
        ENV,
        "WAVEPDE_CHECKER_OUTPUT_DIM",
        string(length(CHESS_BOARD_TARGET_NAMES)),
    )
    ENV["WAVEPDE_TRANSITION_LOSS_WEIGHT"] = get(ENV, "WAVEPDE_TRANSITION_LOSS_WEIGHT", "0.5")
    ENV["WAVEPDE_TRANSITION_CANDIDATES"] = get(ENV, "WAVEPDE_TRANSITION_CANDIDATES", "1")
    ENV["WAVEPDE_CHECKPOINT"] = get(
        ENV,
        "WAVEPDE_CHECKPOINT",
        joinpath(@__DIR__, "..", "checkpoints", "wavepde_chess_reasoning_checkpoint.jls"),
    )
    println("entrypoint=train_chess_reasoning board_fact_supervision=true")
    return run_chess_checker_training()
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_chess_reasoning_training()
end
