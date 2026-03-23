include(joinpath(@__DIR__, "train_chess_state_transition.jl"))

function run_chess_state_transition_airy_training()
    ENV["WAVEPDE_CORE_KIND"] = get(ENV, "WAVEPDE_CORE_KIND", "airy")
    ENV["WAVEPDE_CHECKPOINT"] = get(
        ENV,
        "WAVEPDE_CHECKPOINT",
        joinpath(@__DIR__, "..", "checkpoints", "airy_chess_state_transition_checkpoint.jls"),
    )
    println("entrypoint=train_chess_state_transition_airy")
    return run_chess_state_transition_training()
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_chess_state_transition_airy_training()
end
