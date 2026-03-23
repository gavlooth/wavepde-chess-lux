include(joinpath(@__DIR__, "train_chess_state_transition.jl"))

const STATE_TRANSITION_15M_D_MODEL = 608
const STATE_TRANSITION_15M_PARAM_COUNT = 14858932

function run_chess_state_transition_15m_training()
    ENV["WAVEPDE_D_MODEL"] = get(ENV, "WAVEPDE_D_MODEL", string(STATE_TRANSITION_15M_D_MODEL))
    ENV["WAVEPDE_N_LAYER"] = get(ENV, "WAVEPDE_N_LAYER", "20")
    ENV["WAVEPDE_SOLVER_STEPS"] = get(ENV, "WAVEPDE_SOLVER_STEPS", "4")
    ENV["WAVEPDE_CHECKPOINT"] = get(
        ENV,
        "WAVEPDE_CHECKPOINT",
        joinpath(@__DIR__, "..", "checkpoints", "wavepde_chess_state_transition_15m_checkpoint.jls"),
    )
    println("entrypoint=train_chess_state_transition_15m")
    println("target_d_model=$(ENV["WAVEPDE_D_MODEL"]) target_params≈$(STATE_TRANSITION_15M_PARAM_COUNT)")
    return run_chess_state_transition_training()
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_chess_state_transition_15m_training()
end
