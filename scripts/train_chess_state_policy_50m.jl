include(joinpath(@__DIR__, "train_chess_state_policy.jl"))

const STATE_POLICY_50M_D_MODEL = 1120
const STATE_POLICY_50M_PARAM_COUNT_APPROX = 50323860

function run_chess_state_policy_50m_training()
    ENV["WAVEPDE_D_MODEL"] = get(ENV, "WAVEPDE_D_MODEL", string(STATE_POLICY_50M_D_MODEL))
    ENV["WAVEPDE_N_LAYER"] = get(ENV, "WAVEPDE_N_LAYER", "20")
    ENV["WAVEPDE_SOLVER_STEPS"] = get(ENV, "WAVEPDE_SOLVER_STEPS", "4")
    ENV["WAVEPDE_CFL_SAFETY_FACTOR"] = get(ENV, "WAVEPDE_CFL_SAFETY_FACTOR", "0.9")
    ENV["WAVEPDE_CFL_EPS"] = get(ENV, "WAVEPDE_CFL_EPS", "1e-6")
    ENV["WAVEPDE_STATE_TARGET_MODE"] = get(ENV, "WAVEPDE_STATE_TARGET_MODE", "coarse_only")
    ENV["WAVEPDE_CHECKPOINT"] = get(
        ENV,
        "WAVEPDE_CHECKPOINT",
        joinpath(@__DIR__, "..", "checkpoints", "wavepde_chess_state_policy_50m_coarse_checkpoint.jls"),
    )
    println("entrypoint=train_chess_state_policy_50m")
    println("target_d_model=$(ENV["WAVEPDE_D_MODEL"]) target_params≈$(STATE_POLICY_50M_PARAM_COUNT_APPROX)")
    println("cfl_safety_factor=$(ENV["WAVEPDE_CFL_SAFETY_FACTOR"])")
    println("state_target_mode=$(ENV["WAVEPDE_STATE_TARGET_MODE"])")
    return run_chess_state_transition_training()
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_chess_state_policy_50m_training()
end
