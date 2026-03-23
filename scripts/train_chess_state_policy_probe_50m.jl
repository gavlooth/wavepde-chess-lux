include(joinpath(@__DIR__, "train_chess_state_policy_50m.jl"))

const STATE_POLICY_PROBE_50M_DEFAULT_WEIGHT = 0.5f0

function run_chess_state_policy_probe_50m_training()
    ENV["WAVEPDE_PROBE_LOSS_WEIGHT"] = get(
        ENV,
        "WAVEPDE_PROBE_LOSS_WEIGHT",
        string(STATE_POLICY_PROBE_50M_DEFAULT_WEIGHT),
    )
    ENV["WAVEPDE_CHECKPOINT"] = get(
        ENV,
        "WAVEPDE_CHECKPOINT",
        joinpath(@__DIR__, "..", "checkpoints", "wavepde_chess_state_policy_probe_50m_coarse_checkpoint.jls"),
    )
    println("entrypoint=train_chess_state_policy_probe_50m")
    println("probe_loss_weight=$(ENV["WAVEPDE_PROBE_LOSS_WEIGHT"])")
    return run_chess_state_policy_50m_training()
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_chess_state_policy_probe_50m_training()
end
