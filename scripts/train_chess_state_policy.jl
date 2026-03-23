include(joinpath(@__DIR__, "train_chess_state_transition.jl"))

if !haskey(ENV, "WAVEPDE_POLICY_CONDITION_MODE")
    ENV["WAVEPDE_POLICY_CONDITION_MODE"] = "state_action"
end

if !haskey(ENV, "WAVEPDE_VOCAB_SIZE")
    ENV["WAVEPDE_VOCAB_SIZE"] = string(WavePDEChess.STATE_ACTION_VOCAB_SIZE)
end

if !haskey(ENV, "WAVEPDE_MAX_SEQ_LEN")
    ENV["WAVEPDE_MAX_SEQ_LEN"] = string(
        BOARD_STATE_SEQUENCE_LENGTH + 1 + WavePDEChess.MAX_POLICY_ACTION_TOKENS,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_chess_state_transition_training()
end
