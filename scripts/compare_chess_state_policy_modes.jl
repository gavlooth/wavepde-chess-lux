include(joinpath(@__DIR__, "..", "src", "WavePDEChess.jl"))
using .WavePDEChess

function env_int(name::String, default::Int)
    return parse(Int, get(ENV, name, string(default)))
end

function env_float32(name::String, default::Float32)
    return parse(Float32, get(ENV, name, string(default)))
end

function run_state_policy_mode_comparison()
    train_data_dir = get(
        ENV,
        "CHESS_DATA_DIR",
        joinpath(@__DIR__, "..", "tmp", "pgn_state_dataset"),
    )
    eval_data_dir = get(ENV, "CHESS_EVAL_DIR", train_data_dir)
    output_dir = get(
        ENV,
        "WAVEPDE_COMPARE_OUTPUT_DIR",
        joinpath(@__DIR__, "..", "checkpoints", "state_policy_compare"),
    )

    result = compare_state_transition_training_modes(
        train_data_dir,
        eval_data_dir;
        d_model=env_int("WAVEPDE_D_MODEL", 16),
        n_layer=env_int("WAVEPDE_N_LAYER", 2),
        solver_steps=env_int("WAVEPDE_SOLVER_STEPS", 1),
        dt_init=env_float32("WAVEPDE_DT_INIT", 0.05f0),
        norm_eps=env_float32("WAVEPDE_NORM_EPS", 1f-5),
        batch_size=env_int("WAVEPDE_BATCH_SIZE", 2),
        learning_rate=env_float32("WAVEPDE_LR", 1.0f-3),
        max_iters=env_int("WAVEPDE_MAX_ITERS", 1),
        seed=env_int("WAVEPDE_SEED", 1337),
        output_dir=output_dir,
    )

    println("comparison=state_transition_modes")
    println("train_data_dir=$(result.train_data_dir)")
    println("eval_data_dir=$(result.eval_data_dir)")
    println("state_only_checkpoint=$(result.state_only.checkpoint_path)")
    println("state_only_final_loss=$(result.state_only.final_loss)")
    println("state_only_eval_token_loss=$(result.state_only.eval.token_loss)")
    println("state_only_eval_exact_slot_accuracy=$(result.state_only.eval.exact_slot_accuracy)")
    println("state_action_checkpoint=$(result.state_action.checkpoint_path)")
    println("state_action_final_loss=$(result.state_action.final_loss)")
    println("state_action_eval_token_loss=$(result.state_action.eval.token_loss)")
    println("state_action_eval_exact_slot_accuracy=$(result.state_action.eval.exact_slot_accuracy)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_state_policy_mode_comparison()
end
