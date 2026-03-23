include(joinpath(@__DIR__, "..", "src", "WavePDEChess.jl"))
using .WavePDEChess

function env_int(name::String, default::Int)
    return parse(Int, get(ENV, name, string(default)))
end

function env_symbol(name::String, default::Symbol)
    return Symbol(get(ENV, name, String(default)))
end

function run_eval_chess_state_transition()
    checkpoint_path = get(
        ENV,
        "WAVEPDE_CHECKPOINT",
        joinpath(@__DIR__, "..", "checkpoints", "wavepde_chess_state_transition_checkpoint.jls"),
    )
    data_dir = state_transition_eval_data_dir()
    batch_size = env_int("WAVEPDE_BATCH_SIZE", 8)
    policy_condition_mode = env_symbol("WAVEPDE_POLICY_CONDITION_MODE", :state_only)

    result = evaluate_state_transition_checkpoint(
        checkpoint_path,
        data_dir;
        batch_size=batch_size,
        policy_condition_mode=policy_condition_mode,
    )

    println("entrypoint=eval_chess_state_transition")
    println("checkpoint=$(result.checkpoint_path)")
    println("data_dir=$(result.data_dir)")
    println("policy_condition_mode=$(policy_condition_mode)")
    println("num_examples=$(result.num_examples)")
    println("num_tokens=$(result.num_tokens)")
    println("token_loss=$(result.token_loss)")
    println("approx_perplexity=$(exp(result.token_loss))")
    println("exact_slot_accuracy=$(result.exact_slot_accuracy)")
    println("exact_sequence_match_rate=$(result.exact_sequence_match_rate)")
    println("board_fact_overall_accuracy=$(result.board_fact_metrics.overall_accuracy)")
    println("board_fact_exact_match_rate=$(result.board_fact_metrics.exact_match_rate)")
    println("board_fact_brier_score=$(result.board_fact_metrics.brier_score)")
    println("board_fact_predicted_positive_rate=$(result.board_fact_metrics.predicted_positive_rate)")
    println("board_fact_target_positive_rate=$(result.board_fact_metrics.target_positive_rate)")
    println("board_fact_per_target_accuracy=$(result.board_fact_metrics.per_target_accuracy)")
    println("state_slot_coarse_token_accuracy=$(result.state_slot_family_metrics.coarse_state.token_accuracy)")
    println("state_slot_attack_token_accuracy=$(result.state_slot_family_metrics.attack_maps.token_accuracy)")
    println("state_slot_pressure_token_accuracy=$(result.state_slot_family_metrics.pressure_counts.token_accuracy)")
    println("successor_valid_board_rate=$(result.successor_legality_metrics.valid_board_rate)")
    println("successor_reachable_rate=$(result.successor_legality_metrics.reachable_rate)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_eval_chess_state_transition()
end
