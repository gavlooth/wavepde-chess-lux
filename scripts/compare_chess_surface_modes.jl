include(joinpath(@__DIR__, "..", "src", "WavePDEChess.jl"))
using .WavePDEChess

function env_int(name::String, default::Int)
    return parse(Int, get(ENV, name, string(default)))
end

function env_float32(name::String, default::Float32)
    return parse(Float32, get(ENV, name, string(default)))
end

function prepare_surface_comparison_dirs()
    haskey(ENV, "CHESS_TRANSCRIPT_SOURCE") || throw(ArgumentError(
        "compare_chess_surface_modes requires CHESS_TRANSCRIPT_SOURCE to point at transcript parquet data.",
    ))
    transcript_source = ENV["CHESS_TRANSCRIPT_SOURCE"]
    state_dir = get(
        ENV,
        "CHESS_STATE_DATA_DIR",
        joinpath(@__DIR__, "..", "tmp", "transcript_state_dataset"),
    )
    file_name = get(ENV, "WAVEPDE_TRANSCRIPT_STATE_PARQUET_NAME", "transcript_state_transitions.parquet")
    ensure_transcript_state_parquet(transcript_source, state_dir; file_name=file_name)
    return transcript_source, state_dir
end

function run_surface_mode_comparison()
    transcript_source, state_dir = prepare_surface_comparison_dirs()
    output_dir = get(
        ENV,
        "WAVEPDE_COMPARE_OUTPUT_DIR",
        joinpath(@__DIR__, "..", "checkpoints", "surface_mode_compare"),
    )

    result = compare_surface_training_modes(
        transcript_source,
        state_dir;
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

    println("comparison=surface_modes")
    println("transcript_first_checkpoint=$(result.transcript_first.checkpoint_path)")
    println("transcript_first_loss=$(result.transcript_first.final_loss)")
    println("state_first_checkpoint=$(result.state_first.checkpoint_path)")
    println("state_first_loss=$(result.state_first.final_loss)")
    println("hybrid_checkpoint=$(result.hybrid.checkpoint_path)")
    println("hybrid_total_loss=$(result.hybrid.final_loss)")
    println("hybrid_state_loss=$(result.hybrid.final_state_loss)")
    println("hybrid_transcript_loss=$(result.hybrid.final_transcript_loss)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_surface_mode_comparison()
end
