include(joinpath(@__DIR__, "..", "src", "WavePDEChess.jl"))
using .WavePDEChess

include(joinpath(@__DIR__, "..", "src", "Training", "TransferComparison.jl"))
using .TransferComparison

function env_int(name::String, default::Int)
    return parse(Int, get(ENV, name, string(default)))
end

function env_float32(name::String, default::Float32)
    return parse(Float32, get(ENV, name, string(default)))
end

function env_bool(name::String, default::Bool)
    value = lowercase(get(ENV, name, default ? "true" : "false"))
    value in ("1", "true", "yes", "on") && return true
    value in ("0", "false", "no", "off") && return false
    throw(ArgumentError("Unsupported boolean value for $(name): $(value)"))
end

function run_symbolic_transfer_comparison()
    output_dir = get(
        ENV,
        "WAVEPDE_SYMBOLIC_OUTPUT_DIR",
        joinpath(@__DIR__, "..", "checkpoints", "symbolic_transfer"),
    )
    dataset_dir = get(ENV, "WAVEPDE_SYMBOLIC_DATA_DIR", joinpath(output_dir, "data"))
    source_checkpoint_path = get(
        ENV,
        "WAVEPDE_SOURCE_CHECKPOINT",
        joinpath(@__DIR__, "..", "checkpoints", "wavepde_chess_checkpoint.jls"),
    )

    result = compare_symbolic_transfer(
        output_dir=output_dir,
        source_checkpoint_path=source_checkpoint_path,
        dataset_dir=dataset_dir,
        count_per_task=env_int("WAVEPDE_SYMBOLIC_COUNT_PER_TASK", 32),
        seed=env_int("WAVEPDE_SYMBOLIC_SEED", 1337),
        batch_size=env_int("WAVEPDE_SYMBOLIC_BATCH_SIZE", 8),
        learning_rate=env_float32("WAVEPDE_SYMBOLIC_LR", 6.0f-4),
        max_iters=env_int("WAVEPDE_SYMBOLIC_MAX_ITERS", 2),
        vocab_size=env_int("WAVEPDE_VOCAB_SIZE", symbolic_vocabulary_size()),
        d_model=env_int("WAVEPDE_D_MODEL", 288),
        n_layer=env_int("WAVEPDE_N_LAYER", 20),
        solver_steps=env_int("WAVEPDE_SOLVER_STEPS", 4),
        dt_init=env_float32("WAVEPDE_DT_INIT", 0.05f0),
        norm_eps=env_float32("WAVEPDE_NORM_EPS", 1f-5),
        max_seq_len=env_int("WAVEPDE_MAX_SEQ_LEN", 128),
        rebuild_dataset=env_bool("WAVEPDE_SYMBOLIC_REBUILD_DATASET", true),
    )

    println("entrypoint=run_symbolic_transfer_comparison")
    println("dataset_path=$(result.dataset_path)")
    println("source_checkpoint=$(result.source_checkpoint_path)")
    println("scratch_full final_loss=$(result.scratch_full.final_loss) checkpoint=$(result.scratch_full.checkpoint_path)")
    println("chess_core_frozen final_loss=$(result.chess_core_frozen.final_loss) checkpoint=$(result.chess_core_frozen.checkpoint_path)")
    println("chess_core_finetune final_loss=$(result.chess_core_finetune.final_loss) checkpoint=$(result.chess_core_finetune.checkpoint_path)")
    println("comparison=$(result)")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_symbolic_transfer_comparison()
end
