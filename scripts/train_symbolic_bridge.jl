include(joinpath(@__DIR__, "..", "src", "Training", "SymbolicTasks.jl"))
include(joinpath(@__DIR__, "..", "src", "WavePDEChess.jl"))

using .SymbolicTasks
using .WavePDEChess

function env_int(name::String, default::Int)
    return parse(Int, get(ENV, name, string(default)))
end

function env_float32(name::String, default::Float32)
    return parse(Float32, get(ENV, name, string(default)))
end

function env_symbol(name::String, default::Symbol)
    return Symbol(get(ENV, name, String(default)))
end

function run_symbolic_bridge_training()
    data_dir = get(
        ENV,
        "SYMBOLIC_DATA_DIR",
        joinpath(@__DIR__, "..", "data", "symbolic_bridge"),
    )

    count_per_task = env_int("SYMBOLIC_COUNT_PER_TASK", 256)
    seed = env_int("SYMBOLIC_SEED", 1337)
    dataset_path = ensure_symbolic_bridge_dataset(data_dir; count_per_task=count_per_task, seed=seed)

    model_config = ChessModelConfig(
        adapter=ChessAdapterConfig(
            vocab_size=env_int("WAVEPDE_VOCAB_SIZE", SYMBOLIC_VOCAB_SIZE),
            d_model=env_int("WAVEPDE_D_MODEL", 96),
            pad_token=env_int("WAVEPDE_PAD_TOKEN", 0),
        ),
        core=WavePDECoreConfig(
            d_model=env_int("WAVEPDE_D_MODEL", 96),
            n_layer=env_int("WAVEPDE_N_LAYER", 4),
            solver_steps=env_int("WAVEPDE_SOLVER_STEPS", 2),
            dt_init=env_float32("WAVEPDE_DT_INIT", 0.05f0),
            norm_eps=env_float32("WAVEPDE_NORM_EPS", 1f-5),
            cfl_safety_factor=env_float32("WAVEPDE_CFL_SAFETY_FACTOR", 0.95f0),
            cfl_eps=env_float32("WAVEPDE_CFL_EPS", 1f-6),
            cfl_smoothness=env_float32("WAVEPDE_CFL_SMOOTHNESS", 1000f0),
        ),
        proposer=ChessMoveHeadConfig(
            vocab_size=env_int("WAVEPDE_VOCAB_SIZE", SYMBOLIC_VOCAB_SIZE),
            d_model=env_int("WAVEPDE_D_MODEL", 96),
            tie_embeddings=true,
            bias=false,
        ),
        max_seq_len=env_int("WAVEPDE_MAX_SEQ_LEN", 64),
    )

    train_config = TrainingConfig(
        data_dir=data_dir,
        batch_size=env_int("WAVEPDE_BATCH_SIZE", 16),
        learning_rate=env_float32("WAVEPDE_LR", 6.0f-4),
        max_iters=env_int("WAVEPDE_MAX_ITERS", 100),
        log_interval=env_int("WAVEPDE_LOG_INTERVAL", 10),
        min_tokens=env_int("WAVEPDE_MIN_TOKENS", 8),
        train_file_update_interval=env_int("WAVEPDE_FILE_ROTATE", 10),
        cfl_penalty_weight=env_float32("WAVEPDE_CFL_PENALTY_WEIGHT", 0.0f0),
        training_policy=env_symbol("WAVEPDE_TRAINING_POLICY", :full),
        checkpoint_path=get(
            ENV,
            "WAVEPDE_CHECKPOINT",
            joinpath(@__DIR__, "..", "checkpoints", "wavepde_symbolic_bridge_checkpoint.jls"),
        ),
        seed=env_int("WAVEPDE_SEED", 1337),
    )

    println("entrypoint=train_symbolic_bridge task_surface=synthetic_symbolic")
    println("data_dir=$(train_config.data_dir)")
    println("dataset=$(dataset_path)")
    println("count_per_task=$(count_per_task) seed=$(seed)")
    println("vocab_size=$(model_config.adapter.vocab_size) d_model=$(model_config.adapter.d_model)")
    println("n_layer=$(model_config.core.n_layer) solver_steps=$(model_config.core.solver_steps)")
    println("batch_size=$(train_config.batch_size) max_iters=$(train_config.max_iters) lr=$(train_config.learning_rate)")
    println("training_policy=$(train_config.training_policy)")
    println("checkpoint=$(train_config.checkpoint_path)")

    model = WavePDEChessLM(model_config)
    corpus = ChessParquetCorpus(train_config.data_dir; min_tokens=train_config.min_tokens)
    checkpoint = train!(model, corpus, train_config)

    println("saved checkpoint to $(train_config.checkpoint_path)")
    println("final loss = $(last(checkpoint.losses))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_symbolic_bridge_training()
end
