include(joinpath(@__DIR__, "..", "src", "WavePDEChess.jl"))
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

function run_chess_value_training()
    device = get(ENV, "WAVEPDE_DEVICE", "gpu")
    data_dir = get(
        ENV,
        "CHESS_DATA_DIR",
        joinpath(@__DIR__, "..", "tmp_download", "hf", "stockfish"),
    )
    d_model = env_int("WAVEPDE_D_MODEL", 288)
    max_seq_len = env_int("WAVEPDE_MAX_SEQ_LEN", BOARD_STATE_SEQUENCE_LENGTH)
    cp_scale = env_float32("WAVEPDE_VALUE_CP_SCALE", 400f0)

    model_config = BoardValueModelConfig(
        adapter=ChessAdapterConfig(
            vocab_size=env_int("WAVEPDE_VOCAB_SIZE", BOARD_STATE_VOCAB_SIZE),
            d_model=d_model,
            pad_token=env_int("WAVEPDE_PAD_TOKEN", 0),
        ),
        core=WavePDECoreConfig(
            d_model=d_model,
            n_layer=env_int("WAVEPDE_N_LAYER", 20),
            solver_steps=env_int("WAVEPDE_SOLVER_STEPS", 4),
            dt_init=env_float32("WAVEPDE_DT_INIT", 0.05f0),
            norm_eps=env_float32("WAVEPDE_NORM_EPS", 1f-5),
            cfl_safety_factor=env_float32("WAVEPDE_CFL_SAFETY_FACTOR", 0.95f0),
            cfl_eps=env_float32("WAVEPDE_CFL_EPS", 1f-6),
        ),
        value_head=ChessCheckerHeadConfig(
            d_model=d_model,
            output_dim=1,
            pooling=env_symbol("WAVEPDE_VALUE_POOLING", :mean),
        ),
        max_seq_len=max_seq_len,
    )

    train_config = BoardValueTrainingConfig(
        data_dir=data_dir,
        batch_size=env_int("WAVEPDE_BATCH_SIZE", 32),
        learning_rate=env_float32("WAVEPDE_LR", 6.0f-4),
        max_iters=env_int("WAVEPDE_MAX_ITERS", 100),
        log_interval=env_int("WAVEPDE_LOG_INTERVAL", 10),
        min_tokens=env_int("WAVEPDE_MIN_TOKENS", BOARD_STATE_SEQUENCE_LENGTH),
        train_file_update_interval=env_int("WAVEPDE_FILE_ROTATE", 10),
        training_policy=env_symbol("WAVEPDE_TRAINING_POLICY", :full),
        cp_scale=cp_scale,
        chunk_rows=env_int("WAVEPDE_CHUNK_ROWS", 20_000),
        checkpoint_path=get(
            ENV,
            "WAVEPDE_CHECKPOINT",
            joinpath(@__DIR__, "..", "checkpoints", "wavepde_chess_value_checkpoint.jls"),
        ),
        seed=env_int("WAVEPDE_SEED", 1337),
    )

    println("entrypoint=train_chess_value")
    println("requested_device=$(device)")
    println("data_dir=$(train_config.data_dir)")
    println("batch_size=$(train_config.batch_size) max_iters=$(train_config.max_iters) lr=$(train_config.learning_rate)")
    println("d_model=$(model_config.adapter.d_model) n_layer=$(model_config.core.n_layer) solver_steps=$(model_config.core.solver_steps)")
    println("value_pooling=$(model_config.value_head.pooling) cp_scale=$(train_config.cp_scale) chunk_rows=$(train_config.chunk_rows)")
    println("training_policy=$(train_config.training_policy)")
    println("checkpoint=$(train_config.checkpoint_path)")

    model = BoardValueModel(model_config)
    corpus = BoardValueParquetCorpus(
        train_config.data_dir;
        min_tokens=train_config.min_tokens,
        cp_scale=train_config.cp_scale,
        chunk_rows=train_config.chunk_rows,
    )
    checkpoint = train_value!(model, corpus, train_config)

    println("saved checkpoint to $(train_config.checkpoint_path)")
    println("final loss = $(last(checkpoint.losses))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_chess_value_training()
end
