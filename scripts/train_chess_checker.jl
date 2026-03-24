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

function run_chess_checker_training()
    data_dir = get(
        ENV,
        "CHESS_DATA_DIR",
        joinpath(@__DIR__, "..", "chess-mamba-vs-xformer", "chess-mamba-vs-xformer", "data"),
    )

    model_config = ChessMultiHeadModelConfig(
        adapter=ChessAdapterConfig(
            vocab_size=env_int("WAVEPDE_VOCAB_SIZE", 28),
            d_model=env_int("WAVEPDE_D_MODEL", 288),
            pad_token=env_int("WAVEPDE_PAD_TOKEN", 0),
        ),
        core=WavePDECoreConfig(
            d_model=env_int("WAVEPDE_D_MODEL", 288),
            n_layer=env_int("WAVEPDE_N_LAYER", 20),
            solver_steps=env_int("WAVEPDE_SOLVER_STEPS", 4),
            dt_init=env_float32("WAVEPDE_DT_INIT", 0.05f0),
            norm_eps=env_float32("WAVEPDE_NORM_EPS", 1f-5),
            cfl_safety_factor=env_float32("WAVEPDE_CFL_SAFETY_FACTOR", 0.95f0),
            cfl_eps=env_float32("WAVEPDE_CFL_EPS", 1f-6),
            cfl_smoothness=env_float32("WAVEPDE_CFL_SMOOTHNESS", 1000f0),
        ),
        proposer=ChessMoveHeadConfig(
            vocab_size=env_int("WAVEPDE_VOCAB_SIZE", 28),
            d_model=env_int("WAVEPDE_D_MODEL", 288),
            tie_embeddings=true,
            bias=false,
        ),
        checker=ChessCheckerHeadConfig(
            d_model=env_int("WAVEPDE_D_MODEL", 288),
            output_dim=env_int("WAVEPDE_CHECKER_OUTPUT_DIM", length(CHESS_BOARD_TARGET_NAMES)),
            pooling=env_symbol("WAVEPDE_CHECKER_POOLING", :mean),
        ),
        max_seq_len=env_int("WAVEPDE_MAX_SEQ_LEN", 1536),
    )

    train_config = TrainingConfig(
        data_dir=data_dir,
        batch_size=env_int("WAVEPDE_BATCH_SIZE", 12),
        learning_rate=env_float32("WAVEPDE_LR", 6.0f-4),
        max_iters=env_int("WAVEPDE_MAX_ITERS", 100),
        log_interval=env_int("WAVEPDE_LOG_INTERVAL", 10),
        min_tokens=env_int("WAVEPDE_MIN_TOKENS", 8),
        train_file_update_interval=env_int("WAVEPDE_FILE_ROTATE", 10),
        checker_loss_weight=env_float32("WAVEPDE_CHECKER_LOSS_WEIGHT", 1.0f0),
        cfl_penalty_weight=env_float32("WAVEPDE_CFL_PENALTY_WEIGHT", 0.0f0),
        transition_loss_weight=env_float32("WAVEPDE_TRANSITION_LOSS_WEIGHT", 0.0f0),
        transition_candidates_per_example=env_int("WAVEPDE_TRANSITION_CANDIDATES", 1),
        training_policy=env_symbol("WAVEPDE_TRAINING_POLICY", :full),
        board_target_mode=env_symbol("WAVEPDE_BOARD_TARGET_MODE", :transcript_board_facts),
        checkpoint_path=get(
            ENV,
            "WAVEPDE_CHECKPOINT",
            joinpath(@__DIR__, "..", "checkpoints", "wavepde_chess_checker_checkpoint.jls"),
        ),
        seed=env_int("WAVEPDE_SEED", 1337),
    )

    println("entrypoint=train_chess_checker multi_head=true optional_checker_supervision=true")
    println("data_dir=$(train_config.data_dir)")
    println("batch_size=$(train_config.batch_size) max_iters=$(train_config.max_iters) lr=$(train_config.learning_rate)")
    println("checker_output_dim=$(model_config.checker.output_dim) checker_pooling=$(model_config.checker.pooling)")
    println("checker_loss_weight=$(train_config.checker_loss_weight)")
    println("transition_loss_weight=$(train_config.transition_loss_weight) transition_candidates=$(train_config.transition_candidates_per_example)")
    println("training_policy=$(train_config.training_policy)")
    println("board_target_mode=$(train_config.board_target_mode)")
    println("checkpoint=$(train_config.checkpoint_path)")

    model = ChessMultiHeadModel(model_config)
    corpus = ChessParquetCorpus(
        train_config.data_dir;
        min_tokens=train_config.min_tokens,
        board_target_mode=train_config.board_target_mode,
    )
    checkpoint = train!(model, corpus, train_config)

    println("saved checkpoint to $(train_config.checkpoint_path)")
    println("final loss = $(last(checkpoint.losses))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_chess_checker_training()
end
