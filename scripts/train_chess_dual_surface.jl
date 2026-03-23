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

function maybe_prepare_dual_surface_data()
    if haskey(ENV, "CHESS_TRANSCRIPT_SOURCE")
        source = ENV["CHESS_TRANSCRIPT_SOURCE"]
        output_dir = get(
            ENV,
            "CHESS_DATA_DIR",
            joinpath(@__DIR__, "..", "tmp", "transcript_state_dataset"),
        )
        file_name = get(ENV, "WAVEPDE_TRANSCRIPT_STATE_PARQUET_NAME", "transcript_state_transitions.parquet")
        ensure_transcript_state_parquet(source, output_dir; file_name=file_name)
        return output_dir
    end

    return get(
        ENV,
        "CHESS_DATA_DIR",
        joinpath(@__DIR__, "..", "tmp", "transcript_state_dataset"),
    )
end

function run_chess_dual_surface_training()
    data_dir = maybe_prepare_dual_surface_data()

    model_config = DualSurfaceStateModelConfig(
        adapter=ChessAdapterConfig(
            vocab_size=env_int("WAVEPDE_VOCAB_SIZE", BOARD_STATE_VOCAB_SIZE),
            d_model=env_int("WAVEPDE_D_MODEL", 288),
            pad_token=env_int("WAVEPDE_PAD_TOKEN", 0),
        ),
        core=WavePDECoreConfig(
            d_model=env_int("WAVEPDE_D_MODEL", 288),
            n_layer=env_int("WAVEPDE_N_LAYER", 20),
            solver_steps=env_int("WAVEPDE_SOLVER_STEPS", 4),
            dt_init=env_float32("WAVEPDE_DT_INIT", 0.05f0),
            norm_eps=env_float32("WAVEPDE_NORM_EPS", 1f-5),
        ),
        state_head=ChessMoveHeadConfig(
            vocab_size=env_int("WAVEPDE_VOCAB_SIZE", BOARD_STATE_VOCAB_SIZE),
            d_model=env_int("WAVEPDE_D_MODEL", 288),
            tie_embeddings=true,
            bias=false,
        ),
        transcript_head=ChessMoveHeadConfig(
            vocab_size=length(WavePDEChess.CHESS_TRANSCRIPT_STOI),
            d_model=env_int("WAVEPDE_D_MODEL", 288),
            tie_embeddings=false,
            bias=true,
        ),
        max_seq_len=env_int("WAVEPDE_MAX_SEQ_LEN", BOARD_STATE_SEQUENCE_LENGTH),
    )

    train_config = DualSurfaceTrainingConfig(
        data_dir=data_dir,
        batch_size=env_int("WAVEPDE_BATCH_SIZE", 12),
        learning_rate=env_float32("WAVEPDE_LR", 6.0f-4),
        max_iters=env_int("WAVEPDE_MAX_ITERS", 100),
        log_interval=env_int("WAVEPDE_LOG_INTERVAL", 10),
        min_tokens=env_int("WAVEPDE_MIN_TOKENS", BOARD_STATE_SEQUENCE_LENGTH),
        train_file_update_interval=env_int("WAVEPDE_FILE_ROTATE", 10),
        state_loss_weight=env_float32("WAVEPDE_STATE_LOSS_WEIGHT", 1.0f0),
        transcript_loss_weight=env_float32("WAVEPDE_TRANSCRIPT_LOSS_WEIGHT", 0.2f0),
        training_policy=env_symbol("WAVEPDE_TRAINING_POLICY", :full),
        checkpoint_path=get(
            ENV,
            "WAVEPDE_CHECKPOINT",
            joinpath(@__DIR__, "..", "checkpoints", "wavepde_chess_dual_surface_checkpoint.jls"),
        ),
        seed=env_int("WAVEPDE_SEED", 1337),
    )

    println("entrypoint=train_chess_dual_surface dual_surface=true")
    println("data_dir=$(train_config.data_dir)")
    println("batch_size=$(train_config.batch_size) max_iters=$(train_config.max_iters) lr=$(train_config.learning_rate)")
    println("state_loss_weight=$(train_config.state_loss_weight) transcript_loss_weight=$(train_config.transcript_loss_weight)")
    println("max_seq_len=$(model_config.max_seq_len) training_policy=$(train_config.training_policy)")
    println("checkpoint=$(train_config.checkpoint_path)")

    model = DualSurfaceStateModel(model_config)
    corpus = DualSurfaceParquetCorpus(train_config.data_dir; min_tokens=train_config.min_tokens)
    checkpoint = train_dual_surface!(model, corpus, train_config)

    println("saved checkpoint to $(train_config.checkpoint_path)")
    println("final total loss = $(last(checkpoint.losses))")
    println("final state loss = $(last(checkpoint.state_losses))")
    println("final transcript loss = $(last(checkpoint.transcript_losses))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_chess_dual_surface_training()
end
