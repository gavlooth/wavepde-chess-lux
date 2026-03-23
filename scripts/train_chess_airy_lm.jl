include(joinpath(@__DIR__, "..", "src", "WavePDEChess.jl"))
using .WavePDEChess

function env_int(name::String, default::Int)
    return parse(Int, get(ENV, name, string(default)))
end

function env_float32(name::String, default::Float32)
    return parse(Float32, get(ENV, name, string(default)))
end

function env_bool(name::String, default::Bool)
    return lowercase(get(ENV, name, string(default))) in ("1", "true", "yes", "on")
end

function env_symbol(name::String, default::Symbol)
    return Symbol(get(ENV, name, String(default)))
end

function run_chess_airy_lm_training()
    data_dir = get(
        ENV,
        "CHESS_DATA_DIR",
        joinpath(@__DIR__, "..", "chess-mamba-vs-xformer", "chess-mamba-vs-xformer", "data"),
    )

    airy_config = AiryPDEConfig(
        vocab_size=env_int("WAVEPDE_VOCAB_SIZE", 28),
        d_model=288,
        n_layer=20,
        max_seq_len=1536,
        dt_init=env_float32("AIRYPDE_DT_INIT", 0.05f0),
        dt_floor=env_float32("AIRYPDE_DT_FLOOR", 1f-4),
        alpha_init=env_float32("AIRYPDE_ALPHA_INIT", 0.01f0),
        alpha_floor=env_float32("AIRYPDE_ALPHA_FLOOR", 1f-4),
        beta_init=env_float32("AIRYPDE_BETA_INIT", 0.01f0),
        phase_limit=env_float32("AIRYPDE_PHASE_LIMIT", Float32(pi)),
        decay_limit=env_float32("AIRYPDE_DECAY_LIMIT", 2.0f0),
        residual_init_scale=env_float32("AIRYPDE_RESIDUAL_INIT_SCALE", 0.1f0),
        norm_eps=1f-5,
        stability_eps=env_float32("AIRYPDE_STABILITY_EPS", 1f-6),
        warn_on_clamp=env_bool("AIRYPDE_WARN_ON_CLAMP", true),
    )
    model_config = ChessModelConfig(airy_config)

    train_config = TrainingConfig(
        data_dir=data_dir,
        batch_size=env_int("WAVEPDE_BATCH_SIZE", 12),
        learning_rate=env_float32("WAVEPDE_LR", 6.0f-4),
        max_iters=env_int("WAVEPDE_MAX_ITERS", 100),
        log_interval=env_int("WAVEPDE_LOG_INTERVAL", 10),
        min_tokens=env_int("WAVEPDE_MIN_TOKENS", 8),
        train_file_update_interval=env_int("WAVEPDE_FILE_ROTATE", 10),
        training_policy=env_symbol("WAVEPDE_TRAINING_POLICY", :full),
        checkpoint_path=get(
            ENV,
            "WAVEPDE_CHECKPOINT",
            joinpath(@__DIR__, "..", "checkpoints", "airy_chess_checkpoint.jls"),
        ),
        seed=env_int("WAVEPDE_SEED", 1337),
    )

    println("entrypoint=train_chess_airy_lm proposer_only=true core=airy")
    println("data_dir=$(train_config.data_dir)")
    println("batch_size=$(train_config.batch_size) max_iters=$(train_config.max_iters) lr=$(train_config.learning_rate)")
    println("training_policy=$(train_config.training_policy)")
    println("phase_limit=$(model_config.core.phase_limit) decay_limit=$(model_config.core.decay_limit)")
    println("checkpoint=$(train_config.checkpoint_path)")

    model = ChessModel(airy_config)
    corpus = ChessParquetCorpus(train_config.data_dir; min_tokens=train_config.min_tokens)
    checkpoint = train!(model, corpus, train_config)

    println("saved checkpoint to $(train_config.checkpoint_path)")
    println("final loss = $(last(checkpoint.losses))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_chess_airy_lm_training()
end
