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

function build_state_transition_core_config(d_model::Int, max_seq_len::Int)
    return WavePDECoreConfig(
        d_model=d_model,
        n_layer=env_int("WAVEPDE_N_LAYER", 20),
        solver_steps=env_int("WAVEPDE_SOLVER_STEPS", 4),
        dt_init=env_float32("WAVEPDE_DT_INIT", 0.05f0),
        norm_eps=env_float32("WAVEPDE_NORM_EPS", 1f-5),
        cfl_safety_factor=env_float32("WAVEPDE_CFL_SAFETY_FACTOR", 0.95f0),
        cfl_eps=env_float32("WAVEPDE_CFL_EPS", 1f-6),
        cfl_smoothness=env_float32("WAVEPDE_CFL_SMOOTHNESS", 1000f0),
    )
end

function maybe_prepare_state_transition_data()
    if haskey(ENV, "CHESS_PGN_SOURCE")
        source = ENV["CHESS_PGN_SOURCE"]
        output_dir = get(
            ENV,
            "CHESS_DATA_DIR",
            joinpath(@__DIR__, "..", "tmp", "pgn_state_dataset"),
        )
        file_name = get(ENV, "WAVEPDE_PGN_PARQUET_NAME", "pgn_state_transitions.parquet")
        max_games = env_int("WAVEPDE_PGN_MAX_GAMES", 0)
        max_plies = env_int("WAVEPDE_PGN_MAX_PLIES", 0)
        ensure_pgn_state_parquet(
            source,
            output_dir;
            file_name=file_name,
            max_games=max_games,
            max_plies=max_plies,
        )
        return output_dir
    end

    return get(
        ENV,
        "CHESS_DATA_DIR",
        joinpath(@__DIR__, "..", "tmp", "pgn_state_dataset"),
    )
end

function run_chess_state_transition_training()
    data_dir = maybe_prepare_state_transition_data()
    policy_condition_mode = env_symbol("WAVEPDE_POLICY_CONDITION_MODE", :state_only)
    default_vocab_size = policy_condition_mode == :state_action ? WavePDEChess.STATE_ACTION_VOCAB_SIZE : BOARD_STATE_VOCAB_SIZE
    default_max_seq_len = policy_condition_mode == :state_action ?
        BOARD_STATE_SEQUENCE_LENGTH + 1 + WavePDEChess.MAX_POLICY_ACTION_TOKENS :
        BOARD_STATE_SEQUENCE_LENGTH
    probe_loss_weight = env_float32("WAVEPDE_PROBE_LOSS_WEIGHT", 0.0f0)
    d_model = env_int("WAVEPDE_D_MODEL", 288)
    max_seq_len = env_int("WAVEPDE_MAX_SEQ_LEN", default_max_seq_len)
    core_config = build_state_transition_core_config(d_model, max_seq_len)

    train_config = TrainingConfig(
        data_dir=data_dir,
        batch_size=env_int("WAVEPDE_BATCH_SIZE", 12),
        learning_rate=env_float32("WAVEPDE_LR", 6.0f-4),
        max_iters=env_int("WAVEPDE_MAX_ITERS", 100),
        log_interval=env_int("WAVEPDE_LOG_INTERVAL", 10),
        min_tokens=env_int("WAVEPDE_MIN_TOKENS", BOARD_STATE_SEQUENCE_LENGTH),
        train_file_update_interval=env_int("WAVEPDE_FILE_ROTATE", 10),
        training_policy=env_symbol("WAVEPDE_TRAINING_POLICY", :full),
        probe_loss_weight=probe_loss_weight,
        cfl_penalty_weight=env_float32("WAVEPDE_CFL_PENALTY_WEIGHT", 0.0f0),
        policy_condition_mode=policy_condition_mode,
        state_target_mode=env_symbol("WAVEPDE_STATE_TARGET_MODE", :full),
        checkpoint_path=get(
            ENV,
            "WAVEPDE_CHECKPOINT",
            joinpath(@__DIR__, "..", "checkpoints", "wavepde_chess_state_transition_checkpoint.jls"),
        ),
        seed=env_int("WAVEPDE_SEED", 1337),
    )
    corpus = StateTransitionParquetCorpus(train_config.data_dir; min_tokens=train_config.min_tokens)
    vocab_size = env_int("WAVEPDE_VOCAB_SIZE", default_vocab_size)
    adapter_config = ChessAdapterConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        pad_token=env_int("WAVEPDE_PAD_TOKEN", 0),
    )
    proposer_config = ChessMoveHeadConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        tie_embeddings=true,
        bias=false,
    )
    probe_supervision_enabled = probe_loss_weight > 0 && corpus.active_probe_targets !== nothing
    model_config = if probe_supervision_enabled
        ChessMultiHeadModelConfig(
            adapter=adapter_config,
            core=core_config,
            proposer=proposer_config,
            checker=ChessCheckerHeadConfig(
                d_model=d_model,
                output_dim=corpus.probe_target_dim,
                pooling=:mean,
            ),
            max_seq_len=max_seq_len,
        )
    else
        ChessModelConfig(
            adapter=adapter_config,
            core=core_config,
            proposer=proposer_config,
            max_seq_len=max_seq_len,
        )
    end

    println("entrypoint=train_chess_state_transition paired_state_supervision=true")
    println("data_dir=$(train_config.data_dir)")
    println("batch_size=$(train_config.batch_size) max_iters=$(train_config.max_iters) lr=$(train_config.learning_rate)")
    println("vocab_size=$(model_config.adapter.vocab_size) max_seq_len=$(model_config.max_seq_len)")
    println("training_policy=$(train_config.training_policy) policy_condition_mode=$(train_config.policy_condition_mode) state_target_mode=$(train_config.state_target_mode)")
    println("probe_loss_weight=$(train_config.probe_loss_weight)")
    println("probe_supervision_enabled=$(probe_supervision_enabled)")
    probe_supervision_enabled && println("probe_target_dim=$(corpus.probe_target_dim)")
    println("checkpoint=$(train_config.checkpoint_path)")

    model = probe_supervision_enabled ? ChessMultiHeadModel(model_config) : ChessModel(model_config)
    checkpoint = train!(model, corpus, train_config)

    println("saved checkpoint to $(train_config.checkpoint_path)")
    println("final loss = $(last(checkpoint.losses))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_chess_state_transition_training()
end
