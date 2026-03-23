using Random

include(joinpath(@__DIR__, "..", "src", "WavePDEChess.jl"))
using .WavePDEChess

function env_int(name::String, default::Int)
    return parse(Int, get(ENV, name, string(default)))
end

function env_float32(name::String, default::Float32)
    return parse(Float32, get(ENV, name, string(default)))
end

data_dir = get(
    ENV,
    "CHESS_DATA_DIR",
    joinpath(@__DIR__, "..", "chess-mamba-vs-xformer", "chess-mamba-vs-xformer", "data"),
)

model_config = chess_mamba_11m_config(
    vocab_size=env_int("WAVEPDE_VOCAB_SIZE", 28),
    solver_steps=env_int("WAVEPDE_SOLVER_STEPS", 4),
    dt_init=env_float32("WAVEPDE_DT_INIT", 0.05f0),
)

train_config = TrainingConfig(
    data_dir=data_dir,
    batch_size=env_int("WAVEPDE_BATCH_SIZE", 12),
    learning_rate=env_float32("WAVEPDE_LR", 6.0f-4),
    max_iters=env_int("WAVEPDE_MAX_ITERS", 100),
    log_interval=env_int("WAVEPDE_LOG_INTERVAL", 10),
    min_tokens=env_int("WAVEPDE_MIN_TOKENS", 8),
    train_file_update_interval=env_int("WAVEPDE_FILE_ROTATE", 10),
    checkpoint_path=get(
        ENV,
        "WAVEPDE_CHECKPOINT",
        joinpath(@__DIR__, "..", "checkpoints", "wavepde_chess_checkpoint.jls"),
    ),
    seed=env_int("WAVEPDE_SEED", 1337),
)

println("data_dir=$(train_config.data_dir)")
println("batch_size=$(train_config.batch_size) max_iters=$(train_config.max_iters) lr=$(train_config.learning_rate)")
println("checkpoint=$(train_config.checkpoint_path)")

model = WavePDEChessLM(model_config)
corpus = ChessParquetCorpus(train_config.data_dir; min_tokens=train_config.min_tokens)
checkpoint = train!(model, corpus, train_config)

println("saved checkpoint to $(train_config.checkpoint_path)")
println("final loss = $(last(checkpoint.losses))")
