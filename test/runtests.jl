using Test
using Random
using DuckDB
using DBInterface
using Lux
using Zygote

include("../src/WavePDEChess.jl")
using .WavePDEChess

@testset "WavePDEChess" begin
    rng = Random.default_rng()
    Random.seed!(rng, 0)

    config = WavePDEConfig(
        vocab_size=28,
        d_model=32,
        n_layer=3,
        max_seq_len=64,
        solver_steps=2,
        dt_init=0.05f0,
        norm_eps=1f-5,
        pad_token=0,
    )

    model = WavePDEChessLM(config)
    ps, st = Lux.setup(rng, model)

    tokens = rand(rng, 0:(config.vocab_size - 1), config.max_seq_len, 2)
    logits, st_next = Lux.apply(model, tokens, ps, st)

    @test size(logits) == (config.vocab_size, config.max_seq_len, 2)
    @test all(isfinite, logits)
    @test parameter_count(ps) > 0
    @test length(st_next.blocks) == config.n_layer

    preset = chess_mamba_11m_config()
    @test preset.d_model == 288
    @test preset.n_layer == 20
    @test preset.max_seq_len == 1536
end

@testset "DuckDB Training Path" begin
    rng = MersenneTwister(42)

    config = WavePDEConfig(
        vocab_size=32,
        d_model=16,
        n_layer=2,
        max_seq_len=12,
        solver_steps=1,
        dt_init=0.05f0,
        norm_eps=1f-5,
        pad_token=0,
    )
    model = WavePDEChessLM(config)
    ps, st = Lux.setup(rng, model)

    tempdir = mktempdir()
    parquet_path = joinpath(tempdir, "toy.parquet")
    conn = DBInterface.connect(DuckDB.DB, ":memory:")
    DBInterface.execute(conn, """
        CREATE TABLE toy AS
        SELECT [1, 2, 3, 4, 5, 6] AS tokenized
        UNION ALL
        SELECT [7, 8, 9, 10, 11] AS tokenized
        UNION ALL
        SELECT [12, 13, 14, 15, 16, 17, 18] AS tokenized
    """)
    escaped_parquet_path = replace(parquet_path, "'" => "''")
    DBInterface.execute(conn, "COPY toy TO '$escaped_parquet_path' (FORMAT PARQUET)")

    corpus = ChessParquetCorpus(tempdir; min_tokens=3)
    batch = sample_batch(corpus, rng; batch_size=2, max_seq_len=config.max_seq_len)
    @test size(batch, 2) == 2
    @test size(batch, 1) >= 3

    loss = WavePDEChess.autoregressive_loss(model, ps, st, batch)
    grads = Zygote.gradient(p -> WavePDEChess.autoregressive_loss(model, p, st, batch), ps)[1]

    @test isfinite(loss)
    @test parameter_count(grads) > 0

    train_cfg = TrainingConfig(
        data_dir=tempdir,
        batch_size=2,
        max_iters=1,
        log_interval=1,
        min_tokens=3,
        train_file_update_interval=1,
        checkpoint_path=joinpath(tempdir, "wavepde_ckpt.jls"),
        seed=42,
    )

    checkpoint = train!(model, corpus, train_cfg)
    @test isfile(train_cfg.checkpoint_path)
    @test length(checkpoint.losses) == 1
end
