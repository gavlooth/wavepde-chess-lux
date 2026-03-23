using Test
using Random
using Serialization
using DuckDB
using DBInterface
using Lux
using Zygote

include("../src/WavePDEChess.jl")
include("../src/Training/SymbolicTasks.jl")
include("../src/Training/TransferComparison.jl")
using .WavePDEChess
using .SymbolicTasks
using .TransferComparison

function tree_changed(before, after)
    if before isa AbstractArray && after isa AbstractArray
        size(before) == size(after) || return true
        return any(before .!= after)
    elseif before isa NamedTuple && after isa NamedTuple
        return any(tree_changed(getfield(before, key), getfield(after, key)) for key in keys(before))
    elseif before isa Tuple && after isa Tuple
        return any(tree_changed(x, y) for (x, y) in zip(before, after))
    else
        return !isequal(before, after)
    end
end

function run_policy_training(policy::Symbol)
    rng = MersenneTwister(21)
    tempdir = mktempdir()
    parquet_path = joinpath(tempdir, "toy_checker.parquet")
    conn = DBInterface.connect(DuckDB.DB, ":memory:")
    DBInterface.execute(conn, """
        CREATE TABLE toy AS
        SELECT [1, 2, 3, 4, 5, 6] AS tokenized, [0.0, 1.0] AS checker_targets
        UNION ALL
        SELECT [7, 8, 9, 10, 11] AS tokenized, [1.0, 0.0] AS checker_targets
        UNION ALL
        SELECT [12, 13, 14, 15, 16, 17, 18] AS tokenized, [0.5, 0.5] AS checker_targets
    """)
    escaped_parquet_path = replace(parquet_path, "'" => "''")
    DBInterface.execute(conn, "COPY toy TO '$escaped_parquet_path' (FORMAT PARQUET)")

    config = ChessMultiHeadModelConfig(
        adapter=ChessAdapterConfig(vocab_size=32, d_model=16, pad_token=0),
        core=WavePDECoreConfig(d_model=16, n_layer=2, solver_steps=1, dt_init=0.05f0, norm_eps=1f-5),
        proposer=ChessMoveHeadConfig(vocab_size=32, d_model=16, tie_embeddings=false, bias=true),
        checker=ChessCheckerHeadConfig(d_model=16, output_dim=2, pooling=:mean),
        max_seq_len=12,
    )
    model = ChessMultiHeadModel(config)
    initial_ps, _ = Lux.setup(rng, model)
    corpus = ChessParquetCorpus(tempdir; min_tokens=3)
    train_cfg = TrainingConfig(
        data_dir=tempdir,
        batch_size=2,
        learning_rate=1.0f-1,
        max_iters=1,
        log_interval=1,
        min_tokens=3,
        train_file_update_interval=1,
        checker_loss_weight=1.0f0,
        training_policy=policy,
        checkpoint_path=joinpath(tempdir, "policy_ckpt.jls"),
        seed=21,
    )

    checkpoint = train!(model, corpus, train_cfg)
    return initial_ps, checkpoint.parameters
end

@testset "WavePDEChess" begin
    rng = Random.default_rng()
    Random.seed!(rng, 0)

    flat_config = WavePDEConfig(
        vocab_size=28,
        d_model=32,
        n_layer=3,
        max_seq_len=64,
        solver_steps=2,
        dt_init=0.05f0,
        norm_eps=1f-5,
        pad_token=0,
    )

    config = ChessModelConfig(flat_config)
    model = WavePDEChessLM(flat_config)
    ps, st = Lux.setup(rng, model)

    tokens = rand(rng, 0:(config.adapter.vocab_size - 1), config.max_seq_len, 2)
    logits, st_next = Lux.apply(model, tokens, ps, st)

    @test size(logits) == (config.adapter.vocab_size, config.max_seq_len, 2)
    @test all(isfinite, logits)
    @test parameter_count(ps) > 0
    @test length(st_next.core.blocks) == config.core.n_layer

    adapter = ChessInputAdapter(config.adapter)
    @test adapter isa AbstractInputAdapter
    adapter_ps, adapter_st = Lux.setup(rng, adapter)
    embedded, _ = Lux.apply(adapter, tokens, adapter_ps, adapter_st)
    iface_embedded, _ = input_adapter_output(adapter, tokens, adapter_ps, adapter_st)
    @test size(embedded) == (config.adapter.d_model, config.max_seq_len, 2)
    @test embedded == iface_embedded

    core = WavePDECore(config.core)
    core_ps, core_st = Lux.setup(rng, core)
    hidden, _ = Lux.apply(core, embedded, core_ps, core_st)
    @test size(hidden) == size(embedded)

    proposer = ChessMoveHead(config.proposer)
    proposer_ps, proposer_st = Lux.setup(rng, proposer)
    proposer_logits, _ = proposer_output(proposer, hidden, adapter_ps, proposer_ps, proposer_st)
    @test size(proposer_logits) == size(logits)

    preset = chess_mamba_11m_config()
    @test preset.core.d_model == 288
    @test preset.core.n_layer == 20
    @test preset.max_seq_len == 1536
end

@testset "MultiHead Composition" begin
    rng = MersenneTwister(7)
    config = ChessMultiHeadModelConfig(
        adapter=ChessAdapterConfig(vocab_size=28, d_model=16, pad_token=0),
        core=WavePDECoreConfig(d_model=16, n_layer=2, solver_steps=1, dt_init=0.05f0, norm_eps=1f-5),
        proposer=ChessMoveHeadConfig(vocab_size=28, d_model=16, tie_embeddings=true, bias=false),
        checker=ChessCheckerHeadConfig(d_model=16, output_dim=3, pooling=:mean),
        max_seq_len=12,
    )
    model = ChessMultiHeadModel(config)
    ps, st = Lux.setup(rng, model)
    tokens = rand(rng, 0:27, config.max_seq_len, 2)
    outputs, _ = Lux.apply(model, tokens, ps, st)

    @test size(outputs.proposer) == (28, config.max_seq_len, 2)
    @test size(outputs.checker) == (3, 2)
end

@testset "MultiHead Reranking" begin
    rng = MersenneTwister(11)
    config = ChessMultiHeadModelConfig(
        adapter=ChessAdapterConfig(vocab_size=4, d_model=1, pad_token=0),
        core=WavePDECoreConfig(d_model=1, n_layer=0, solver_steps=1, dt_init=0.05f0, norm_eps=1f-5),
        proposer=ChessMoveHeadConfig(vocab_size=4, d_model=1, tie_embeddings=false, bias=true),
        checker=ChessCheckerHeadConfig(d_model=1, output_dim=2, pooling=:mean),
        max_seq_len=8,
    )
    model = ChessMultiHeadModel(config)
    _, st = Lux.setup(rng, model)
    ps = (
        adapter=(weight=Float32[-1 1 1 1],),
        core=(blocks=(), norm=(scale=Float32[1],)),
        proposer=(weight=zeros(Float32, 4, 1), bias=Float32[3, 2, 1, 0]),
        checker=(weight=reshape(Float32[1, 3], 2, 1), bias=Float32[0, 0]),
    )
    st = (
        adapter=NamedTuple(),
        core=(blocks=(), norm=NamedTuple()),
        proposer=NamedTuple(),
        checker=NamedTuple(),
    )
    tokens = reshape(Int32[1, 1], 2, 1)
    outputs, _ = Lux.apply(model, tokens, ps, st)

    topk = proposer_topk(outputs.proposer; top_k=3, timestep=2)
    reranked = rerank_next_token_candidates(model, tokens, ps, st; top_k=3, checker_weight=2.0f0)

    @test topk.indices[:, 1] == [0, 1, 2]
    @test reranked.proposer.indices[:, 1] == [0, 1, 2]
    @test reranked.candidate_checker_scores[1, 1] < reranked.candidate_checker_scores[2, 1]
    @test reranked.reranked.indices[:, 1] == [1, 2, 0]
    @test reranked.reranked.scores[1, 1] > reranked.proposer.scores[1, 1]
end

@testset "Checker Metrics" begin
    prediction_metrics = checker_prediction_metrics(
        Float32[1.0 3.0; -1.0 5.0],
        Float32[0.0 1.0; 1.0 2.0],
    )

    @test isapprox(prediction_metrics.mse, 4.5; atol=1f-6)
    @test isapprox(prediction_metrics.mae, 2.0; atol=1f-6)
    @test isapprox(prediction_metrics.rmse, sqrt(4.5); atol=1f-6)
    @test isapprox(prediction_metrics.max_abs_error, 3.0; atol=1f-6)

    rerank_metrics = rerank_comparison_metrics(
        (indices=reshape(Int[0, 1, 2], 3, 1), scores=reshape(Float32[3.0, 2.0, 1.0], 3, 1)),
        (indices=reshape(Int[1, 2, 0], 3, 1), scores=reshape(Float32[4.0, 3.0, 2.0], 3, 1)),
        [1];
    )

    @test rerank_metrics.proposer_top1_accuracy == 0.0
    @test rerank_metrics.reranked_top1_accuracy == 1.0
    @test rerank_metrics.accuracy_delta == 1.0
    @test rerank_metrics.rerank_win_rate == 1.0
    @test rerank_metrics.rerank_loss_rate == 0.0
    @test rerank_metrics.agreement_rate == 0.0

    board_metrics = board_fact_metrics(
        Float32[0.9 0.2; 0.1 0.8],
        Float32[1.0 0.0; 0.0 1.0],
    )

    @test isapprox(board_metrics.overall_accuracy, 1.0; atol=1f-6)
    @test isapprox(board_metrics.exact_match_rate, 1.0; atol=1f-6)
    @test isapprox(board_metrics.brier_score, 0.025; atol=1f-6)
    @test all(isapprox.(board_metrics.per_target_accuracy, 1.0; atol=1f-6))

    legality_metrics = candidate_legality_metrics(Float32[0.9, 0.2, 0.1], Float32[1.0, 0.0, 0.0])
    @test isapprox(legality_metrics.accuracy, 1.0; atol=1f-6)
    @test isapprox(legality_metrics.brier_score, 0.02; atol=1f-6)
    @test isapprox(legality_metrics.predicted_legal_rate, 1 / 3; atol=1f-6)
    @test isapprox(legality_metrics.target_legal_rate, 1 / 3; atol=1f-6)
end

@testset "Chess Transcript Targets" begin
    transcript = "1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O b5 6.Bb3 Bb7"
    encoded = encode_chess_transcript(transcript)
    decoded = decode_chess_tokens(encoded)
    board_targets = extract_board_targets_from_transcript(transcript)
    legality = candidate_legality_targets(transcript, ["Ng5", "Qh5", "O-O"])
    transition = transition_board_targets(transcript, ["Ng5", "Qh5", "O-O"])

    @test decoded == WavePDEChess.normalize_chess_transcript(transcript)
    @test encoded isa Vector{Int32}
    @test length(board_targets) == length(CHESS_BOARD_TARGET_NAMES)
    @test board_targets == Float32[1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
    @test legality == Float32[1, 0, 0]
    @test transition.legality == legality
    @test size(transition.targets) == (length(CHESS_BOARD_TARGET_NAMES), 3)
    @test transition.targets[:, 1] == Float32[0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
    @test transition.targets[:, 2] == zeros(Float32, length(CHESS_BOARD_TARGET_NAMES))
    @test transition.targets[:, 3] == zeros(Float32, length(CHESS_BOARD_TARGET_NAMES))
end

@testset "Transition Consistency Training" begin
    rng = MersenneTwister(19)

    config = ChessMultiHeadModelConfig(
        adapter=ChessAdapterConfig(vocab_size=28, d_model=16, pad_token=0),
        core=WavePDECoreConfig(d_model=16, n_layer=2, solver_steps=1, dt_init=0.05f0, norm_eps=1f-5),
        proposer=ChessMoveHeadConfig(vocab_size=28, d_model=16, tie_embeddings=true, bias=false),
        checker=ChessCheckerHeadConfig(d_model=16, output_dim=length(CHESS_BOARD_TARGET_NAMES), pooling=:mean),
        max_seq_len=96,
    )
    model = ChessMultiHeadModel(config)
    ps, st = Lux.setup(rng, model)

    tempdir = mktempdir()
    parquet_path = joinpath(tempdir, "toy_transition.parquet")
    conn = DBInterface.connect(DuckDB.DB, ":memory:")
    DBInterface.execute(conn, """
        CREATE TABLE toy AS
        SELECT '1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7' AS transcript
        UNION ALL
        SELECT '1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 Be7 5.e3 O-O' AS transcript
        UNION ALL
        SELECT '1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6' AS transcript
    """)
    escaped_parquet_path = replace(parquet_path, "'" => "''")
    DBInterface.execute(conn, "COPY toy TO '$escaped_parquet_path' (FORMAT PARQUET)")

    corpus = ChessParquetCorpus(tempdir; min_tokens=8, board_target_mode=:transcript_board_facts)
    batch = WavePDEChess.sample_training_batch(
        corpus,
        rng;
        batch_size=2,
        max_seq_len=config.max_seq_len,
        transition_candidates_per_example=1,
    )

    @test batch.transition_tokens !== nothing
    @test batch.transition_targets !== nothing
    @test size(batch.transition_targets, 1) == length(CHESS_BOARD_TARGET_NAMES)
    @test size(batch.transition_tokens, 2) == size(batch.transition_targets, 2)

    loss = WavePDEChess.autoregressive_loss(
        model,
        ps,
        st,
        batch;
        checker_loss_weight=1.0f0,
        transition_loss_weight=0.5f0,
    )
    grads = Zygote.gradient(
        p -> WavePDEChess.autoregressive_loss(
            model,
            p,
            st,
            batch;
            checker_loss_weight=1.0f0,
            transition_loss_weight=0.5f0,
        ),
        ps,
    )[1]

    @test isfinite(loss)
    @test parameter_count(grads) > 0

    train_cfg = TrainingConfig(
        data_dir=tempdir,
        batch_size=2,
        max_iters=1,
        log_interval=1,
        min_tokens=8,
        train_file_update_interval=1,
        checker_loss_weight=1.0f0,
        transition_loss_weight=0.5f0,
        transition_candidates_per_example=1,
        board_target_mode=:transcript_board_facts,
        checkpoint_path=joinpath(tempdir, "wavepde_transition_ckpt.jls"),
        seed=19,
    )

    checkpoint = train!(model, corpus, train_cfg)
    @test isfile(train_cfg.checkpoint_path)
    @test length(checkpoint.losses) == 1
end

@testset "Symbolic Bridge Tasks" begin
    rng = MersenneTwister(23)
    examples = generate_symbolic_bridge_examples(count_per_task=2, seed=23)
    families = Set(example.family for example in examples)

    @test length(examples) == 8
    @test families == Set(["propositional_logic", "entailment", "contradiction_detection", "simple_rule_chaining"])

    first_tokens = examples[1].tokenized
    @test first_tokens isa Vector{Int32}
    @test occursin("<bos>", decode_symbolic_tokens(first_tokens))

    tempdir = mktempdir()
    parquet_path = write_symbolic_bridge_parquet(tempdir; count_per_task=2, seed=23)
    @test isfile(parquet_path)

    corpus = ChessParquetCorpus(tempdir; min_tokens=6)
    config = ChessModelConfig(
        adapter=ChessAdapterConfig(vocab_size=SYMBOLIC_VOCAB_SIZE, d_model=24, pad_token=0),
        core=WavePDECoreConfig(d_model=24, n_layer=2, solver_steps=1, dt_init=0.05f0, norm_eps=1f-5),
        proposer=ChessMoveHeadConfig(vocab_size=SYMBOLIC_VOCAB_SIZE, d_model=24, tie_embeddings=true, bias=false),
        max_seq_len=64,
    )
    model = WavePDEChessLM(config)
    ps, st = Lux.setup(rng, model)
    batch = WavePDEChess.sample_training_batch(corpus, rng; batch_size=2, max_seq_len=config.max_seq_len)
    loss = WavePDEChess.autoregressive_loss(model, ps, st, batch)

    @test isfinite(loss)

    train_cfg = TrainingConfig(
        data_dir=tempdir,
        batch_size=2,
        max_iters=1,
        log_interval=1,
        min_tokens=6,
        train_file_update_interval=1,
        checkpoint_path=joinpath(tempdir, "wavepde_symbolic_ckpt.jls"),
        seed=23,
    )

    checkpoint = train!(model, corpus, train_cfg)
    @test isfile(train_cfg.checkpoint_path)
    @test length(checkpoint.losses) == 1
end

@testset "DuckDB Training Path" begin
    rng = MersenneTwister(42)

    config = ChessModelConfig(
        adapter=ChessAdapterConfig(vocab_size=32, d_model=16, pad_token=0),
        core=WavePDECoreConfig(d_model=16, n_layer=2, solver_steps=1, dt_init=0.05f0, norm_eps=1f-5),
        proposer=ChessMoveHeadConfig(vocab_size=32, d_model=16, tie_embeddings=true, bias=false),
        max_seq_len=12,
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
    training_batch = WavePDEChess.sample_training_batch(corpus, rng; batch_size=2, max_seq_len=config.max_seq_len)
    @test training_batch.checker_targets === nothing

    loss = WavePDEChess.autoregressive_loss(model, ps, st, training_batch)
    grads = Zygote.gradient(p -> WavePDEChess.autoregressive_loss(model, p, st, training_batch), ps)[1]

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

@testset "Transcript-Derived Checker Targets" begin
    rng = MersenneTwister(31)

    config = ChessMultiHeadModelConfig(
        adapter=ChessAdapterConfig(vocab_size=28, d_model=16, pad_token=0),
        core=WavePDECoreConfig(d_model=16, n_layer=2, solver_steps=1, dt_init=0.05f0, norm_eps=1f-5),
        proposer=ChessMoveHeadConfig(vocab_size=28, d_model=16, tie_embeddings=true, bias=false),
        checker=ChessCheckerHeadConfig(d_model=16, output_dim=length(CHESS_BOARD_TARGET_NAMES), pooling=:mean),
        max_seq_len=96,
    )
    model = ChessMultiHeadModel(config)
    ps, st = Lux.setup(rng, model)

    tempdir = mktempdir()
    parquet_path = joinpath(tempdir, "toy_transcript.parquet")
    conn = DBInterface.connect(DuckDB.DB, ":memory:")
    DBInterface.execute(conn, """
        CREATE TABLE toy AS
        SELECT '1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7' AS transcript
        UNION ALL
        SELECT '1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 Be7 5.e3 O-O' AS transcript
        UNION ALL
        SELECT '1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6' AS transcript
    """)
    escaped_parquet_path = replace(parquet_path, "'" => "''")
    DBInterface.execute(conn, "COPY toy TO '$escaped_parquet_path' (FORMAT PARQUET)")

    corpus = ChessParquetCorpus(tempdir; min_tokens=8, board_target_mode=:transcript_board_facts)
    batch = WavePDEChess.sample_training_batch(corpus, rng; batch_size=2, max_seq_len=config.max_seq_len)

    @test batch.checker_targets !== nothing
    @test size(batch.checker_targets) == (length(CHESS_BOARD_TARGET_NAMES), 2)
    @test size(batch.tokens, 1) >= 8

    loss = WavePDEChess.autoregressive_loss(model, ps, st, batch)
    grads = Zygote.gradient(p -> WavePDEChess.autoregressive_loss(model, p, st, batch), ps)[1]

    @test isfinite(loss)
    @test parameter_count(grads) > 0

    train_cfg = TrainingConfig(
        data_dir=tempdir,
        batch_size=2,
        max_iters=1,
        log_interval=1,
        min_tokens=8,
        train_file_update_interval=1,
        board_target_mode=:transcript_board_facts,
        checkpoint_path=joinpath(tempdir, "wavepde_transcript_ckpt.jls"),
        seed=31,
    )

    checkpoint = train!(model, corpus, train_cfg)
    @test isfile(train_cfg.checkpoint_path)
    @test length(checkpoint.losses) == 1
end

@testset "Checker Supervision" begin
    rng = MersenneTwister(21)

    config = ChessMultiHeadModelConfig(
        adapter=ChessAdapterConfig(vocab_size=32, d_model=16, pad_token=0),
        core=WavePDECoreConfig(d_model=16, n_layer=2, solver_steps=1, dt_init=0.05f0, norm_eps=1f-5),
        proposer=ChessMoveHeadConfig(vocab_size=32, d_model=16, tie_embeddings=true, bias=false),
        checker=ChessCheckerHeadConfig(d_model=16, output_dim=2, pooling=:mean),
        max_seq_len=12,
    )
    model = ChessMultiHeadModel(config)
    ps, st = Lux.setup(rng, model)

    tempdir = mktempdir()
    parquet_path = joinpath(tempdir, "toy_checker.parquet")
    conn = DBInterface.connect(DuckDB.DB, ":memory:")
    DBInterface.execute(conn, """
        CREATE TABLE toy AS
        SELECT [1, 2, 3, 4, 5, 6] AS tokenized, [0.0, 1.0] AS checker_targets
        UNION ALL
        SELECT [7, 8, 9, 10, 11] AS tokenized, [1.0, 0.0] AS checker_targets
        UNION ALL
        SELECT [12, 13, 14, 15, 16, 17, 18] AS tokenized, [0.5, 0.5] AS checker_targets
    """)
    escaped_parquet_path = replace(parquet_path, "'" => "''")
    DBInterface.execute(conn, "COPY toy TO '$escaped_parquet_path' (FORMAT PARQUET)")

    corpus = ChessParquetCorpus(tempdir; min_tokens=3)
    batch = WavePDEChess.sample_training_batch(corpus, rng; batch_size=2, max_seq_len=config.max_seq_len)

    @test batch.checker_targets !== nothing
    @test size(batch.checker_targets) == (2, 2)

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
        checkpoint_path=joinpath(tempdir, "wavepde_multihead_ckpt.jls"),
        seed=21,
    )

    checkpoint = train!(model, corpus, train_cfg)
    @test isfile(train_cfg.checkpoint_path)
    @test length(checkpoint.losses) == 1
end

@testset "Training Policy" begin
    @test TrainingConfig().training_policy == :full

    initial_ps, adapter_only_ps = run_policy_training(:adapters_only)
    @test tree_changed(initial_ps.adapter, adapter_only_ps.adapter)
    @test !tree_changed(initial_ps.core, adapter_only_ps.core)
    @test !tree_changed(initial_ps.proposer, adapter_only_ps.proposer)
    @test !tree_changed(initial_ps.checker, adapter_only_ps.checker)

    initial_ps, heads_only_ps = run_policy_training(:heads_only)
    @test !tree_changed(initial_ps.adapter, heads_only_ps.adapter)
    @test !tree_changed(initial_ps.core, heads_only_ps.core)
    @test tree_changed(initial_ps.proposer, heads_only_ps.proposer)
    @test tree_changed(initial_ps.checker, heads_only_ps.checker)

    initial_ps, full_ps = run_policy_training(:full)
    @test tree_changed(initial_ps.adapter, full_ps.adapter)
    @test tree_changed(initial_ps.core, full_ps.core)
    @test tree_changed(initial_ps.proposer, full_ps.proposer)
    @test tree_changed(initial_ps.checker, full_ps.checker)
end

@testset "Transfer Comparison" begin
    rng = MersenneTwister(29)
    tempdir = mktempdir()
    source_checkpoint_path = joinpath(tempdir, "source_checkpoint.jls")

    source_config = symbolic_model_config(
        d_model=16,
        n_layer=2,
        solver_steps=1,
        max_seq_len=32,
    )
    source_model = WavePDEChessLM(source_config)
    source_ps, _ = Lux.setup(rng, source_model)
    open(source_checkpoint_path, "w") do io
        serialize(io, (parameters=source_ps,))
    end

    result = compare_symbolic_transfer(
        output_dir=joinpath(tempdir, "comparison"),
        source_checkpoint_path=source_checkpoint_path,
        dataset_dir=joinpath(tempdir, "comparison", "data"),
        count_per_task=2,
        seed=29,
        batch_size=2,
        learning_rate=1.0f-3,
        max_iters=1,
        d_model=16,
        n_layer=2,
        solver_steps=1,
        max_seq_len=32,
    )

    @test isfile(result.dataset_path)
    @test isfile(result.scratch_full.checkpoint_path)
    @test isfile(result.chess_core_frozen.checkpoint_path)
    @test isfile(result.chess_core_finetune.checkpoint_path)
    @test result.scratch_full.transplanted_core == false
    @test result.chess_core_frozen.transplanted_core == true
    @test result.chess_core_frozen.frozen_core == true
    @test result.chess_core_finetune.transplanted_core == true
    @test result.chess_core_finetune.frozen_core == false
    @test isfinite(result.scratch_full.final_loss)
    @test isfinite(result.chess_core_frozen.final_loss)
    @test isfinite(result.chess_core_finetune.final_loss)
end
