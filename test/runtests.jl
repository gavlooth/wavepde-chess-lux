using Test
using Random
using Serialization
using DuckDB
using DBInterface
using Lux
using Zygote

include("../src/WavePDEChess.jl")
include("../src/Training/CheckerMetrics.jl")
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

    checkpoint = WavePDEChess.train!(model, corpus, train_cfg)
    return initial_ps, checkpoint.parameters
end

function write_test_pgn(path::AbstractString)
    open(path, "w") do io
        write(io, """
[Event "WavePDE Test"]
[Site "Local"]
[Date "2026.03.23"]
[Round "1"]
[White "White"]
[Black "Black"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0
""")
    end
    return path
end

function write_test_transcript_parquet(path::AbstractString)
    conn = DBInterface.connect(DuckDB.DB, ":memory:")
    DBInterface.execute(conn, """
        CREATE TABLE toy AS
        SELECT '1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7' AS transcript
        UNION ALL
        SELECT '1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 Be7 5.e3 O-O' AS transcript
        UNION ALL
        SELECT '1.c4 e5 2.Nc3 Nf6 3.Nf3 Nc6 4.g3 d5 5.cxd5 Nxd5' AS transcript
    """)
    escaped_path = replace(path, "'" => "''")
    DBInterface.execute(conn, "COPY toy TO '$escaped_path' (FORMAT PARQUET)")
    return path
end

function write_test_value_parquet(path::AbstractString)
    conn = DBInterface.connect(DuckDB.DB, ":memory:")
    DBInterface.execute(conn, """
        CREATE TABLE toy AS
        SELECT 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' AS fen, 20 AS depth, 0.0 AS cp, CAST(NULL AS DOUBLE) AS mate
        UNION ALL
        SELECT 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1' AS fen, 24 AS depth, 120.0 AS cp, CAST(NULL AS DOUBLE) AS mate
        UNION ALL
        SELECT 'r1bqkbnr/pppp1ppp/2n5/4p3/1b2P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3' AS fen, 28 AS depth, CAST(NULL AS DOUBLE) AS cp, 3.0 AS mate
    """)
    escaped_path = replace(path, "'" => "''")
    DBInterface.execute(conn, "COPY toy TO '$escaped_path' (FORMAT PARQUET)")
    return path
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

    dt_safe, raw_cfl_safe, _, _, _, clamped_safe = WavePDEChess.wavepde_cfl_control(
        fill(0.5f0, 1, 1, 1),
        0.1f0,
        0.9f0,
        1f-6,
        10f0,
        1f-6,
    )
    @test clamped_safe
    @test raw_cfl_safe ≈ 0.05f0 atol=1f-6

    dt_clamped, raw_cfl_clamped, _, _, _, clamped = WavePDEChess.wavepde_cfl_control(
        fill(2.0f0, 1, 1, 1),
        1.0f0,
        0.9f0,
        1f-6,
        10f0,
        1f-6,
    )
    @test clamped
    @test raw_cfl_clamped ≈ 2.0f0 atol=1f-5
    @test dt_clamped * raw_cfl_clamped < raw_cfl_clamped

    # The CFL warning path must remain outside the differentiated graph.
    mixer = WavePDEChess.WavePDESpectralMixer(16, 1, 0.05f0, 1f-4, 0.9f0, 1f-6, 10f0)
    mixer_ps = Lux.initialparameters(rng, mixer)
    mixer_st = Lux.initialstates(rng, mixer)
    mixer_input = rand(rng, Float32, 16, 32, 2)
    unstable_mixer_ps = (; mixer_ps..., log_dt=10.0f0)
    unstable_grads = Zygote.gradient(
        p -> sum(first(mixer(mixer_input, p, mixer_st))),
        unstable_mixer_ps,
    )[1]
    @test parameter_count(unstable_grads) > 0

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

@testset "Policy Targets" begin
    transcript = "1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6"
    transcript_candidates = policy_legal_candidates(transcript)
    transcript_bundle = policy_target_bundle(transcript, "O-O")

    @test transcript_bundle.candidates == transcript_candidates
    @test sum(transcript_bundle.labels) == 1.0f0
    @test transcript_bundle.candidates[transcript_bundle.target_index] == "O-O"
    @test transcript_bundle.target_move == "O-O"

    explicit_candidates = policy_legal_candidates(["Qh5", "Nf3", "Nf3", "O-O"])
    explicit_bundle = policy_target_bundle(explicit_candidates, "Nf3")

    @test explicit_candidates == ["Qh5", "Nf3", "O-O"]
    @test explicit_bundle.labels == Float32[0.0, 1.0, 0.0]
    @test policy_target_index(explicit_candidates, "O-O") == 3
    @test_throws ArgumentError policy_target_bundle(explicit_candidates, "Qh4")

    encoded_action = WavePDEChess.encode_policy_action_san("O-O")
    conditioned_tokens = WavePDEChess.append_policy_action_tokens(Int32[1, 2, 3], "O-O")

    @test !isempty(encoded_action)
    @test all(token -> token >= WavePDEChess.POLICY_ACTION_TOKEN_BASE, encoded_action)
    @test conditioned_tokens[1:3] == Int32[1, 2, 3]
    @test conditioned_tokens[4] == WavePDEChess.POLICY_ACTION_SEPARATOR_TOKEN
    @test conditioned_tokens[5:end] == encoded_action
end

@testset "PGN State Transitions" begin
    rng = MersenneTwister(37)
    tempdir = mktempdir()
    pgn_path = write_test_pgn(joinpath(tempdir, "toy_game.pgn"))

    discovered_files = discover_pgn_files(tempdir)
    examples = parse_pgn_state_examples(tempdir)
    first_example = first(examples)

    @test discovered_files == [pgn_path]
    @test length(examples) == 10
    @test length(first_example.state_tokens) == BOARD_STATE_SEQUENCE_LENGTH
    @test length(first_example.next_state_tokens) == BOARD_STATE_SEQUENCE_LENGTH
    @test length(first_example.attacked_white) == 64
    @test length(first_example.attacked_black) == 64
    @test first_example.in_check == Float32[0, 0]
    @test first_example.pinned_count == Float32[0, 0]
    @test first_example.move_san == "e4"
    @test first_example.state_tokens != first_example.next_state_tokens

    probe = WavePDEChess.board_probe_targets_from_transcript("1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7")
    @test length(probe.attacked_white) == 64
    @test length(probe.attacked_black) == 64
    @test probe.in_check == Float32[0, 0]
    @test length(probe.pinned_count) == 2
    @test all(isfinite, probe.pinned_count)
    @test length(probe.king_pressure) == 2
    @test all(isfinite, probe.king_pressure)
    @test length(probe.attacked_piece_count) == 2
    @test probe.attacked_piece_count[2] >= 1
    @test length(probe.mobility) == 2
    @test all(>(0), probe.mobility)

    parquet_dir = joinpath(tempdir, "dataset")
    parquet_path = write_pgn_state_parquet(tempdir, parquet_dir)
    @test isfile(parquet_path)

    corpus = StateTransitionParquetCorpus(parquet_dir; min_tokens=BOARD_STATE_SEQUENCE_LENGTH)
    @test corpus.active_moves !== nothing
    @test corpus.active_probe_targets !== nothing
    @test corpus.probe_target_dim == BOARD_PROBE_TARGET_LENGTH
    @test first(corpus.active_moves) == "e4"
    batch = WavePDEChess.sample_training_batch(
        corpus,
        rng;
        batch_size=2,
        max_seq_len=BOARD_STATE_SEQUENCE_LENGTH,
    )
    @test size(batch.tokens) == (BOARD_STATE_SEQUENCE_LENGTH, 2)
    @test size(batch.target_tokens) == (BOARD_STATE_SEQUENCE_LENGTH, 2)
    @test size(batch.checker_targets) == (BOARD_PROBE_TARGET_LENGTH, 2)
    split_probes = WavePDEChess.split_board_probe_targets(batch.checker_targets)
    @test size(split_probes.attacked_white) == (64, 2)
    @test size(split_probes.mobility) == (2, 2)
    family_metrics = state_slot_family_metrics(batch.target_tokens, batch.target_tokens)
    legality_metrics = successor_legality_metrics(batch.tokens, batch.target_tokens)
    @test isapprox(family_metrics.coarse_state.token_accuracy, 1.0; atol=1f-6)
    @test isapprox(family_metrics.attack_maps.exact_match_rate, 1.0; atol=1f-6)
    @test isapprox(family_metrics.pressure_counts.token_accuracy, 1.0; atol=1f-6)
    @test isapprox(legality_metrics.valid_board_rate, 1.0; atol=1f-6)
    @test isapprox(legality_metrics.reachable_rate, 1.0; atol=1f-6)
    @test isapprox(legality_metrics.reachable_strict_rate, 1.0; atol=1f-6)

    relaxed_target = copy(batch.target_tokens[:, 1:1])
    relaxed_target[71, 1] = Int32(WavePDEChess.BOARD_STATE_HALFMOVE_BASE + 5)
    relaxed_legality = successor_legality_metrics(batch.tokens[:, 1:1], relaxed_target)
    @test isapprox(relaxed_legality.valid_board_rate, 1.0; atol=1f-6)
    @test isapprox(relaxed_legality.reachable_rate, 1.0; atol=1f-6)
    @test isapprox(relaxed_legality.reachable_strict_rate, 0.0; atol=1f-6)

    config = ChessModelConfig(
        adapter=ChessAdapterConfig(vocab_size=BOARD_STATE_VOCAB_SIZE, d_model=16, pad_token=0),
        core=WavePDECoreConfig(d_model=16, n_layer=2, solver_steps=1, dt_init=0.05f0, norm_eps=1f-5),
        proposer=ChessMoveHeadConfig(vocab_size=BOARD_STATE_VOCAB_SIZE, d_model=16, tie_embeddings=true, bias=false),
        max_seq_len=BOARD_STATE_SEQUENCE_LENGTH,
    )
    model = WavePDEChessLM(config)
    ps, st = Lux.setup(rng, model)
    loss = WavePDEChess.autoregressive_loss(model, ps, st, batch)
    grads = Zygote.gradient(p -> WavePDEChess.autoregressive_loss(model, p, st, batch), ps)[1]

    @test isfinite(loss)
    @test parameter_count(grads) > 0

    train_cfg = TrainingConfig(
        data_dir=parquet_dir,
        batch_size=2,
        max_iters=1,
        log_interval=1,
        min_tokens=BOARD_STATE_SEQUENCE_LENGTH,
        train_file_update_interval=1,
        checkpoint_path=joinpath(tempdir, "wavepde_state_transition_ckpt.jls"),
        seed=37,
    )

    checkpoint = WavePDEChess.train!(model, corpus, train_cfg)
    @test isfile(train_cfg.checkpoint_path)
    @test length(checkpoint.losses) == 1
end

@testset "Policy Conditioned State Transitions" begin
    rng = MersenneTwister(53)
    tempdir = mktempdir()
    pgn_path = write_test_pgn(joinpath(tempdir, "toy_game.pgn"))
    parquet_dir = joinpath(tempdir, "dataset")
    write_pgn_state_parquet(dirname(pgn_path), parquet_dir)

    corpus = StateTransitionParquetCorpus(parquet_dir; min_tokens=BOARD_STATE_SEQUENCE_LENGTH)
    state_only_batch = WavePDEChess.sample_training_batch(
        corpus,
        rng;
        batch_size=2,
        max_seq_len=BOARD_STATE_SEQUENCE_LENGTH,
        policy_condition_mode=:state_only,
    )
    state_action_batch = WavePDEChess.sample_training_batch(
        corpus,
        rng;
        batch_size=2,
        max_seq_len=BOARD_STATE_SEQUENCE_LENGTH + 1 + WavePDEChess.MAX_POLICY_ACTION_TOKENS,
        policy_condition_mode=:state_action,
    )
    state_action_coarse_batch = WavePDEChess.sample_training_batch(
        corpus,
        rng;
        batch_size=2,
        max_seq_len=BOARD_STATE_SEQUENCE_LENGTH + 1 + WavePDEChess.MAX_POLICY_ACTION_TOKENS,
        policy_condition_mode=:state_action,
        state_target_mode=:coarse_only,
    )

    @test size(state_only_batch.tokens) == (BOARD_STATE_SEQUENCE_LENGTH, 2)
    @test size(state_only_batch.target_mask) == size(state_only_batch.target_tokens)
    @test count(state_only_batch.target_mask) == BOARD_STATE_SEQUENCE_LENGTH * 2
    @test size(state_action_batch.tokens, 1) > size(state_only_batch.tokens, 1)
    @test size(state_action_batch.target_mask) == size(state_action_batch.target_tokens)
    @test count(state_action_batch.target_mask) == BOARD_STATE_SEQUENCE_LENGTH * 2
    @test count(state_action_coarse_batch.target_mask) == BOARD_STATE_COARSE_LENGTH * 2
    @test state_action_batch.tokens[BOARD_STATE_SEQUENCE_LENGTH + 1, 1] == WavePDEChess.POLICY_ACTION_SEPARATOR_TOKEN

    model_config = ChessModelConfig(
        adapter=ChessAdapterConfig(vocab_size=WavePDEChess.STATE_ACTION_VOCAB_SIZE, d_model=16, pad_token=0),
        core=WavePDECoreConfig(d_model=16, n_layer=2, solver_steps=1, dt_init=0.05f0, norm_eps=1f-5),
        proposer=ChessMoveHeadConfig(vocab_size=WavePDEChess.STATE_ACTION_VOCAB_SIZE, d_model=16, tie_embeddings=true, bias=false),
        max_seq_len=BOARD_STATE_SEQUENCE_LENGTH + 1 + WavePDEChess.MAX_POLICY_ACTION_TOKENS,
    )
    model = WavePDEChessLM(model_config)
    ps, st = Lux.setup(rng, model)
    loss = WavePDEChess.autoregressive_loss(model, ps, st, state_action_batch)
    grads = Zygote.gradient(p -> WavePDEChess.autoregressive_loss(model, p, st, state_action_batch), ps)[1]

    @test isfinite(loss)
    @test parameter_count(grads) > 0

    train_cfg = TrainingConfig(
        data_dir=parquet_dir,
        batch_size=2,
        max_iters=1,
        log_interval=1,
        min_tokens=BOARD_STATE_SEQUENCE_LENGTH,
        train_file_update_interval=1,
        policy_condition_mode=:state_action,
        checkpoint_path=joinpath(tempdir, "wavepde_state_action_ckpt.jls"),
        seed=53,
    )

    checkpoint = WavePDEChess.train!(model, corpus, train_cfg)
    @test isfile(train_cfg.checkpoint_path)
    @test checkpoint.training_config.policy_condition_mode == :state_action
    @test length(checkpoint.losses) == 1
end

@testset "Board Probe Metrics" begin
    probe_metrics = WavePDEChess.CheckerMetrics.board_probe_metrics(
        (
            attacked_white=Float32[1 0; 0 1],
            attacked_black=Float32[0 1; 1 0],
            in_check=Float32[1 0; 0 1],
            pinned_count=Float32[0 1; 1 2],
            king_pressure=Float32[2 1; 0 0],
            mobility=Float32[10 20; 30 40],
            attacked_piece_count=Float32[1 0; 2 3],
        ),
        (
            attacked_white=Float32[1 0; 0 1],
            attacked_black=Float32[0 1; 1 0],
            in_check=Float32[1 0; 0 1],
            pinned_count=Float32[0 1; 1 2],
            king_pressure=Float32[2 1; 0 0],
            mobility=Float32[10 20; 30 40],
            attacked_piece_count=Float32[1 0; 2 3],
        ),
    )

    @test isapprox(probe_metrics.attacked_white.overall_accuracy, 1.0; atol=1f-6)
    @test isapprox(probe_metrics.attacked_black.exact_match_rate, 1.0; atol=1f-6)
    @test isapprox(probe_metrics.in_check.brier_score, 0.0; atol=1f-6)
    @test isapprox(probe_metrics.pinned_count.mse, 0.0; atol=1f-6)
    @test isapprox(probe_metrics.king_pressure.mae, 0.0; atol=1f-6)
    @test isapprox(probe_metrics.mobility.rmse, 0.0; atol=1f-6)
    @test isapprox(probe_metrics.attacked_piece_count.max_abs_error, 0.0; atol=1f-6)
end

@testset "Transcript Parquet State Transitions" begin
    tempdir = mktempdir()
    parquet_path = joinpath(tempdir, "toy_transcript.parquet")
    conn = DBInterface.connect(DuckDB.DB, ":memory:")
    DBInterface.execute(conn, """
        CREATE TABLE toy AS
        SELECT '1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7' AS transcript
        UNION ALL
        SELECT '1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 Be7 5.e3 O-O' AS transcript
    """)
    escaped_parquet_path = replace(parquet_path, "'" => "''")
    DBInterface.execute(conn, "COPY toy TO '$escaped_parquet_path' (FORMAT PARQUET)")

    examples = parse_transcript_parquet_state_examples(tempdir)
    @test length(examples) == 20
    @test first(examples).move_san == "e4"
    @test length(first(examples).state_tokens) == BOARD_STATE_SEQUENCE_LENGTH

    output_dir = joinpath(tempdir, "state_dataset")
    state_parquet = write_transcript_state_parquet(tempdir, output_dir)
    @test isfile(state_parquet)

    corpus = StateTransitionParquetCorpus(output_dir; min_tokens=BOARD_STATE_SEQUENCE_LENGTH)
    @test !isempty(corpus.active_states)
    @test length(first(corpus.active_states)) == BOARD_STATE_SEQUENCE_LENGTH
    @test length(first(corpus.active_next_states)) == BOARD_STATE_SEQUENCE_LENGTH
    @test corpus.active_probe_targets !== nothing
    @test corpus.probe_target_dim == BOARD_PROBE_TARGET_LENGTH
end

@testset "State Transition Evaluation" begin
    rng = MersenneTwister(41)
    tempdir = mktempdir()
    train_dir = joinpath(tempdir, "train")
    eval_dir = joinpath(tempdir, "eval")
    mkpath(train_dir)
    mkpath(eval_dir)

    write_test_pgn(joinpath(train_dir, "train_game.pgn"))
    open(joinpath(eval_dir, "eval_game.pgn"), "w") do io
        write(io, """
[Event "WavePDE Eval"]
[Site "Local"]
[Date "2026.03.23"]
[Round "1"]
[White "White"]
[Black "Black"]
[Result "0-1"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 0-1
""")
    end

    train_parquet_dir = joinpath(tempdir, "train_parquet")
    eval_parquet_dir = joinpath(tempdir, "eval_parquet")
    write_pgn_state_parquet(train_dir, train_parquet_dir)
    write_pgn_state_parquet(eval_dir, eval_parquet_dir)

    config = ChessModelConfig(
        adapter=ChessAdapterConfig(vocab_size=BOARD_STATE_VOCAB_SIZE, d_model=16, pad_token=0),
        core=WavePDECoreConfig(d_model=16, n_layer=2, solver_steps=1, dt_init=0.05f0, norm_eps=1f-5),
        proposer=ChessMoveHeadConfig(vocab_size=BOARD_STATE_VOCAB_SIZE, d_model=16, tie_embeddings=true, bias=false),
        max_seq_len=BOARD_STATE_SEQUENCE_LENGTH,
    )
    model = WavePDEChessLM(config)
    corpus = StateTransitionParquetCorpus(train_parquet_dir; min_tokens=BOARD_STATE_SEQUENCE_LENGTH)
    train_cfg = TrainingConfig(
        data_dir=train_parquet_dir,
        batch_size=2,
        max_iters=1,
        log_interval=1,
        min_tokens=BOARD_STATE_SEQUENCE_LENGTH,
        train_file_update_interval=1,
        checkpoint_path=joinpath(tempdir, "wavepde_state_eval_ckpt.jls"),
        seed=41,
    )

    checkpoint = WavePDEChess.train!(model, corpus, train_cfg)
    @test isfile(train_cfg.checkpoint_path)
    @test length(checkpoint.losses) == 1

    result = evaluate_state_transition_checkpoint(train_cfg.checkpoint_path, eval_parquet_dir; batch_size=2)
    @test result.checkpoint_path == train_cfg.checkpoint_path
    @test result.data_dir == eval_parquet_dir
    @test result.num_examples > 0
    @test result.num_tokens > 0
    @test isfinite(result.token_loss)
    @test 0.0 <= result.exact_slot_accuracy <= 1.0
    @test 0.0 <= result.exact_sequence_match_rate <= 1.0
    @test 0.0 <= result.board_fact_metrics.overall_accuracy <= 1.0
    @test 0.0 <= result.board_fact_metrics.exact_match_rate <= 1.0
    @test isfinite(result.board_fact_metrics.brier_score)
    @test 0.0 <= result.state_slot_family_metrics.coarse_state.token_accuracy <= 1.0
    @test 0.0 <= result.state_slot_family_metrics.attack_maps.exact_match_rate <= 1.0
    @test 0.0 <= result.state_slot_family_metrics.pressure_counts.token_accuracy <= 1.0
    @test 0.0 <= result.successor_legality_metrics.valid_board_rate <= 1.0
    @test 0.0 <= result.successor_legality_metrics.reachable_rate <= 1.0
    @test 0.0 <= result.successor_legality_metrics.reachable_strict_rate <= 1.0
end

@testset "State Probe Supervision" begin
    rng = MersenneTwister(141)
    tempdir = mktempdir()
    train_dir = joinpath(tempdir, "train")
    eval_dir = joinpath(tempdir, "eval")
    mkpath(train_dir)
    mkpath(eval_dir)
    write_test_pgn(joinpath(train_dir, "train_game.pgn"))
    write_test_pgn(joinpath(eval_dir, "eval_game.pgn"))

    train_parquet_dir = joinpath(tempdir, "train_parquet")
    eval_parquet_dir = joinpath(tempdir, "eval_parquet")
    write_pgn_state_parquet(train_dir, train_parquet_dir)
    write_pgn_state_parquet(eval_dir, eval_parquet_dir)

    corpus = StateTransitionParquetCorpus(train_parquet_dir; min_tokens=BOARD_STATE_SEQUENCE_LENGTH)
    batch = WavePDEChess.sample_training_batch(
        corpus,
        rng;
        batch_size=2,
        max_seq_len=BOARD_STATE_SEQUENCE_LENGTH + 1 + WavePDEChess.MAX_POLICY_ACTION_TOKENS,
        policy_condition_mode=:state_action,
        state_target_mode=:coarse_only,
    )
    @test batch.checker_targets !== nothing
    @test size(batch.checker_targets) == (BOARD_PROBE_TARGET_LENGTH, 2)

    config = ChessMultiHeadModelConfig(
        adapter=ChessAdapterConfig(vocab_size=WavePDEChess.STATE_ACTION_VOCAB_SIZE, d_model=16, pad_token=0),
        core=WavePDECoreConfig(d_model=16, n_layer=2, solver_steps=1, dt_init=0.05f0, norm_eps=1f-5),
        proposer=ChessMoveHeadConfig(vocab_size=WavePDEChess.STATE_ACTION_VOCAB_SIZE, d_model=16, tie_embeddings=true, bias=false),
        checker=ChessCheckerHeadConfig(d_model=16, output_dim=BOARD_PROBE_TARGET_LENGTH, pooling=:mean),
        max_seq_len=BOARD_STATE_SEQUENCE_LENGTH + 1 + WavePDEChess.MAX_POLICY_ACTION_TOKENS,
    )
    model = ChessMultiHeadModel(config)
    ps, st = Lux.setup(rng, model)
    loss = WavePDEChess.autoregressive_loss(model, ps, st, batch; checker_loss_weight=0.25f0)
    @test isfinite(loss)

    train_cfg = TrainingConfig(
        data_dir=train_parquet_dir,
        batch_size=2,
        max_iters=1,
        log_interval=1,
        min_tokens=BOARD_STATE_SEQUENCE_LENGTH,
        train_file_update_interval=1,
        probe_loss_weight=0.25f0,
        policy_condition_mode=:state_action,
        state_target_mode=:coarse_only,
        checkpoint_path=joinpath(tempdir, "wavepde_state_probe_ckpt.jls"),
        seed=141,
    )
    checkpoint = WavePDEChess.train!(model, corpus, train_cfg)
    @test isfile(train_cfg.checkpoint_path)
    @test checkpoint.model_config isa ChessMultiHeadModelConfig

    loaded = load_state_transition_checkpoint(train_cfg.checkpoint_path)
    @test loaded.model isa ChessMultiHeadModel

    result = evaluate_state_transition_checkpoint(
        train_cfg.checkpoint_path,
        eval_parquet_dir;
        batch_size=2,
        policy_condition_mode=:state_action,
        state_target_mode=:coarse_only,
    )
    @test result.probe_metrics !== nothing
    @test 0.0 <= result.probe_metrics.attacked_white.overall_accuracy <= 1.0
    @test 0.0 <= result.probe_metrics.attacked_black.exact_match_rate <= 1.0
    @test isfinite(result.probe_metrics.pinned_count.mse)
    @test isfinite(result.probe_metrics.mobility.rmse)
end

@testset "State Transition Mode Comparison" begin
    tempdir = mktempdir()
    pgn_dir = joinpath(tempdir, "pgn")
    mkpath(pgn_dir)
    write_test_pgn(joinpath(pgn_dir, "compare_game.pgn"))

    train_parquet_dir = joinpath(tempdir, "train_parquet")
    eval_parquet_dir = joinpath(tempdir, "eval_parquet")
    write_pgn_state_parquet(pgn_dir, train_parquet_dir)
    write_pgn_state_parquet(pgn_dir, eval_parquet_dir)

    result = compare_state_transition_training_modes(
        train_parquet_dir,
        eval_parquet_dir;
        d_model=8,
        n_layer=1,
        solver_steps=1,
        batch_size=2,
        learning_rate=1.0f-3,
        max_iters=1,
        seed=61,
        output_dir=joinpath(tempdir, "compare_output"),
    )

    @test isfile(result.state_only.checkpoint_path)
    @test isfile(result.state_action.checkpoint_path)
    @test isfinite(result.state_only.final_loss)
    @test isfinite(result.state_action.final_loss)
    @test isfinite(result.state_only.eval.token_loss)
    @test isfinite(result.state_action.eval.token_loss)
    @test 0.0 <= result.state_only.eval.exact_slot_accuracy <= 1.0
    @test 0.0 <= result.state_action.eval.exact_slot_accuracy <= 1.0
end

@testset "Board Value Training" begin
    rng = MersenneTwister(73)
    tempdir = mktempdir()
    parquet_path = write_test_value_parquet(joinpath(tempdir, "toy_value.parquet"))

    @test WavePDEChess.value_target_from_engine_eval(0.0, missing; cp_scale=400f0) ≈ 0.0f0 atol=1f-6
    @test WavePDEChess.value_target_from_engine_eval(400.0, missing; cp_scale=400f0) ≈ tanh(1.0f0) atol=1f-6
    @test WavePDEChess.value_target_from_engine_eval(missing, 3.0; cp_scale=400f0) == 1.0f0

    corpus = BoardValueParquetCorpus(tempdir; chunk_rows=2, cp_scale=400f0)
    @test length(corpus.active_states) == 2
    @test length(corpus.active_values) == 2
    @test corpus.file_row_counts[parquet_path] == 3
    @test first(corpus.active_states) == board_state_tokens_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    WavePDEChess.reload_file!(corpus, parquet_path; offset=2)
    @test length(corpus.active_states) == 1
    @test corpus.active_values[1] == 1.0f0

    WavePDEChess.reload_file!(corpus, parquet_path; offset=0)
    batch = WavePDEChess.sample_board_value_batch(corpus, rng; batch_size=2, max_seq_len=BOARD_STATE_SEQUENCE_LENGTH)
    @test size(batch.tokens) == (BOARD_STATE_SEQUENCE_LENGTH, 2)
    @test size(batch.value_targets) == (1, 2)

    config = BoardValueModelConfig(
        adapter=ChessAdapterConfig(vocab_size=BOARD_STATE_VOCAB_SIZE, d_model=16, pad_token=0),
        core=WavePDECoreConfig(d_model=16, n_layer=2, solver_steps=1, dt_init=0.05f0, norm_eps=1f-5),
        value_head=ChessCheckerHeadConfig(d_model=16, output_dim=1, pooling=:mean),
        max_seq_len=BOARD_STATE_SEQUENCE_LENGTH,
    )
    model = BoardValueModel(config)
    ps, st = Lux.setup(rng, model)
    loss, _ = WavePDEChess.board_value_loss(model, ps, st, batch)
    grads = Zygote.gradient(
        p -> first(WavePDEChess.board_value_loss(model, p, st, batch)),
        ps,
    )[1]
    @test isfinite(loss)
    @test parameter_count(grads) > 0

    train_cfg = BoardValueTrainingConfig(
        data_dir=tempdir,
        batch_size=2,
        learning_rate=1.0f-3,
        max_iters=1,
        log_interval=1,
        min_tokens=BOARD_STATE_SEQUENCE_LENGTH,
        train_file_update_interval=1,
        cp_scale=400f0,
        chunk_rows=2,
        checkpoint_path=joinpath(tempdir, "wavepde_value_ckpt.jls"),
        seed=73,
    )
    checkpoint = train_value!(model, corpus, train_cfg)
    @test isfile(train_cfg.checkpoint_path)
    @test length(checkpoint.losses) == 1

    loaded = load_board_value_checkpoint(train_cfg.checkpoint_path)
    @test loaded.model isa BoardValueModel

    result = evaluate_board_value_checkpoint(
        train_cfg.checkpoint_path,
        tempdir;
        batch_size=2,
        max_examples=3,
        chunk_rows=2,
    )
    @test result.num_examples == 3
    @test isfinite(result.mse)
    @test isfinite(result.rmse)
    @test 0.0 <= result.direction_accuracy <= 1.0
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

    checkpoint = WavePDEChess.train!(model, corpus, train_cfg)
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

    checkpoint = WavePDEChess.train!(model, corpus, train_cfg)
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

    checkpoint = WavePDEChess.train!(model, corpus, train_cfg)
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

    checkpoint = WavePDEChess.train!(model, corpus, train_cfg)
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

    checkpoint = WavePDEChess.train!(model, corpus, train_cfg)
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

@testset "Dual Surface Training" begin
    rng = MersenneTwister(51)
    tempdir = mktempdir()
    transcript_path = write_test_transcript_parquet(joinpath(tempdir, "toy_transcript.parquet"))
    state_dir = joinpath(tempdir, "state_dataset")
    lm_dir = joinpath(tempdir, "transcript_lm")

    state_parquet = write_transcript_state_parquet(dirname(transcript_path), state_dir)
    lm_parquet = write_transcript_language_parquet(dirname(transcript_path), lm_dir)

    @test isfile(state_parquet)
    @test isfile(lm_parquet)

    corpus = DualSurfaceParquetCorpus(state_dir; min_tokens=BOARD_STATE_SEQUENCE_LENGTH)
    batch = sample_dual_surface_batch(
        corpus,
        rng;
        batch_size=2,
        max_seq_len=BOARD_STATE_SEQUENCE_LENGTH,
    )

    @test size(batch.tokens) == (BOARD_STATE_SEQUENCE_LENGTH, 2)
    @test size(batch.target_tokens) == (BOARD_STATE_SEQUENCE_LENGTH, 2)
    @test size(batch.transcript_targets) == (BOARD_STATE_SEQUENCE_LENGTH, 2)
    @test size(batch.transcript_mask) == (BOARD_STATE_SEQUENCE_LENGTH, 2)
    @test any(batch.transcript_mask)

    config = DualSurfaceStateModelConfig(
        adapter=ChessAdapterConfig(vocab_size=BOARD_STATE_VOCAB_SIZE, d_model=16, pad_token=0),
        core=WavePDECoreConfig(d_model=16, n_layer=2, solver_steps=1, dt_init=0.05f0, norm_eps=1f-5),
        state_head=ChessMoveHeadConfig(vocab_size=BOARD_STATE_VOCAB_SIZE, d_model=16, tie_embeddings=true, bias=false),
        transcript_head=ChessMoveHeadConfig(vocab_size=length(WavePDEChess.CHESS_TRANSCRIPT_STOI), d_model=16, tie_embeddings=false, bias=true),
        max_seq_len=BOARD_STATE_SEQUENCE_LENGTH,
    )
    model = DualSurfaceStateModel(config)
    ps, st = Lux.setup(rng, model)
    outputs, _ = Lux.apply(model, batch.tokens, ps, st)
    total_loss, parts = dual_surface_loss(model, ps, st, batch)

    @test size(outputs.state) == (BOARD_STATE_VOCAB_SIZE, BOARD_STATE_SEQUENCE_LENGTH, 2)
    @test size(outputs.transcript) == (length(WavePDEChess.CHESS_TRANSCRIPT_STOI), BOARD_STATE_SEQUENCE_LENGTH, 2)
    @test isfinite(total_loss)
    @test isfinite(parts.state_loss)
    @test isfinite(parts.transcript_loss)

    train_cfg = DualSurfaceTrainingConfig(
        data_dir=state_dir,
        batch_size=2,
        learning_rate=1.0f-3,
        max_iters=1,
        log_interval=1,
        min_tokens=BOARD_STATE_SEQUENCE_LENGTH,
        train_file_update_interval=1,
        checkpoint_path=joinpath(tempdir, "dual_surface_ckpt.jls"),
        seed=51,
    )
    checkpoint = train_dual_surface!(model, corpus, train_cfg)

    @test isfile(train_cfg.checkpoint_path)
    @test length(checkpoint.losses) == 1
    @test length(checkpoint.state_losses) == 1
    @test length(checkpoint.transcript_losses) == 1
end

@testset "Surface Mode Comparison" begin
    tempdir = mktempdir()
    transcript_path = write_test_transcript_parquet(joinpath(tempdir, "toy_transcript.parquet"))
    state_dir = joinpath(tempdir, "state_dataset")
    write_transcript_state_parquet(dirname(transcript_path), state_dir)

    result = compare_surface_training_modes(
        dirname(transcript_path),
        state_dir;
        d_model=8,
        n_layer=1,
        solver_steps=1,
        batch_size=2,
        learning_rate=1.0f-3,
        max_iters=1,
        seed=59,
        output_dir=joinpath(tempdir, "surface_compare"),
    )

    @test isfile(result.transcript_first.checkpoint_path)
    @test isfile(result.state_first.checkpoint_path)
    @test isfile(result.hybrid.checkpoint_path)
    @test isfinite(result.transcript_first.final_loss)
    @test isfinite(result.state_first.final_loss)
    @test isfinite(result.hybrid.final_loss)
    @test isfinite(result.hybrid.final_state_loss)
    @test isfinite(result.hybrid.final_transcript_loss)
end
