Base.@kwdef struct ChessMultiHeadModelConfig
    adapter::ChessAdapterConfig = ChessAdapterConfig()
    core::SequenceCoreConfig = WavePDECoreConfig()
    proposer::ChessMoveHeadConfig = ChessMoveHeadConfig()
    checker::ChessCheckerHeadConfig = ChessCheckerHeadConfig()
    max_seq_len::Int = 1536
end

struct ChessMultiHeadModel{A, C, P, K} <: Lux.AbstractLuxLayer
    config::ChessMultiHeadModelConfig
    adapter::A
    core::C
    proposer::P
    checker::K
end

function validate_chess_multihead_config(config::ChessMultiHeadModelConfig)
    config.max_seq_len > 0 || throw(ArgumentError("max_seq_len must be positive"))
    config.adapter.d_model == core_d_model(config.core) || throw(ArgumentError("adapter d_model must match core d_model"))
    config.proposer.d_model == core_d_model(config.core) || throw(ArgumentError("proposer d_model must match core d_model"))
    config.checker.d_model == core_d_model(config.core) || throw(ArgumentError("checker d_model must match core d_model"))
    config.proposer.vocab_size == config.adapter.vocab_size || throw(ArgumentError("proposer vocab_size must match adapter vocab_size"))
    return nothing
end

function ChessMultiHeadModel(config::ChessMultiHeadModelConfig)
    validate_chess_multihead_config(config)
    return ChessMultiHeadModel(
        config,
        ChessInputAdapter(config.adapter),
        build_sequence_core(config.core),
        ChessMoveHead(config.proposer),
        ChessCheckerHead(config.checker),
    )
end

Lux.initialparameters(rng::AbstractRNG, model::ChessMultiHeadModel) = (
    adapter=Lux.initialparameters(rng, model.adapter),
    core=Lux.initialparameters(rng, model.core),
    proposer=Lux.initialparameters(rng, model.proposer),
    checker=Lux.initialparameters(rng, model.checker),
)

Lux.initialstates(rng::AbstractRNG, model::ChessMultiHeadModel) = (
    adapter=Lux.initialstates(rng, model.adapter),
    core=Lux.initialstates(rng, model.core),
    proposer=Lux.initialstates(rng, model.proposer),
    checker=Lux.initialstates(rng, model.checker),
)

function (model::ChessMultiHeadModel)(tokens::AbstractArray{<:Integer}, ps, st)
    size(tokens, 1) > model.config.max_seq_len && throw(ArgumentError(
        "Sequence length $(size(tokens, 1)) exceeds configured max_seq_len $(model.config.max_seq_len).",
    ))
    hidden, st_adapter = model.adapter(tokens, ps.adapter, st.adapter)
    hidden, st_core = model.core(hidden, ps.core, st.core)
    logits, st_proposer = proposer_output(model.proposer, hidden, ps.adapter, ps.proposer, st.proposer)
    checker_scores, st_checker = checker_output(model.checker, hidden, ps.checker, st.checker)
    return (
        proposer=logits,
        checker=checker_scores,
    ), (
        adapter=st_adapter,
        core=st_core,
        proposer=st_proposer,
        checker=st_checker,
    )
end

function checker_scalarize(checker_scores::AbstractVector{<:Real})
    return mean(checker_scores)
end

function checker_scalarize(checker_scores::AbstractMatrix{<:Real})
    return vec(mean(checker_scores; dims=1))
end

function proposer_topk(logits::AbstractMatrix{T}; top_k::Int=5) where {T<:Real}
    top_k > 0 || throw(ArgumentError("top_k must be positive"))
    vocab_size, batch_size = size(logits)
    vocab_size > 0 || throw(ArgumentError("logits must have a positive vocabulary dimension"))

    selected_k = min(top_k, vocab_size)
    indices = Matrix{Int}(undef, selected_k, batch_size)
    scores = Matrix{T}(undef, selected_k, batch_size)

    for batch_index in 1:batch_size
        column = view(logits, :, batch_index)
        candidate_order = partialsortperm(column, 1:selected_k; rev=true)
        indices[:, batch_index] .= candidate_order .- 1
        scores[:, batch_index] .= column[candidate_order]
    end

    return (
        indices=indices,
        scores=scores,
    )
end

function proposer_topk(logits::AbstractArray{T, 3}; top_k::Int=5, timestep::Int=size(logits, 2)) where {T<:Real}
    ndims(logits) == 3 || throw(ArgumentError("proposer_topk expects a 3D logits tensor or a 2D vocabulary/batch matrix"))
    1 <= timestep <= size(logits, 2) || throw(ArgumentError("timestep $(timestep) is out of bounds for sequence length $(size(logits, 2))"))
    return proposer_topk(view(logits, :, timestep, :); top_k=top_k)
end

function append_candidate_token(tokens::AbstractMatrix{<:Integer}, batch_index::Int, candidate_token::Integer)
    seq_len = size(tokens, 1)
    extended = Matrix{eltype(tokens)}(undef, seq_len + 1, 1)
    extended[1:seq_len, 1] .= view(tokens, :, batch_index)
    extended[end, 1] = candidate_token
    return extended
end

function rerank_next_token_candidates(
    model::ChessMultiHeadModel,
    tokens::AbstractMatrix{<:Integer},
    ps,
    st;
    top_k::Int=5,
    timestep::Int=size(tokens, 1),
    checker_weight::Real=1,
)
    outputs, _ = Lux.apply(model, tokens, ps, st)
    proposer = proposer_topk(outputs.proposer; top_k=top_k, timestep=timestep)
    selected_k, batch_size = size(proposer.indices)

    score_type = promote_type(eltype(proposer.scores), typeof(checker_weight))
    candidate_checker_scores = Matrix{score_type}(undef, selected_k, batch_size)
    combined_scores = Matrix{score_type}(undef, selected_k, batch_size)
    reranked_indices = Matrix{Int}(undef, selected_k, batch_size)
    reranked_scores = Matrix{score_type}(undef, selected_k, batch_size)

    for batch_index in 1:batch_size
        for candidate_rank in 1:selected_k
            candidate_token = proposer.indices[candidate_rank, batch_index]
            extended_tokens = append_candidate_token(tokens, batch_index, candidate_token)
            candidate_outputs, _ = Lux.apply(model, extended_tokens, ps, st)
            candidate_checker_score = only(checker_scalarize(candidate_outputs.checker))
            candidate_checker_scores[candidate_rank, batch_index] = candidate_checker_score
            combined_scores[candidate_rank, batch_index] = proposer.scores[candidate_rank, batch_index] + checker_weight * candidate_checker_score
        end

        rerank_order = sortperm(view(combined_scores, :, batch_index); rev=true)
        reranked_indices[:, batch_index] .= proposer.indices[rerank_order, batch_index]
        reranked_scores[:, batch_index] .= combined_scores[rerank_order, batch_index]
    end

    return (
        proposer=proposer,
        candidate_checker_scores=candidate_checker_scores,
        candidate_combined_scores=combined_scores,
        reranked=(
            indices=reranked_indices,
            scores=reranked_scores,
        ),
    )
end
