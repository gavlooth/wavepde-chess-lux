Base.@kwdef struct PolicyTargetBundle
    candidates::Vector{String}
    labels::Vector{Float32}
    target_index::Union{Nothing, Int}
    target_move::String
end

const POLICY_ACTION_SEPARATOR_TOKEN = Int32(BOARD_STATE_VOCAB_SIZE)
const POLICY_ACTION_TOKEN_BASE = Int32(BOARD_STATE_VOCAB_SIZE + 1)
const POLICY_ACTION_TOKEN_VOCAB_SIZE = Int32(length(CHESS_TRANSCRIPT_STOI))
const STATE_ACTION_VOCAB_SIZE = Int(POLICY_ACTION_TOKEN_BASE + POLICY_ACTION_TOKEN_VOCAB_SIZE)
const MAX_POLICY_ACTION_TOKENS = 12

function normalize_policy_candidates(candidates::AbstractVector{<:AbstractString})
    normalized = String[]
    seen = Set{String}()
    for candidate in candidates
        candidate_str = String(candidate)
        isempty(candidate_str) && continue
        candidate_str in seen && continue
        push!(normalized, candidate_str)
        push!(seen, candidate_str)
    end
    return normalized
end

function policy_legal_candidates(context::AbstractString; limit::Integer=0)
    candidates = legal_san_candidates_from_transcript(context; limit=limit)
    return String[candidate for candidate in candidates]
end

function policy_legal_candidates(context::AbstractVector{<:AbstractString}; limit::Integer=0)
    candidates = normalize_policy_candidates(context)
    if limit > 0
        return candidates[1:min(limit, length(candidates))]
    end
    return candidates
end

function policy_labels(candidates::AbstractVector{<:AbstractString}, target_move::AbstractString; strict::Bool=true)
    normalized_candidates = String[candidate for candidate in candidates]
    labels = zeros(Float32, length(normalized_candidates))
    target_index = findfirst(==(String(target_move)), normalized_candidates)

    if target_index === nothing
        strict && throw(ArgumentError(
            "Target move $(repr(target_move)) is not present in the provided legal candidate set.",
        ))
        return labels, nothing
    end

    labels[target_index] = 1.0f0
    return labels, target_index
end

function policy_target_bundle(context::AbstractString, target_move::AbstractString; limit::Integer=0, strict::Bool=true)
    candidates = policy_legal_candidates(context; limit=limit)
    labels, target_index = policy_labels(candidates, target_move; strict=strict)
    return PolicyTargetBundle(
        candidates=candidates,
        labels=labels,
        target_index=target_index,
        target_move=String(target_move),
    )
end

function policy_target_bundle(context::AbstractVector{<:AbstractString}, target_move::AbstractString; limit::Integer=0, strict::Bool=true)
    candidates = policy_legal_candidates(context; limit=limit)
    labels, target_index = policy_labels(candidates, target_move; strict=strict)
    return PolicyTargetBundle(
        candidates=candidates,
        labels=labels,
        target_index=target_index,
        target_move=String(target_move),
    )
end

function policy_target_index(context, target_move::AbstractString; limit::Integer=0, strict::Bool=true)
    bundle = policy_target_bundle(context, target_move; limit=limit, strict=strict)
    return bundle.target_index
end

function encode_policy_action_san(candidate::AbstractString)
    encoded = encode_chess_candidate_san(candidate)
    return Int32[POLICY_ACTION_TOKEN_BASE + token for token in encoded]
end

function append_policy_action_tokens(state_tokens::AbstractVector{<:Integer}, candidate::AbstractString)
    action_tokens = encode_policy_action_san(candidate)
    return vcat(Int32.(state_tokens), POLICY_ACTION_SEPARATOR_TOKEN, action_tokens)
end
