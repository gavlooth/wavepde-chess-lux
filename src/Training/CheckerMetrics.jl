module CheckerMetrics

export checker_prediction_metrics,
    rerank_comparison_metrics,
    board_fact_metrics,
    candidate_legality_metrics,
    board_probe_metrics,
    state_slot_family_metrics

function checker_prediction_metrics(predictions::AbstractArray{<:Real}, targets::AbstractArray{<:Real})
    size(predictions) == size(targets) || throw(ArgumentError(
        "checker_prediction_metrics expects predictions and targets with the same shape, got $(size(predictions)) and $(size(targets)).",
    ))
    n = length(predictions)
    n > 0 || throw(ArgumentError("checker_prediction_metrics requires non-empty inputs."))

    total_squared_error = 0.0
    total_absolute_error = 0.0
    max_absolute_error = 0.0

    for idx in eachindex(predictions, targets)
        diff = Float64(predictions[idx]) - Float64(targets[idx])
        abs_diff = abs(diff)
        total_squared_error += diff * diff
        total_absolute_error += abs_diff
        max_absolute_error = max(max_absolute_error, abs_diff)
    end

    mse = total_squared_error / n
    return (
        mse=mse,
        mae=total_absolute_error / n,
        rmse=sqrt(mse),
        max_abs_error=max_absolute_error,
    )
end

function rerank_comparison_metrics(
    proposer_indices::AbstractMatrix{<:Integer},
    reranked_indices::AbstractMatrix{<:Integer},
    labels::AbstractArray{<:Integer};
    label_offset::Integer=0,
)
    size(proposer_indices, 2) == size(reranked_indices, 2) || throw(ArgumentError(
        "rerank_comparison_metrics expects proposer and reranked candidate matrices with the same batch dimension.",
    ))

    labels_vec = vec(labels)
    batch_size = size(proposer_indices, 2)
    length(labels_vec) == batch_size || throw(ArgumentError(
        "rerank_comparison_metrics expects one label per example, got $(length(labels_vec)) labels for batch size $(batch_size).",
    ))
    batch_size > 0 || throw(ArgumentError("rerank_comparison_metrics requires a non-empty batch."))

    proposer_top1 = view(proposer_indices, 1, :)
    reranked_top1 = view(reranked_indices, 1, :)

    proposer_hits = 0
    reranked_hits = 0
    rerank_wins = 0
    rerank_losses = 0
    agreements = 0

    for idx in 1:batch_size
        label = Int(labels_vec[idx]) - label_offset
        proposer_hit = proposer_top1[idx] == label
        reranked_hit = reranked_top1[idx] == label

        proposer_hits += proposer_hit
        reranked_hits += reranked_hit
        rerank_wins += reranked_hit && !proposer_hit
        rerank_losses += proposer_hit && !reranked_hit
        agreements += proposer_hit == reranked_hit
    end

    total = Float64(batch_size)
    return (
        proposer_top1_accuracy=proposer_hits / total,
        reranked_top1_accuracy=reranked_hits / total,
        accuracy_delta=(reranked_hits - proposer_hits) / total,
        rerank_win_rate=rerank_wins / total,
        rerank_loss_rate=rerank_losses / total,
        agreement_rate=agreements / total,
    )
end

function rerank_comparison_metrics(
    proposer_topk::NamedTuple,
    reranked::NamedTuple,
    labels::AbstractArray{<:Integer};
    label_offset::Integer=0,
)
    haskey(proposer_topk, :indices) || throw(ArgumentError("proposer_topk named tuple must contain an :indices field."))
    haskey(reranked, :indices) || throw(ArgumentError("reranked named tuple must contain an :indices field."))
    return rerank_comparison_metrics(proposer_topk.indices, reranked.indices, labels; label_offset=label_offset)
end

function metrics_matrix(values::AbstractArray{<:Real})
    if ndims(values) == 1
        return reshape(values, :, 1)
    elseif ndims(values) == 2
        return values
    end
    throw(ArgumentError(
        "Classification-style checker metrics expect a vector or matrix, got array with size $(size(values)).",
    ))
end

function board_fact_metrics(
    predictions::AbstractArray{<:Real},
    targets::AbstractArray{<:Real};
    threshold::Real=0.5,
)
    pred_matrix = metrics_matrix(predictions)
    target_matrix = metrics_matrix(targets)
    size(pred_matrix) == size(target_matrix) || throw(ArgumentError(
        "board_fact_metrics expects predictions and targets with the same shape, got $(size(pred_matrix)) and $(size(target_matrix)).",
    ))

    num_targets, batch_size = size(pred_matrix)
    num_targets > 0 || throw(ArgumentError("board_fact_metrics requires at least one target dimension."))
    batch_size > 0 || throw(ArgumentError("board_fact_metrics requires a non-empty batch."))

    per_target_correct = zeros(Float64, num_targets)
    per_target_predicted_positive = zeros(Float64, num_targets)
    per_target_target_positive = zeros(Float64, num_targets)
    total_correct = 0.0
    exact_matches = 0.0
    total_brier = 0.0

    for batch_idx in 1:batch_size
        row_correct = true
        for target_idx in 1:num_targets
            prediction = Float64(pred_matrix[target_idx, batch_idx])
            target = Float64(target_matrix[target_idx, batch_idx])
            predicted_positive = prediction >= threshold
            target_positive = target >= 0.5
            correct = predicted_positive == target_positive

            total_correct += correct
            per_target_correct[target_idx] += correct
            per_target_predicted_positive[target_idx] += predicted_positive
            per_target_target_positive[target_idx] += target_positive
            total_brier += (prediction - target) ^ 2
            row_correct &= correct
        end
        exact_matches += row_correct
    end

    total = Float64(num_targets * batch_size)
    return (
        overall_accuracy=total_correct / total,
        exact_match_rate=exact_matches / Float64(batch_size),
        brier_score=total_brier / total,
        predicted_positive_rate=sum(per_target_predicted_positive) / total,
        target_positive_rate=sum(per_target_target_positive) / total,
        per_target_accuracy=per_target_correct / Float64(batch_size),
        per_target_predicted_positive_rate=per_target_predicted_positive / Float64(batch_size),
        per_target_target_positive_rate=per_target_target_positive / Float64(batch_size),
    )
end

function candidate_legality_metrics(
    predictions::AbstractArray{<:Real},
    targets::AbstractArray{<:Real};
    threshold::Real=0.5,
)
    metrics = board_fact_metrics(predictions, targets; threshold=threshold)
    return (
        accuracy=metrics.overall_accuracy,
        brier_score=metrics.brier_score,
        predicted_legal_rate=metrics.predicted_positive_rate,
        target_legal_rate=metrics.target_positive_rate,
    )
end

function board_probe_metrics(predictions::NamedTuple, targets::NamedTuple; threshold::Real=0.5)
    required = (
        :attacked_white,
        :attacked_black,
        :in_check,
        :pinned_count,
        :king_pressure,
        :mobility,
        :attacked_piece_count,
    )
    for field in required
        haskey(predictions, field) || throw(ArgumentError("board_probe_metrics requires a $(field) prediction field."))
        haskey(targets, field) || throw(ArgumentError("board_probe_metrics requires a $(field) target field."))
    end

    return (
        attacked_white=board_fact_metrics(predictions.attacked_white, targets.attacked_white; threshold=threshold),
        attacked_black=board_fact_metrics(predictions.attacked_black, targets.attacked_black; threshold=threshold),
        in_check=board_fact_metrics(predictions.in_check, targets.in_check; threshold=threshold),
        pinned_count=checker_prediction_metrics(predictions.pinned_count, targets.pinned_count),
        king_pressure=checker_prediction_metrics(predictions.king_pressure, targets.king_pressure),
        mobility=checker_prediction_metrics(predictions.mobility, targets.mobility),
        attacked_piece_count=checker_prediction_metrics(predictions.attacked_piece_count, targets.attacked_piece_count),
    )
end

function slot_family_metrics(
    predictions::AbstractMatrix{<:Integer},
    targets::AbstractMatrix{<:Integer},
    slot_range::AbstractUnitRange{<:Integer},
)
    size(predictions) == size(targets) || throw(ArgumentError(
        "slot_family_metrics expects predictions and targets with the same shape, got $(size(predictions)) and $(size(targets)).",
    ))
    first(slot_range) >= 1 || throw(ArgumentError("slot ranges must be 1-based."))
    last(slot_range) <= size(predictions, 1) || throw(ArgumentError(
        "slot range $(slot_range) exceeds the available token sequence length $(size(predictions, 1)).",
    ))

    num_examples = size(predictions, 2)
    num_slots = length(slot_range)
    num_examples > 0 || throw(ArgumentError("slot_family_metrics requires a non-empty batch."))
    num_slots > 0 || throw(ArgumentError("slot_family_metrics requires a non-empty slot range."))

    total_correct = 0
    exact_matches = 0
    for batch_idx in 1:num_examples
        row_exact = true
        for slot_idx in slot_range
            correct = predictions[slot_idx, batch_idx] == targets[slot_idx, batch_idx]
            total_correct += correct
            row_exact &= correct
        end
        exact_matches += row_exact
    end

    total = Float64(num_slots * num_examples)
    return (
        token_accuracy=total_correct / total,
        exact_match_rate=exact_matches / Float64(num_examples),
        num_tokens=Int(total),
        num_examples=num_examples,
    )
end

function state_slot_family_metrics(predictions::AbstractMatrix{<:Integer}, targets::AbstractMatrix{<:Integer})
    size(predictions) == size(targets) || throw(ArgumentError(
        "state_slot_family_metrics expects predictions and targets with the same shape, got $(size(predictions)) and $(size(targets)).",
    ))
    size(predictions, 1) >= 72 || throw(ArgumentError(
        "state_slot_family_metrics expects at least 72 state slots, got $(size(predictions, 1)).",
    ))
    nan_metrics = (
        token_accuracy=NaN,
        exact_match_rate=NaN,
        num_tokens=0,
        num_examples=size(predictions, 2),
    )
    if size(predictions, 1) < 210
        return (
            coarse_state=slot_family_metrics(predictions, targets, 1:72),
            attack_maps=nan_metrics,
            pressure_counts=nan_metrics,
        )
    end
    return (
        coarse_state=slot_family_metrics(predictions, targets, 1:72),
        attack_maps=slot_family_metrics(predictions, targets, 73:200),
        pressure_counts=slot_family_metrics(predictions, targets, 201:210),
    )
end

end
