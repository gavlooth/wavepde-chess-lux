abstract type AbstractProposerHead end
abstract type AbstractCheckerHead end

function proposer_output(head::AbstractProposerHead, hidden, adapter_ps, ps, st)
    throw(MethodError(proposer_output, (head, hidden, adapter_ps, ps, st)))
end

function checker_output(head::AbstractCheckerHead, hidden, ps, st)
    throw(MethodError(checker_output, (head, hidden, ps, st)))
end
