abstract type AbstractInputAdapter <: Lux.AbstractLuxLayer end
abstract type AbstractProposerHead end
abstract type AbstractCheckerHead end

function input_adapter_output(adapter::AbstractInputAdapter, tokens, ps, st)
    throw(MethodError(input_adapter_output, (adapter, tokens, ps, st)))
end

function (adapter::AbstractInputAdapter)(tokens, ps, st)
    return input_adapter_output(adapter, tokens, ps, st)
end

function proposer_output(head::AbstractProposerHead, hidden, adapter_ps, ps, st)
    throw(MethodError(proposer_output, (head, hidden, adapter_ps, ps, st)))
end

function checker_output(head::AbstractCheckerHead, hidden, ps, st)
    throw(MethodError(checker_output, (head, hidden, ps, st)))
end
