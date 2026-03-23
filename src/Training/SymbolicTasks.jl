module SymbolicTasks

using DBInterface
using DuckDB
using Random

export SYMBOLIC_SYMBOLS,
    SYMBOLIC_TASK_FAMILIES,
    SYMBOLIC_VOCAB_SIZE,
    decode_symbolic_tokens,
    ensure_symbolic_bridge_dataset,
    encode_symbolic_tokens,
    generate_symbolic_bridge_examples,
    write_symbolic_bridge_parquet

const SYMBOLIC_SYMBOLS = (
    "<pad>",
    "<bos>",
    "<eos>",
    "<sep>",
    "QUERY",
    "PROP",
    "ENTAIL",
    "CONTRA",
    "CHAIN",
    "AND",
    "OR",
    "NOT",
    "IMPLIES",
    "LP",
    "RP",
    "COMMA",
    "YES",
    "NO",
    "A",
    "B",
    "C",
    "D",
)

const SYMBOLIC_VOCAB_SIZE = length(SYMBOLIC_SYMBOLS)
const SYMBOLIC_TASK_FAMILIES = (
    :propositional_logic,
    :entailment,
    :contradiction_detection,
    :simple_rule_chaining,
)
const SYMBOLIC_ATOMS = (:A, :B, :C, :D)
const SYMBOLIC_ATOM_POOL = [(:lit, atom, negated) for atom in SYMBOLIC_ATOMS for negated in (false, true)]

const SYMBOLIC_STOI = Dict{String, Int32}(symbol => Int32(idx - 1) for (idx, symbol) in pairs(SYMBOLIC_SYMBOLS))
const SYMBOLIC_ITOS = Dict{Int32, String}(value => key for (key, value) in SYMBOLIC_STOI)
const SYMBOLIC_TASK_TOKENS = Dict(
    :propositional_logic => "PROP",
    :entailment => "ENTAIL",
    :contradiction_detection => "CONTRA",
    :simple_rule_chaining => "CHAIN",
)

_symbolic_text(token::AbstractString) = String(token)
_symbolic_text(token::Symbol) = String(token)

function symbolic_sql_escape(text::AbstractString)
    return replace(text, "'" => "''")
end

function symbolic_sql_list(tokens::AbstractVector{<:Integer})
    return "[" * join(Int.(tokens), ",") * "]"
end

function symbolic_token_id(token)
    text = _symbolic_text(token)
    haskey(SYMBOLIC_STOI, text) || throw(ArgumentError("Unsupported symbolic token $(repr(text))."))
    return SYMBOLIC_STOI[text]
end

function encode_symbolic_tokens(tokens::AbstractVector)
    return Int32[symbolic_token_id(token) for token in tokens]
end

function decode_symbolic_tokens(tokens::AbstractVector{<:Integer})
    return join((SYMBOLIC_ITOS[Int32(token)] for token in tokens), " ")
end

literal(atom::Symbol, negated::Bool=false) = (:lit, atom, negated)
rule(antecedent, consequent) = (:rule, antecedent, consequent)
is_literal(value) = value isa Tuple && length(value) == 3 && value[1] == :lit

literal_atom(lit) = lit[2]
literal_negated(lit) = lit[3]
rule_antecedent(r) = r[2]
rule_consequent(r) = r[3]

literal_key(lit) = literal_negated(lit) ? "NOT_$(literal_atom(lit))" : String(literal_atom(lit))
negate_literal(lit) = literal(literal_atom(lit), !literal_negated(lit))

function literal_tokens(lit)
    if literal_negated(lit)
        return String["NOT", String(literal_atom(lit))]
    end
    return String[String(literal_atom(lit))]
end

function literal_string(lit)
    return join(literal_tokens(lit), " ")
end

function rule_tokens(r)
    return vcat(literal_tokens(rule_antecedent(r)), String["IMPLIES"], literal_tokens(rule_consequent(r)))
end

function clause_tokens(parts::AbstractVector{<:AbstractVector{String}})
    tokens = String[]
    for (idx, part) in enumerate(parts)
        append!(tokens, part)
        idx < length(parts) && push!(tokens, "COMMA")
    end
    return tokens
end

function assignment_literals(rng::AbstractRNG)
    return [literal(atom, rand(rng, Bool)) for atom in SYMBOLIC_ATOMS]
end

function assignment_map(literals)
    return Dict{Symbol, Bool}(literal_atom(lit) => !literal_negated(lit) for lit in literals)
end

function all_literal_nodes()
    return [literal(atom, negated) for atom in SYMBOLIC_ATOMS for negated in (false, true)]
end

function literal_nodes_excluding(keys::AbstractSet{<:AbstractString})
    nodes = Any[]
    for node in all_literal_nodes()
        literal_key(node) in keys || push!(nodes, node)
    end
    return nodes
end

function choose_literal_not_in(rng::AbstractRNG, keys::AbstractSet{<:AbstractString})
    candidates = literal_nodes_excluding(keys)
    isempty(candidates) && throw(ArgumentError("No available symbolic literal left to choose from."))
    return rand(rng, candidates)
end

function evaluate_formula(formula, valuation::AbstractDict{<:Symbol, <:Bool})
    tag = formula[1]
    if tag == :atom
        return valuation[formula[2]]
    elseif tag == :not
        return !evaluate_formula(formula[2], valuation)
    elseif tag == :and
        return evaluate_formula(formula[2], valuation) && evaluate_formula(formula[3], valuation)
    elseif tag == :or
        return evaluate_formula(formula[2], valuation) || evaluate_formula(formula[3], valuation)
    elseif tag == :implies
        return !evaluate_formula(formula[2], valuation) || evaluate_formula(formula[3], valuation)
    end
    throw(ArgumentError("Unsupported formula tag $(tag)."))
end

function random_formula(rng::AbstractRNG, depth::Int)
    if depth <= 0 || rand(rng) < 0.3
        return (:atom, rand(rng, SYMBOLIC_ATOMS))
    end

    tag = rand(rng, (:not, :and, :or, :implies))
    if tag == :not
        return (:not, random_formula(rng, depth - 1))
    end
    left_depth = max(depth - 1, 0)
    right_depth = max(depth - 1, 0)
    return (tag, random_formula(rng, left_depth), random_formula(rng, right_depth))
end

function formula_tokens(formula)
    tag = formula[1]
    if tag == :atom
        return String[String(formula[2])]
    elseif tag == :not
        return vcat(String["LP", "NOT"], formula_tokens(formula[2]), String["RP"])
    elseif tag == :and || tag == :or || tag == :implies
        op_token = tag == :and ? "AND" : tag == :or ? "OR" : "IMPLIES"
        return vcat(String["LP"], formula_tokens(formula[2]), String[op_token], formula_tokens(formula[3]), String["RP"])
    end
    throw(ArgumentError("Unsupported formula tag $(tag)."))
end

function build_sequence(task_token::AbstractString, context_tokens::AbstractVector{String}, query_tokens::AbstractVector{String}, answer::Bool)
    answer_token = answer ? "YES" : "NO"
    return String[
        "<bos>",
        task_token,
        context_tokens...,
        "<sep>",
        "QUERY",
        query_tokens...,
        "<sep>",
        answer_token,
        "<eos>",
    ]
end

function closure_from_rules(facts, rules)
    seen = Set{String}(literal_key.(facts))
    frontier = collect(seen)
    while !isempty(frontier)
        current = popfirst!(frontier)
        for r in rules
            if literal_key(rule_antecedent(r)) == current
                next_key = literal_key(rule_consequent(r))
                if !(next_key in seen)
                    push!(seen, next_key)
                    push!(frontier, next_key)
                end
            end
        end
    end
    return seen
end

function chain_problem(rng::AbstractRNG, chain_length::Int; add_contradiction::Bool=false)
    chain_nodes = shuffle!(rng, copy(all_literal_nodes()))[1:(chain_length + 1)]
    facts = [chain_nodes[1]]
    rules = [rule(chain_nodes[idx], chain_nodes[idx + 1]) for idx in 1:chain_length]
    if add_contradiction
        push!(rules, rule(chain_nodes[1], negate_literal(chain_nodes[end])))
    end
    return facts, rules, chain_nodes
end

function propositional_example(rng::AbstractRNG)
    valuation_literals = assignment_literals(rng)
    valuation = assignment_map(valuation_literals)
    formula = random_formula(rng, rand(rng, 2:3))
    answer = evaluate_formula(formula, valuation)
    context_tokens = clause_tokens([literal_tokens(lit) for lit in valuation_literals])
    query_tokens = formula_tokens(formula)
    sequence = encode_symbolic_tokens(build_sequence("PROP", context_tokens, query_tokens, answer))
    return (
        family="propositional_logic",
        prompt=join(build_sequence("PROP", context_tokens, query_tokens, answer)[1:end-2], " "),
        answer=answer ? "YES" : "NO",
        label=Int32(answer),
        tokenized=sequence,
    )
end

function entailment_example(rng::AbstractRNG, index::Int)
    chain_length = isodd(index) ? 1 : 2
    facts, rules, chain_nodes = chain_problem(rng, chain_length)
    closure = closure_from_rules(facts, rules)
    positive = isodd(index)
    query = positive ? chain_nodes[end] : choose_literal_not_in(rng, closure)
    answer = query |> literal_key in closure
    context_tokens = clause_tokens(vcat([literal_tokens(fact) for fact in facts], [rule_tokens(r) for r in rules]))
    query_tokens = literal_tokens(query)
    sequence = encode_symbolic_tokens(build_sequence("ENTAIL", context_tokens, query_tokens, answer))
    return (
        family="entailment",
        prompt=join(build_sequence("ENTAIL", context_tokens, query_tokens, answer)[1:end-2], " "),
        answer=answer ? "YES" : "NO",
        label=Int32(answer),
        tokenized=sequence,
    )
end

function contradiction_example(rng::AbstractRNG, index::Int)
    positive = isodd(index)
    facts, rules, chain_nodes = chain_problem(rng, positive ? 2 : 1; add_contradiction=positive)
    closure = closure_from_rules(facts, rules)
    if positive
        query = chain_nodes[end]
        answer = literal_key(negate_literal(query)) in closure
    else
        negative_candidates = [
            node for node in all_literal_nodes()
            if literal_key(node) in closure && !(literal_key(negate_literal(node)) in closure)
        ]
        isempty(negative_candidates) && throw(ArgumentError("Unable to build a negative contradiction example."))
        query = rand(rng, negative_candidates)
        answer = false
    end
    context_tokens = clause_tokens(vcat([literal_tokens(fact) for fact in facts], [rule_tokens(r) for r in rules]))
    query_tokens = literal_tokens(query)
    sequence = encode_symbolic_tokens(build_sequence("CONTRA", context_tokens, query_tokens, answer))
    return (
        family="contradiction_detection",
        prompt=join(build_sequence("CONTRA", context_tokens, query_tokens, answer)[1:end-2], " "),
        answer=answer ? "YES" : "NO",
        label=Int32(answer),
        tokenized=sequence,
    )
end

function rule_chaining_example(rng::AbstractRNG, index::Int)
    chain_length = 3
    facts, rules, chain_nodes = chain_problem(rng, chain_length)
    closure = closure_from_rules(facts, rules)
    positive = isodd(index)
    query = positive ? chain_nodes[end] : choose_literal_not_in(rng, closure)
    answer = literal_key(query) in closure
    context_tokens = clause_tokens(vcat([literal_tokens(fact) for fact in facts], [rule_tokens(r) for r in rules]))
    query_tokens = literal_tokens(query)
    sequence = encode_symbolic_tokens(build_sequence("CHAIN", context_tokens, query_tokens, answer))
    return (
        family="simple_rule_chaining",
        prompt=join(build_sequence("CHAIN", context_tokens, query_tokens, answer)[1:end-2], " "),
        answer=answer ? "YES" : "NO",
        label=Int32(answer),
        tokenized=sequence,
    )
end

function generate_symbolic_bridge_examples(; count_per_task::Int=256, seed::Int=1337)
    count_per_task > 0 || throw(ArgumentError("count_per_task must be positive."))
    rng = MersenneTwister(seed)
    examples = NamedTuple[]
    sizehint!(examples, count_per_task * length(SYMBOLIC_TASK_FAMILIES))

    for idx in 1:count_per_task
        push!(examples, propositional_example(rng))
        push!(examples, entailment_example(rng, idx))
        push!(examples, contradiction_example(rng, idx))
        push!(examples, rule_chaining_example(rng, idx))
    end

    return examples
end

function write_symbolic_bridge_parquet(output_dir::AbstractString; count_per_task::Int=256, seed::Int=1337, file_name::AbstractString="symbolic_bridge.parquet")
    mkpath(output_dir)
    examples = generate_symbolic_bridge_examples(; count_per_task=count_per_task, seed=seed)
    isempty(examples) && throw(ArgumentError("No symbolic examples were generated."))

    parquet_path = joinpath(output_dir, file_name)
    db = DuckDB.DB()
    conn = DBInterface.connect(db)
    DBInterface.execute(
        conn,
        """
        CREATE TABLE symbolic_bridge (
            tokenized INTEGER[],
            family VARCHAR,
            prompt VARCHAR,
            answer VARCHAR,
            label INTEGER
        )
        """,
    )

    values = String[]
    sizehint!(values, length(examples))
    for example in examples
        push!(values, "($(symbolic_sql_list(example.tokenized)), '$(symbolic_sql_escape(example.family))', '$(symbolic_sql_escape(example.prompt))', '$(symbolic_sql_escape(example.answer))', $(Int(example.label)))")
    end

    chunk_size = 128
    for chunk_start in 1:chunk_size:length(values)
        chunk_end = min(chunk_start + chunk_size - 1, length(values))
        chunk_values = join(@view(values[chunk_start:chunk_end]), ",")
        DBInterface.execute(conn, "INSERT INTO symbolic_bridge VALUES $(chunk_values)")
    end

    isfile(parquet_path) && rm(parquet_path; force=true)
    DBInterface.execute(conn, "COPY symbolic_bridge TO '$(symbolic_sql_escape(parquet_path))' (FORMAT PARQUET)")
    return parquet_path
end

function ensure_symbolic_bridge_dataset(output_dir::AbstractString; count_per_task::Int=256, seed::Int=1337, file_name::AbstractString="symbolic_bridge.parquet")
    mkpath(output_dir)
    parquet_path = joinpath(output_dir, file_name)
    isfile(parquet_path) || write_symbolic_bridge_parquet(output_dir; count_per_task=count_per_task, seed=seed, file_name=file_name)
    return parquet_path
end

end
