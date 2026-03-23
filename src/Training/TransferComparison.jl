module TransferComparison

using DBInterface
using DuckDB
using Lux
using Optimisers
using Random
using Serialization
using Zygote

using Main.WavePDEChess

export SYMBOLIC_TASK_NAMES,
    SYMBOLIC_TOKEN_STOI,
    SYMBOLIC_TOKEN_ITOS,
    compare_symbolic_transfer,
    load_transfer_checkpoint,
    symbolic_model_config,
    symbolic_vocabulary_size,
    transplant_core_parameters,
    write_symbolic_dataset

const SYMBOLIC_TASK_NAMES = (
    :propositional_logic,
    :entailment,
    :contradiction_detection,
    :simple_rule_chaining,
)

const SYMBOLIC_VOCABULARY = (
    "<pad>",
    "<bos>",
    "<eos>",
    "PROP",
    "ENT",
    "CON",
    "RULE",
    "IF",
    "THEN",
    "SO",
    "IMPLIES",
    "AND",
    "OR",
    "NOT",
    "BECAUSE",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "X",
    "Y",
    "Z",
    "ALL",
    "SOME",
)

const SYMBOLIC_TOKEN_STOI = Dict(token => Int32(index - 1) for (index, token) in enumerate(SYMBOLIC_VOCABULARY))
const SYMBOLIC_TOKEN_ITOS = Dict(value => key for (key, value) in SYMBOLIC_TOKEN_STOI)

symbolic_vocabulary_size() = length(SYMBOLIC_VOCABULARY)

function symbolic_model_config(;
    vocab_size::Int=symbolic_vocabulary_size(),
    d_model::Int=288,
    n_layer::Int=20,
    solver_steps::Int=4,
    dt_init::Float32=0.05f0,
    norm_eps::Float32=1f-5,
    max_seq_len::Int=128,
)
    return WavePDEChess.ChessModelConfig(
        adapter=WavePDEChess.ChessAdapterConfig(vocab_size=vocab_size, d_model=d_model, pad_token=0),
        core=WavePDEChess.WavePDECoreConfig(
            d_model=d_model,
            n_layer=n_layer,
            solver_steps=solver_steps,
            dt_init=dt_init,
            norm_eps=norm_eps,
        ),
        proposer=WavePDEChess.ChessMoveHeadConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            tie_embeddings=true,
            bias=false,
        ),
        max_seq_len=max_seq_len,
    )
end

function symbolic_tokenize(tokens::AbstractVector{<:AbstractString})
    encoded = Int32[]
    for token in tokens
        haskey(SYMBOLIC_TOKEN_STOI, token) || throw(ArgumentError("Unsupported symbolic token $(repr(token))."))
        push!(encoded, SYMBOLIC_TOKEN_STOI[token])
    end
    return encoded
end

function symbolic_detokenize(tokens::AbstractVector{<:Integer})
    decoded = String[]
    for token in tokens
        token_id = Int32(token)
        haskey(SYMBOLIC_TOKEN_ITOS, token_id) || throw(ArgumentError("Unsupported symbolic token id $(token_id)."))
        push!(decoded, SYMBOLIC_TOKEN_ITOS[token_id])
    end
    return join(decoded, " ")
end

function symbolic_symbols(example_index::Int)
    alphabet = ("A", "B", "C", "D", "E", "F", "G", "H", "X", "Y", "Z")
    length(alphabet) >= 3 || throw(ArgumentError("Symbolic alphabet must contain at least three symbols."))
    a = alphabet[(example_index - 1) % length(alphabet) + 1]
    b = alphabet[example_index % length(alphabet) + 1]
    c = alphabet[(example_index + 1) % length(alphabet) + 1]
    return a, b, c
end

function symbolic_sequence(task::Symbol, example_index::Int)
    a, b, c = symbolic_symbols(example_index)
    if task == :propositional_logic
        tokens = ["<bos>", "PROP", a, "AND", b, "IMPLIES", c, "<eos>"]
    elseif task == :entailment
        tokens = ["<bos>", "ENT", "IF", a, "THEN", b, "SO", c, "<eos>"]
    elseif task == :contradiction_detection
        tokens = ["<bos>", "CON", a, "NOT", a, "BECAUSE", b, "<eos>"]
    elseif task == :simple_rule_chaining
        tokens = ["<bos>", "RULE", a, "IMPLIES", b, b, "IMPLIES", c, a, "SO", c, "<eos>"]
    else
        throw(ArgumentError("Unsupported symbolic task $(task)."))
    end

    return (
        task=String(task),
        example_index=example_index,
        sequence_text=join(tokens, " "),
        tokenized=symbolic_tokenize(tokens),
    )
end

function symbolic_rows(; count_per_task::Int=32, seed::Int=1337)
    count_per_task > 0 || throw(ArgumentError("count_per_task must be positive."))
    rng = MersenneTwister(seed)
    rows = NamedTuple[]

    for task in SYMBOLIC_TASK_NAMES
        for example_index in 1:count_per_task
            push!(rows, symbolic_sequence(task, example_index))
        end
    end

    shuffle!(rng, rows)
    return rows
end

function sql_escape(value::AbstractString)
    return replace(value, "'" => "''")
end

function write_symbolic_dataset(output_dir::AbstractString; count_per_task::Int=32, seed::Int=1337, force::Bool=true)
    mkpath(output_dir)
    dataset_path = joinpath(output_dir, "symbolic_bridge.parquet")
    if isfile(dataset_path) && !force
        return dataset_path
    end

    rows = symbolic_rows(; count_per_task=count_per_task, seed=seed)
    conn = DBInterface.connect(DuckDB.DB, ":memory:")
    DBInterface.execute(conn, """
        CREATE TABLE symbolic_bridge (
            tokenized INTEGER[],
            task VARCHAR,
            example_index INTEGER,
            sequence_text VARCHAR
        )
    """)

    for row in rows
        token_literal = "[" * join(Int.(row.tokenized), ", ") * "]"
        task_literal = sql_escape(row.task)
        text_literal = sql_escape(row.sequence_text)
        DBInterface.execute(
            conn,
            "INSERT INTO symbolic_bridge VALUES ($(token_literal), '$(task_literal)', $(row.example_index), '$(text_literal)')",
        )
    end

    escaped_dataset_path = sql_escape(dataset_path)
    DBInterface.execute(conn, "COPY symbolic_bridge TO '$(escaped_dataset_path)' (FORMAT PARQUET)")
    return dataset_path
end

function load_transfer_checkpoint(path::AbstractString)
    isfile(path) || throw(ArgumentError("Checkpoint not found: $(path)"))
    open(path, "r") do io
        return deserialize(io)
    end
end

function extract_checkpoint_parameters(payload)
    if payload isa NamedTuple && haskey(payload, :parameters)
        return payload.parameters
    elseif payload isa AbstractDict && haskey(payload, :parameters)
        return payload[:parameters]
    elseif payload isa AbstractDict && haskey(payload, "parameters")
        return payload["parameters"]
    end
    return payload
end

function tree_shape_matches(a, b)
    if a isa AbstractArray && b isa AbstractArray
        return size(a) == size(b)
    elseif a isa NamedTuple && b isa NamedTuple
        keys(a) == keys(b) || return false
        for key in keys(a)
            tree_shape_matches(getfield(a, key), getfield(b, key)) || return false
        end
        return true
    elseif a isa Tuple && b isa Tuple
        length(a) == length(b) || return false
        for (left, right) in zip(a, b)
            tree_shape_matches(left, right) || return false
        end
        return true
    elseif a === nothing || b === nothing
        return a === b
    elseif a isa Number && b isa Number
        return true
    else
        return typeof(a) == typeof(b)
    end
end

function replace_namedtuple_field(nt::NamedTuple, field::Symbol, value)
    haskey(nt, field) || throw(ArgumentError("NamedTuple does not contain field $(field)."))
    return (; (key => (key == field ? value : getfield(nt, key)) for key in keys(nt))...)
end

function transplant_core_parameters(target_parameters, source_parameters)
    target_parameters isa NamedTuple || throw(ArgumentError("Target parameters must be a NamedTuple."))
    source_parameters = extract_checkpoint_parameters(source_parameters)
    source_parameters isa NamedTuple || throw(ArgumentError("Source parameters must be a NamedTuple or a checkpoint payload containing :parameters."))
    haskey(target_parameters, :core) || throw(ArgumentError("Target parameter tree does not contain a :core field."))
    haskey(source_parameters, :core) || throw(ArgumentError("Source parameter tree does not contain a :core field."))
    tree_shape_matches(target_parameters.core, source_parameters.core) || throw(ArgumentError(
        "Core parameter trees are incompatible. Check d_model, n_layer, and solver configuration before transplanting.",
    ))
    return replace_namedtuple_field(target_parameters, :core, source_parameters.core)
end

function zero_like_tree(x::AbstractArray)
    y = similar(x)
    fill!(y, zero(eltype(x)))
    return y
end

zero_like_tree(x::Number) = zero(x)
zero_like_tree(x::Nothing) = nothing
zero_like_tree(x::Tuple) = tuple((zero_like_tree(v) for v in x)...)

function zero_like_tree(x::NamedTuple)
    return (; (key => zero_like_tree(getfield(x, key)) for key in keys(x))...)
end

zero_like_tree(x) = zero(x)

function mask_core_gradients(grads)
    grads isa NamedTuple || return grads
    haskey(grads, :core) || return grads
    return replace_namedtuple_field(grads, :core, zero_like_tree(grads.core))
end

function save_transfer_checkpoint(path::AbstractString, payload)
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, payload)
    end
    return path
end

function train_symbolic_setting(
    scenario::Symbol,
    model_config::WavePDEChess.ChessModelConfig,
    corpus::WavePDEChess.ChessParquetCorpus;
    seed::Int,
    batch_size::Int,
    learning_rate::Float32,
    max_iters::Int,
    source_checkpoint_path::Union{Nothing, AbstractString}=nothing,
    freeze_core::Bool=false,
    checkpoint_path::AbstractString,
)
    rng = MersenneTwister(seed)
    model = WavePDEChess.WavePDEChessLM(model_config)
    target_ps, st = Lux.setup(rng, model)

    transplanted = false
    if source_checkpoint_path !== nothing
        payload = load_transfer_checkpoint(source_checkpoint_path)
        source_parameters = extract_checkpoint_parameters(payload)
        target_ps = transplant_core_parameters(target_ps, source_parameters)
        transplanted = true
    end

    optimizer = Optimisers.Adam(learning_rate)
    opt_state = Optimisers.setup(optimizer, target_ps)
    losses = Float32[]

    for step in 1:max_iters
        batch = WavePDEChess.sample_training_batch(corpus, rng; batch_size=batch_size, max_seq_len=model.config.max_seq_len)
        loss = WavePDEChess.autoregressive_loss(model, target_ps, st, batch)
        grads = Zygote.gradient(p -> WavePDEChess.autoregressive_loss(model, p, st, batch), target_ps)[1]
        freeze_core && (grads = mask_core_gradients(grads))
        opt_state, target_ps = Optimisers.update(opt_state, target_ps, grads)
        push!(losses, Float32(loss))
    end

    checkpoint = (
        scenario=scenario,
        model_config=model_config,
        source_checkpoint_path=source_checkpoint_path,
        transplanted_core=transplanted,
        frozen_core=freeze_core,
        parameters=target_ps,
        state=st,
        losses=losses,
    )
    save_transfer_checkpoint(checkpoint_path, checkpoint)

    return (
        scenario=scenario,
        checkpoint_path=checkpoint_path,
        source_checkpoint_path=source_checkpoint_path,
        transplanted_core=transplanted,
        frozen_core=freeze_core,
        losses=losses,
        final_loss=last(losses),
        parameters=target_ps,
    )
end

function compare_symbolic_transfer(;
    output_dir::AbstractString,
    source_checkpoint_path::AbstractString,
    dataset_dir::AbstractString=joinpath(output_dir, "data"),
    count_per_task::Int=32,
    seed::Int=1337,
    batch_size::Int=8,
    learning_rate::Float32=6.0f-4,
    max_iters::Int=2,
    vocab_size::Int=symbolic_vocabulary_size(),
    d_model::Int=288,
    n_layer::Int=20,
    solver_steps::Int=4,
    dt_init::Float32=0.05f0,
    norm_eps::Float32=1f-5,
    max_seq_len::Int=128,
    rebuild_dataset::Bool=true,
)
    source_checkpoint_path = abspath(source_checkpoint_path)
    isfile(source_checkpoint_path) || throw(ArgumentError("source_checkpoint_path does not exist: $(source_checkpoint_path)"))

    mkpath(output_dir)
    dataset_path = write_symbolic_dataset(dataset_dir; count_per_task=count_per_task, seed=seed, force=rebuild_dataset)
    corpus = WavePDEChess.ChessParquetCorpus(dataset_dir; min_tokens=6)
    model_config = symbolic_model_config(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layer=n_layer,
        solver_steps=solver_steps,
        dt_init=dt_init,
        norm_eps=norm_eps,
        max_seq_len=max_seq_len,
    )

    scratch_result = train_symbolic_setting(
        :scratch_full,
        model_config,
        corpus;
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_iters=max_iters,
        checkpoint_path=joinpath(output_dir, "scratch_full.jls"),
    )

    frozen_result = train_symbolic_setting(
        :chess_core_frozen,
        model_config,
        corpus;
        seed=seed + 1,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_iters=max_iters,
        source_checkpoint_path=source_checkpoint_path,
        freeze_core=true,
        checkpoint_path=joinpath(output_dir, "chess_core_frozen.jls"),
    )

    finetune_result = train_symbolic_setting(
        :chess_core_finetune,
        model_config,
        corpus;
        seed=seed + 2,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_iters=max_iters,
        source_checkpoint_path=source_checkpoint_path,
        freeze_core=false,
        checkpoint_path=joinpath(output_dir, "chess_core_finetune.jls"),
    )

    result = (
        output_dir=abspath(output_dir),
        dataset_dir=abspath(dataset_dir),
        dataset_path=abspath(dataset_path),
        source_checkpoint_path=source_checkpoint_path,
        model_config=model_config,
        scratch_full=scratch_result,
        chess_core_frozen=frozen_result,
        chess_core_finetune=finetune_result,
    )
    save_transfer_checkpoint(joinpath(output_dir, "symbolic_transfer_comparison.jls"), result)
    return result
end

end
