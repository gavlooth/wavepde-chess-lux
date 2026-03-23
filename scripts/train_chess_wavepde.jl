include(joinpath(@__DIR__, "train_chess_lm.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    run_chess_lm_training()
end
