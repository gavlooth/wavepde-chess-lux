include(joinpath(@__DIR__, "train_chess_airy_lm.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    run_chess_airy_lm_training()
end
