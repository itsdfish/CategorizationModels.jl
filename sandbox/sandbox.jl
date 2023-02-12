############################################################################################################
#                                         load packages
############################################################################################################
cd(@__DIR__)
using Pkg 
Pkg.activate("..")
using CategorizationModels
using Test 

parms = (μ = 80.0,
        σ = 20.0,
        υ_k_k = 1.0,
        υ_s_k = 2.0,
        υ_k_s = 3.0,
        υ_s_s = 4.0,
        λ_k_k = .5,
        λ_s_k = .5,
        λ_k_s = .5,
        λ_s_s = .5)

n_states = 12
n_options = 6

model = MarkovModel(;parms..., n_states)

preds = generate_predictions(model, n_options)
