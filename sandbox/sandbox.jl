############################################################################################################
#                                         load packages
############################################################################################################
cd(@__DIR__)
using Pkg 
Pkg.activate("..")
using Revise, Random, CategorizationModels
using CategorizationModels: compute_initial_state
using CategorizationModels: anti_diagonal
using CategorizationModels: make_projector
using CategorizationModels: joint_distribution

parms = (μk = 80.0,
σk = 20.0,
μs = 90.0,
σs = 20.0)

n_states = 96
n_options = 6

model = RationalModel(;parms..., n_states)

preds = generate_predictions(model, n_options)


initial_states = Vector{Vector{Float64}}(undef,0)
# initial state k then s for k stimulus
push!(initial_states, compute_initial_state(μk, σk, n_states))
#initial state s then k for k stimulus
push!(initial_states, reverse(initial_states[1]))
# initial state s then k for s stimulus
push!(initial_states, compute_initial_state(μs, σs, n_states))
# initial state k then s for s stimulus
push!(initial_states, reverse(initial_states[3]))

# change of basis matrix 
R = anti_diagonal(n_states)

# a projector for each option 
projectors = map(i -> 
                make_projector(n_states, n * (i - 1) + 1, i * n),
                 1:n_options)

# compute joint probability for each condition
preds = map(i -> 
            joint_distribution(i, projectors, R),
            initial_states)
return preds 
