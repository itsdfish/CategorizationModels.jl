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
using CategorizationModels: make_intensity_matrix
using CategorizationModels: make_transition_matrix

μk = 80.0
σk = 20.0
μs = 90.0
σs = 20.0

υ_ks_k = 5.0
υ_sk_s = 4.0
λ_ks_k = .50
λ_sk_s = .50

model = BayesianModel(rand(8)..., 1)

n_states = 12
n_options = 6
n = div(n_states, n_options)

T = Float64

initial_states = Vector{Vector{T}}(undef,4)
# initial state k then s for k stimulus
initial_states[1] = compute_initial_state(μk, σk, n_states)
# initial state s then k for s stimulus
initial_states[3] = compute_initial_state(μs, σs, n_states)

# intensity matrix k then s for k stimulus
κ_ks_k = make_intensity_matrix(n_states, υ_ks_k)
# intensity matrix s then k for s stimulus 
κ_sk_s = make_intensity_matrix(n_states, υ_sk_s)

t_matrices = Vector{Matrix{T}}(undef,4)
# transition matrix k then s for k stimulus
t_matrices[1] = exp(λ_ks_k * κ_ks_k)
# transition matrix s then k for s stimulus 
t_matrices[3] = exp(λ_sk_s * κ_sk_s)

# initial state s then k for k stimulus (why are there negative states?)
initial_states[2] = t_matrices[1] * initial_states[1]
# initial state k then s for s stimulus 
initial_states[4] = t_matrices[3] * initial_states[3]

# transition matrix s then k for k stimulus
t_matrices[2] = make_transition_matrix(model, 
                                       initial_states[2], 
                                       initial_states[1], 
                                       t_matrices[1])

# transition matrix k then s for s stimulus
t_matrices[4] = make_transition_matrix(model, 
                                       initial_states[4], 
                                       initial_states[3], 
                                       t_matrices[3])

# a projector for each option 
projectors = map(i -> 
                    make_projector(n_states, n * (i - 1) + 1, i * n),
                    1:n_options)
                     
preds = map((s,t) -> 
            joint_distribution(model, s, projectors, t),
            initial_states,
            t_matrices)



using CategorizationModels
using Test 

parms = (μk = 80.0,
            σk = 20.0,
            μs = 90.0,
            σs = 20.0,
            υ_ks_k = 5.0,
            υ_sk_s = 4.0,
            λ_ks_k = .50,
            λ_sk_s = .50)

n_states = 12
n_options = 6

model = BayesianModel(;parms..., n_states)


@time generate_predictions(model, n_options)

data = rand(model, preds, 100)
@time sumlogpdf(model, preds, data)