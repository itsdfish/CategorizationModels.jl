############################################################################################################
#                                         load packages
############################################################################################################
cd(@__DIR__)
using Pkg 
Pkg.activate("..")
using CategorizationModels
using Random 
using Test 

Random.seed!(5)

# model predictions 
parms = (μ = 80.0,
        σ = 20.0,
        υ_k_k = 1.0,
        υ_s_k = 2.0,
        υ_k_s = 3.0,
        υ_s_s = 4.0,
        λ_k_k = .2,
        λ_s_k = .3,
        λ_k_s = .4,
        λ_s_s = .5)

# number of evidence states 
n_states = 96

# number of rating options 
n_options = 6

# create a model object 
model = QuantumModel(;parms..., n_states)

# generate predictions for all conditions 
preds = predict(model, n_options)

# generate 100 trials of data per condition 
data = rand(model, preds, 100)

# compute the logpdf of each data point
LLs = logpdf(model, preds, data)



function make_hamiltonian1(n_states, υ, σ)
    mat = zeros(n_states, n_states)
    for c ∈ 1:n_states, r ∈ 1:n_states
        if (c - r) == 1
            mat[r,c] = σ
        elseif (c - r) == -1
            mat[r,c] = σ
        elseif r == c
            mat[r,c] = (r / n_states) * υ
        end
    end
    return mat
end

function make_hamiltonian2(n_states, υ, σ)
    mat = zeros(n_states, n_states)
    for i ∈ 1:n_states
        mat[i,i] = (i / n_states) * υ
        i < n_states ? mat[i,i+1] = σ : nothing 
        println(i > 1 )
        i > 1 ? (mat[i,i-1] = σ) : nothing 
    end
    return mat
end