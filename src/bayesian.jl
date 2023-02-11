"""
    BayesianModel <: Model 

A model object for the Bayesian model. 

# Field Names 
- `μk`: the mean of the initial state distribution for stimulus k when k is evaluated first
- `μs`: the mean of the initial state distribution for stimulus s when s is evaluated first
- `σk`: the standard deviation of the initial state distribution for stimulus k when it is evaluated first
- `σs`: the standard deviation of the initial state distribution for stimulus s when it is evaluated first
- `υ_ks_k`: drift rate for stimulus k when k is evaluated first 
- `υ_sk_s`: drift rate for stimulus k when s is evaluated first 
- `λ_ks_k`: diffusion for stimulus k when k is evaluated first 
- `λ_sk_s`: diffusion for stimulus s when s is evaluated first 
- `n_states`: the number of evidence states
"""
@concrete struct BayesianModel <: Model 
    μk
    μs 
    σk
    σs
    υ_ks_k
    υ_sk_s
    λ_ks_k
    λ_sk_s
    n_states
end

function BayesianModel(;μk,
                        μs, 
                        σk, 
                        σs, 
                        υ_ks_k, 
                        υ_sk_s,
                        λ_ks_k,
                        λ_sk_s,
                        n_states)

    return BayesianModel(μk,
                         μs, 
                         σk, 
                         σs, 
                         υ_ks_k, 
                         υ_sk_s,
                         λ_ks_k,
                         λ_sk_s,
                         n_states)
end

function generate_predictions(model::BayesianModel{T}, n_options) where {T}
    (;μk,μs,σk,σs,n_states) = model 
    (;υ_ks_k, υ_sk_s,λ_ks_k,λ_sk_s) = model 

    n = div(n_states, n_options)
    initial_states = Vector{Vector{T}}(undef,4)
    # initial state k then s for k stimulus
    initial_states[1] = compute_initial_state(μk, σk, n_states)
    # initial state s then k for s stimulus
    initial_states[3] = compute_initial_state(μs, σs, n_states)

    # intensity matrix k then s for k stimulus
    κ_ks_k = make_intensity_matrix(model, n_states, υ_ks_k)
    # intensity matrix s then k for s stimulus 
    κ_sk_s = make_intensity_matrix(model, n_states, υ_sk_s)

    transitions = Vector{Matrix{T}}(undef,4)
    # transition matrix k then s for k stimulus
    transitions[1] = exp(λ_ks_k * κ_ks_k)
    # transition matrix s then k for s stimulus 
    transitions[3] = exp(λ_sk_s * κ_sk_s)

    # initial state s then k for k stimulus (why are there negative states?)
    initial_states[2] = transitions[1] * initial_states[1]
    # initial state k then s for s stimulus 
    initial_states[4] = transitions[3] * initial_states[3]

    # transition matrix s then k for k stimulus
    transitions[2] = make_transition_matrix(model, 
                                        initial_states[2], 
                                        initial_states[1], 
                                        transitions[1])

    # transition matrix k then s for s stimulus
    transitions[4] = make_transition_matrix(model, 
                                        initial_states[4], 
                                        initial_states[3], 
                                        transitions[3])

    # a projector for each option 
    projectors = map(i -> 
                        make_projector(n_states, n * (i - 1) + 1, i * n),
                        1:n_options)
                        
    preds = map((s,t) -> 
                    joint_distribution(model, s, projectors, t),
                    initial_states,
                    transitions)
    return preds 
end

function make_intensity_matrix(model::BayesianModel, n_states, υ)
    mat = zeros(n_states, n_states)
    for c ∈ 1:n_states, r ∈ 1:n_states
        if r == (c - 1)
            mat[r,c] = 1 - υ
        elseif c == (r - 1)
            mat[r,c] = 1 + υ
        elseif r == c
            mat[r,c] = -2
        end
    end
    mat[1,1] = -1 - υ
    mat[end,end] = υ - 1
    return mat
end

function make_transition_matrix(model::BayesianModel, s0_sk, s0_ks, T)
    n = length(s0_sk)
    probs = fill(0.0, n, n)
    for k ∈ 1:n, j ∈ 1:n
        probs[j,k] = s0_ks[j] * T[k,j] / s0_sk[k]
    end
    return probs 
end