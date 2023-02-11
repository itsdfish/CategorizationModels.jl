"""
    MarkovModel <: Model 

A model object for the Markov model. 

# Field Names 
- `μ`: the mean of the initial state distribution
- `σ`: the standard deviation of the initial state distribution
- `υ_k_k`: drift rate for stimulus k when k is presented 
- `υ_s_k`: drift rate for stimulus s when s is presented 
- `υ_k_s`: drift rate for stimulus k when s is presented  
- `υ_s_s`: drift rate for stimulus s when s is presented 
- `λ_k_k`: diffusion for stimulus k when k is presented
- `λ_s_k`: diffusion for stimulus s when k is presented
- `λ_k_s`: diffusion for stimulus k when s is presented
- `λ_s_s`: diffusion for stimulus s when s is presented
- `n_states`: the number of evidence states
"""
@concrete struct MarkovModel <: Model 
    μ
    σ
    υ_k_k
    υ_s_k
    υ_k_s
    υ_s_s
    λ_k_k
    λ_s_k
    λ_k_s
    λ_s_s
    n_states
end

function MarkovModel(;μ,
                    σ,
                    υ_k_k,
                    υ_s_k,
                    υ_k_s,
                    υ_s_s,
                    λ_k_k,
                    λ_s_k,
                    λ_k_s,
                    λ_s_s,
                    n_states)

    return MarkovModel( μ,
                        σ,
                        υ_k_k,
                        υ_s_k,
                        υ_k_s,
                        υ_s_s,
                        λ_k_k,
                        λ_s_k,
                        λ_k_s,
                        λ_s_s,
                        n_states)
end

function generate_predictions(model::MarkovModel{T}, n_options) where {T}
    (;μ,σ,σ,n_states) = model 
    (;υ_k_k,υ_s_k,υ_k_s,υ_s_s) = model 
    (;λ_k_k,λ_s_k,λ_k_s,λ_s_s) = model 
    n = div(n_states, n_options)

    initial_states = Vector{Vector{T}}(undef,4)
    # initial state k then s for k stimulus
    initial_states[1] = compute_initial_state(μk, σk, n_states)
    # initial state s then k for s stimulus
    initial_states[3] = compute_initial_state(μs, σs, n_states)

    # intensity matrix k then s for k stimulus
    κ_ks_k = make_intensity_matrix(n_states, υ_ks_k)
    # intensity matrix s then k for s stimulus 
    κ_sk_s = make_intensity_matrix(n_states, υ_sk_s)

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

function make_intensity_matrix(model::MarkovModel, n_states, υ)
    mat = zeros(n_states, n_states)
    for c ∈ 1:n_states, r ∈ 1:n_states
        if r == (c - 1)
            mat[r,c] = υ
        elseif c == (r - 1)
            mat[r,c] = 1
        elseif r == c
            mat[r,c] = -υ - 1
        end
    end
    mat[1,1] = -1
    mat[end,end] = -υ
    return mat
end