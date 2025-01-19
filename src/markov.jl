"""
    MarkovModel{T<:Real} <: Model 

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
struct MarkovModel{T <: Real} <: Model
    μ::T
    σ::T
    υ_k_k::T
    υ_s_k::T
    υ_k_s::T
    υ_s_s::T
    λ_k_k::T
    λ_s_k::T
    λ_k_s::T
    λ_s_s::T
    n_states::Int
end

function MarkovModel(; μ,
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
    return MarkovModel(μ,
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

function MarkovModel(
    μ,
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
    parms = promote(
        μ,
        σ,
        υ_k_k,
        υ_s_k,
        υ_k_s,
        υ_s_s,
        λ_k_k,
        λ_s_k,
        λ_k_s,
        λ_s_s)

    return MarkovModel(parms..., n_states)
end

"""
    predict(model::MarkovModel{T}, n_options) where {T}

Generate predictions for the Markov model. 

# Arguments 

- `model::MarkovModel{T}`: a Markov model object 
- `n_options`: the number of response options 

# Returns 

The predictions are organized as a vector of four n × n matrices representing the joint probability 
distribution of rating both stimuli in two orders. The joint distributions are as follows:

1. The joint probability distribution for k then s given stimulus k where element `pred[i,j]` is the probability of rating stimulus `k` as `i` and stimulus `s` as `j` 

2. The joint probability distribution for s then k given stimulus k where element `pred[i,j]` is the probability of rating stimulus `s` as `i` and stimulus `k` as `j` 

3. The joint probability distribution for k then s given stimulus s where element `pred[i,j]` is the probability of rating stimulus `k` as `i` and stimulus `s` as `j` 

4. The joint probability distribution for k then s given stimulus s where element `pred[i,j]` is the probability of rating stimulus `s` as `i` and stimulus `k` as `j` 
"""
function predict(model::MarkovModel{T}, n_options) where {T}
    (; μ, σ, n_states) = model
    (; υ_k_k, υ_s_k, υ_k_s, υ_s_s) = model
    (; λ_k_k, λ_s_k, λ_k_s, λ_s_s) = model
    n = div(n_states, n_options)

    initial_state = compute_initial_state(model, μ, σ, n_states)

    # intensity matrix for option k given k stimulus
    κ_k_k = make_intensity_matrix(model, n_states, υ_k_k)
    # intensity matrix for option s given k stimulus 
    κ_s_k = make_intensity_matrix(model, n_states, υ_s_k)
    # intensity matrix for option k given s stimulus 
    κ_k_s = make_intensity_matrix(model, n_states, υ_k_s)
    # intensity matrix for option s given s stimulus 
    κ_s_s = make_intensity_matrix(model, n_states, υ_s_s)

    # transition matrix for option k given stimulus k 
    T_k_k = exp(λ_k_k * κ_k_k)
    # transition matrix for option s given stimulus k 
    T_s_k = exp(λ_s_k * κ_s_k)
    # transition matrix for option k given stimulus s
    T_k_s = exp(λ_k_s * κ_k_s)
    # transition matrix for option s given stimulus s 
    T_s_s = exp(λ_s_s * κ_s_s)

    # a projector for each option 
    projectors = map(i ->
            make_projector(n_states, n * (i - 1) + 1, i * n),
        1:n_options)

    preds = Vector{Matrix{T}}(undef, 4)

    # joint probability distribution for k then s given stimulus k 
    preds[1] = make_joint_dist(model, initial_state, projectors, T_k_k, T_s_k)
    # joint probability distribution for s then k given stimulus k 
    preds[2] = make_joint_dist(model, initial_state, projectors, T_s_k, T_k_k)
    # joint probability distribution for s then k given stimulus s 
    preds[3] = make_joint_dist(model, initial_state, projectors, T_s_s, T_k_s)
    # joint probability distribution for k then s given stimulus s 
    preds[4] = make_joint_dist(model, initial_state, projectors, T_k_s, T_s_s)
    return preds
end

"""
    make_intensity_matrix(model::MarkovModel, n_states, υ)

Returns the intensity matrix for the Markov model. 

# Arguments 

- `model::MarkovModel{T}`: a Markov model object 
- `n_states`: the number of evidence states 
- `υ`: drift rate 

# Example:

```julia-repl
v = make_intensity_matrix(model, 5, 3)

5×5 Matrix{Float64}:
 -1.0   3.0   0.0   0.0   0.0
  1.0  -4.0   3.0   0.0   0.0
  0.0   1.0  -4.0   3.0   0.0
  0.0   0.0   1.0  -4.0   3.0
  0.0   0.0   0.0   1.0  -3.0
```
"""
function make_intensity_matrix(model::MarkovModel, n_states, υ)
    mat = zeros(n_states, n_states)
    for c ∈ 1:n_states, r ∈ 1:n_states
        if r == (c - 1)
            mat[r, c] = υ
        elseif c == (r - 1)
            mat[r, c] = 1
        elseif r == c
            mat[r, c] = -υ - 1
        end
    end
    mat[1, 1] = -1
    mat[end, end] = -υ
    return mat
end

"""
    make_joint_dist(model::MarkovModel, s0, projectors, T)

Returns an n × n matrix representing the joint probability distribution of response ratings. 

# Arguments

- `model::MarkovModel`: a Markov model object 
- `s0`: the initial state vector 
- `projectors`: a vector of projector matrices 
- `T1`: first transition matrix 
- `T2`: second transition matrix 
"""
function make_joint_dist(model::MarkovModel, s0, projectors, T1, T2)
    n = length(projectors)
    probs = fill(0.0, n, n)
    for j ∈ 1:n, i ∈ 1:n
        probs[i, j] = sum(abs.(projectors[j] * T2 * projectors[i] * T1 * s0))
    end
    probs .= max.(probs, eps())
    return probs
end
