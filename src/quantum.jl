"""
    QuantumModel <: Model 

A model object for the quantum model. 

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
struct QuantumModel{T<:Real} <: Model 
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
    α::T
    n_states::Int
end

function QuantumModel(;μ,
                    σ,
                    υ_k_k,
                    υ_s_k,
                    υ_k_s,
                    υ_s_s,
                    λ_k_k,
                    λ_s_k,
                    λ_k_s,
                    λ_s_s,
                    α = 1000.0,
                    n_states)

    return QuantumModel(μ,
                        σ,
                        υ_k_k,
                        υ_s_k,
                        υ_k_s,
                        υ_s_s,
                        λ_k_k,
                        λ_s_k,
                        λ_k_s,
                        λ_s_s,
                        α,
                        n_states)
end

function QuantumModel(
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
    α,
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
        λ_s_s,
        α)

    return QuantumModel(parms..., n_states)
end

"""
    predict(model::QuantumModel{T}, n_options) where {T}

Generate predictions for the Quantum model. 

# Arguments 

- `model::QuantumModel{T}`: a Markov model object 
- `n_options`: the number of response options 

# Returns 

The predictions are organized as a vector of four n × n matrices representing the joint probability 
distribution of rating both stimuli in two orders. The joint distributions are as follows:

1. The joint probability distribution for k then s given stimulus k where element `pred[i,j]` is the probability of rating stimulus `k` as `i` and stimulus `s` as `j` 

2. The joint probability distribution for s then k given stimulus k where element `pred[i,j]` is the probability of rating stimulus `s` as `i` and stimulus `k` as `j` 

3. The joint probability distribution for k then s given stimulus s where element `pred[i,j]` is the probability of rating stimulus `k` as `i` and stimulus `s` as `j` 

4. The joint probability distribution for k then s given stimulus s where element `pred[i,j]` is the probability of rating stimulus `s` as `i` and stimulus `k` as `j` 
"""
function predict(model::QuantumModel{T}, n_options) where {T}
    (;μ,σ,α,n_states) = model 
    (;υ_k_k,υ_s_k,υ_k_s,υ_s_s) = model 
    (;λ_k_k,λ_s_k,λ_k_s,λ_s_s) = model 
    n = div(n_states, n_options)

    initial_state = compute_initial_state(model, μ, σ, n_states)

    # Hamiltonian for option k given k stimulus
    ℋ_k_k = make_hamiltonian(model, n_states, υ_k_k, α)
    # Hamiltonian for option s given k stimulus 
    ℋ_s_k = make_hamiltonian(model, n_states, υ_s_k, α)
    # Hamiltonian for option k given s stimulus 
    ℋ_k_s = make_hamiltonian(model, n_states, υ_k_s, α)
    # Hamiltonian for option s given s stimulus 
    ℋ_s_s = make_hamiltonian(model, n_states, υ_s_s, α)

    # transition matrix for option k given stimulus k 
    U_k_k = exp(-im * λ_k_k * ℋ_k_k)
    # transition matrix for option s given stimulus k 
    U_s_k = exp(-im * λ_s_k * ℋ_s_k)
    # transition matrix for option k given stimulus s
    U_k_s = exp(-im * λ_k_s * ℋ_k_s)
    # transition matrix for option s given stimulus s 
    U_s_s = exp(-im * λ_s_s * ℋ_s_s)

    # a projector for each option 
    projectors = map(i -> 
                        make_projector(n_states, n * (i - 1) + 1, i * n),
                        1:n_options)

    preds = Vector{Matrix{T}}(undef,4)

    # joint probability distribution for k then s given stimulus k 
    preds[1] = make_joint_dist(model, initial_state, projectors, U_k_k, U_s_k)
    # joint probability distribution for s then k given stimulus k 
    preds[2] = make_joint_dist(model, initial_state, projectors, U_s_k, U_k_k)
    # joint probability distribution for s then k given stimulus s 
    preds[3] = make_joint_dist(model, initial_state, projectors, U_s_s, U_k_s)
    # joint probability distribution for k then s given stimulus s 
    preds[4] = make_joint_dist(model, initial_state, projectors, U_k_s, U_s_s)
    return preds 
end

"""
    make_hamiltonian(model::QuantumModel, n_states, υ)

Returns the Hamiltonian matrix for the quantum model. 

# Arguments 

- `model::QuantumModel{T}`: a quantum model object 
- `n_states`: the number of evidence states 
- `υ`: drift rate 

# Example:

```julia-repl
julia> v = make_hamiltonian(model, 5, 2, .5)
5×5 Matrix{Float64}:
 0.4  0.5  0.0  0.0  0.0
 0.5  0.8  0.5  0.0  0.0
 0.0  0.5  1.2  0.5  0.0
 0.0  0.0  0.5  1.6  0.5
 0.0  0.0  0.0  0.5  2.0
```
"""
function make_hamiltonian(model::QuantumModel, n_states, υ, σ)
    mat = zeros(n_states, n_states)
    for i ∈ 1:n_states
        mat[i,i] = (i / n_states) * υ
        i < n_states ? mat[i,i+1] = σ : nothing 
        i > 1 ? (mat[i,i-1] = σ) : nothing 
    end
    return mat
end

"""
    make_joint_dist(model::QuantumModel, s0, projectors, U1, U2)

Returns an n × n matrix representing the joint probability distribution of response ratings. 

# Arguments

- `model::QuantumModel`: a Markov model object 
- `s0`: the initial state vector 
- `projectors`: a vector of projector matrices 
- `U1`: first unitary matrix 
- `U2`: second unitary matrix 
"""
function make_joint_dist(model::QuantumModel, s0, projectors, U1, U2)
    n = length(projectors)
    probs = fill(0.0, n, n)
    for j ∈ 1:n, i ∈ 1:n
        # why abs instead of real?
        probs[i,j] = sum(abs.(projectors[j] * (U2 * U1') * projectors[i] * U1 * s0).^2)
    end
    probs .= max.(probs, eps())
    return probs 
end

"""
    compute_initial_state(model::QuantumModel, μ, σ, n_states)

Create a normally distribution initial state probability vector 

# Arguments

- `μ`: mean of intitial state distribution
- `σ`: standard deviation of initial state distribution
- `n_states`: the number of evidence states 
"""
function compute_initial_state(model::QuantumModel, μ, σ, n_states)
    p = pdf.(Normal(μ, σ), 1:n_states)
    p ./= sum(p)
    return p.^(1/2)
end
