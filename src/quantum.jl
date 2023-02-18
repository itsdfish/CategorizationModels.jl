"""
    QuantumModel <: Model 

A model object for the quantum model. 

# Field Names 
- `μ`: the mean of the initial state distribution
- `σ`: the standard deviation of the initial state distribution
- `υ_k_s_k`: drift rate for k then s given stimulus k
- `υ_s_k_k`: drift rate for s then k given stimulus k
- `υ_k_s_s`: drift rate for k then s given stimulus s
- `υ_s_k_s`: drift rate for s then k given stimulus s
- `λ_k_s_k`: diffusion rate for k then s given stimulus k
- `λ_s_k_k`: diffusion rate for s then k given stimulus k
- `λ_k_s_s`: diffusion rate for k then s given stimulus s
- `λ_s_k_s`: diffusion rate for s then k given stimulus s
- `n_states`: the number of evidence states
"""
@concrete struct QuantumModel <: Model 
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
    α
    n_states
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

"""
    generate_predictions(model::QuantumModel{T}, n_options) where {T}

Generate predictions for the Markov model. 

# Arguments 

- `model::QuantumModel{T}`: a Markov model object 
- `n_options`: the number of response options 
"""
function generate_predictions(model::QuantumModel{T}, n_options) where {T}
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

"""
    make_joint_dist(model::QuantumModel, s0, projectors, T)

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
