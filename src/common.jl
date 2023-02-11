abstract type Model <: ContinuousUnivariateDistribution end

"""
    compute_initial_state(μ, σ, n_states)

Create a normally distribution initial state probability vector 

# Arguments

- `μ`: mean of intitial state distribution
- `σ`: standard deviation of initial state distribution
- `n_states`: the number of evidence states 
"""
function compute_initial_state(μ, σ, n_states)
    p = pdf.(Normal(μ, σ), 1:n_states)
    p ./= sum(p)
    return p
end

"""
    make_projector(n, lb, ub)

Creates a projection matrix across evidences states spanned by 
`lb` and `ub`.

# Arguments

- `n`: number of evidence states 
- `lb`: the lower index state 
- `ub`: the upper index state 
"""
function make_projector(n, lb, ub)
    x = zeros(n)
    x[lb:ub] .= 1
    return diagm(x)
end

function joint_distribution(model::Model, s0, projectors, T)
    n = length(projectors)
    probs = fill(0.0, n, n)
    for j ∈ 1:n, i ∈ 1:n
        probs[i,j] = sum(projectors[j] * T * projectors[i] * s0)
    end
    return probs 
end

"""
    sample(c_dist)

Samples indices of an n × n joint probability matrix. 

# Arguments

- `c_dist`: an n × n matrix of the cummulative joint distribution
"""
function sample(c_dist)
    urand = rand()
    n_rows,n_cols = size(c_dist)
    for c ∈ 1:n_cols,r ∈ 1:n_rows
        c_dist[r,c] > urand ? (return r, c) : nothing 
    end 
    return n_rows,n_cols
end

"""
    cummulative_dist(x)

Returns an n × n matrix representing the cummulative joint distribution.

# Arguments

- `x`: an n × n matrix representing a joint probability distribution
"""
function cummulative_dist(x)
    y = similar(x)
    y[1] = x[1]
    for i ∈ 2:length(x)
        y[i] = y[i-1] + x[i]
    end
    return y
end

"""
    rand(model::Model, preds::Array{T,2}, n) where {T}

Samples `n` indices (row,column) from a joint distribution.

# Arguments

- `model`: a model object 
- `preds`: an n × n joint probability matrices
- `n`: the number of samples 
"""
function rand(model::Model, preds::Array{T,2}, n) where {T}
    c_dist = cummulative_dist(preds)
    return map(_ -> sample(c_dist), 1:n)
end

"""
    rand(model::Model, preds::Vector{Array{T,2}}, n) where {T}

Samples `n` indices (row,column) from a vector of joint distributions.

# Arguments

- `model`: a model object 
- `preds`: a vector of n × n joint probability matrices
- `n`: the number of samples 
"""
function rand(model::Model, preds::Vector{Array{T,2}}, n) where {T}
    return map(p -> rand(model, p, n), preds)
end