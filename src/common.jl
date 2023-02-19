abstract type Model <: ContinuousUnivariateDistribution end

"""
    compute_initial_state(μ, σ, n_states)

Create a normally distribution initial state probability vector 

# Arguments

- `μ`: mean of intitial state distribution
- `σ`: standard deviation of initial state distribution
- `n_states`: the number of evidence states 
"""
function compute_initial_state(model::Model, μ, σ, n_states)
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

"""
    make_joint_dist(model::Model, s0, projectors, T)

Returns an n × n matrix representing the joint probability distribution of response ratings. 

# Arguments

- `model::Model`: a model object 
- `s0`: the initial state vector 
- `projectors`: a vector of projector matrices 
- `T`: a transition matrix 
"""
function make_joint_dist(model::Model, s0, projectors, T)
    n = length(projectors)
    probs = fill(0.0, n, n)
    for j ∈ 1:n, i ∈ 1:n
        probs[i,j] = sum(projectors[j] * T * projectors[i] * s0)
    end
    probs .= max.(probs, eps())
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

Samples `n` indices (i,j) from a vector of joint distributions.

# Arguments

- `model`: a model object 
- `preds`: a vector of n × n joint probability matrices
- `n`: the number of samples 

# Returns 

The data are organized as a vector of four n × n matrices representing ratings of both stimuli in two orders. The data are as follows:

1. Data for k then s given stimulus k where element `(i,j)` is the rating index for rating stimulus `k` as `i` and stimulus `s` as `j` 

2. Data for s then k given stimulus k where element `(i,j)` is the rating index for rating stimulus `s` as `i` and stimulus `k` as `j`

3. Data for k then s given stimulus s where element `(i,j)` is the rating index for rating stimulus `k` as `i` and stimulus `s` as `j`

4. Data for k then s given stimulus s where element `(i,j)`  is the rating index for rating stimulus `s` as `i` and stimulus `k` as `j`
"""
function rand(model::Model, preds::Vector{Array{T,2}}, n) where {T}
    return map(p -> rand(model, p, n), preds)
end

"""
    logpdf(model::Model, preds::Array{T,2}, data) where {T}

Computes the log pdf of responses within a given condition.

# Arguments

- `model`: a model object 
- `preds`: an n × n joint probability matrices
- `data`: an array of vectors, one per condition
"""
function logpdf(model::Model, preds::Array{T,2}, data) where {T}
    n = length(data)
    LLs = zeros(n)
    for i ∈ 1:n
        LLs[i] = log(preds[data[i]...])
    end
    return LLs 
end

"""
    logpdf(model::Model, preds::Vector{Array{T,2}}, data) where {T}

Computes the log pdf of responses across all conditions. 

# Arguments

- `model`: a model object 
- `preds`: a vector of n × n joint probability matrices
- `n`: a vector of data

# Returns 

The logpdfs are organized as four vectors of responses, one vector for each stimulus type and ordering permutation. The vectors are:

1. logpdfs for the condition in which stimulus k is rated as k then s  
2. logpdfs for the condition in which stimulus k is rated as s then k 
3. logpdfs for the condition in which stimulus s is rated as k then s   
4. logpdfs for the condition in which stimulus s is rated as s then k  
"""
function logpdf(model::Model, preds::Vector{Array{T,2}}, data) where {T}
    return map(i -> logpdf(model, preds[i], data[i]), 1:length(data))
end

"""
    sumlogpdf(model::Model, n_options, data) 

Returns the sum of the log likelihood across all trials and conditions. 

# Arguments

- `model`: a model object 
- `n_options`: the number of rating options
- `data`: an array of vectors, one per condition
"""
function sumlogpdf(model::Model, n_options, data) 
    preds = generate_predictions(model, n_options)
    return sum(sum(logpdf(model, preds, data)))
end