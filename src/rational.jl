"""
    RationalModel  <: Model 

A model object for the rational model. 

# Field Names 
- `μk`: the mean of the initial state distribution for stimulus k when k is evaluated first
- `μs`: the mean of the initial state distribution for stimulus s when s is evaluated first
- `σk`: the standard deviation of the initial state distribution for stimulus k when it is evaluated first
- `σs`: the standard deviation of the initial state distribution for stimulus s when it is evaluated first
- `n_states`: the number of evidence states
"""
@concrete struct RationalModel <: Model 
    μk
    μs 
    σk
    σs
    n_states
end

function RationalModel(;μk, μs, σk, σs, n_states)
    return RationalModel(μk, μs, σk, σs, n_states)
end

"""
    predict(model::RationalModel{T}, n_options) where {T}

Generate predictions for the rational model. 

# Arguments 

- `model::RationalModel{T}`: a rational model object 
- `n_options`: the number of response options 

# Returns 

The predictions are organized as a vector of four n × n matrices representing the joint probability 
distribution of rating both stimuli in two orders. The joint distributions are as follows:

1. The joint probability distribution for k then s given stimulus k where element `pred[i,j]` is the probability of rating stimulus `k` as `i` and stimulus `s` as `j` 

2. The joint probability distribution for s then k given stimulus k where element `pred[i,j]` is the probability of rating stimulus `s` as `i` and stimulus `k` as `j` 

3. The joint probability distribution for k then s given stimulus s where element `pred[i,j]` is the probability of rating stimulus `k` as `i` and stimulus `s` as `j` 

4. The joint probability distribution for k then s given stimulus s where element `pred[i,j]` is the probability of rating stimulus `s` as `i` and stimulus `k` as `j` 
"""
function predict(model::RationalModel{T}, n_options) where {T}
    (;μk,μs,σk,σs,n_states) = model 
    n = div(n_states, n_options)
    
    initial_states = Vector{Vector{T}}(undef,0)
    # initial state k then s for k stimulus
    push!(initial_states, compute_initial_state(model, μk, σk, n_states))
    #initial state s then k for k stimulus
    push!(initial_states, reverse(initial_states[1]))
    # initial state s then k for s stimulus
    push!(initial_states, compute_initial_state(model, μs, σs, n_states))
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
                make_joint_dist(model, i, projectors, R),
                initial_states)
    return preds 
end

"""
    make_joint_dist(s0, projectors, R)

# Arguments

- `s0`: initial state vector 
- `projectors`: array of projectors 
- `R`: change of basis matrix 
"""
function make_joint_dist(model::RationalModel, s0, projectors, R)
    n = length(projectors)
    probs = fill(eps(), n, n)
    for j ∈ 1:n
        i = n - j + 1
        probs[i,j] = sum(projectors[j] * R * projectors[i] * s0)
    end
    probs .= max.(probs, eps())
    return probs 
end

"""
    anti_diagonal(n)

Creates a n × n matrix with 1's along the off-diagonal and 0's elsewhere. 

# Arguments

- `n`: the number of evidence states 
"""
function anti_diagonal(n)
    x = zeros(n, n)
    for i ∈ 1:n
        x[i,n-i+1] = 1
    end
    return x 
end
