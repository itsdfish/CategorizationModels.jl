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

function generate_predictions(model::RationalModel{T}, n_options) where {T}
    (;μk,μs,σk,σs,n_states) = model 
    n = div(n_states, n_options)
    
    initial_states = Vector{Vector{T}}(undef,0)
    # initial state k then s for k stimulus
    push!(initial_states, compute_initial_state(μk, σk, n_states))
    #initial state s then k for k stimulus
    push!(initial_states, reverse(initial_states[1]))
    # initial state s then k for s stimulus
    push!(initial_states, compute_initial_state(μs, σs, n_states))
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
                joint_distribution(model, i, projectors, R),
                initial_states)
    return preds 
end

"""
    joint_distribution(s0, projectors, R)

# Arguments

- `s0`: initial state vector 
- `projectors`: array of projectors 
- `R`: change of basis matrix 
"""
function joint_distribution(model::RationalModel, s0, projectors, R)
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
