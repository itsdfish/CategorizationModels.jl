abstract type Model end

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