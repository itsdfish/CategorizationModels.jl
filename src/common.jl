abstract type Model <: ContinuousUnivariateDistribution end

function compute_initial_state(μ, σ, n_states)
    p = pdf.(Normal(μ, σ), 1:n_states)
    p ./= sum(p)
    return p
end

function anti_diagonal(n)
    x = zeros(n, n)
    for i ∈ 1:n
        x[i,n-i+1] = 1
    end
    return x 
end

function make_projector(n, lb, ub)
    x = zeros(n)
    x[lb:ub] .= 1
    return diagm(x)
end

function rand(model::Model, preds, n)
    c_dist = cummulative_dist(preds)
    

end

function sample(c_dist)
    urand = rand()
    n_rows,n_cols = size(c_dist)
    for c ∈ 1:n_cols,r ∈ 1:n_rows
        c_dist[r,c] > urand ? (return r, c) : nothing 
    end 
    return n_rows,n_cols
end
    

function cummulative_dist(x)
    y = similar(x)
    y[1] = x[1]
    for i ∈ 2:length(x)
        y[i] = y[i-1] + x[i]
    end
    return y
end

function joint_distribution(model::Model, s0, projectors, T)
    n = length(projectors)
    probs = fill(0.0, n, n)
    for j ∈ 1:n, i ∈ 1:n
        probs[i,j] = sum(projectors[j] * T * projectors[i] * s0)
    end
    return probs 
end