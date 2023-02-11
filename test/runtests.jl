using SafeTestsets

@safetestset "cummulative_dist" begin 
    using CategorizationModels
    using CategorizationModels: cummulative_dist
    using Test 

    probs = [.2 .4; .3 .1]
    c_probs = cummulative_dist(probs)
    true_c_probs = [.2 .9; .5 1.0]

    @test c_probs ≈ true_c_probs atol = 1e-5
end

@safetestset "sample" begin 
    using CategorizationModels
    using CategorizationModels: cummulative_dist
    using CategorizationModels: sample
    using Statistics
    using Random
    using Test 

    Random.seed!(9874)
    probs = [.2 .4; .3 .1]
    c_probs = cummulative_dist(probs)
    samples = map(_ -> sample(c_probs), 1:100_000)
    
    @test probs[1,1] ≈ mean(x -> x == (1,1), samples) atol = 5e-3
    @test probs[2,1] ≈ mean(x -> x == (2,1), samples) atol = 5e-3
    @test probs[1,2] ≈ mean(x -> x == (1,2), samples) atol = 5e-3
    @test probs[2,2] ≈ mean(x -> x == (2,2), samples) atol = 5e-3
end

@safetestset "compute_initial_state" begin 
    using CategorizationModels
    using CategorizationModels: compute_initial_state
    using Test 
    
    n_states = 96
    μ = 20
    σ = 20
    initial_state = compute_initial_state(μ, σ, n_states)

    @test length(initial_state) == n_states
    @test sum(initial_state) ≈ 1 atol = 1e-8
    mx,mx_id = findmax(initial_state)
    @test mx_id ≈ μ atol = 1e-3

    n_states = 96
    μ = 80
    σ = 20
    initial_state = compute_initial_state(μ, σ, n_states)

    @test length(initial_state) == n_states
    @test sum(initial_state) ≈ 1 atol = 1e-8
    mx,mx_id = findmax(initial_state)
    @test mx_id ≈ μ atol = 1e-3
end

@safetestset "RationalModel" begin
    using CategorizationModels
    using Test 
    
    parms = (μk = 80.0,
             σk = 20.0,
             μs = 90.0,
             σs = 20.0)
    
    n_states = 96
    n_options = 6

    model = RationalModel(;parms..., n_states)

    preds = generate_predictions(model, n_options)

    p1 = [ 0.0    0.0    0.0    0.0    0.0   0.001;
        0.0    0.0    0.0    0.0    0.01  0.0;
        0.0    0.0    0.0    0.061  0.0   0.0;
        0.0    0.0    0.203  0.0    0.0   0.0;
        0.0    0.366  0.0    0.0    0.0   0.0;
        0.359  0.0    0.0    0.0    0.0   0.0]

    p2 = [ 0.0    0.0   0.0    0.0    0.0    0.359
        0.0    0.0   0.0    0.0    0.366  0.0;
        0.0    0.0   0.0    0.203  0.0    0.0;
        0.0    0.0   0.061  0.0    0.0    0.0;
        0.0    0.01  0.0    0.0    0.0    0.0;
        0.001  0.0   0.0    0.0    0.0    0.0]

    p3 = [ 0.0    0.0    0.0    0.0    0.0    0.0;
        0.0    0.0    0.0    0.0    0.003  0.0;
        0.0    0.0    0.0    0.027  0.0    0.0;
        0.0    0.0    0.131  0.0    0.0    0.0;
        0.0    0.345  0.0    0.0    0.0    0.0;
        0.494  0.0    0.0    0.0    0.0    0.0]

    p4 = [ 0.0  0.0    0.0    0.0    0.0    0.494;
        0.0  0.0    0.0    0.0    0.345  0.0;
        0.0  0.0    0.0    0.131  0.0    0.0;
        0.0  0.0    0.027  0.0    0.0    0.0;
        0.0  0.003  0.0    0.0    0.0    0.0;
        0.0  0.0    0.0    0.0    0.0    0.0]

    @test preds[1] ≈ p1 atol = 1e3
    @test preds[2] ≈ p2 atol = 1e3
    @test preds[3] ≈ p3 atol = 1e3
    @test preds[4] ≈ p4 atol = 1e3
end


@safetestset "BayesianModel" begin
    using CategorizationModels
    using Test 
    
    parms = (μk = 80.0,
             σk = 20.0,
             μs = 90.0,
             σs = 20.0,
             υ_ks_k = 5.0,
             υ_sk_s = 4.0,
             λ_ks_k = .50,
             λ_sk_s = .50)
    
    n_states = 12
    n_options = 6

    model = BayesianModel(;parms..., n_states)

    preds = generate_predictions(model, n_options)

    # note these have been transposed from original
    p1 = [  -0.00400   0.00700   0.03000   0.01700   0.00400   0.00100;
             0.00300  -0.00800   0.00600   0.04600   0.02600   0.00700;
             0.00100   0.00200  -0.01100   0.01000   0.06700   0.04700;
            -0.00100   0.00200   0.00400  -0.01500   0.02000   0.15800;
            -0.00000  -0.00100   0.00300   0.00900   0.01700   0.21300;
            -0.00000  -0.00000   0.00100   0.02800   0.12800   0.18300]

    p2 = [  -0.00400   0.00300   0.00100  -0.00100  -0.00000  -0.00000;
            0.00700  -0.00800   0.00200   0.00200  -0.00100  -0.00000;
            0.03000   0.00600  -0.01100   0.00400   0.00300   0.00100;
            0.01700   0.04600   0.01000  -0.01500   0.00900   0.02800;
            0.00400   0.02600   0.06700   0.02000   0.01700   0.12800;
            0.00100   0.00700   0.04700   0.15800   0.21300   0.18300]

    p3 = [  -0.00400   0.01800   0.02300   0.00800   0.00100   0.00000;
             0.00300  -0.01100   0.02700   0.03700   0.01200   0.00200;
            -0.00100   0.00500  -0.01600   0.04200   0.05600   0.02100;
            -0.00000  -0.00100   0.00800  -0.02400   0.06500   0.11600;
            -0.00000  -0.00100  -0.00100   0.01300  -0.02000   0.25500;
            -0.00000  -0.00000  -0.00000   0.00700   0.10300   0.25600]

    p4 = [  -0.00400   0.00300  -0.00100  -0.00000  -0.00000  -0.00000;
             0.01800  -0.01100   0.00500  -0.00100  -0.00100  -0.00000;
             0.02300   0.02700  -0.01600   0.00800  -0.00100  -0.00000;
             0.00800   0.03700   0.04200  -0.02400   0.01300   0.00700;
             0.00100   0.01200   0.05600   0.06500  -0.02000   0.10300;
             0.00000   0.00200   0.02100   0.11600   0.25500   0.25600]

    @test preds[1] ≈ p1 atol = 1e3
    @test preds[2] ≈ p2 atol = 1e3
    @test preds[3] ≈ p3 atol = 1e3
    @test preds[4] ≈ p4 atol = 1e3
end

@safetestset "make_intensity_matrix" begin 
    @safetestset "BayesianModel" begin 
        using CategorizationModels
        using CategorizationModels: make_intensity_matrix
        using Test 
    
        parms = (μk = 80.0,
                σk = 20.0,
                μs = 90.0,
                σs = 20.0,
                υ_ks_k = 5.0,
                υ_sk_s = 4.0,
                λ_ks_k = .50,
                λ_sk_s = .50)
    
        model = BayesianModel(;parms..., n_states=5)
    
        x = [ -1.5   0.5   0.0   0.0   0.0;
               1.5  -2.0   0.5   0.0   0.0;
               0.0   1.5  -2.0   0.5   0.0;
               0.0   0.0   1.5  -2.0   0.5;
               0.0   0.0   0.0   1.5  -0.5]
    
        v = make_intensity_matrix(model, 5, .5)
    
        @test x ≈ v atol = 1e-3
    
        for i ∈ 1:30
            θ = rand() * 5
            n = rand(3:100)
            v = make_intensity_matrix(model, n, θ)
            sums = sum(v, dims=1)
            @test sums ≈ zeros(1,n) atol = 1e-8
        end
    end

    @safetestset "MarkovModel" begin 
        using CategorizationModels
        using CategorizationModels: make_intensity_matrix
        using Test 
    
        parms = (μ = 80,
                σ = 20,
                υ_k_k = 2.0,
                υ_s_k = 1.0,
                υ_k_s = 1.0,
                υ_s_s = 2.0,
                λ_k_k = .5,
                λ_s_k = .5,
                λ_k_s = .5,
                λ_s_s = .5,
                n_states=5)
    
        model = MarkovModel(;parms...)
    
        x = [  -1   3   0   0   0;
                1  -4   3   0   0;
                0   1  -4   3   0;
                0   0   1  -4   3;
                0   0   0   1  -3]
    
        v = make_intensity_matrix(model, 5, 3)
    
        @test x ≈ v atol = 1e-3
    
        for i ∈ 1:30
            θ = rand() * 5
            n = rand(3:100)
            v = make_intensity_matrix(model, n, θ)
            sums = sum(v, dims=1)
            @test sums ≈ zeros(1,n) atol = 1e-8
        end
    end
end

@safetestset "rand" begin
    @safetestset "BayesianModel" begin
        using CategorizationModels
        using Test 
        
        parms = (μk = 80.0,
                 σk = 20.0,
                 μs = 90.0,
                 σs = 20.0,
                 υ_ks_k = 5.0,
                 υ_sk_s = 4.0,
                 λ_ks_k = .50,
                 λ_sk_s = .50)
        
        n_states = 12
        n_options = 6
        n_trials = 100
    
        model = BayesianModel(;parms..., n_states)
    
        preds = generate_predictions(model, n_options)

        data = rand(model, preds, n_trials)
        @test isa(data, Vector{Vector{Tuple{Int64, Int64}}})
        @test length(data) == 4
        @test all(length.(data) .== n_trials)

        data1 = rand(model, preds[1], n_trials)
        @test isa(data1, Vector{Tuple{Int64, Int64}})
        @test length(data1) == n_trials
    end

    @safetestset "RationalModel" begin
        using CategorizationModels
        using Test 
        
        parms = (μk = 80.0,
                 σk = 20.0,
                 μs = 90.0,
                 σs = 20.0)
        
        n_states = 12
        n_options = 6
        n_trials = 100
    
        model = RationalModel(;parms..., n_states)
    
        preds = generate_predictions(model, n_options)

        data = rand(model, preds, n_trials)
        @test isa(data, Vector{Vector{Tuple{Int64, Int64}}})
        @test length(data) == 4
        @test all(length.(data) .== n_trials)

        data1 = rand(model, preds[1], n_trials)
        @test isa(data1, Vector{Tuple{Int64, Int64}})
        @test length(data1) == n_trials
    end
end

@safetestset "logpdf" begin 

    @safetestset "RationalModel" begin 
        using CategorizationModels
        using Random
        using Test 

        Random.seed!(985)

        μk = 80.0
        σk = 20.0
        μs = 90.0
        σs = 20.0
        n_states = 96

        parms = (;μk,
                σk,
                μs,
                σs,
                n_states)

        n_options = 6
        n_trials = 10_000

        model = RationalModel(;parms...)

        preds =  generate_predictions(model, n_options)
        data = rand(model, preds, n_trials)
        μks = range(μk * .8, μk * 1.2, length = 100)
        LLs = map(μk -> sumlogpdf(RationalModel(;parms..., μk), n_options, data), μks)
        _,idx = findmax(LLs)
        @test μks[idx] ≈ μk rtol = .05

        σks = range(σk * .8, σk * 1.2, length = 100)
        LLs = map(σk -> sumlogpdf(RationalModel(;parms..., σk), n_options, data), σks)
        _,idx = findmax(LLs)
        @test σks[idx] ≈ σk rtol = .05

        μss = range(μs * .8, μs * 1.2, length = 100)
        LLs = map(μs -> sumlogpdf(RationalModel(;parms..., μs), n_options, data), μss)
        _,idx = findmax(LLs)
        @test μss[idx] ≈ μs rtol = .05

        σss = range(σs * .8, σs * 1.2, length = 100)
        LLs = map(σs -> sumlogpdf(RationalModel(;parms..., σs), n_options, data), σss)
        _,idx = findmax(LLs)
        @test σss[idx] ≈ σs rtol = .05
    end

    @safetestset "BayesianModel" begin 
        using CategorizationModels
        using Random
        using Test 

        Random.seed!(985)

        μk = 80.0
        σk = 20.0
        μs = 90.0
        σs = 20.0
        υ_ks_k = 5.0
        υ_sk_s = 4.0
        λ_ks_k = .50
        λ_sk_s = .50
        n_states = 96

        parms = (;μk,
                σk,
                μs,
                σs,
                υ_ks_k,
                υ_sk_s,
                λ_ks_k,
                λ_sk_s,
                n_states)

        n_options = 6
        n_trials = 10_000

        model = BayesianModel(;parms...)

        preds =  generate_predictions(model, n_options)
        data = rand(model, preds, n_trials)
        μks = range(μk * .8, μk * 1.2, length = 100)
        LLs = map(μk -> sumlogpdf(BayesianModel(;parms..., μk), n_options, data), μks)
        _,idx = findmax(LLs)
        @test μks[idx] ≈ μk rtol = .05

        σks = range(σk * .8, σk * 1.2, length = 100)
        LLs = map(σk -> sumlogpdf(BayesianModel(;parms..., σk), n_options, data), σks)
        _,idx = findmax(LLs)
        @test σks[idx] ≈ σk rtol = .05

        μss = range(μs * .8, μs * 1.2, length = 100)
        LLs = map(μs -> sumlogpdf(BayesianModel(;parms..., μs), n_options, data), μss)
        _,idx = findmax(LLs)
        @test μss[idx] ≈ μs rtol = .05

        σss = range(σs * .8, σs * 1.2, length = 100)
        LLs = map(σs -> sumlogpdf(BayesianModel(;parms..., σs), n_options, data), σss)
        _,idx = findmax(LLs)
        @test σss[idx] ≈ σs rtol = .05

        υ_ks_ks = range(υ_ks_k * .8, υ_ks_k * 1.2, length = 100)
        LLs = map(υ_ks_k -> sumlogpdf(BayesianModel(;parms..., υ_ks_k), n_options, data), υ_ks_ks)
        _,idx = findmax(LLs)
        @test υ_ks_ks[idx] ≈ υ_ks_k rtol = .05

        υ_sk_ss = range(υ_sk_s * .8, υ_sk_s * 1.2, length = 100)
        LLs = map(υ_sk_s -> sumlogpdf(BayesianModel(;parms..., υ_sk_s), n_options, data), υ_sk_ss)
        _,idx = findmax(LLs)
        @test υ_sk_ss[idx] ≈ υ_sk_s rtol = .05

        λ_ks_ks = range(λ_ks_k * .8, λ_ks_k * 1.2, length = 100)
        LLs = map(λ_ks_k -> sumlogpdf(BayesianModel(;parms..., λ_ks_k), n_options, data), λ_ks_ks)
        _,idx = findmax(LLs)
        @test λ_ks_ks[idx] ≈ λ_ks_k rtol = .05

        λ_sk_ss = range(λ_sk_s * .8, λ_sk_s * 1.2, length = 100)
        LLs = map(λ_sk_s -> sumlogpdf(BayesianModel(;parms..., λ_sk_s), n_options, data), λ_sk_ss)
        _,idx = findmax(LLs)
        @test λ_sk_ss[idx] ≈ λ_sk_s rtol = .05
    end

end