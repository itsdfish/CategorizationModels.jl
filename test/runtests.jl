using SafeTestsets

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
    using CategorizationModels
    using CategorizationModels: make_intensity_matrix
    using Test 

    x = [ -1.5   0.5   0.0   0.0   0.0;
           1.5  -2.0   0.5   0.0   0.0;
           0.0   1.5  -2.0   0.5   0.0;
           0.0   0.0   1.5  -2.0   0.5;
           0.0   0.0   0.0   1.5  -0.5]

    v = make_intensity_matrix(5, .5)

    @test x ≈ v atol = 1e-3
end