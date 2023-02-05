using SafeTestsets

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
