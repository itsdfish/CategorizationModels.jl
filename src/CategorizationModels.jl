module CategorizationModels

    using ConcreteStructs
    using Distributions
    using LinearAlgebra

    import Distributions: ContinuousUnivariateDistribution
    import Distributions: rand
    import Distributions: logpdf
        
    export BayesianModel
    export MarkovModel
    export QuantumModel
    export RationalModel

    export generate_predictions
    export logpdf 
    export rand
    export sumlogpdf
    
    include("common.jl")
    include("rational.jl")
    include("bayesian.jl")
    include("markov.jl")
    include("quantum.jl")
end
