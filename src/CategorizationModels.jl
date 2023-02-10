module CategorizationModels

    using ConcreteStructs
    using Distributions
    using LinearAlgebra

    import Distributions: ContinuousUnivariateDistribution
    import Distributions: rand
    import Distributions: logpdf
        
    export BayesianModel
    export RationalModel

    export generate_predictions
    export logpdf 
    export rand
    
    include("common.jl")
    include("rational.jl")
    include("bayesian.jl")
end
