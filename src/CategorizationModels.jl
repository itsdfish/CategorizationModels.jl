module CategorizationModels

    using ConcreteStructs
    using Distributions
    using LinearAlgebra
    
    export RationalModel

    export generate_predictions


    include("structs.jl")
    include("functions.jl")
end
