# CategorizationModels

This repo contains a Julia implementation of category learning models implemented in Epping & Busemeyer (2023). The models are used to evaluate order effects in category judgment ratings. 

# API

The package provides an API for generating predictions, generating random data, and computing the logpdf of a model. 

## Models

All models are a subtype of `Model <: ContinuousUnivariateDistribution`. The models include

- `RationalModel`

- `BayesianModel`

- `MarkovModel`

- `QuantumModel`

## Methods

This package impliments the following three methods

- `generate_predictions`: generates a vector of choices for each condition

- `logpdf`: computes logpdf of data for choices in one condition or all conditions

- `sumlogpdf`: returns the sum of logpdfs of data for choices in one condition or all conditions

- `rand`: returns a vector of choices from the model for each condition 

# Internal Methods 

In addition, to the API, there are several internal methods that one may use for extending the package to new models. The generic versions of these methods can be found in `src/common.jl`. The methods include:

- `compute_initial_state`: computes the initial state probability vector based on a normal distribution across states. 

- `make_joint_dist`: computes an n × n matrix representing the joint rating distribution of categories for a given order. 

- `make_projector`: creates a projector matrix for computing the probability of a given category rating. 

# Help

String docs can be accessed via the REPL by switching to help model with `?`, e.g.,

```julia 

help?> RationalModel
search: RationalModel

  RationalModel  <: Model

  A model object for the rational model.

  Field Names
  ≡≡≡≡≡≡≡≡≡≡≡≡≡

    •  μk: the mean of the initial state distribution for stimulus k when k is evaluated first

    •  μs: the mean of the initial state distribution for stimulus s when s is evaluated first

    •  σk: the standard deviation of the initial state distribution for stimulus k when it is evaluated first

    •  σs: the standard deviation of the initial state distribution for stimulus s when it is evaluated first

    •  n_states: the number of evidence states

```

# Example 

```julia
using CategorizationModels
using Random 

Random.seed!(5)

# model predictions 
parms = (μ = 80.0,
        σ = 20.0,
        υ_k_k = 1.0,
        υ_s_k = 2.0,
        υ_k_s = 3.0,
        υ_s_s = 4.0,
        λ_k_k = .2,
        λ_s_k = .3,
        λ_k_s = .4,
        λ_s_s = .5)

# number of evidence states 
n_states = 96

# number of rating options 
n_options = 6

# create a model object 
model = QuantumModel(;parms..., n_states)

# generate predictions for all conditions 
preds = generate_predictions(model, n_options)

# generate 100 trials of data per condition 
data = rand(model, preds, 100)

# compute the logpdf of each data point
LLs = logpdf(model, preds, data)
```
# References

Epping, G. P., & Busemeyer, J. R. (2023). Using diverging predictions from classical and quantum models to dissociate between categorization systems. Journal of Mathematical Psychology, 112, 102738.