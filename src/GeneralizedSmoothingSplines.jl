module GeneralizedSmoothingSplines

# Using Relevant Packages
using DataFrames
using LinearAlgebra
using SymSemiseparableMatrices
using Optim
using SpecialFunctions
using MLJModelInterface
const MMI = MLJModelInterface


# Defining a `fit` and `predict` for SmoothingSplines
include("SmoothingSpline.jl")

export SmoothingSpline


end
