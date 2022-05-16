module GeneralizedSmoothingSplines

#==========================================================================================
                                Using Relevant Packages
==========================================================================================#
using DataFrames
using LinearAlgebra
using SymSemiseparableMatrices
using Optim
using MLJ
using SpecialFunctions
using MLJModelInterface
import JuMP
import Ipopt
const MMI = MLJModelInterface

#==========================================================================================
                                Including code files
==========================================================================================#
include("SmoothingSpline.jl")

#==========================================================================================
                            Exporting user-faced functions 
==========================================================================================#
export SmoothingSpline, tune!

end
