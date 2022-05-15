using Test
using GeneralizedSmoothingSplines
using DataFrames
using CSV
using MLJ

@testset "GeneralizedSmoothingSplines.jl" begin
    include("smoothing_spline.jl")
end
