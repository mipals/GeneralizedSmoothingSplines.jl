###
using Test
using GeneralizedSmoothingSplines
using DataFrames
using CSV
using MLJ

# Load Data
include("motor_preds.jl") # Load pre-compute solution
df = DataFrame(CSV.File("datasets/motor.csv"))
y,X = unpack(df, ==(:accel), ==(:times))
# Define Spline + fit
spl = SmoothingSpline(λ = 1e-3)
mach = machine(spl,X,y)
tune!(mach)
# Extract fit and compute fit
interp = predict(mach)
# Compute prediction in interval
a,b = extrema(X)
Xnew = collect(a+1:1:b-1)

##
shapes = GeneralizedSmoothingSplines.shapes
for (index,shape_restriction) in enumerate(shapes)
    spl = SmoothingSpline(λ = 1e-5,shape_restrictions=(shape_restriction,))
    mach = machine(spl,X,y) 
    tune!(mach)
    interp = predict(mach)
    preds  = predict(mach,Xnew)
    # Tests
    @test isapprox(interp, Interps[:,index], atol=1e-4)
    @test isapprox(preds, Preds[:,index], atol=1e-4)
end
