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
preds = predict(mach,Xnew)
# Tests
@test isapprox(interp, motor_interp, atol=1e-4)
@test isapprox(preds, motor_preds, atol=1e-4)

# Used for testing
# scatter(X, y, ms=2, label="Observations", xlabel="time (s)", ylabel="Acceleration (m/s^2)") 
# plot!(X,interp, label="Spline Fit")
# scatter!(Xnew,preds)
