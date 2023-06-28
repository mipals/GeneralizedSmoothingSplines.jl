# Loading Relevant Packages
using GeneralizedSmoothingSplines
using DataFrames
using CSV
using MLJ
# Load Data
df = DataFrame(CSV.File("test/datasets/motor.csv"))
y,X = unpack(df, ==(:accel), ==(:times))
# Define Spline + fit
spl = SmoothingSpline(p=2) # p is the order of the penalized derivative (p=2, Cubic splines)
mach = machine(spl,X,y)
tune!(mach)
# Extract fit and compute fit
interp = predict(mach)
# Compute prediction in interval
a,b = extrema(X)
Xnew = collect(a+1:1:b-1)
preds = predict(mach,Xnew)
# Plotting Results
scatter(X, y, ms=2, label="Observations", xlabel="Time (s)", ylabel="Acceleration (m/s^2)")
plot!(X,interp, label="Spline Fit")
scatter!(Xnew,preds)
