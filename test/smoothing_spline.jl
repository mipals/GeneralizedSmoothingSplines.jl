# Load Data
include("motor_preds.jl") # Load pre-compute solution
df = DataFrame(CSV.File("datasets/motor.csv"))
y,X = unpack(df, ==(:accel), ==(:times))
# Define Spline + fit
spl = SmoothingSpline()
mach = machine(spl,X,y)
fit!(mach;verbosity=0)
# Extract fit and compute fit
c = mach.fitresult[:c]
d = mach.fitresult[:d]
K = mach.fitresult[:K]
H = mach.fitresult[:H]
interp = K*c + H*d
# Compute prediction in interval
a,b = extrema(X)
Xnew = collect(a+1:1:b-1)
preds = predict(mach,Xnew)
# Tests
@test interp ≈ motor_interp
@test preds ≈ motor_preds

# Used for testing
# scatter(X, y, ms=2, label="Observations", xlabel="time (s)", ylabel="Acceleration (m/s^2)") 
# plot!(X,interp, label="Spline Fit")
# scatter!(Xnew,preds)
