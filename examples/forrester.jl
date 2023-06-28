using Plots
using Printf
using GeneralizedSmoothingSplines
using MLJ

n       = 100        # Number of samples (large to check for scalability)
sigma   = 2.0          # Noise standard deviation
a,b     = -0.2,1.1     # Interval [a,b]
delta   = b - a        # Interval Width

X = Float32.(a .+ sort(rand(n))*delta)
# FORRESTER ET AL. (2008) FUNCTION
f(t) = (6*t .- 2).^2 .* sin.(12*t .- 4)
y = f(X) + convert.(eltype(X),sigma*randn(length(X)))

# Defining spline with roughness penalty λ
spl = SmoothingSpline(lambda = 1f-3,sigma=1f0) # Defining spline with roughness penalty λ
preds = predict(spl,X,y)  # Fiting spline to the data

# Plotting
scatter(X, y, ms=2, label="Observations", xlims=(a,b), xlabel="t")
plot!(a:delta/n:b, f(a:delta/n:b), label="f(t)",color=:blue, ls=:dash,lw=2)
plot!(X,preds,label="Spline Fit",color=:red,lw=2)

# Computing optimal λ with respect to the generalized marginal likelihood
mach = machine(spl,X,y)         # Combining spline with data
tune!(mach;show_trace=true)     # Optimizing

# Evaluating between knots
m = 100
Xnew = convert.(eltype(X),sort(a .+ sort(rand(m))*delta))
new_preds = predict(mach,Xnew)

# Extracting predictions of the spline
preds = predict(mach)

scatter(X, y, ms=2, label="Observations", xlims=(a,b), xlabel="t",legend=:topleft)
plot!(X, preds, label="fit",color=:red, lw=2)
scatter!(Xnew,new_preds,color=:green,label="Predictions")
plot!(a:delta/n:b, f(a:delta/n:b), label="f(t)",color=:blue, ls=:dash,lw=2)
