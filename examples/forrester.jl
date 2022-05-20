using Plots
using Printf
using DataFrames
using SpecialFunctions
using GeneralizedSmoothingSplines
using MLJ
using RDatasets

n       = 10000        # Number of samples (large to check for scalability)
sigma   = 2.0          # noise standard deviation
a,b     = -0.2,1.1     # interval [a,b]
delta   = b - a        # Interval Width

X = a .+ sort(rand(n))*delta;
# FORRESTER ET AL. (2008) FUNCTION
f(t) = (6.0*t .- 2.0).^2 .* sin.(12.0*t .- 4.0)
y = f(X) + sigma*randn(length(X))

# Defining spline with roughness penalty λ 
spl = SmoothingSpline(lambda = 1e-5) # Defining spline with roughness penalty λ 
preds = predict(spl,X,y)  # Fiting spline to the data

# Plotting
scatter(X, y, ms=2, label="Observations", xlims=(a,b), xlabel="t") 
plot!(a:delta/n:b, f(a:delta/n:b), label="f(t)",color=:blue, ls=:dash,lw=2)
plot!(X,preds,label="Interpolation",color=:red,lw=2)


# Computing optimal λ with respect to the generalized marginal likelihood
mach = machine(spl,X,y) # Combining spline with data
tune!(mach;show_trace=true)         # Optimizing
fit!(mach)

# Evaluating between knots
m = 100
Xnew = sort(a .+ sort(rand(m))*delta)
new_preds = predict(mach,Xnew)

# Extracting predictions of the spline
preds = predict(mach)

scatter(X, y, ms=2, label="Observations", xlims=(a,b), xlabel="t",legend=:topleft) 
plot!(X, preds, label="fit",color=:red, lw=2)
scatter!(Xnew,new_preds,color=:green,label="Predictions")
plot!(a:delta/n:b, f(a:delta/n:b), label="f(t)",color=:blue, ls=:dash,lw=2)


# p = spl.p
# L = cholesky(SymSemiseparableMatrix(Ut*delta^(2p-1),Vt), n*smoothingspline.lambda)
# H = compute_H(t,p)
# B = L\H
# F = qr(B)
# S = L'\Matrix(F.Q)

# dH = zeros(n)
# ei = zeros(n)
# for i = n:-1:1
#   ei[i] = 1.0
#   dH[i] = norm((L\ei)[i:end])^2
#   ei[i] = 0.0
# end

# diagH = 1.0 .- n*smoothingspline.lambda*(dH - sum(S.*S,dims=2))
# α = 0.05
# β = sqrt(2.0)*erfinv(1.0 - α)
# sigma = sqrt(n*smoothingspline.lambda*dot(y,c)/(n-p))

# scatter(t, y, ms=2, label="Observations", xlims=(a,b), xlabel="t", legend=false);
# plot!(a:delta/n:b, f(a:delta/n:b), label="f(t)", ls=:dash,color=:black,lw=1, ribbon=β*sigma, alpha=0.8, fillalpha=0.1)
# plot!(t,Σ*c + H*d, label="fit",color=:red,lw=2,
#  title="λ="*@sprintf("%.3e", smoothingspline.lambda)*", "*  "σ="* @sprintf("%.3e", sigma), 
#  ribbon=β*sigma*sqrt.(diagH), fillalpha=0.5)
