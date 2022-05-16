###
using Plots, MLJ, LinearAlgebra, GeneralizedSmoothingSplines
n       = 55            # number of samples
sigma   = 0.1          # noise standard deviation
a,b     = 0.0,5.0     # interval [a,b]
delta   = b - a        # Interval Width

X = a .+ sort(rand(n))*delta;
# FORRESTER ET AL. (2008) FUNCTION
f(t) = exp.(-X)
y = f(X) + sigma*randn(length(X))
scatter(X,y)
plot!(X,f(X))

# Positive
pos_spl = SmoothingSpline(λ = 1e-5,shape_restriction=:positive)
pos_mach = machine(pos_spl,X,y) 
tune!(pos_mach)
pos_preds = pos_mach.fitresult[:fit]
# Negative
neg_spl = SmoothingSpline(λ = 1e-5,shape_restriction=:negative)
neg_mach = machine(neg_spl,X,y) 
tune!(neg_mach)
neg_preds = neg_mach.fitresult[:fit]
# Increasing
inc_spl = SmoothingSpline(λ = 1e-5,shape_restriction=:increasing)
inc_mach = machine(inc_spl,X,y) 
tune!(inc_mach)
inc_preds = inc_mach.fitresult[:fit]
# Decreasing
dec_spl = SmoothingSpline(λ = 1e-5,shape_restriction=:decreasing)
dec_mach = machine(dec_spl,X,y) 
tune!(dec_mach)
dec_preds = dec_mach.fitresult[:fit]
# Convex
convex_spl = SmoothingSpline(λ = 1e-5,shape_restriction=:convex)
convex_mach = machine(convex_spl,X,y) 
tune!(convex_mach)
convex_preds = convex_mach.fitresult[:fit]
# Concave
concave_spl = SmoothingSpline(λ = 1e-5,shape_restriction=:concave)
concave_mach = machine(concave_spl,X,y) 
tune!(concave_mach)
concave_preds = concave_mach.fitresult[:fit]

# Plotting everyhing together
scatter(X,y)
plot!(X,f(X), label="True Solution",lw=2)
plot!(X,pos_preds,label="Positive",ls=:dash)
plot!(X,neg_preds,label="Negative",ls=:dash)
plot!(X,inc_preds,label="Increasing",ls=:dash)
plot!(X,dec_preds,label="Decreasing",ls=:dash)
plot!(X,convex_preds,label="Convex",ls=:dash)
plot!(X,concave_preds,label="Concave",ls=:dash)
