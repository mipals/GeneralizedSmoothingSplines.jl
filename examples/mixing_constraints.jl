# It is possible to add multiple shape restrictions at once
using Plots, MLJ, LinearAlgebra, GeneralizedSmoothingSplines
n       = 25        # number of samples
sigma   = 0.2       # noise standard deviation
a,b     = -π,π      # interval [a,b]
delta   = b - a     # Interval width

# Defing data
X = a .+ sort(rand(n))*delta
f(t) = sin.(t)
y = f(X) + sigma*randn(length(X))

# Fitting splines
con_spl = SmoothingSpline(λ = 1e-5,shape_restrictions=(:convex,))
con_mach = machine(con_spl,X,y) 
tune!(con_mach)
pos_spl = SmoothingSpline(λ = 1e-5,shape_restrictions=(:lowerbound,))
pos_mach = machine(pos_spl,X,y) 
tune!(pos_mach)
conpos_spl = SmoothingSpline(λ = 1e-5,shape_restrictions=(:convex,:lowerbound))
conpos_mach = machine(conpos_spl,X,y) 
tune!(conpos_mach)
scatter(X,y, label="Data",legend=:topleft)
plot!(X,f(X), label="True Solution",lw=2)
plot!(X, con_mach.fitresult[:fit],ls=:dash,label="Convex")
plot!(X, pos_mach.fitresult[:fit],ls=:dash,label="Positive")
plot!(X, conpos_mach.fitresult[:fit],ls=:dash,label="Convex-positive")
