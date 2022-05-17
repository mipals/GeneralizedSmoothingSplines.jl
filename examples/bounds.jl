# It is possible to add multiple shape restrictions at once
using Plots, MLJ, LinearAlgebra, GeneralizedSmoothingSplines
n       = 50        # number of samples
sigma   = 0.2       # noise standard deviation
a,b     = -π,π      # interval [a,b]
delta   = b - a     # Interval width

# Defing data
X = a .+ sort(rand(n))*delta
f(t) = 1.0 ./ (1.0 .+ exp.(-5.0*t))
y = f(X) + sigma*randn(length(X))

m = 10
Xnew = a .+ sort(rand(m))*delta

# Fitting splines
uncon_spl = SmoothingSpline()
uncon_mach = machine(uncon_spl,X,y) 
tune!(uncon_mach)
fit!(uncon_mach)
uncon_preds = predict(uncon_mach,Xnew)

bounds = (0.0,1,0.0)
spl = SmoothingSpline(shape_restrictions=(:lowerbound,:upperbound,:increasing),bounds=bounds)
mach = machine(spl,X,y) 
tune!(mach)
fit!(mach)
preds = predict(mach,Xnew)

scatter(X,y, label="Data",legend=:topleft,ms=2)
plot!(a:delta/100:b,f(a:delta/100:b), label="True Solution",lw=2)
plot!(X, uncon_mach.fitresult[:fit],ls=:dash,label="Unconstrained")
scatter!(Xnew,uncon_preds,label="Unconstrained-Preds")
plot!(X, mach.fitresult[:fit],ls=:dash,label="Lower/Upper/Increasing")
scatter!(Xnew,preds,label="Constrained-Preds")
