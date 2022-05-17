###
using Plots, MLJ, LinearAlgebra, GeneralizedSmoothingSplines
n       = 55        # number of samples
sigma   = 0.2       # noise standard deviation
a,b     = 0.0,5.0   # interval [a,b]
delta   = b - a     # Interval width

X = a .+ sort(rand(n))*delta
f(t) = exp.(-t)
y = f(X) + sigma*randn(length(X))

plt = scatter(X,y,label="Data")
plot!(a:delta/100:b,f(a:delta/100:b), label="True Solution",lw=2)
for shape_restriction in GeneralizedSmoothingSplines.shapes
    spl = SmoothingSpline(λ = 1e-5,shape_restrictions=(shape_restriction,))
    mach = machine(spl,X,y) 
    tune!(mach)
    plot!(plt, X,mach.fitresult[:fit],label=String(shape_restriction),ls=:dash)
end
plt
