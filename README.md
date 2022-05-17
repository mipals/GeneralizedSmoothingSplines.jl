# GeneralizedSmoothingSplines.jl

[![Build Status](https://github.com/mipals/GeneralizedSmoothingSplines.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mipals/GeneralizedSmoothingSplines.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mipals/GeneralizedSmoothingSplines.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mipals/GeneralizedSmoothingSplines.jl)

A (experimental) Julia package for nonparametric regression with Smoothing Splines of nth order. The implementation uses so-called extended generator representable semiseparable matrices (EGRSS-matrices) as described in [1]. As such the implementation uses heavily the functionalities of the package [SymSemiseparableMatrices.jl](https://github.com/mipals/SymSemiseparableMatrices.jl).

In additional to the standard spline fit the package also includes the possibility constraining the spline to be positive/negative/increasing/decreasing/convex/convex. These ideas follow that of [2], but also does not assume equidistant spacing. Note that the implementation is *sloppy* in that it transform everything to dense matrices and solves a Quadratic Program (QP). As a result it is recommended to only use shape restrictions in the case of few data points, which is usually the case when shape restrictions is required.

The package aims to be compatible with the style of the [MLJ](https://github.com/alan-turing-institute/MLJ.jl) framework.

## Usage
```julia
using Plots, MLJ, LinearAlgebra, GeneralizedSmoothingSplines
n       = 100       # number of samples
sigma   = 0.2       # noise standard deviation
a,b     = -π,π      # interval [a,b]
delta   = b - a     # Interval width

# Simulating Data
X = a .+ sort(rand(n))*delta
f(t) = 1.0 ./ (1.0 .+ exp.(-5.0*t))
y = f(X) + sigma*randn(length(X))

# Creating 10 data points to predict
Xnew = a .+ sort(rand(10))*delta

# Fitting Smoothing Spline
spl = SmoothingSpline()
mach = machine(spl,X,y) 
tune!(mach)
preds = predict(mach,Xnew)

# Fitting Shape Restricted Spline
bounds = (0.0,1,0.0) # Min/Max/Strictly increasing
res_spl = SmoothingSpline(shape_restrictions=(:lowerbound,:upperbound,:increasing),bounds=bounds)
res_mach = machine(res_spl,X,y) 
tune!(res_mach)
res_preds = predict(res_mach,Xnew)

# Plotting Results
scatter(X,y, label="Data",legend=:topleft,ms=2)
plot!(a:delta/100:b,f(a:delta/100:b), label="True Solution",ls=:dash)
plot!(X, mach.fitresult[:fit],label="Unconstrained")
scatter!(Xnew,preds,label="Unconstrained-Preds")
plot!(X, res_mach.fitresult[:fit],label="Lower/Upper/Increasing")
scatter!(Xnew,res_preds,label="Constrained-Preds")
```
![](readme_plot.png)


## Examples
* General usage *examples/forrester.jl* & *examples/motor.jl*.
* Shape restricted curves *examples/constrained.jl*, *examples/mixing_constraints.jl* & *examples/bounds.jl*.

## Related Packages
* [SmoothingSplines.jl](https://github.com/nignatiadis/SmoothingSplines.jl): Follows the style of R's `smooth.spline`. Does not have an automatic selection of λ. Does not include shape constraints.

## References
[1] M. S. Andersen and T. Chen, “Smoothing Splines and Rank Structured Matrices: Revisiting the Spline Kernel,” SIAM Journal on Matrix Analysis and Applications, 2020.

[2] Helene Charlotte Rytgaard, “Statistical models for robust spline smoothing”. MA thesis. University of Copenhagen, 2016.
