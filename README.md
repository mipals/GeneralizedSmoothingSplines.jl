# GeneralizedSmoothingSplines.jl

[![Build Status](https://github.com/mipals/GeneralizedSmoothingSplines.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mipals/GeneralizedSmoothingSplines.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mipals/GeneralizedSmoothingSplines.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mipals/GeneralizedSmoothingSplines.jl)

A (experimental) Julia package for nonparametric regression with Smoothing Splines of nth order. The implementation uses so-called extended generator representable semiseparable matrices (EGRSS-matrices) as described in [1]. As such the implementation uses heavily the functionalities of the package [SymSemiseparableMatrices.jl](https://github.com/mipals/SymSemiseparableMatrices.jl).

In additional to the standard spline fit the package also includes the possibility constraining the spline to be positive/negative/increasing/decreasing/convex/convex. These ideas follow that of [2], but also does not assume equidistant spacing. Note that the implementation is *sloppy* in that it transform everything to dense matrices and solves a Quadratic Program (QP). This could potentially be done in linear complexity, but requires a QP-solver that supports structured matrices.

The package aims to be compatible with the style of the [MLJ](https://github.com/alan-turing-institute/MLJ.jl) framework.

## Example
* General usage *examples/forrester.jl* & *examples/motor.jl*.
* Shape restricted curves *examples/constrained.jl*.

## Related Packages
* [SmoothingSplines.jl](https://github.com/nignatiadis/SmoothingSplines.jl): Follows the style of R's `smooth.spline`. Does not have an automatic selection of λ. Does not include shape constraints.

## References
[1] M. S. Andersen and T. Chen, “Smoothing Splines and Rank Structured Matrices: Revisiting the Spline Kernel,” SIAM Journal on Matrix Analysis and Applications, 2020.

[2] Helene Charlotte Rytgaard, “Statistical models for robust spline smoothing”. MA thesis. University of Copenhagen, 2016.
