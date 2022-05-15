# GeneralizedSmoothingSplines

[![Build Status](https://github.com/mipals/GeneralizedSmoothingSplines.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mipals/GeneralizedSmoothingSplines.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mipals/GeneralizedSmoothingSplines.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mipals/GeneralizedSmoothingSplines.jl)

A (experimental) Julia package for nonparametric regression with Smoothing Splines of nth order. The implementation uses so-called extended generator representable semiseparable matrices (EGRSS-matrices) as described in [1]. As such the implementation uses heavily the functionalities of the package [SymSemiseparableMatrices.jl](https://github.com/mipals/SymSemiseparableMatrices.jl).

The package (tries) to be compatible with the style of [MLJ](https://github.com/alan-turing-institute/MLJ.jl) framework.

## Example
See *examples/forrest.jl*

## TODO
Would be nice to have some tests that does not depend on CSV.


## References
[1] M. S. Andersen and T. Chen, “Smoothing Splines and Rank Structured Matrices: Revisiting the Spline Kernel,” SIAM Journal on Matrix Analysis and Applications, 2020.

## Related Packages
* [SmoothingSplines.jl](https://github.com/nignatiadis/SmoothingSplines.jl): Follows the style of R's `smooth.spline`. Does not have an automatic selection of λ. 
