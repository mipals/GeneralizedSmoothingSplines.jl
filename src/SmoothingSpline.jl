#==========================================================================================
                            Defining spline relevantfunctions
==========================================================================================#
"""
    compute_H(t,p)

Returns `H` with the ith columns equal to the ith Taylor term evaluated on `t`.
"""
function compute_H(t,p)
    n = length(t)
    H = ones(eltype(t), n, p )
    for i = 2:p
        H[:,i] = t.^(i - 1)/factorial(i - 1)
    end
    return H
end
"""
    compute_coefficients(Σ,H,y,λ)

Solves the quadratic program of a smoothing splines.\\
Returns coefficients `c` and `d` as well as log(generalized marginal likelihood).\\
See [section 2.7 of GP for ML](http://gaussianprocess.org/gpml/chapters/RW.pdf)
"""
function compute_coefficients(Σ,H,y,λ)
    n, p = size(H)
    L = cholesky(Σ, n*λ)
    v = L'\(L\y)
    A = (H'*(L'\(L\H)))
    d = A\(H'*v)
    c = L'\(L\(y - H*d))
    log_gml = log(dot(y,c))  + 2.0*logdet(L)/(n - p) + logdet(A)/(n - p)
    return c, d, log_gml
end
"""
    log_gml(v,K,H,y)

Returns the log of the generalized marginal likelihood.\\
See [section 2.7 of GP for ML](http://gaussianprocess.org/gpml/chapters/RW.pdf)
"""
function log_gml(v,K,H,y)
    _,_,log_gml = compute_coefficients(K, H, y, 10.0^v)
    return log_gml
end


#==========================================================================================
                                MLJ functionalities
==========================================================================================#
MMI.@mlj_model mutable struct SmoothingSpline <: MMI.Deterministic
    λ::AbstractFloat = 1.0::(_ > 0.0)
    η::AbstractFloat = 1.0::(_ > 0.0)
    p::Integer       = 2::(_ > 0)
end

"""
fit(model::SmoothingSpline, verbosity::Int, X, y)

Computes relevant coefficients used when evaluating the spline on the data `(X,y)`.
"""
function MMI.fit(model::SmoothingSpline, verbosity::Int, X, y)
    # Extracting relevant data
    if !(typeof(X) <: AbstractArray)
        t = X[!,Tables.schema(X).names...]
    else
        t = X
    end
    if !issorted(t)
        error("The data is not sorted")
    end
    p = model.p
    n = length(t)
    a,b = extrema(t)
    δ = b - a
    # Create taylor polynomial basis
    H = compute_H(t,p)
    # Create Kernel matrix
    Ut, Vt = SymSemiseparableMatrices.spline_kernel((t' .- a)/δ,p)
    Σ = SymSemiseparableMatrix(Ut*δ^(2p-1), Vt)
    # Computing Coefficients
    c,d,_   = compute_coefficients(Σ,H,y,model.λ)
    model.η = n*model.λ * dot(c,y)/(n - p)
    # Saving Output
    fitresults = (c = c, d=d, t=t, K=Σ, H=H, fit = Σ*c + H*d)
    cache   = nothing
    report  = NamedTuple{}()
    return fitresults, cache, report
end

"""
predict(model::SmoothingSpline, X::AbstractVector, y)

Evaluates the spline model using the data `(X,y)`.
"""
function MMI.predict(model::SmoothingSpline, X::AbstractVector, y)
    fitresults, _, _ = MMI.fit(model::SmoothingSpline, 0, X, y)
    return fitresults[:fit]
end
mergesorted(a,b) = sort!(vcat(a,b))
mergeperm(a,b)   = sortperm(vcat(a,b))
"""
predict(model::SmoothingSpline, fitresult, Xnew)

Evaluates the spline model at the values defined in `Xnew`.
"""
function MMI.predict(model::SmoothingSpline, fitresult, Xnew)
    # Extract Coefficients and relevant infromation
    c = fitresult[:c]
    d = fitresult[:d]
    t = fitresult[:t]
    n = length(t)
    p = model.p
    a,b = extrema(t)
    δ = b - a
    # Merge data with new observations
    perm = mergeperm(t,Xnew)
    T    = mergesorted(t,Xnew)    
    # Create Taylor Basis
    Hnew = compute_H(Xnew,p)
    # Creating New Kernel Matrix
    UT, VT = SymSemiseparableMatrices.spline_kernel((T' .- a)/δ,p)
    Σ = SymSemiseparableMatrix(UT*δ^(2p-1),VT)
    # Computing Linear Interpolation
    C = zeros(length(T))
    C[perm .<= n] .= c
    E = (Σ*C)
    # Extract the values which are relevant for the predictions
    x0 = E[perm .> n]
    # Return Predictions
    return x0 + Hnew*d
end

"""
optimize!(machine::MLJ.Machine,show_trace=false)

Computing optimal roughness penealty with respect to the marginal likelihood.\\
See [section 2.7 of GP for ML](http://gaussianprocess.org/gpml/chapters/RW.pdf)
"""
function optimize!(machine::MLJ.Machine,show_trace=false)
    optimize!(machine.model,machine.args[1].data,machine.args[2].data,show_trace)
    fit!(machine)
end

"""
optimize!(model::SmoothingSpline, X, y::AbstractArray, show_trace=false)

Computing optimal roughness penealty with respect to the marginal likelihood.
See [section 2.7 of GP for ML](http://gaussianprocess.org/gpml/chapters/RW.pdf)
"""
function optimize!(model::SmoothingSpline, X, y::AbstractArray, show_trace=false)
    # Extracting relevant data
    if !(typeof(X) <: AbstractArray)
        t = X[!,Tables.schema(X).names...]
    else
        t = X
    end
    if !issorted(t)
        error("The data is not sorted")
    end
    p = model.p
    n = length(t)
    a,b = extrema(t)
    δ = b - a
    # Create taylor polynomial basis
    H = compute_H(t,p)
    # Create Kernel matrix
    Ut, Vt = SymSemiseparableMatrices.spline_kernel((t' .- a)/δ,p)
    Σ = SymSemiseparableMatrix(Ut*δ^(2p-1), Vt)
    # Optimize hyper parameter
    res = optimize(v -> log_gml(v,Σ,H,y), -10.0, 0.0, GoldenSection(),show_trace=show_trace)
    # Extract optimized values and refit
    model.λ = 10.0^res.minimizer
    c,d,_   = compute_coefficients(Σ,H,y,model.λ)
    model.η = n*model.λ * dot(c,y)/(n - p)
end
