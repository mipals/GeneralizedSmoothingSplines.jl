using Plots
using Printf
using DataFrames
using GeneralizedSmoothingSplines
using MLJ
using RDatasets

n       = 300          # number of samples
sigma   = 2.0          # noise standard deviation
a,b     = -0.2,1.1     # interval [a,b]
delta   = b - a        # Interval Width

t = a .+ sort(rand(n))*delta;
# FORRESTER ET AL. (2008) FUNCTION
f(t) = (6.0*t .- 2.0).^2 .* sin.(12.0*t .- 4.0)
targets = f(t) + sigma*randn(length(t));


# cars = dataset("datasets","cars")
# t = map(Float64,convert(Array,cars[!,:Speed]))
# targets = map(Float64,convert(Array,cars[!,:Dist]))

# Plotting
scatter(t, targets, ms=2, label="Observations", xlims=(a,b), xlabel="t") 
plot!(a:delta/n:b, f(a:delta/n:b), label="f(t)",color=:black, ls=:dash,lw=1)
# Defining Spline
spl = SmoothingSpline()
df = DataFrame(t = t)


mach = machine(spl,df,targets)
fit!(mach)

m = 100
Xnew = sort(a .+ sort(rand(m))*delta)
preds = predict(mach,Xnew)

c = mach.fitresult[:c]
d = mach.fitresult[:d]
K = mach.fitresult[:K]
H = mach.fitresult[:H]

scatter(t, targets, ms=2, label="Observations", xlims=(a,b), xlabel="t",legend=:topleft) 
plot!(a:delta/n:b, f(a:delta/n:b), label="f(t)",color=:black, ls=:dash,lw=1)
plot!(t, K*c + H*d, label="fit",color=:red, lw=2)
scatter!(Xnew,preds,color=:green)


L = cholesky(SymSemiseparableMatrix(Ut*δ^(2p-1),Vt), n*smoothingspline.λ)
H = compute_H(t,p)
B = L\H
F = qr(B)
S = L'\Matrix(F.Q)

dH = zeros(n)
ei = zeros(n)
for i = n:-1:1
  ei[i] = 1.0
  dH[i] = norm((L\ei)[i:end])^2
  ei[i] = 0.0
end

diagH = 1.0 .- n*smoothingspline.λ*(dH - sum(S.*S,dims=2))
α = 0.05
β = sqrt(2.0)*erfinv(1.0 - α)
sigma = sqrt(n*smoothingspline.λ*dot(y,c)/(n-p))

scatter(t, y, ms=2, label="Observations", xlims=(a,b), xlabel="t", legend=false);
plot!(a:δ/n:b, f(a:δ/n:b), label="f(t)", ls=:dash,color=:black,lw=1, ribbon=β*sigma, alpha=0.8, fillalpha=0.1)
plot!(t,Σ*c + H*d, label="fit",color=:red,lw=2,title="λ="*@sprintf("%.3e", smoothingspline.λ)*", "*  "σ="* @sprintf("%.3e", sigma), ribbon=β*sigma*sqrt.(diagH), fillalpha=0.5)

