# Load Data
include("motor_preds.jl") # Load pre-compute solution
df = DataFrame(CSV.File("datasets/motor.csv"))
y,X = unpack(df, ==(:accel), ==(:times))
# Defining prediction data set
a,b  = extrema(X)
Xnew = collect(a+1:1:b-1)

# Fitting and evaluating splines of all restrictions
for (index, shape_restriction) in enumerate(GeneralizedSmoothingSplines.SHAPES)
    spl = SmoothingSpline(lambda = 1e-5,shape_restrictions=(shape_restriction,))
    mach = machine(spl,X,y) 
    tune!(mach)
    interp = predict(mach)
    preds  = predict(mach,Xnew)
    # Tests
    @test isapprox(interp, Interps[:,index], atol=1e-4)
    @test isapprox(preds, Preds[:,index], atol=1e-4)
end
