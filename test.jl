using GLMakie

include("./SL.jl")
X = data[:,[:A, :W]]
y = data[:, :Y];

# train model 
slm = machine(super_learner, X, y)
fit!(slm)


# predict (SL)
xs = LinRange(min((X[:, :A])...), max((X[:, :A])...), 100) 

ys = LinRange(min((X[:, :W])...), max((X[:, :W])...), 100) 
zs = [MLJ.predict(slm, hcat(x, y))[1] for x in xs, y in ys]

surface(xs, ys, zs, axis=(type=Axis3,))


# partial derivative (SL)
zs = [((MLJ.predict(slm, hcat(x .+ 0.01, y)) .- MLJ.predict(slm, hcat(x .- 0.01, y))) / (2 * 0.01))[1] for x in xs, y in ys]
surface(xs, ys, zs, axis=(type=Axis3,))