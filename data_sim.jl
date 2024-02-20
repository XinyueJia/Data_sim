using DataFrames, CSV, Random, CairoMakie

function make_mediation_data(n_obs=100)
    # baseline covariate -- continuous
    W = randn(n_obs)

    # create treatment based on baseline W
    A = W + randn(n_obs) .* 0.3

    
    # create outcome 
    Y = (a > 0.5 ? a : a^2 + 3 for a in A) .+ W .+ randn(n_obs) .* 0.2

   

    # full data structure
    data = DataFrame(Y=Y, A=A, W=W)
    return data
end

Random.seed!(123)

data = make_mediation_data(100)
scatter(data.A, data.Y)
#save("./data_sim/data_sim_nonlinear.png", f)
