# packages
using MLJ
using MLJLinearModels
using EvoTrees
using XGBoost

XGBoostRegressor = @load XGBoostRegressor pkg = XGBoost
Random.seed!(123)
# importa dataset


include("./data_sim.jl")

# data = CSV.File("data_sim/data_sim.csv", normalizenames=true) |> DataFrame

# super learner
function base_models()
    lambdas = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    linear_models = [RidgeRegressor(lambda=l) for l in lambdas]
    linear_models = NamedTuple{Tuple(Symbol("lr_$i") for i in eachindex(lambdas))}(linear_models)

    max_depths = [1, 2, 3, 4, 5]
    evo_trees = [EvoTreeRegressor(ax_depth=m) for m in max_depths]
    evo_trees = NamedTuple{Tuple(Symbol("tree_$i") for i in eachindex(max_depths))}(evo_trees)

    xgboost_models = [XGBoostRegressor(max_depth=m) for m in max_depths]
    xgboost_models = NamedTuple{Tuple(Symbol("xgboost_$i") for i in eachindex(max_depths))}(xgboost_models)
    return merge(linear_models, xgboost_models)
end

super_learner = Stack(;
    metalearner=EvoTreeRegressor(fit_intercept=true),
    resampling=CV(nfolds=5),
    base_models()...
);