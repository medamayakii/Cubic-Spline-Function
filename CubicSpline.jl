# load library
using CSV, Plots, DataFrames, Polynomials, LinearAlgebra

# define cubic spline function and return x and fitted polynomial function
function cubic_regression_spline(x, y, K, k)
    X = [j ≤ 3 ? x[i]^j : max(0, (x[i] - k[j-3])^3) for i in eachindex(x), j in 0:K+3]
    # calculate linear regression
    β = (X' * X) \ X' * y
    # declare Polynomial array
    g = Polynomial[Polynomial(β[1:4])]
    # calculate spline function for all cases
    for i in 1:K
        push!(g, g[end] + β[i+4] * Polynomial([-k[i]^3, 3k[i]^2, -3k[i], 1]))
    end
    return X, g
end

# define function in order to plot the spline function
function plot_spline(x, y, K, g, st)
    scatter(x, y, label="data points", legend=:topleft)
    for i in 1:K+1
        i == 1 ? plot!(g[i], label="spline curve", xlims=(st * (i - 1), st * i), c="red", lw=3) :
        plot!(g[i], label="", xlims=(st * (i - 1), st * i), c="red", lw=3)
        plot!(g[i], label="\$g($(round(st * (i - 1), digits=1))<x<$(round(st * i, digits=1)))\$",
            xlims=(0, 10), ylims=(-2, 21), xaxis=0:2:10, alpha=0.3, lw=2)
    end
    savefig("Result/$K.pdf")
end

# load data
data = Matrix(CSV.read("TrainingDataForAssingment5.csv", DataFrame))[:, 2:3]
x, y = data[:, 1], data[:, 2]
scatter(x, y, label="data points", legend=:topleft)

K = 4 # number of knots
k = [2, 4, 6, 8] # knots
X, g = cubic_regression_spline(x, y, K, k)
plot_spline(x, y, K, g, 2)

# ----------------------------------------------------------------------------------------------------------- #
# C) Leave on out cross validation using hat matrix (magic foemula)
function calc_CVLOO_magic(X, x, g, y, st)
    diag_H = diag(X * ((X' * X) \ X'))
    f = [g[Int(ceil(i / st))](i) for i in x]
    CV_LOO = sum([((y[i] - f[i]) / (1 - diag_H[i]))^2 for i in 1:length(y)]) / length(y)
    println(CV_LOO)
end
calc_CVLOO_magic(X, x, g, y, 2)

# ----------------------------------------------------------------------------------------------------------- #
# D) Leave on out cross validation without magic formula
CV_LOO_2 = 0.0
for iter in 1:length(y)
    # remove the first element of x and y
    x_, y_ = popfirst!(x), popfirst!(y)
    # fit by cubic spline function
    local X, g = cubic_regression_spline(x, y, K, k)
    # calculate CVLOO for each element
    global CV_LOO_2 += (y_ - g[Int(ceil(x_ / 2))](x_))^2
    # append x_ and y_ as the last element of x and y respectively
    push!(x, x_)
    push!(y, y_)
end
println(CV_LOO_2 / length(y))

# ----------------------------------------------------------------------------------------------------------- #
# use CVLOO to determine the number of knots among 1,2,...,15
for K_ in 1:15
    # calculate uniform knots
    k_ = [10i / (K_ + 1) for i in 1:K_]
    # calculate distance between 2 konts
    st = 10 / (K_ + 1)
    local X, g = cubic_regression_spline(x, y, K_, k_)
    plot_spline(x, y, K_, g, st)
    calc_CVLOO_magic(X, x, g, y, st)
end
