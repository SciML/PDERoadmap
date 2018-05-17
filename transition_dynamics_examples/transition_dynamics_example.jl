using DifferentialEquations, NamedTuples, BenchmarkTools

function time_stepping_example(c_tilde, sigma_tilde, mu_tilde, x_min, x_max, M, T, rho, algorithm, dt=0.0001, adaptive=true)

    #Setting up the general discretization matrices for the upwind L_1+ and the L_2 central differences
    #These would be automaitcally generated from the DIffEqOperators.jl library
    M_bar = M + 2
    x = linspace(x_min, x_max, M)
    Delta = x[2]-x[1]

    L_1_plus = Tridiagonal(zeros(M_bar-1), -1 * ones(M_bar), ones(M_bar-1))[2: end-1, :]/Delta
    L_1_plus[:, 2] = L_1_plus[:, 2] + L_1_plus[:, 1]
    L_1_plus[:, end-1] = L_1_plus[:, end-1] + L_1_plus[:, end]
    L_1_plus = L_1_plus[:, 2: end-1]

    L_2 = Tridiagonal(ones(M_bar-1), -2 * ones(M_bar), ones(M_bar-1))[2: end-1, :]/(Delta^2)
    L_2[:, 2] = L_2[:, 2] + L_2[:, 1]
    L_2[:, end-1] = L_2[:, end-1] + L_2[:, end]
    L_2 = L_2[:, 2: end-1]

    #Calculating the stationary solution
    @assert minimum(mu_tilde.(T, x)) >= 0.0 #Note: The above assumes that the drift is positive!  It is not checking the other t, though.
    L_T = rho*I - Diagonal(mu_tilde.(T, x)) * L_1_plus - Diagonal(sigma_tilde.(T, x).^2/2) * L_2
    u_T = L_T \ c_tilde.(T, x)

    function f(du,u,p,t)
        L = rho*I  - Diagonal(mu_tilde.(t, x)) * L_1_plus - Diagonal(sigma_tilde.(t, x).^2/2) * L_2
        A_mul_B!(du,L,u)
        du .-= c_tilde.(t, x)
    end

    tspan=(0.0, T)
    prob = ODEProblem(f, u_T, tspan)
    if dt==nothing
        solve(prob, algorithm)
    else
        solve(prob, algorithm, dt=dt, adaptive=adaptive)
    end
end

x_min = 0.01
x_max = 1.0
M = 50
T = 0.4
rho = 0.05
sigma_bar = 0.1
c_tilde(t, x) = exp(x)
sigma_tilde(t, x) =  sigma_bar * x
mu_tilde(t, x) = 0.1*x + t/1500.0 #How is this so sensitive?
#algorithm = Tsit5()
#algorithm = AutoTsit5(Rosenbrock23())
#algorithm = Rosenbrock23()
algorithm = ImplicitEuler()
#algorithm = Euler()
dt = nothing #0.01#0.0001
adaptive =true# true#false

@time sol = time_stepping_example(c_tilde, sigma_tilde, mu_tilde, x_min, x_max, M, T, rho, algorithm, dt, adaptive)
#time_stepping_example(x_min, x_max, M, T, N, rho, sigma_bar, algorithm)
using Plots
plot(sol, vars=1:5:M)
