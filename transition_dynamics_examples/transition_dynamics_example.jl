using DifferentialEquations, NamedTuples, BenchmarkTools, Plots

function time_stepping_example(c_tilde, sigma_tilde, mu_tilde, x_min, x_max, M, T, rho, algorithm, dt=0.0001, adaptive=true)

    #Setting up the general discretization matrices for the upwind L_1+ and the L_2 central differences
    #These would be automaitcally generated from the DIffEqOperators.jl library
    x = linspace(x_min, x_max, M)
    Δ = x[2]-x[1]

    dl_1 = zeros(M-1)
    d_1 = -1 * ones(M)
    d_1[end] = 0
    du_1 = ones(M-1)
    L_1_plus = Tridiagonal(dl_1, d_1, du_1)/Δ

    dl_2 = ones(M-1)
    d_2 = -2 * ones(M)
    d_2[1] = -1
    d_2[end] = -1
    du_2 = ones(M-1)
    L_2 = Tridiagonal(dl_2, d_2, du_2)/(Δ^2)

    #Calculating the stationary solution
    @assert minimum(mu_tilde.(T, x)) >= 0.0 #Note: The above assumes that the drift is positive!  It is not checking interior t
    @assert minimum(mu_tilde.(0, x)) >= 0.0

    L_T = rho*I - Diagonal(mu_tilde.(T, x)) * L_1_plus - Diagonal(sigma_tilde.(T, x).^2/2) * L_2
    u_T = L_T \ c_tilde.(T, x)

    @assert(issorted(u_T)) #We are only solving versions that are increasing for now

    function f(du,u,p,t)
        L = rho*I  - Diagonal(mu_tilde.(t, x)) * L_1_plus - Diagonal(sigma_tilde.(t, x).^2/2) * L_2
        A_mul_B!(du,L,u)
        du .+= c_tilde.(t, x)
    end

    tspan = (T, 0.0)
    prob = ODEProblem(f, u_T, tspan)
    if dt==nothing
        solve(prob, algorithm)
    else
        solve(prob, algorithm, dt=dt, adaptive=adaptive)
    end
end

# mu_tilde is time varying but sigma_tilde is not.
x_min = 0.01
x_max = 1.0
M = 50
T = 0.3
rho = 0.05
sigma_bar = 0.1
c_tilde(t, x) = exp(x)
sigma_tilde(t, x) =  sigma_bar * x
mu_tilde(t, x) = 0.1*x + t/1500.0 #How is this so sensitive?
#Algorithms which don't work well: Rosenbrock23(), Rodas4(), KenCarp4()
#algorithm = ImplicitEuler() #OK
#algorithm = TRBDF2() # Converge to steady state with fixed dt = 0.01
algorithm = CVODE_BDF()

dt = nothing #0.01#0.0001
adaptive =true# true#false

@time sol_1 = time_stepping_example(c_tilde, sigma_tilde, mu_tilde, x_min, x_max, M, T, rho, algorithm, dt, adaptive)
#time_stepping_example(x_min, x_max, M, T, N, rho, sigma_bar, algorithm)
plot(sol_1, vars=1:5:M, xlims=[0.0 T], xflip=false)
@assert(issorted(sol_1[end]))

# Increase the time variation
c_tilde(t, x) = exp(x)
sigma_tilde(t, x) =  sigma_bar * x
mu_tilde(t, x) = 0.1*x + t/15.0 #increase the time variation
@time sol_2 = time_stepping_example(c_tilde, sigma_tilde, mu_tilde, x_min, x_max, M, T, rho, algorithm, dt, adaptive)
plot(sol_2, vars=1:5:M, xlims=[0.0 T], xflip=false)
@assert(issorted(sol_2[end]))

@time sol_2 = time_stepping_example(c_tilde, sigma_tilde, mu_tilde, x_min, x_max, M, T, rho, ImplicitEuler(), 0.0001, true)
plot(sol_2, vars=1:5:M, xlims=[0.0 T], xflip=false)
@assert(issorted(sol_2[end]))

# # Both mu_tilde and sigma_tilde are time varying
# c_tilde(t, x) = exp(x)
# sigma_tilde(t, x) =  sigma_bar * x + t/10
# mu_tilde(t, x) = 0.1*x + t/1500.0
#
# @time sol_4 = time_stepping_example(c_tilde, sigma_tilde, mu_tilde, x_min, x_max, M, T, rho, algorithm, dt, adaptive)
# plot(sol_4, vars=1:5:M, xlims=[0.0 T], xflip=false)
# @assert(issorted(sol_4[end]))
#
# # Allow c_tilde and mu_tilde change in time
# c_tilde(t, x) = exp(x) + t/100
# sigma_tilde(t, x) =  sigma_bar * x
# mu_tilde(t, x) = 0.1*x + t/1500.0
#
# @time sol_5 = time_stepping_example(c_tilde, sigma_tilde, mu_tilde, x_min, x_max, M, T, rho, algorithm, dt, adaptive)
# plot(sol_5, vars=1:5:M, xlims=[0.0 T], xflip=false)
# @assert(issorted(sol_5[end]))
