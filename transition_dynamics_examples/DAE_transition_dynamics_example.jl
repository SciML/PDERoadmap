using DifferentialEquations, BenchmarkTools, Plots

function time_stepping_example(c_tilde, sigma_tilde, mu_tilde, x_min, x_max, M, T, rho, algorithm)
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
    @assert minimum(mu_tilde.(0, x)) >= 0.0 #Note: The above assumes that the drift is positive!  It is not checking the other t, though.
    L_0 = rho*I - Diagonal(mu_tilde.(0, x)) * L_1_plus - Diagonal(sigma_tilde.(0, x).^2/2) * L_2
    u_0 = L_0 \ c_tilde.(0, x)

    function f(du,u,p,t)
        L = rho*I  - Diagonal(mu_tilde.(t, x)) * L_1_plus - Diagonal(sigma_tilde.(t, x).^2/2) * L_2
        A_mul_B!(du,L,u)
        du .-= c_tilde.(t, x)
    end

    tspan=(0.0, T)
    prob = ODEProblem(f, u_0, tspan)
    solve(prob, algorithm)
end


function time_stepping_DAE_example(c_tilde, sigma_tilde, mu_tilde, x_min, x_max, M, T, rho, algorithm)
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
    @assert minimum(mu_tilde.(0, x)) >= 0.0 #Note: The above assumes that the drift is positive!  It is not checking the other t, though.
    L_0 = rho*I - Diagonal(mu_tilde.(0, x)) * L_1_plus - Diagonal(sigma_tilde.(0, x).^2/2) * L_2
    u_0 = L_0 \ c_tilde.(0, x)
    du_0 = zeros(u_0)

    function f(resid,du,u,p,t)
        L = rho*I  - Diagonal(mu_tilde.(t, x)) * L_1_plus - Diagonal(sigma_tilde.(t, x).^2/2) * L_2
        resid .= L*u - c_tilde.(t, x) - du
    end


    #Should Verifying that the initial condition is solid
    #resid_0 = zeros(u_0) #preallocation
    #f(resid_0, du_0,  u_0, nothing, 0)
    #Initial condition
    #@show norm(resid0)

    tspan=(0.0, T)

    prob = DAEProblem(f, zeros(u_0), u_0, tspan, differential_vars = trues(u_0)) #the zeros(u_0) is the du initial condition.
    solve(prob, algorithm)
end


x_min = 0.01
x_max = 1.0
M = 100
T = 0.2
rho = 0.05
sigma_bar = 0.1
c_tilde(t, x) = exp(x) + t*10
sigma_tilde(t, x) =  sigma_bar * x
mu_tilde(t, x) = 0.1*x *(1+ t/100)
algorithm =CVODE_BDF()#Rosenbrock23()#CVODE_BDF() #AutoTsit5(Rosenbrock23()) ImplicitEuler()
@time sol = time_stepping_example(c_tilde, sigma_tilde, mu_tilde, x_min, x_max, M, T, rho, algorithm)
plot(sol, vars=1:5:M)


algorithmDAE = IDA()
@time solDAE = time_stepping_DAE_example(c_tilde, sigma_tilde, mu_tilde, x_min, x_max, M, T, rho, algorithmDAE)
plot(solDAE, vars=1:5:M)

#How close are they?
@show norm(sol.u[1] - solDAE.u[1])
@show norm(sol.u[end] - solDAE.u[end])
