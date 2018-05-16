using DifferentialEquations

function time_stepping_example(x_min, x_max, M, T, N, rho, sigma_bar, algorithm)
    M_bar = M + 2
    x = linspace(x_min, x_max, M)
    Delta = x[2]-x[1]
    mu_tilde_T = -0.1+T+0.1*x
    sigma_tilde_T = sigma_bar*x
    c_tilde = exp.(x)

    L_1_plus = Tridiagonal(zeros(M_bar-1), -1 * ones(M_bar), ones(M_bar-1))[2: end-1, :]/Delta
    L_1_plus[:, 2] = L_1_plus[:, 2] + L_1_plus[:, 1]
    L_1_plus[:, end-1] = L_1_plus[:, end-1] + L_1_plus[:, end]
    L_1_plus = L_1_plus[:, 2: end-1]

    L_2 = Tridiagonal(ones(M_bar-1), -2 * ones(M_bar), ones(M_bar-1))[2: end-1, :]/(Delta^2)
    L_2[:, 2] = L_2[:, 2] + L_2[:, 1]
    L_2[:, end-1] = L_2[:, end-1] + L_2[:, end]
    L_2 = L_2[:, 2: end-1]

    L_tilde = Diagonal(rho*ones(M)) - (Diagonal(mu_tilde_T)*L_1_plus + Diagonal(sigma_tilde_T.^2/2)*L_2)


    u_T = L_tilde\c_tilde

    function f(du,u,p,t)
        mu_tilde = -0.1+0.01*t+0.1*x#mu_tilde_T#-0.1+t+0.1*x
        L = Diagonal(rho*ones(M)) - (Diagonal(mu_tilde)*L_1_plus + Diagonal(sigma_tilde_T.^2/2)*L_2)
        A_mul_B!(du,L,u)
        du .-= c_tilde
    end

    tstops=(0.0, T)
    prob = ODEProblem(f, u_T, tstops)
    solve(prob, algorithm)
end

x_min = 0.01
x_max = 1
M = 40
T = 0.3
N = 10 #This is not being used right now.
rho = 0.05
sigma_bar = 0.1
algorithm = Tsit5()

sol = time_stepping_example(x_min, x_max, M, T, N, rho, sigma_bar, algorithm)

using Plots
plot(sol)
