using DifferentialEquations

x_min = 0.01
x_max = 1
M = 20
T = 0.5
N = 10 #This is not being used right now.
rho = 0.05
sigma_bar = 0.1

sol = time_stepping(x_min, x_max, M, T, N, rho, sigma_bar)
