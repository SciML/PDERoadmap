# Huggett Model with Two Income Types
# ====================================
#
# This implements the continuous-time Huggett economy with two income states
# from Achdou, Han, Lasry, Lions, and Moll (2022) "Income and Wealth Distribution
# in Macroeconomics: A Continuous-Time Approach", Review of Economic Studies.
#
# Reference: http://benjaminmoll.com/codes/
# Numerical Appendix: https://benjaminmoll.com/wp-content/uploads/2020/02/HACT_Numerical_Appendix.pdf
#
# The model solves:
# 1. Hamilton-Jacobi-Bellman (HJB) equation for optimal consumption/savings
# 2. Kolmogorov Forward Equation (KFE) for the stationary wealth distribution

using LinearAlgebra, SparseArrays, Printf

"""
    HuggettTwoTypes

Parameters for the Huggett model with two income types.

# Fields
- `s`: CRRA utility parameter (risk aversion)
- `r`: Interest rate
- `rho`: Discount rate
- `z`: Income levels for the two states [z1, z2]
- `la`: Poisson transition intensities [la1, la2]
  - la1: transition rate from state 1 to state 2
  - la2: transition rate from state 2 to state 1
- `I`: Number of grid points for assets
- `amin`: Minimum asset level (borrowing constraint)
- `amax`: Maximum asset level
"""
struct HuggettTwoTypes{T<:Real}
    s::T      # CRRA utility parameter
    r::T      # Interest rate
    rho::T    # Discount rate
    z::Vector{T}   # Income levels [z1, z2]
    la::Vector{T}  # Poisson intensities [la1, la2]
    I::Int    # Number of grid points
    amin::T   # Borrowing constraint
    amax::T   # Maximum asset level
end

"""
    HuggettTwoTypes(; kwargs...)

Create a HuggettTwoTypes model with default parameters from Achdou et al.
"""
function HuggettTwoTypes(;
    s = 1.2,       # CRRA utility parameter
    r = 0.035,     # Interest rate
    rho = 0.05,    # Discount rate
    z1 = 0.1,      # Income in state 1 (low)
    z2 = 0.2,      # Income in state 2 (high)
    la1 = 1.5,     # Transition rate from state 1 to 2
    la2 = 1.0,     # Transition rate from state 2 to 1
    I = 500,       # Number of asset grid points
    amin = -0.02,  # Borrowing constraint
    amax = 3.0     # Maximum asset level
)
    HuggettTwoTypes(s, r, rho, [z1, z2], [la1, la2], I, amin, amax)
end

"""
    solve_hjb(p::HuggettTwoTypes; maxit=100, crit=1e-6, Delta=1000)

Solve the Hamilton-Jacobi-Bellman equation for the Huggett model with two income types.

Returns:
- `V`: Value function (I x 2 matrix)
- `c`: Consumption policy (I x 2 matrix)
- `s`: Savings policy (I x 2 matrix)
- `A`: Transition matrix for Kolmogorov Forward Equation
- `a`: Asset grid
- `dist`: Convergence history
"""
function solve_hjb(p::HuggettTwoTypes; maxit=100, crit=1e-6, Delta=1000)
    s_param = p.s
    r = p.r
    rho = p.rho
    z = p.z
    la = p.la
    Igrid = p.I
    amin = p.amin
    amax = p.amax

    # Asset grid
    a = range(amin, amax, length=Igrid)
    da = step(a)

    # Grids for both states
    aa = hcat(collect(a), collect(a))
    zz = ones(Igrid) * z'

    # Preallocate
    dVf = zeros(Igrid, 2)  # Forward difference
    dVb = zeros(Igrid, 2)  # Backward difference
    c = zeros(Igrid, 2)    # Consumption

    # Transition matrix for income states (Poisson process)
    # Aswitch encodes: with rate la[1], state 1 transitions to state 2
    #                  with rate la[2], state 2 transitions to state 1
    Aswitch = [
        spdiagm(0 => -la[1] * ones(Igrid))  spdiagm(0 => la[1] * ones(Igrid))
        spdiagm(0 => la[2] * ones(Igrid))   spdiagm(0 => -la[2] * ones(Igrid))
    ]

    # Initial guess: steady state consumption value
    v = zeros(Igrid, 2)
    for j in 1:2
        v[:, j] = (z[j] .+ r .* collect(a)) .^ (1 - s_param) / (1 - s_param) / rho
    end

    dist = Float64[]
    local A  # Declare A in outer scope

    for n in 1:maxit
        V = copy(v)

        # Forward difference (for positive drift)
        dVf[1:Igrid-1, :] = (V[2:Igrid, :] - V[1:Igrid-1, :]) / da
        dVf[Igrid, :] = (z .+ r * amax) .^ (-s_param)  # State constraint at amax

        # Backward difference (for negative drift)
        dVb[2:Igrid, :] = (V[2:Igrid, :] - V[1:Igrid-1, :]) / da
        dVb[1, :] = (z .+ r * amin) .^ (-s_param)  # State constraint at amin

        # Consumption and savings with forward difference
        cf = dVf .^ (-1 / s_param)
        ssf = zz + r * aa - cf

        # Consumption and savings with backward difference
        cb = dVb .^ (-1 / s_param)
        ssb = zz + r * aa - cb

        # Consumption at steady state
        c0 = zz + r * aa
        dV0 = c0 .^ (-s_param)

        # Upwind scheme: choose difference direction based on drift sign
        If = ssf .> 0  # Forward if drift > 0
        Ib = ssb .< 0  # Backward if drift < 0
        I0 = 1 .- If .- Ib  # Steady state otherwise

        dV_Upwind = dVf .* If + dVb .* Ib + dV0 .* I0
        c = dV_Upwind .^ (-1 / s_param)
        u = c .^ (1 - s_param) / (1 - s_param)

        # Construct transition matrix A
        # X: lower diagonal (backward drift)
        # Y: main diagonal
        # Z: upper diagonal (forward drift)
        X = -min.(ssb, 0) / da
        Y = -max.(ssf, 0) / da + min.(ssb, 0) / da
        Z = max.(ssf, 0) / da

        # Build tridiagonal matrices for each income state
        A1 = spdiagm(0 => Y[:, 1], -1 => X[2:Igrid, 1], 1 => Z[1:Igrid-1, 1])
        A2 = spdiagm(0 => Y[:, 2], -1 => X[2:Igrid, 2], 1 => Z[1:Igrid-1, 2])

        # Full transition matrix (asset dynamics + income switching)
        A = [
            A1                     spzeros(Igrid, Igrid)
            spzeros(Igrid, Igrid)  A2
        ] + Aswitch

        # Check transition matrix rows sum to zero
        if maximum(abs.(sum(A, dims=2))) > 1e-9
            @warn "Improper transition matrix at iteration $n"
            break
        end

        # Implicit time stepping: (1/Delta + rho)V = u + V/Delta + A*V
        B = (1 / Delta + rho) * sparse(1.0I, 2*Igrid, 2*Igrid) - A

        u_stacked = vcat(u[:, 1], u[:, 2])
        V_stacked = vcat(V[:, 1], V[:, 2])

        b = u_stacked + V_stacked / Delta
        V_stacked_new = B \ b

        V_new = hcat(V_stacked_new[1:Igrid], V_stacked_new[Igrid+1:2*Igrid])

        Vchange = V_new - v
        v = V_new

        push!(dist, maximum(abs.(Vchange)))
        if dist[end] < crit
            println("Value Function Converged, Iteration = $n")
            break
        end
    end

    # Compute savings policy
    savings = zz + r * aa - c

    return (V=v, c=c, s=savings, A=A, a=collect(a), dist=dist)
end

"""
    solve_kfe(A, Igrid, da)

Solve the Kolmogorov Forward Equation for the stationary distribution.

# Arguments
- `A`: Transition matrix from HJB solution
- `Igrid`: Number of asset grid points
- `da`: Grid spacing

# Returns
- `g`: Stationary distribution (Igrid x 2 matrix)
"""
function solve_kfe(A, Igrid, da)
    # Transpose of A for Kolmogorov Forward Equation
    AT = sparse(A')

    # The stationary distribution satisfies A'g = 0 with integral(g) = 1
    # We fix one value to make the system non-singular
    b = zeros(2*Igrid)
    b[1] = 0.1  # Fix one value

    # Modify the first row
    AT_modified = copy(AT)
    AT_modified[1, :] .= 0
    AT_modified[1, 1] = 1

    # Solve the linear system
    gg = AT_modified \ b

    # Normalize so that the distribution integrates to 1
    g_sum = sum(gg) * da
    gg = gg / g_sum

    # Reshape to Igrid x 2 matrix
    g = hcat(gg[1:Igrid], gg[Igrid+1:2*Igrid])

    return g
end

"""
    compute_aggregate_assets(g, a, da)

Compute aggregate asset supply (average assets in the economy).
"""
function compute_aggregate_assets(g, a, da)
    return dot(g[:, 1], a) * da + dot(g[:, 2], a) * da
end

"""
    run_huggett_example()

Run the full Huggett model with two types and return results.
"""
function run_huggett_example()
    println("Solving Huggett Model with Two Income Types")
    println("=" ^ 50)

    # Create model with default parameters
    p = HuggettTwoTypes()

    println("\nModel Parameters:")
    println("  CRRA parameter s = $(p.s)")
    println("  Interest rate r = $(p.r)")
    println("  Discount rate rho = $(p.rho)")
    println("  Income states z = $(p.z)")
    println("  Transition rates la = $(p.la)")
    println("  Asset grid: $(p.I) points on [$(p.amin), $(p.amax)]")

    # Solve HJB
    println("\nSolving HJB equation...")
    @time hjb_result = solve_hjb(p)

    da = hjb_result.a[2] - hjb_result.a[1]

    # Solve KFE
    println("\nSolving Kolmogorov Forward Equation...")
    @time g = solve_kfe(hjb_result.A, p.I, da)

    # Compute aggregates
    S = compute_aggregate_assets(g, hjb_result.a, da)
    println("\nAggregate Asset Supply: $S")

    # Check distribution masses
    mass1 = sum(g[:, 1]) * da
    mass2 = sum(g[:, 2]) * da
    println("Mass in state 1: $mass1")
    println("Mass in state 2: $mass2")
    println("Total mass: $(mass1 + mass2)")

    # Expected fractions from Poisson rates
    frac1 = p.la[2] / (p.la[1] + p.la[2])
    frac2 = p.la[1] / (p.la[1] + p.la[2])
    println("\nExpected fractions from Poisson rates:")
    println("  State 1: $frac1")
    println("  State 2: $frac2")

    return (
        params = p,
        V = hjb_result.V,
        c = hjb_result.c,
        s = hjb_result.s,
        g = g,
        a = hjb_result.a,
        A = hjb_result.A,
        aggregate_assets = S,
        convergence = hjb_result.dist
    )
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = run_huggett_example()

    # Optional: create simple ASCII visualization
    println("\n" * "=" ^ 50)
    println("Savings Policy at Selected Asset Levels:")
    println("=" ^ 50)
    a = results.a
    s = results.s
    selected_indices = [1, div(length(a), 4), div(length(a), 2), div(3*length(a), 4), length(a)]
    println("Asset\t\tSavings(z1)\tSavings(z2)")
    for i in selected_indices
        @printf("%.4f\t\t%.4f\t\t%.4f\n", a[i], s[i, 1], s[i, 2])
    end
end
