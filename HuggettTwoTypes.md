# Huggett Model with Two Income Types

This implements the continuous-time Huggett economy with two income states from Achdou, Han, Lasry, Lions, and Moll (2022) "Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach", Review of Economic Studies.

**Reference Paper**: [HACT Paper](https://benjaminmoll.com/wp-content/uploads/2019/07/HACT.pdf)
**Numerical Appendix**: [HACT Numerical Appendix](https://benjaminmoll.com/wp-content/uploads/2020/02/HACT_Numerical_Appendix.pdf)
**Original MATLAB Codes**: [Benjamin Moll's Codes](https://benjaminmoll.com/codes/)

## Model Description

The Huggett model is a workhorse heterogeneous agent model in macroeconomics. Agents face idiosyncratic income risk and can self-insure by accumulating a risk-free asset, subject to a borrowing constraint.

### Agent's Problem

Each agent maximizes expected discounted utility:
$$
\max_{c_t} \mathbb{E}_0 \int_0^\infty e^{-\rho t} u(c_t) dt
$$

subject to the budget constraint:
$$
\dot{a}_t = r a_t + z_t - c_t
$$

and the borrowing constraint:
$$
a_t \geq \underline{a}
$$

where:
- $a_t$ is asset holdings
- $c_t$ is consumption
- $z_t \in \{z_1, z_2\}$ is income (two states)
- $r$ is the interest rate
- $\rho$ is the discount rate
- $\underline{a}$ is the borrowing limit

The income process follows a continuous-time Markov chain with transition intensities $\lambda_1$ (from state 1 to 2) and $\lambda_2$ (from state 2 to 1).

### Hamilton-Jacobi-Bellman Equation

The value function $v_j(a)$ for income state $j \in \{1, 2\}$ satisfies:

$$
\rho v_j(a) = \max_c \left\{ u(c) + v_j'(a)(ra + z_j - c) + \lambda_j[v_{-j}(a) - v_j(a)] \right\}
$$

With CRRA utility $u(c) = \frac{c^{1-s}}{1-s}$, the first-order condition gives:
$$
c = (v_j'(a))^{-1/s}
$$

The optimal savings policy is:
$$
s_j(a) = ra + z_j - c_j(a)
$$

### Kolmogorov Forward Equation (KFE)

The stationary wealth distribution $g_j(a)$ satisfies:
$$
0 = -\partial_a[s_j(a) g_j(a)] - \lambda_j g_j(a) + \lambda_{-j} g_{-j}(a)
$$

with the normalization condition:
$$
\int_{\underline{a}}^{\bar{a}} [g_1(a) + g_2(a)] da = 1
$$

## Numerical Method

### Upwind Finite Difference Scheme

The HJB equation is discretized using an upwind finite difference scheme. For the derivative $v'(a)$:
- Use **forward difference** when drift $s(a) > 0$ (saving)
- Use **backward difference** when drift $s(a) < 0$ (dissaving)

This ensures numerical stability by respecting the direction of information flow.

Forward difference:
$$
v'_f(a_i) = \frac{v(a_{i+1}) - v(a_i)}{\Delta a}
$$

Backward difference:
$$
v'_b(a_i) = \frac{v(a_i) - v(a_{i-1})}{\Delta a}
$$

### Implicit Time Stepping

The value function iteration uses implicit time stepping for stability:
$$
\frac{v^{n+1} - v^n}{\Delta} = \rho v^{n+1} - u(c^n) - A^n v^{n+1}
$$

where $A^n$ is the transition matrix constructed from the optimal policy at iteration $n$.

### Solving the KFE

The stationary distribution is found by solving $A' g = 0$ subject to $\int g \, da = 1$. Since $A'$ is singular (rows sum to zero by construction), we fix one element and normalize.

## Julia Implementation

The Julia implementation is in `operator_examples/huggett_two_types.jl`.

### Quick Start

```julia
include("operator_examples/huggett_two_types.jl")

# Create model with default parameters
p = HuggettTwoTypes()

# Solve HJB equation
hjb_result = solve_hjb(p)

# Solve KFE for stationary distribution
da = hjb_result.a[2] - hjb_result.a[1]
g = solve_kfe(hjb_result.A, p.I, da)

# Compute aggregate asset supply
S = compute_aggregate_assets(g, hjb_result.a, da)
```

### Custom Parameters

```julia
# Custom model parameters
p = HuggettTwoTypes(
    s = 2.0,        # Higher risk aversion
    r = 0.04,       # Interest rate
    rho = 0.05,     # Discount rate
    z1 = 0.1,       # Low income state
    z2 = 0.3,       # High income state
    la1 = 1.0,      # Transition rate 1 -> 2
    la2 = 1.0,      # Transition rate 2 -> 1
    I = 1000,       # Finer grid
    amin = -0.05,   # More generous borrowing limit
    amax = 5.0      # Higher asset limit
)

results = run_huggett_example()
```

### Outputs

The `solve_hjb` function returns:
- `V`: Value function (I x 2 matrix)
- `c`: Optimal consumption policy (I x 2 matrix)
- `s`: Optimal savings policy (I x 2 matrix)
- `A`: Transition matrix
- `a`: Asset grid
- `dist`: Convergence history

## Default Parameters

| Parameter | Symbol | Default Value | Description |
|-----------|--------|---------------|-------------|
| `s` | $s$ | 1.2 | CRRA utility parameter |
| `r` | $r$ | 0.035 | Interest rate |
| `rho` | $\rho$ | 0.05 | Discount rate |
| `z1` | $z_1$ | 0.1 | Low income state |
| `z2` | $z_2$ | 0.2 | High income state |
| `la1` | $\lambda_1$ | 1.5 | Transition rate from state 1 to 2 |
| `la2` | $\lambda_2$ | 1.0 | Transition rate from state 2 to 1 |
| `I` | - | 500 | Number of asset grid points |
| `amin` | $\underline{a}$ | -0.02 | Borrowing constraint |
| `amax` | $\bar{a}$ | 3.0 | Maximum asset level |

## Key Economic Results

1. **Wealth Distribution**: The model generates a non-degenerate wealth distribution due to idiosyncratic income risk.

2. **State Fractions**: In steady state, the fraction of agents in each income state equals the ergodic distribution of the Markov chain:
   - Fraction in state 1: $\frac{\lambda_2}{\lambda_1 + \lambda_2}$
   - Fraction in state 2: $\frac{\lambda_1}{\lambda_1 + \lambda_2}$

3. **Borrowing Constraint**: Low-income agents may hit the borrowing constraint, leading to consumption equal to income: $c = z_1 + ra$.

4. **Aggregate Asset Supply**: The function $S(r) = \int a \cdot g(a) da$ gives the aggregate asset supply as a function of the interest rate.

## Extensions

This basic implementation can be extended to:
- Find the equilibrium interest rate where $S(r) = 0$ (zero net supply)
- Add transition dynamics
- Include diffusion in income (continuous income shocks)
- Add aggregate shocks for HANK (Heterogeneous Agent New Keynesian) models

See `operator_examples/huggett_diffusion.jl` (if available) for the version with continuous income shocks following an Ornstein-Uhlenbeck process.

## References

1. Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., & Moll, B. (2022). Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach. *Review of Economic Studies*, 89(1), 45-86.

2. Huggett, M. (1993). The Risk-Free Rate in Heterogeneous-Agent Incomplete-Insurance Economies. *Journal of Economic Dynamics and Control*, 17(5-6), 953-969.

3. Aiyagari, S. R. (1994). Uninsured Idiosyncratic Risk and Aggregate Saving. *Quarterly Journal of Economics*, 109(3), 659-684.
