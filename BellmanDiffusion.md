# Solving the Bellman Equation with a Simple Univariate Diffusion
## Setup
Take the stochastic process
$$
d x_t = \mu(t, x) dt + \sigma(t, x) d W_t
$$
where $W_t$ is Brownian motion and reflecting barriers at $x \in (x^{\min},x^{\max})$

The partial differential operator (infinitesimal generator) associated with the stochastic process is

\begin{align}
\tilde{L_1} \equiv \tilde{\mu}(t, x)  \partial_x + \frac{\tilde{\sigma}(t, x)^2}{2}\partial_{xx}
\end{align}

Then, if the payoff in state $x$ is $c(x) = x^2$, and payoffs are discounted at rate $\rho$, then the Bellman equation is,
$$
\rho \tilde{u}(t, x) = \tilde{c}(t, x) + \tilde{L}_1 \tilde{u}(t, x) + \partial_t \tilde{u}(t,x)
$$
With boundary values $\partial_x \tilde{u}(t, x^{\min}) = 0$ and $\partial_x \tilde{u}(t, x^{\max}) = 0$ for all $t$

We can combine these to form the operator,
\begin{align}
\tilde{L} = \rho - \tilde{L_1}
\end{align}
and the boundary condition operator (using the $|$ for "evaluated at"),
\begin{align}
\tilde{B} = \begin{bmatrix}
	\partial_x \Big|_{x=x^{\min},t}\\
	\partial_x \Big|_{x=x^{\max},t}
\end{bmatrix}
\end{align}

which leads to the PDE,
$$
\partial_t \tilde{u}(t,x) = \tilde{L}_t \tilde{u}(t,x) - \tilde{c}(t,x)
$$
and boundary conditions at every $t$,
$$
 \tilde{B} \tilde{u}(t,x) = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$


## Example Functions
As a numerical example, start with something like
- $x^{\min} = 0.01$
- $x^{\max} = 1.0$
- $\tilde{\mu}(t,x) = -0.1 + t + .1 x$
   - Note, that this keeps $\tilde{\mu}(t,x) \geq 0$ for all $t,x$.  Hence, we know the correct upwind direction.
- $\tilde{\sigma}(t,x) = \bar{\sigma} x$ for $\bar{\sigma} = 0.1$
- $\tilde{c}(t,x) = e^x$
- $\rho = 0.05$

## Discretization
Do a discretization of the $\tilde{L}$ operator subject to the $\tilde{B}$, using the standard technique (and knowing that the positive drift ensures we can use a single upwind direction).  the value function is then $u(t) \in R^M$, an operator is $L(t) \in R^M$, and a vector of payoffs $c(t) \in R^M$.  This leads to the following system of ODEs,
$$
\partial_t u(t) = L(t) u(t) - c(t)
$$

The stationary solution, at a $t=T$ is the solution to the linear system,
$$
u(T) = L(T) \backslash c(T)
$$

Given this solution, we can solve for the transition dynamics by going back in time from the $u(T)$ initial condition.
