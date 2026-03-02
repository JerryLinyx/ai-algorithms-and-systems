## Introduction

Generative models aim to transform a simple base distribution, such as a Gaussian, into a complex target data distribution. The core idea is to construct a generator that maps samples from the base distribution into data space. To compute likelihoods, this generator must be invertible. If a generator transforms $Z$ into $X = G(Z)$, then $Z = G^{-1}(X)$ must exist, and the density is computed using the change of variables:

$P_X(X) = P_Z(G^{-1}(X)) \left|\det(\partial G^{-1}(X)/\partial X)\right|$

or equivalently:

$P_X(X) = P_Z(Z) / \left|\det(\partial X/\partial Z)\right|$

Stacking $K$ invertible generators gives:

$P_X(X) = P_Z(G^{-1}(X)) \prod_{k=1}^K \left|\det(\partial G_k^{-1}(X_k)/\partial X_k)\right|$

The main difficulty is designing expressive yet tractable invertible transformations with efficiently computable Jacobian determinants.

## Residual Flows: Discrete-Time Transformations

A residual flow layer has the simple form:

$X = Z + U(Z)$

To make this invertible, $U$ must be a contractive mapping so that the fixed point equation:

$Z = X - U(Z)$

has a unique solution. This allows the layer to be inverted by iterative fixed-point updates.

The Jacobian determinant is:

$\det(\partial X/\partial Z) = \det(I + \partial U(Z)/\partial Z)$

Direct determinant computation is expensive. A common trick is estimating traces using Monte Carlo:

$\mathrm{Tr}(A) = E_v[v^T A v]$

with $v$ sampled from a standard Gaussian.

Stacking $K$ residual layers produces:

$X_k = X_{k-1} + U_k(X_{k-1})$

which gradually warps the simple base distribution into a complex target distribution.

## Continuous Normalizing Flows (CNF)

Taking the limit of infinitely many residual layers transforms discrete updates into an ODE. From:

$X_k - X_{k-1} = U_k(X_{k-1})$

letting the step size go to zero yields:

$dX(t)/dt = U(X(t), t, \theta)$

This defines a Neural ODE where $U$ is a time-varying vector field.

### Density Evolution via the Continuity Equation

The probability density $P_t(x)$ evolves according to:

$\partial P_t(x)/\partial t + \nabla \cdot (P_t(x) U(x,t)) = 0$

This describes conservation of probability mass as samples move under the vector field.

### Training Difficulty

Training CNFs requires solving ODEs inside the likelihood objective for every data point, making CNFs computationally expensive and slow to scale.

## Flow Matching: Scalable CNF Training

Flow Matching avoids ODE integration by directly learning the vector field $U(x,t)$.

### The Flow Matching Objective

Define the ideal objective:

$L_{FM}(\theta) = E_{P_t(x)} \|U_\theta(x,t) - U_{target}(x,t)\|^2$

This trains $U_\theta$ to match the true vector field $U_{target}$. However, both $P_t(x)$ and $U_{target}(x,t)$ are unknown.

### Conditional Flow Matching

The key idea is to design conditional probability paths $P_t(x|Z)$ that are analytically tractable. Sampling over these conditional paths recovers the marginal objective in expectation.

A common conditional path:

$P_t(x|Z) = \mathcal{N}(\alpha_t Z, \beta_t^2 I)$

with $\alpha_0=0$, $\beta_0=1$, $\alpha_1=1$, $\beta_1=0$, smoothly interpolating from noise to the data point $Z$.

For this path, the conditional vector field:

$U_{target}(x,t|Z) = \dot{\alpha}_t Z + (\dot{\beta}_t/\beta_t^2)(x - \alpha_t Z)$

A simpler special case is the rectified flow path:

$x_t = tZ + (1-t)\epsilon$

where $\epsilon$ is sampled from the base distribution. Then the target vector field is constant:

$U_{target} = Z - \epsilon$

### Conditional Flow Matching Objective

$L_{CFM}(\theta) = E_{t \sim U(0,1), Z \sim P_{data}, x \sim P_t(x|Z)} \|U_\theta(x,t) - U_{target}(x,t|Z)\|^2$

A theorem states:

$L_{FM}(\theta) = L_{CFM}(\theta) + C$

where $C$ is independent of $\theta$. Minimizers of both objectives are identical, so training with the conditional loss recovers the true flow.

### Training Loop

1. Sample a data point $Z$.
2. Sample a time $t$ uniformly in $[0,1]$.
3. Sample $x$ from $P_t(x|Z)$.
4. Compute $U_{target}(x,t|Z)$.
5. Train $U_\theta$ using the L2 loss.
6. Update parameters with gradient descent.

No ODE solving is required during training.

## Comparison with Diffusion Models

Flow Matching generalizes diffusion models:

- It does not require the forward process to be a fixed Gaussian diffusion.
- It directly designs a path from noise to data.
- It avoids computing or simulating a forward diffusion chain.
- The training objective is a simple regression problem.

Diffusion models produce samples by reversing a stochastic diffusion process, while Flow Matching learns a continuous vector field mapping noise to data directly.

## Conclusion

Residual flows introduced discrete invertible transformations. CNFs extended these to continuous time, governed by ODEs and the continuity equation. However, CNFs are computationally expensive to train because they require ODE integration inside the loss.

Flow Matching resolves this by replacing likelihood-based training with a tractable conditional regression objective that teaches the model to approximate the true vector field. This preserves the expressive power of continuous flows while greatly improving scalability and efficiency. As a result, Flow Matching has become a foundational framework for high-resolution generative modeling.
