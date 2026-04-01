# Theoretical Analysis: Effect of the Proposal on RKPCN

## Setup

Consider the joint Markov chain $(u_t, f_t)$ defined by:

**f-update (pCN, no MH correction):**
$$
f_{t+1} = \mu + \rho(f_t - \mu) + \sqrt{1 - \rho^2}\,\xi_t, \quad \xi_t \sim \mathcal{N}(0, k)
$$

**u-update (symmetric MH):**
$$
\tilde{u} \sim \mathcal{N}(u_t, \Sigma_Q), \quad \alpha = \min\!\Big\{1,\; \frac{\pi_0(\tilde{u})\,L(\tilde{u};\,f_{t+1})}{\pi_0(u_t)\,L(u_t;\,f_{t+1})}\Big\}
$$
Accept $u_{t+1} = \tilde{u}$ with probability $\alpha$, else $u_{t+1} = u_t$.

Here $L(u; f) = \exp(f(u))$ in the log-density emulation case, $\pi_0$ is the prior, and $\Sigma_Q$ is the proposal covariance.

The goal is to sample from the expected posterior:
$$
\hat{\pi}(u) = \mathbb{E}_f\!\big[\pi(u; f)\big] = \mathbb{E}_f\!\bigg[\frac{\pi_0(u)\,L(u; f)}{Z(f)}\bigg]
$$

## Why the exact MwG is intractable

The exact Metropolis-within-Gibbs scheme for the f-update requires:
$$
\alpha_f = \min\!\bigg\{1,\; \frac{L(u;\tilde{f})}{L(u;f)} \cdot \frac{Z(f)}{Z(\tilde{f})}\bigg\}
$$

The normalizing constant ratio $Z(f)/Z(\tilde{f})$ is intractable. RKPCN drops this accept-reject step entirely, setting $\alpha_f \equiv 1$. This introduces an approximation whose quality depends on $\rho$ and on the u-proposal $\Sigma_Q$.

## Approximation error from dropping the f-acceptance step

### The true acceptance probability

For the exact MwG, the f-acceptance probability is:
$$
\alpha_f(f, \tilde{f}; u) = \min\!\bigg\{1,\; \frac{L(u;\tilde{f})}{L(u;f)} \cdot \frac{Z(f)}{Z(\tilde{f})}\bigg\}
$$

The RKPCN approximation sets $\alpha_f \approx 1$, which is equivalent to assuming:
$$
\frac{L(u;\tilde{f})}{L(u;f)} \cdot \frac{Z(f)}{Z(\tilde{f})} \approx 1
$$

### Decomposition into local and global terms

Write $\log \alpha_f = A_{\text{local}} + A_{\text{global}}$ where:

$$
A_{\text{local}}(u) = \log L(u;\tilde{f}) - \log L(u;f) = \tilde{f}(u) - f(u)
$$

$$
A_{\text{global}} = \log Z(f) - \log Z(\tilde{f}) = \log \int \pi_0(u')\exp(f(u'))\,du' - \log \int \pi_0(u')\exp(\tilde{f}(u'))\,du'
$$

Both terms are controlled by how much $\tilde{f}$ differs from $f$.

### Effect of $\rho$

Under the pCN proposal:
$$
\tilde{f} - f = -(1-\rho)(f - \mu) + \sqrt{1-\rho^2}\,\xi
$$

So:
$$
\tilde{f}(u) - f(u) = -(1-\rho)(f(u) - \mu(u)) + \sqrt{1-\rho^2}\,\xi(u)
$$

The variance of this perturbation at any point $u$ is:
$$
\text{Var}[\tilde{f}(u) - f(u)] = (1-\rho)^2(f(u) - \mu(u))^2 + (1-\rho^2)\,k(u,u)
$$

For $\rho \to 1$: $(1-\rho^2) = (1-\rho)(1+\rho) \approx 2(1-\rho)$, so $\text{Var} = O(1-\rho)$. This means $A_{\text{local}} = O_p(\sqrt{1-\rho})$.

Similarly, $A_{\text{global}}$ is controlled by the integrated effect of $\tilde{f} - f$ across the domain. The key insight is that **the global term depends on the full function perturbation, not just at the current point $u$**.

## The role of the u-proposal covariance $\Sigma_Q$

### Key observation: the implicit target depends on $\Sigma_Q$

Since RKPCN drops the f-acceptance step, detailed balance is violated for the joint chain. The marginal stationary distribution of $u$ (the "implicit target") depends on the entire transition kernel, including the u-proposal $\Sigma_Q$.

### Intuition for why $\Sigma_Q$ matters

Consider the limiting cases:

**Case 1: $\Sigma_Q \to 0$ (proposal variance shrinks to zero).**

When the u-proposal is very small, $u$ barely moves between iterations. Each u-update is essentially:
- Draw $\tilde{u} \approx u + \epsilon$ for tiny $\epsilon$
- The acceptance ratio $L(\tilde{u}; f) / L(u; f) \approx 1$ for any $f$, so the u-update almost always accepts

But the f-update proceeds with pCN steps that explore the GP posterior. Over many iterations, $f$ decorrelates and converges to its marginal $\mathcal{N}(\mu, k)$. At each f-value, the u-chain is approximately stationary at a distribution that depends on the current $f$.

In the limit, the u-marginal becomes:
$$
u \sim \frac{\pi_0(u)\,L(u; f^*)}{Z(f^*)}
$$
for whatever $f^*$ the chain happens to be visiting. Over time, this averages over $f$-values, but because u barely moves, the chain gets "stuck" tracking whichever $f$ trajectory it's on. **The implicit target approaches the plug-in mean approximation** because the chain can't escape fast enough as $f$ changes.

More precisely: when $\Sigma_Q$ is small, the u-chain moves slowly relative to the f-chain. Each time $f$ changes, $u$ doesn't have time to re-equilibrate to the new posterior $\pi(\cdot; f)$. So the chain effectively samples $u$ from the posterior conditioned on a slowly-changing $f$, which weights nearby trajectories more heavily. This is biased toward the MAP-like behavior of $\mu$.

**Case 2: $\Sigma_Q \to \infty$ (proposal variance grows).**

When the u-proposal is very large, most proposals $\tilde{u}$ are far from $u$. The acceptance probability becomes:
$$
\alpha \approx \min\{1, \pi_0(\tilde{u})L(\tilde{u}; f) / \pi_0(u)L(u; f)\}
$$

For very large proposals, $\tilde{u}$ lands in low-density regions of $\pi(\cdot; f)$, so $\alpha \to 0$. The chain barely moves in $u$-space, leading to the same problem as Case 1 but for a different reason: the chain is stuck because proposals are always rejected.

**Case 3: Well-tuned $\Sigma_Q$ (optimal regime).**

In the well-tuned regime, $u$ moves efficiently within the support of $\pi(\cdot; f)$ for the current trajectory $f$. As $f$ changes (slowly, when $\rho \approx 1$), $u$ can track the shifting posterior.

The implicit target in this regime most closely approximates the EP, because:
1. For each $f$, the u-chain approximately samples from $\pi(\cdot; f)$
2. The f-chain samples from $\mathcal{N}(\mu, k)$ (its marginal)
3. The u-marginal is approximately $\int \pi(u; f)\,\nu(df) = \hat{\pi}(u)$

### Quantifying proposal sensitivity

Let $\hat{\pi}_Q(u)$ denote the implicit stationary distribution of $u$ under proposal covariance $\Sigma_Q$. Consider the Wasserstein distance $W_2(\hat{\pi}_Q, \hat{\pi})$ between the implicit target and the true EP.

**Conjecture**: For well-behaved problems:
$$
W_2(\hat{\pi}_Q, \hat{\pi}) \leq C_1(\rho, Q) \cdot \sqrt{1 - \rho} + C_2(\rho, Q) \cdot \tau_u(Q)^{-1}
$$

where:
- The first term captures the approximation error from dropping the f-acceptance step ($\to 0$ as $\rho \to 1$)
- $\tau_u(Q)$ is the mixing time of the u-chain within a frozen $\pi(\cdot; f)$
- The second term captures the bias from insufficient u-mixing between f-updates

**As $\rho \to 1$**: The first term vanishes. The second term depends on $\tau_u(Q)$, which is a function of $\Sigma_Q$ and the geometry of $\pi(\cdot; f)$. So the sensitivity to $\Sigma_Q$ persists even as $\rho \to 1$, but the contribution of the f-update approximation error goes to zero.

**However**, there's a subtle interaction: as $\rho \to 1$, the f-chain moves more slowly, giving $u$ more iterations to equilibrate between f-updates. This suggests $C_2(\rho, Q)$ may decrease with $\rho$, partially offsetting the reduced mixing per iteration.

### The time-scale separation perspective

The key insight is that RKPCN involves **two coupled chains operating at different time scales**:

1. The f-chain has decorrelation time $\tau_f \sim 1/(1-\rho^2) \approx 1/(2(1-\rho))$
2. The u-chain (within frozen $f$) has decorrelation time $\tau_u(\Sigma_Q)$

For the algorithm to work well, we need $\tau_u \ll \tau_f$: the u-chain should mix many times before $f$ changes appreciably. This gives the condition:
$$
\tau_u(\Sigma_Q) \ll \frac{1}{1 - \rho}
$$

When this is satisfied, the u-chain approximately equilibrates within each "f-epoch", and the implicit target is close to the true EP.

When $\tau_u \gtrsim \tau_f$ (u-chain too slow relative to f-chain), the u-chain can't keep up with the changing f-landscape. The implicit target then depends on the path-dependent correlation between $u$ and $f$, which is where the $\Sigma_Q$-dependence enters.

## Practical implications

### Choosing $\Sigma_Q$

1. **Match $\Sigma_Q$ to the posterior geometry**: The u-chain should mix efficiently within $\pi(\cdot; f)$ for typical $f$. Since $\pi(\cdot; f)$ varies with $f$, the ideal $\Sigma_Q$ should reflect an "average" posterior geometry. Using the covariance of the EUP or plug-in mean posterior is a reasonable starting point.

2. **Target acceptance rate**: Standard MH theory suggests targeting $\sim 0.234$ acceptance for the u-update. But RKPCN is not a standard MH — the implicit target changes between iterations. Empirically, a moderate acceptance rate (0.2–0.4) seems reasonable.

3. **Avoid extremes**: Both too-small and too-large $\Sigma_Q$ cause the u-chain to move slowly (for different reasons), increasing the implicit-target bias.

### Adaptive $\Sigma_Q$

An adaptive scheme that tunes $\Sigma_Q$ during burn-in could help:
- Adapt based on the acceptance rate of the u-update
- Use the running sample covariance of the u-chain (similar to adaptive MCMC)
- Be cautious: since the target is implicit and proposal-dependent, adaptation changes the target. But if adaptation converges (stops updating), the final $\Sigma_Q$ determines a fixed implicit target.

### Choosing $\rho$

The choice of $\rho$ involves a trade-off:
- Larger $\rho$: smaller f-update approximation error, but slower f-mixing ($\tau_f$ increases)
- Smaller $\rho$: faster f-mixing, but larger approximation error

The optimal $\rho$ balances these. For problems where the normalizing constant ratio varies a lot (large $\text{Var}[Z(f)]$), smaller $\rho$ may be better despite the approximation error, because the chain can explore more f-trajectories.

### State-dependent $\rho$

A state-dependent $\rho_t(u_t, f_t)$ could adapt the f-step size based on local conditions:
- In regions where $k(u,u)$ is large (high GP uncertainty), the pCN step has larger effect on the local log-density ratio. Use a larger $\rho$ (smaller step) in these regions.
- In regions where $k(u,u)$ is small (near design points), the f-update barely changes the local value, so smaller $\rho$ is safe.

This could be implemented as:
$$
\rho(u) = 1 - (1-\rho_0)\,\cdot\, g(k(u,u))
$$
where $g$ is a scaling function. When $k(u,u)$ is small, $\rho(u) \approx 1$; when $k(u,u)$ is large, $\rho(u) \approx \rho_0$.

**Caution**: state-dependent $\rho$ changes the f-proposal distribution, which may break the pCN structure and its desirable properties. Need to verify that the resulting kernel is still valid.

## Connection to observed behavior

### Why rkpcn99 concentrates as a horizontal band

With $\rho = 0.99$, the f-chain moves very slowly ($\tau_f \approx 50$). The u-chain, using the adapted proposal from the exact posterior ($\sigma_{av} \approx 0.06$, $\sigma_{vi} \approx 0.2$), can move within the support of $\pi(\cdot; f)$ for the current $f$.

But the shape of $\pi(\cdot; f)$ is determined by the GP prediction at $u$. Since the GP has different predictive variances in different regions:
- **Near design points**: GP variance is low, so $\pi(\cdot; f)$ is similar across different $f$ trajectories
- **Away from design points**: GP variance is high, so different $f$ trajectories give very different $\pi(\cdot; f)$

The horizontal band pattern suggests that the chain finds a region where the GP predictive mean is high (along a horizontal slice) and the predictive variance is moderate. The u-chain moves efficiently along this "ridge" but doesn't easily cross to vertically-separated regions because:
1. The GP prediction changes significantly in the vertical direction
2. The adapted proposal has small variance in the horizontal (av) direction ($\sigma_{av} \approx 0.06$), limiting exploration

### Why smaller proposal changes the shape

With the 0.1× scaled proposal, the u-chain moves more slowly but stays closer to the current high-density region. When $f$ changes (slowly, due to $\rho = 0.99$), the u-chain can't move fast enough to follow the shifting posterior. This causes it to "fall behind" and sample more from the time-averaged posterior around the current region — which happens to be closer to the EP in this case.

This is consistent with the time-scale separation analysis: the smaller proposal increases $\tau_u$, which should make the implicit target **worse** (further from EP). But empirically it looks better, suggesting that the "optimal" $\Sigma_Q$ for approximating the EP may not be the one that maximizes u-mixing speed.

An alternative explanation: the smaller proposal reduces the effective exploration radius, which concentrates the samples in the region where the EP has most mass. This could be a "coincidental" improvement rather than a systematic one.

## Summary of recommendations

1. **Tier 1 (immediate)**: Test adaptive $\Sigma_Q$ (warm-up adaptation). Compare against fixed adapted proposal.
2. **Tier 2 (short-term)**: Implement the IAT-scaled iteration counts. Test across multiple replicates.
3. **Tier 3 (medium-term)**: Explore state-dependent $\rho$ based on local GP variance. Develop rigorous bounds on the time-scale separation condition.
