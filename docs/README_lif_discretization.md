# Time discretization of the LIF neuron in this repo

## 1. The continuous equation

The leaky integrate-and-fire (LIF) sub-threshold dynamics are a first-order linear ODE:

$$
\tau_m \frac{dV}{dt} = -\bigl(V(t) - V_{\text{rest}}\bigr) + R\, I_{\text{syn}}(t)
$$

with the spike-and-reset rule:

$$
\text{if } V(t) \ge V_{\text{thresh}} \;\Rightarrow\; V(t^+) = V_{\text{reset}},\quad \text{neuron refractory for } t_{\text{ref}}.
$$

Parameters in [include/lif_neuron.h](../include/lif_neuron.h): $\tau_m=20$ ms, $V_{\text{rest}}=-65$ mV, $V_{\text{thresh}}=-50$ mV, $V_{\text{reset}}=-65$ mV, $R=1\,\text{M}\Omega$, $\Delta t=1$ ms, $t_{\text{ref}}=2$ ms.

## 2. Discretization used in the C++ code

In [src/lif_neuron.cpp](../src/lif_neuron.cpp) the ODE is replaced by a **forward (explicit) Euler** update on a uniform grid $t_n = n\,\Delta t$:

$$
V_{n+1} \;=\; V_n + \frac{\Delta t}{\tau_m}\Bigl[-\bigl(V_n - V_{\text{rest}}\bigr) + R\, I_{\text{syn},n}\Bigr]
$$

Two more discrete-time approximations are layered on top of this:

- **Threshold detection** is checked only at grid points: a spike fires if $V_{n+1} \ge V_{\text{thresh}}$.
- **Refractory period** is implemented as an integer countdown of `ref_steps = round(t_ref / dt)` timesteps. With $t_{\text{ref}}=2$ ms and $\Delta t=1$ ms this is exactly 2 steps; for non-integer ratios you also incur a rounding error of up to $\Delta t/2$ in the refractory length.

So the simulator advances the population by repeatedly calling `step(I_syn)` once per millisecond.

## 3. Where the numerical error comes from

Because forward Euler approximates the derivative by a finite difference, three distinct error sources appear:

### (a) Local truncation error (per step)
Taylor-expanding the exact solution around $t_n$:

$$
V(t_{n+1}) = V_n + \Delta t\, \dot V_n + \tfrac{1}{2}\Delta t^2\, \ddot V_n + O(\Delta t^3)
$$

Forward Euler keeps only the first two terms, so the per-step error is

$$
\tau_{\text{loc}} = \tfrac{1}{2}\Delta t^2 \,\ddot V(\xi) \;=\; O(\Delta t^2).
$$

### (b) Global error (over a simulation of length $T$)
Accumulating $T/\Delta t$ steps gives global accuracy

$$
\bigl\|V_n - V(t_n)\bigr\| = O(\Delta t).
$$

Forward Euler is therefore **first-order accurate** — halving $\Delta t$ halves the error.

### (c) Stability constraint
For the linear leak term, the amplification factor of forward Euler is

$$
V_{n+1} - V_{\text{rest}} = \Bigl(1 - \tfrac{\Delta t}{\tau_m}\Bigr)(V_n - V_{\text{rest}}) + \tfrac{\Delta t}{\tau_m} R I_{\text{syn},n}.
$$

Stability (no spurious oscillation/blow-up) requires

$$
\left|1 - \frac{\Delta t}{\tau_m}\right| < 1 \;\;\Longleftrightarrow\;\; 0 < \Delta t < 2\tau_m = 40\text{ ms}.
$$

With $\Delta t = 1$ ms and $\tau_m = 20$ ms the ratio $\Delta t/\tau_m = 0.05$, which is comfortably inside the stable region and keeps the leading-order error small (~5 % of $\dot V$ contribution per step in the worst case).

### (d) Spike-time quantization error
Even if $V(t)$ crossed threshold at $t_n + \alpha\Delta t$ with $0<\alpha<1$, the simulator only registers the spike at $t_{n+1}$. The error in spike timing is therefore bounded by $\Delta t$ and biases inter-spike intervals upward. This is *not* reduced by going to a higher-order ODE integrator unless threshold-crossing interpolation is added.

## 4. Putting numbers on it (with the repo's defaults)

| Quantity | Value |
|---|---|
| Step size $\Delta t$ | 1 ms |
| Membrane time constant $\tau_m$ | 20 ms |
| $\Delta t/\tau_m$ | 0.05 |
| Local truncation error per step | $\sim\tfrac{1}{2}(\Delta t/\tau_m)^2 \cdot |V-V_{\text{rest}}| \approx 1.25\times10^{-3}\cdot|V-V_{\text{rest}}|$ |
| Global voltage error over 1 s | $O(\Delta t) \sim$ few % of the driving amplitude |
| Spike-time error | $\le 1$ ms per spike |
| Refractory rounding (here) | 0 ms (exact, since 2 ms / 1 ms is integer) |

## 5. Why the package backends do not have this Euler error

For comparison, NEST's `iaf_psc_alpha` ([scripts/nest_export.py](../scripts/nest_export.py)) and GeNN's built-in `LIF` model ([scripts/genn_benchmark.py](../scripts/genn_benchmark.py)) both use the **exact exponential solution** of the leaky equation between spikes:

$$
V_{n+1} = V_{\text{rest}} + (V_n - V_{\text{rest}})\,e^{-\Delta t/\tau_m} + R\bigl(1 - e^{-\Delta t/\tau_m}\bigr) I_{\text{syn},n}
$$

This is exact for piecewise-constant input, so it has **no $O(\Delta t)$ sub-threshold error and no stability bound on $\Delta t$**. The only residual discretization errors there are (i) the spike-time quantization at the grid (same $\le \Delta t$ bound) and (ii) the assumption that $I_{\text{syn}}$ is constant over a step. That is the main numerical reason why your CPU simulator and the NEST/GeNN backends will not produce bit-identical voltage traces even with identical parameters.
