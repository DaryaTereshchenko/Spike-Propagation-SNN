# Leaky Integrate-and-Fire (LIF) Neuron Model

## Overview

The Leaky Integrate-and-Fire (LIF) model is the most widely used
point-neuron model in computational neuroscience.  It captures the
essential dynamics of neuronal membrane potential integration while
remaining analytically tractable and computationally efficient.

---

## Continuous-Time Equations

The subthreshold membrane potential $V(t)$ of a LIF neuron obeys:

$$
\tau_m \frac{dV}{dt} = -(V - V_{\text{rest}}) + R \, I_{\text{syn}}(t)
$$

where:

| Symbol | Meaning | Default value |
|--------|---------|---------------|
| $\tau_m$ | Membrane time constant | 20 ms |
| $V_{\text{rest}}$ | Resting (leak) potential | −65 mV |
| $R$ | Membrane resistance | 1.0 MΩ (normalized) |
| $I_{\text{syn}}(t)$ | Total synaptic input current | (computed) |

When $V(t)$ reaches the threshold $V_{\text{thresh}} = -50$ mV, the neuron
emits a spike, $V$ is reset to $V_{\text{rest}}$, and the neuron enters an
absolute refractory period of $t_{\text{ref}} = 2$ ms during which $V$
remains clamped at $V_{\text{rest}}$.

---

## Discrete-Time Forward Euler Integration

The continuous ODE is discretised with a forward (explicit) Euler step of
size $\Delta t = 1$ ms:

$$
V^{(n+1)} = V^{(n)} + \frac{\Delta t}{\tau_m} \left[ -(V^{(n)} - V_{\text{rest}}) + R \, I_{\text{syn}}^{(n)} \right]
$$

### Spike and Reset Logic

```
if V[i] >= V_thresh:
    spike[i] = true
    V[i]     = V_rest
    ref[i]   = ref_steps        # ceil(t_ref / dt)
```

During the refractory period ($\text{ref}[i] > 0$), integration is skipped
and the counter is decremented each timestep.

### Stability Condition

Forward Euler is conditionally stable for the LIF equation; the scheme is
stable when:

$$
0 < \frac{\Delta t}{\tau_m} < 2
$$

With $\Delta t = 1$ ms and $\tau_m = 20$ ms we have
$\Delta t / \tau_m = 0.05 \ll 2$, so the scheme is well within the stability
region.

### Approximation Error

The forward Euler method introduces a local truncation error of
$O(\Delta t^2)$ per step, giving a global error of $O(\Delta t)$.  For
benchmarking purposes this accuracy is sufficient; production simulators
(NEST, Brian2) use exact integration or higher-order methods.

---

## Biological Context

The LIF model omits many biophysical details (sodium/potassium channel
kinetics, dendritic morphology, adaptation currents), but faithfully
reproduces:

1. **Membrane time constant filtering** — low-pass integration of synaptic
   inputs.
2. **Threshold crossing** — all-or-none spike generation.
3. **Refractory period** — minimum inter-spike interval, preventing
   unrealistically high firing rates.

For the purpose of comparing sparse-format performance, the LIF model
provides a computationally inexpensive per-neuron update so that runtime
is dominated by the spike-propagation kernel (matrix–vector operation).

---

## API Reference

### `LIFParams` (parameters)

```cpp
struct LIFParams {
    double V_rest    = -65.0;    // mV
    double V_thresh  = -50.0;    // mV
    double tau_m     = 20.0;     // ms
    double R         = 1.0;      // MΩ (normalized)
    double dt        = 1.0;      // ms
    double t_ref     = 2.0;      // ms
};
```

### `LIFPopulation` (state + methods)

```cpp
class LIFPopulation {
public:
    explicit LIFPopulation(int N, const LIFParams& p = LIFParams{});

    void step(const std::vector<double>& I_syn,
              std::vector<bool>& spikes);
    void reset();

    const std::vector<double>& voltages() const;
    int size() const;
};
```

- `step()` — performs one forward-Euler timestep; fills `spikes[i]=true` for
  each neuron that crossed threshold.
- `reset()` — reinitialises all voltages to $V_{\text{rest}}$ and clears
  refractory counters.

### Files

| File | Purpose |
|------|---------|
| `include/lif_neuron.h` | `LIFParams` and `LIFPopulation` declarations |
| `src/lif_neuron.cpp`   | Forward Euler integration implementation |
