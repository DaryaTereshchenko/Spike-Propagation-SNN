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

Backward Euler (implicit) integration is also possible
Centered Euler

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

## Background Current Injection

### Motivation

With default parameters the recurrent weight normalisation
($w = G/\sqrt{K_{\text{avg}}}$) scales synaptic weights inversely with
network size.  At large $N$ (e.g. 10 000) this scaling causes the network
to fall silent after the initial seed spikes at $t = 0$: the synaptic input
from ~100 initial spikes is far below the 15 mV threshold gap
($V_{\text{thresh}} - V_{\text{rest}}$), and there is no sustained drive to
keep neurons firing.  A silent network means the benchmark only measures
cache pollution from an empty scatter call — not real spike-propagation cost.

### Solution

A constant **background current** $I_{\text{bg}}$ is added to every neuron's
synaptic input at each timestep:

$$
I_{\text{syn},i}(t) \leftarrow I_{\text{syn},i}(t) + I_{\text{bg}}
$$

This is applied *after* the recurrent scatter and external Poisson drive,
before the LIF integration step.

### Parameter Choice

The equilibrium membrane potential under constant input $I$ is:

$$
V_{\text{eq}} = V_{\text{rest}} + R \cdot I
$$

Setting $I_{\text{bg}} \approx 14$ mV places $V_{\text{eq}} = -65 + 14 = -51$ mV,
just above $V_{\text{thresh}} = -50$ mV.  Combined with the stochastic Poisson
drive and recurrent input, this produces a sustained firing rate of roughly
5–8% of the population per timestep — a biologically plausible operating
point for cortical networks and sufficient to stress-test the spike-
propagation kernel across all timesteps.

### CLI Usage

```bash
./build/spike_benchmark --bg-current 14.0 ...
```

Default: `0.0` (no background current — original behaviour preserved).

---

## Controlled Spike Injection

### Motivation

Even with background current the actual spike rate depends on LIF dynamics,
Poisson drive, and recurrent feedback — making it difficult to isolate the
effect of spike rate on sparse-format performance.  To measure the pure
cost curve (scatter/gather time as a function of network activity) an
independent spike-rate control is needed.

### Solution

When `--inject-rate F` is set ($F \in (0, 1]$), the LIF spike output is
**overridden** each timestep: instead of using the neurons that crossed
threshold, a random fraction $F$ of all $N$ neurons is selected as spiking
(uniformly, independently per timestep).  The LIF integration still runs
(to preserve its cache/compute overhead in the timing), but its spikes are
discarded.

This decouples the sparse-matrix workload from the network dynamics,
enabling controlled experiments such as:

- **Format crossover analysis**: at what activity level does ELL's strided
  access outperform CSR's row-indexed access?
- **Scaling studies**: how does scatter time grow with spike fraction for
  each format at fixed $N$ and density?

### CLI Usage

```bash
# 5% of neurons spike each timestep (independent of LIF)
./build/spike_benchmark --inject-rate 0.05 ...

# Sweep injection rates
./build/spike_benchmark --sweep --inject-rate 0.10 \
    --sweep-rates "5 10 15 20 25 30" ...
```

Default: `0.0` (disabled — LIF dynamics determine spikes).

### Expected Outcomes

| Inject rate | Expected spikes/step (N=5000) | Purpose |
|-------------|-------------------------------|---------|
| 0.01 | ~50 | Sparse activity — CSR should dominate |
| 0.05 | ~250 | Moderate activity — realistic cortical regime |
| 0.10 | ~500 | High activity — memory bandwidth becomes bottleneck |
| 0.25 | ~1250 | Stress test — reveals format scaling limits |

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
