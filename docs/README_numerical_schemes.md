# Numerical schemes: who specifies them?

This note answers: *for each simulation backend used in this repo, who chooses the
numerical integration scheme — the user, or the package?*

## Summary

| Backend | Who specifies the scheme | Scheme used | User-controlled knobs |
|---|---|---|---|
| C++ `LIFPopulation` (this repo) | **You** (hand-written) | Explicit forward Euler | `dt`, `tau_m`, `t_ref`, thresholds |
| NEST (`iaf_psc_alpha`) | **Package** (per neuron model) | Exact exponential integration (propagator matrix) | parameters + simulation `dt` only |
| GeNN (built-in `LIF`) | **Package** (code-generated kernel) | Exact exponential update of leaky ODE | parameters + `model.dt` only |

So: **only the C++ backend exposes the numerical scheme as an explicit
implementation choice. NEST and GeNN ship the integrator with the neuron model;
you only configure parameters and the timestep.**

---

## 1. Your C++ implementation — you specify the scheme

In [src/lif_neuron.cpp](../src/lif_neuron.cpp) the LIF ODE is integrated with an
explicit **forward Euler** scheme, written by hand:

```cpp
V_[i] += (params_.dt / params_.tau_m)
       * (-(V_[i] - params_.V_rest) + params_.R * I_syn[i]);
```

The threshold is checked at grid points and the absolute refractory period is an
integer countdown of `round(t_ref / dt)` timesteps.

There is no library choice involved: the integrator, the threshold rule, and the
refractory bookkeeping are all part of your source code. If you wanted RK4,
exponential Euler, or sub-step spike-time interpolation, you would have to
modify [src/lif_neuron.cpp](../src/lif_neuron.cpp) yourself.

Default parameters live in [include/lif_neuron.h](../include/lif_neuron.h):
$\tau_m=20$ ms, $V_{\text{rest}}=-65$ mV, $V_{\text{thresh}}=-50$ mV,
$V_{\text{reset}}=-65$ mV, $R=1\,\text{M}\Omega$, $\Delta t=1$ ms, $t_{\text{ref}}=2$ ms.

## 2. NEST — the package specifies the scheme

In [scripts/nest_export.py](../scripts/nest_export.py) the network is built with
the standard model `iaf_psc_alpha`:

```python
exc_pop = nest.Create("iaf_psc_alpha", N_exc, params=neuron_params)
inh_pop = nest.Create("iaf_psc_alpha", N_inh, params=neuron_params)
```

For `iaf_psc_alpha`, NEST integrates the linear sub-threshold dynamics with an
**exact (analytic) exponential integration** based on a precomputed propagator
matrix (Rotter & Diesmann, 1999). The user does **not** pick the integrator —
NEST chooses it per neuron model. You only set:

- the biophysical parameters (`tau_m`, `V_th`, `V_reset`, `E_L`, `t_ref`, …),
- the simulation resolution / `dt`,
- and which neuron model to use (which implicitly picks the integrator).

Note that `iaf_psc_alpha` also adds alpha-shaped synaptic currents (also
integrated exactly), whereas the C++ model takes `I_syn` directly. So the C++
and NEST backends differ both in numerical scheme **and** in synaptic model.

## 3. GeNN — the package specifies the scheme

In [scripts/genn_benchmark.py](../scripts/genn_benchmark.py) the network is built
from GeNN's built-in models:

```python
pop = model.add_neuron_population("pop", N, "LIF", params, LIF_INIT)
wu_init = init_weight_update("StaticPulse", {}, {"g": w})
ps_init = init_postsynaptic("DeltaCurr")
```

GeNN's standard `"LIF"` neuron model uses an **exact exponential update** of the
membrane equation each timestep (derived from the analytic solution of the
leaky equation between spikes). The actual CUDA kernel is *generated* by GeNN
from this model definition; you don't write the integrator. The only knobs you
control are:

- the model class (`"LIF"`, `"StaticPulse"`, `"DeltaCurr"`, …),
- the parameters (`TauM`, `Vthresh`, `Vreset`, `TauRefrac`, …),
- and the timestep via `model.dt = 1.0`.

Connectivity mode (`DENSE` / `SPARSE` / `BITMASK`) only affects the synaptic
*data layout*, not the numerical scheme.

## 4. Practical consequence

Because the three backends use different integrators (forward Euler vs. exact
exponential) and, in NEST's case, different synaptic kinetics, they will **not**
produce bit-identical voltage traces or spike times even with matched
parameters. Any cross-backend validation in this repo should therefore compare
*statistics* (firing rates, ISI distributions, population sync) rather than
exact spike times. See also [docs/README_lif_discretization.md](README_lif_discretization.md)
for the error analysis of the C++ forward-Euler scheme.
