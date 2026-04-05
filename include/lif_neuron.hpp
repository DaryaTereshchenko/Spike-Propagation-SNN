#pragma once
#include <vector>
#include <cstddef>

// Minimal Leaky Integrate-and-Fire (LIF) neuron population.
//
// Dynamics (forward Euler):
//   tau_m * dV/dt = -(V - V_rest) + R * I_syn
//   V[t+1] = V[t] + dt/tau_m * (-(V[t] - V_rest) + R * I_syn[t])
//
// Spike condition:  V[t+1] >= V_thresh
// After spike:      V is reset to V_reset, and the neuron enters a
//                   refractory period of refrac_steps timesteps.
class LifNeuronPop {
public:
    struct Params {
        double tau_m   = 20.0;   // membrane time constant (ms)
        double V_rest  = -65.0;  // resting potential (mV)
        double V_reset = -65.0;  // reset potential after spike (mV)
        double V_thresh = -50.0; // spike threshold (mV)
        double R        = 1.0;   // membrane resistance (MOhm)
        double dt       = 0.1;   // timestep (ms)
        int    refrac_steps = 20;// absolute refractory period (in timesteps)
    };

    // Construct with default Leaky Integrate-and-Fire parameters.
    explicit LifNeuronPop(std::size_t n);
    // Construct with custom parameters.
    LifNeuronPop(std::size_t n, const Params& p);

    // Advance one timestep.
    // I_syn[i] is the total synaptic current (in nA) arriving at neuron i.
    // Returns a bool vector: true if neuron i fired this step.
    std::vector<bool> step(const std::vector<double>& I_syn);

    // Reset all membrane potentials and refractory counters to their
    // initial state (uniformly distributed in [V_rest, V_thresh]).
    void reset(unsigned seed = 42);

    std::size_t size() const { return v_.size(); }
    const std::vector<double>& voltages() const { return v_; }

private:
    Params params_;
    std::vector<double> v_;       // membrane voltages
    std::vector<int>    refrac_;  // remaining refractory steps per neuron
};
