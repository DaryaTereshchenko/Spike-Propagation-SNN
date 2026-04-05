#pragma once

#include <vector>

/// Parameters for the Leaky Integrate-and-Fire (LIF) neuron model.
struct LIFParams {
    double tau_m    = 20.0;     // Membrane time constant (ms)
    double V_rest   = -65.0;    // Resting potential (mV)
    double V_thresh = -50.0;    // Spike threshold (mV)
    double V_reset  = -65.0;    // Reset potential after spike (mV)
    double R        = 1.0;      // Membrane resistance (MΩ)
    double dt       = 1.0;      // Simulation timestep (ms)
    double t_ref    = 2.0;      // Absolute refractory period (ms)
};

/// A population of N leaky integrate-and-fire neurons.
///
/// Each neuron's membrane potential evolves via forward Euler:
///   V[i] += (dt / tau_m) * (-(V[i] - V_rest) + R * I_syn[i])
///
/// When V[i] >= V_thresh, neuron i emits a spike, its potential is reset to
/// V_reset, and it enters an absolute refractory period of t_ref ms during
/// which no updates occur.
class LIFPopulation {
public:
    /// Create a population of @p N neurons with the given parameters.
    /// All membrane potentials are initialized to V_rest.
    explicit LIFPopulation(int N, LIFParams params = {});

    /// Advance the population by one timestep.
    /// @param I_syn  Synaptic input current for each neuron (size N).
    /// @return Indices of neurons that spiked during this timestep.
    std::vector<int> step(const std::vector<double>& I_syn);

    /// Reset all neurons to initial state (V = V_rest, no refractory).
    void reset();

    /// Access membrane potentials (for testing / inspection).
    const std::vector<double>& voltages() const { return V_; }

    int size() const { return N_; }

private:
    int       N_;
    LIFParams params_;
    std::vector<double> V_;             // membrane potentials
    std::vector<int>    ref_counter_;   // refractory countdown (in timesteps)
};
