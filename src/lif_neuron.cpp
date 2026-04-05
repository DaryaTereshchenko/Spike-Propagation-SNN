#include "lif_neuron.hpp"
#include <random>
#include <stdexcept>

LifNeuronPop::LifNeuronPop(std::size_t n)
    : params_(Params{}), v_(n), refrac_(n, 0) {
    reset(42);
}

LifNeuronPop::LifNeuronPop(std::size_t n, const Params& p)
    : params_(p), v_(n), refrac_(n, 0) {
    reset(42);
}

void LifNeuronPop::reset(unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(params_.V_rest, params_.V_thresh);
    for (auto& v : v_) {
        v = dist(rng);
    }
    std::fill(refrac_.begin(), refrac_.end(), 0);
}

std::vector<bool> LifNeuronPop::step(const std::vector<double>& I_syn) {
    const std::size_t n = v_.size();
    std::vector<bool> spikes(n, false);

    const double alpha = params_.dt / params_.tau_m;

    for (std::size_t i = 0; i < n; ++i) {
        if (refrac_[i] > 0) {
            --refrac_[i];
            v_[i] = params_.V_reset;
            continue;
        }

        // Forward Euler LIF update.
        v_[i] += alpha * (-(v_[i] - params_.V_rest) + params_.R * I_syn[i]);

        if (v_[i] >= params_.V_thresh) {
            spikes[i]   = true;
            v_[i]       = params_.V_reset;
            refrac_[i]  = params_.refrac_steps;
        }
    }

    return spikes;
}
