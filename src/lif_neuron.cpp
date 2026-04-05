#include "lif_neuron.h"
#include <cmath>

LIFPopulation::LIFPopulation(int N, LIFParams params)
    : N_(N), params_(params)
{
    V_.assign(N_, params_.V_rest);
    ref_counter_.assign(N_, 0);
}

std::vector<int> LIFPopulation::step(const std::vector<double>& I_syn)
{
    std::vector<int> spikes;
    const int ref_steps = static_cast<int>(std::round(params_.t_ref / params_.dt));

    for (int i = 0; i < N_; ++i) {
        // If in refractory period, decrement counter and skip.
        if (ref_counter_[i] > 0) {
            --ref_counter_[i];
            continue;
        }

        // Forward Euler update.
        V_[i] += (params_.dt / params_.tau_m)
               * (-(V_[i] - params_.V_rest) + params_.R * I_syn[i]);

        // Threshold crossing → spike.
        if (V_[i] >= params_.V_thresh) {
            spikes.push_back(i);
            V_[i] = params_.V_reset;
            ref_counter_[i] = ref_steps;
        }
    }
    return spikes;
}

void LIFPopulation::reset()
{
    std::fill(V_.begin(), V_.end(), params_.V_rest);
    std::fill(ref_counter_.begin(), ref_counter_.end(), 0);
}
