#pragma once
#include <cstddef>
#include <vector>

// Abstract interface for sparse connectivity matrices used in spike propagation.
// All formats expose three operations:
//   scatter  – push-based delivery: for each spiking neuron i, add w[i][j] to out[j]
//   gather   – pull-based delivery: for each target neuron j, sum incoming weights from spiking neurons
//   memory_bytes – memory footprint of the stored format
class SparseMatrix {
public:
    virtual ~SparseMatrix() = default;

    // Returns number of pre-synaptic (source) neurons.
    virtual std::size_t rows() const = 0;

    // Returns number of post-synaptic (target) neurons.
    virtual std::size_t cols() const = 0;

    // Returns total number of non-zero (synaptic) entries.
    virtual std::size_t nnz() const = 0;

    // Push-based spike delivery.
    // spikes[i] == true means neuron i fired.
    // Adds weight * spikes[i] contributions to output[j] for every synapse (i, j).
    virtual void scatter(const std::vector<bool>& spikes,
                         std::vector<double>& output) const = 0;

    // Pull-based spike delivery.
    // spikes[i] == true means neuron i fired.
    // For each target neuron j, sums incoming weights from spiking source neurons.
    virtual void gather(const std::vector<bool>& spikes,
                        std::vector<double>& output) const = 0;

    // Returns the total memory used by the format's arrays (in bytes).
    virtual std::size_t memory_bytes() const = 0;
};
