#include "coo_matrix.h"
#include <algorithm>

COOMatrix::COOMatrix(const COOTriplets& t)
    : nrows_(t.N), ncols_(t.N), row_(t.rows), col_(t.cols), val_(t.vals)
{
}

void COOMatrix::scatter(const std::vector<int>& spike_sources,
                        std::vector<double>&    out_buffer) const
{
    // Build a fast boolean lookup for spiking neurons.
    std::vector<bool> is_spiking(nrows_, false);
    for (int s : spike_sources) {
        is_spiking[s] = true;
    }

    // Scan every stored non-zero.
    const size_t nnz = row_.size();
    for (size_t i = 0; i < nnz; ++i) {
        if (is_spiking[row_[i]]) {
            out_buffer[col_[i]] += val_[i];
        }
    }
}

double COOMatrix::gather(int target,
                         const std::vector<int>& spike_sources) const
{
    std::vector<bool> is_spiking(nrows_, false);
    for (int s : spike_sources) {
        is_spiking[s] = true;
    }

    double sum = 0.0;
    const size_t nnz = row_.size();
    for (size_t i = 0; i < nnz; ++i) {
        if (col_[i] == target && is_spiking[row_[i]]) {
            sum += val_[i];
        }
    }
    return sum;
}

size_t COOMatrix::memory_bytes() const
{
    return row_.size() * sizeof(int)
         + col_.size() * sizeof(int)
         + val_.size() * sizeof(double);
}
