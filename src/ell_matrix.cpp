#include "ell_matrix.h"
#include <algorithm>

ELLMatrix::ELLMatrix(const COOTriplets& t)
    : nrows_(t.N), ncols_(t.N), nnz_(t.nnz()), max_nnz_(0)
{
    // Count non-zeros per row to determine max_nnz_per_row.
    std::vector<int> row_counts(nrows_, 0);
    for (size_t i = 0; i < nnz_; ++i) {
        row_counts[t.rows[i]]++;
    }
    max_nnz_ = *std::max_element(row_counts.begin(), row_counts.end());

    // Allocate padded arrays.
    const size_t total = static_cast<size_t>(nrows_) * max_nnz_;
    indices_.assign(total, -1);       // sentinel
    values_.assign(total, 0.0);

    // Fill in entries row by row.
    std::vector<int> cursor(nrows_, 0);   // next free slot per row
    for (size_t i = 0; i < nnz_; ++i) {
        int r   = t.rows[i];
        int pos = r * max_nnz_ + cursor[r];
        indices_[pos] = t.cols[i];
        values_[pos]  = t.vals[i];
        cursor[r]++;
    }
}

void ELLMatrix::scatter(const std::vector<int>& spike_sources,
                        std::vector<double>&    out_buffer) const
{
    for (int r : spike_sources) {
        const int base = r * max_nnz_;
        for (int k = 0; k < max_nnz_; ++k) {
            int c = indices_[base + k];
            if (c < 0) break;           // sentinel → no more entries
            out_buffer[c] += values_[base + k];
        }
    }
}

double ELLMatrix::gather(int target,
                         const std::vector<int>& spike_sources) const
{
    // ELL is row-oriented → gather requires scanning spiking rows.
    std::vector<bool> is_spiking(nrows_, false);
    for (int s : spike_sources) {
        is_spiking[s] = true;
    }

    double sum = 0.0;
    for (int r = 0; r < nrows_; ++r) {
        if (!is_spiking[r]) continue;
        const int base = r * max_nnz_;
        for (int k = 0; k < max_nnz_; ++k) {
            int c = indices_[base + k];
            if (c < 0) break;
            if (c == target) {
                sum += values_[base + k];
            }
        }
    }
    return sum;
}

size_t ELLMatrix::memory_bytes() const
{
    return indices_.size() * sizeof(int)
         + values_.size()  * sizeof(double);
}
