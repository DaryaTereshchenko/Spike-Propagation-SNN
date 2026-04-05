#include "ell_matrix.hpp"
#include <algorithm>
#include <stdexcept>

EllMatrix EllMatrix::from_coo(std::size_t rows, std::size_t cols,
                               const std::vector<std::size_t>& row_idx,
                               const std::vector<std::size_t>& col_idx,
                               const std::vector<double>&      values) {
    const std::size_t nz = row_idx.size();

    // Compute degree of each row to find max_cols.
    std::vector<std::size_t> degree(rows, 0);
    for (std::size_t k = 0; k < nz; ++k) {
        degree[row_idx[k]]++;
    }
    const std::size_t max_cols = *std::max_element(degree.begin(), degree.end());

    // Allocate and initialise with padding sentinel.
    EllMatrix m;
    m.rows_     = rows;
    m.cols_     = cols;
    m.nnz_      = nz;
    m.max_cols_ = max_cols;
    m.col_idx_.assign(rows * max_cols, cols);   // sentinel = cols (out of range)
    m.values_.assign(rows * max_cols, 0.0);

    // Fill in actual entries, tracking position within each row.
    std::vector<std::size_t> pos(rows, 0);
    for (std::size_t k = 0; k < nz; ++k) {
        std::size_t r = row_idx[k];
        std::size_t slot = r * max_cols + pos[r];
        m.col_idx_[slot] = col_idx[k];
        m.values_[slot]  = values[k];
        pos[r]++;
    }

    return m;
}

void EllMatrix::scatter(const std::vector<bool>& spikes,
                        std::vector<double>& output) const {
    for (std::size_t i = 0; i < rows_; ++i) {
        if (!spikes[i]) continue;
        const std::size_t base = i * max_cols_;
        for (std::size_t k = 0; k < max_cols_; ++k) {
            const std::size_t j = col_idx_[base + k];
            if (j < cols_) {   // skip padding sentinel
                output[j] += values_[base + k];
            }
        }
    }
}

void EllMatrix::gather(const std::vector<bool>& spikes,
                       std::vector<double>& output) const {
    // Pull: same traversal order as scatter for ELL.
    for (std::size_t i = 0; i < rows_; ++i) {
        if (!spikes[i]) continue;
        const std::size_t base = i * max_cols_;
        for (std::size_t k = 0; k < max_cols_; ++k) {
            const std::size_t j = col_idx_[base + k];
            if (j < cols_) {
                output[j] += values_[base + k];
            }
        }
    }
}

std::size_t EllMatrix::memory_bytes() const {
    return col_idx_.size() * sizeof(std::size_t)
         + values_.size()  * sizeof(double);
}
