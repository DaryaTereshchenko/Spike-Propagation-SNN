#include "csc_matrix.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

CscMatrix::CscMatrix(std::size_t rows, std::size_t cols,
                     std::vector<std::size_t> colptr,
                     std::vector<std::size_t> rowidx,
                     std::vector<double>      values)
    : rows_(rows), cols_(cols),
      colptr_(std::move(colptr)),
      rowidx_(std::move(rowidx)),
      values_(std::move(values)) {}

CscMatrix CscMatrix::from_coo(std::size_t rows, std::size_t cols,
                                const std::vector<std::size_t>& row_idx,
                                const std::vector<std::size_t>& col_idx,
                                const std::vector<double>&      values) {
    const std::size_t nz = col_idx.size();

    // Count entries per column.
    std::vector<std::size_t> colptr(cols + 1, 0);
    for (std::size_t k = 0; k < nz; ++k) {
        colptr[col_idx[k] + 1]++;
    }
    std::partial_sum(colptr.begin(), colptr.end(), colptr.begin());

    // Fill row indices and values.
    std::vector<std::size_t> pos(colptr.begin(), colptr.end() - 1);
    std::vector<std::size_t> rowidx_out(nz);
    std::vector<double>      values_out(nz);
    for (std::size_t k = 0; k < nz; ++k) {
        std::size_t c = col_idx[k];
        std::size_t p = pos[c]++;
        rowidx_out[p] = row_idx[k];
        values_out[p] = values[k];
    }

    return CscMatrix(rows, cols,
                     std::move(colptr),
                     std::move(rowidx_out),
                     std::move(values_out));
}

void CscMatrix::scatter(const std::vector<bool>& spikes,
                        std::vector<double>& output) const {
    // Push: for CSC, we must scan all columns to find spiking sources.
    // Less cache-friendly than CSR for scatter.
    for (std::size_t j = 0; j < cols_; ++j) {
        for (std::size_t p = colptr_[j]; p < colptr_[j + 1]; ++p) {
            if (spikes[rowidx_[p]]) {
                output[j] += values_[p];
            }
        }
    }
}

void CscMatrix::gather(const std::vector<bool>& spikes,
                       std::vector<double>& output) const {
    // Pull: iterate column j, sum weights from spiking source neurons.
    // Contiguous rowidx per column gives good cache behaviour for gather.
    for (std::size_t j = 0; j < cols_; ++j) {
        double acc = 0.0;
        for (std::size_t p = colptr_[j]; p < colptr_[j + 1]; ++p) {
            if (spikes[rowidx_[p]]) {
                acc += values_[p];
            }
        }
        output[j] += acc;
    }
}

std::size_t CscMatrix::memory_bytes() const {
    return colptr_.size() * sizeof(std::size_t)
         + rowidx_.size() * sizeof(std::size_t)
         + values_.size()  * sizeof(double);
}
