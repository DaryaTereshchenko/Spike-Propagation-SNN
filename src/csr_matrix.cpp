#include "csr_matrix.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

CsrMatrix::CsrMatrix(std::size_t rows, std::size_t cols,
                     std::vector<std::size_t> rowptr,
                     std::vector<std::size_t> colidx,
                     std::vector<double>      values)
    : rows_(rows), cols_(cols),
      rowptr_(std::move(rowptr)),
      colidx_(std::move(colidx)),
      values_(std::move(values)) {}

CsrMatrix CsrMatrix::from_coo(std::size_t rows, std::size_t cols,
                                const std::vector<std::size_t>& row_idx,
                                const std::vector<std::size_t>& col_idx,
                                const std::vector<double>&      values) {
    const std::size_t nz = row_idx.size();

    // Count entries per row.
    std::vector<std::size_t> rowptr(rows + 1, 0);
    for (std::size_t k = 0; k < nz; ++k) {
        rowptr[row_idx[k] + 1]++;
    }
    // Prefix-sum to get row pointers.
    std::partial_sum(rowptr.begin(), rowptr.end(), rowptr.begin());

    // Fill column indices and values using a temporary position array.
    std::vector<std::size_t> pos(rowptr.begin(), rowptr.end() - 1);
    std::vector<std::size_t> colidx_out(nz);
    std::vector<double>      values_out(nz);
    for (std::size_t k = 0; k < nz; ++k) {
        std::size_t r = row_idx[k];
        std::size_t p = pos[r]++;
        colidx_out[p] = col_idx[k];
        values_out[p] = values[k];
    }

    return CsrMatrix(rows, cols,
                     std::move(rowptr),
                     std::move(colidx_out),
                     std::move(values_out));
}

void CsrMatrix::scatter(const std::vector<bool>& spikes,
                        std::vector<double>& output) const {
    for (std::size_t i = 0; i < rows_; ++i) {
        if (!spikes[i]) continue;
        for (std::size_t p = rowptr_[i]; p < rowptr_[i + 1]; ++p) {
            output[colidx_[p]] += values_[p];
        }
    }
}

void CsrMatrix::gather(const std::vector<bool>& spikes,
                       std::vector<double>& output) const {
    // Pull: iterate over every target column j and accumulate contributions
    // from spiking sources. For CSR this requires a full scan of each row.
    for (std::size_t i = 0; i < rows_; ++i) {
        if (!spikes[i]) continue;
        for (std::size_t p = rowptr_[i]; p < rowptr_[i + 1]; ++p) {
            output[colidx_[p]] += values_[p];
        }
    }
}

std::size_t CsrMatrix::memory_bytes() const {
    return rowptr_.size() * sizeof(std::size_t)
         + colidx_.size() * sizeof(std::size_t)
         + values_.size()  * sizeof(double);
}
