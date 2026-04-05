#include "csr_matrix.h"
#include <algorithm>
#include <numeric>

CSRMatrix::CSRMatrix(const COOTriplets& t)
    : nrows_(t.N), ncols_(t.N)
{
    const size_t nnz = t.nnz();

    // Sort indices by (row, col).
    std::vector<size_t> perm(nnz);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
        if (t.rows[a] != t.rows[b]) return t.rows[a] < t.rows[b];
        return t.cols[a] < t.cols[b];
    });

    // Build CSR arrays.
    row_ptr_.assign(nrows_ + 1, 0);
    col_idx_.resize(nnz);
    val_.resize(nnz);

    for (size_t i = 0; i < nnz; ++i) {
        size_t idx  = perm[i];
        col_idx_[i] = t.cols[idx];
        val_[i]     = t.vals[idx];
        row_ptr_[t.rows[idx] + 1]++;
    }

    // Prefix sum to get row pointers.
    for (int r = 0; r < nrows_; ++r) {
        row_ptr_[r + 1] += row_ptr_[r];
    }
}

void CSRMatrix::scatter(const std::vector<int>& spike_sources,
                        std::vector<double>&    out_buffer) const
{
    for (int r : spike_sources) {
        const int start = row_ptr_[r];
        const int end   = row_ptr_[r + 1];
        for (int j = start; j < end; ++j) {
            out_buffer[col_idx_[j]] += val_[j];
        }
    }
}

double CSRMatrix::gather(int target,
                         const std::vector<int>& spike_sources) const
{
    // CSR is not optimal for gather – we must scan each spiking row's entries
    // looking for the target column.
    double sum = 0.0;
    for (int r : spike_sources) {
        const int start = row_ptr_[r];
        const int end   = row_ptr_[r + 1];
        for (int j = start; j < end; ++j) {
            if (col_idx_[j] == target) {
                sum += val_[j];
                break;  // at most one entry per (row, col) pair
            }
        }
    }
    return sum;
}

size_t CSRMatrix::memory_bytes() const
{
    return row_ptr_.size()  * sizeof(int)
         + col_idx_.size()  * sizeof(int)
         + val_.size()      * sizeof(double);
}
