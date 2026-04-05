#include "csc_matrix.h"
#include <algorithm>
#include <numeric>

CSCMatrix::CSCMatrix(const COOTriplets& t)
    : nrows_(t.N), ncols_(t.N)
{
    const size_t nnz = t.nnz();

    // Sort indices by (col, row).
    std::vector<size_t> perm(nnz);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
        if (t.cols[a] != t.cols[b]) return t.cols[a] < t.cols[b];
        return t.rows[a] < t.rows[b];
    });

    // Build CSC arrays.
    col_ptr_.assign(ncols_ + 1, 0);
    row_idx_.resize(nnz);
    val_.resize(nnz);

    for (size_t i = 0; i < nnz; ++i) {
        size_t idx  = perm[i];
        row_idx_[i] = t.rows[idx];
        val_[i]     = t.vals[idx];
        col_ptr_[t.cols[idx] + 1]++;
    }

    // Prefix sum to get column pointers.
    for (int c = 0; c < ncols_; ++c) {
        col_ptr_[c + 1] += col_ptr_[c];
    }
}

void CSCMatrix::scatter(const std::vector<int>& spike_sources,
                        std::vector<double>&    out_buffer) const
{
    // CSC is not optimal for scatter – we must scan every column and check
    // whether each entry's row is a spiking source.
    std::vector<bool> is_spiking(nrows_, false);
    for (int s : spike_sources) {
        is_spiking[s] = true;
    }

    for (int c = 0; c < ncols_; ++c) {
        const int start = col_ptr_[c];
        const int end   = col_ptr_[c + 1];
        for (int j = start; j < end; ++j) {
            if (is_spiking[row_idx_[j]]) {
                out_buffer[c] += val_[j];
            }
        }
    }
}

double CSCMatrix::gather(int target,
                         const std::vector<int>& spike_sources) const
{
    // Optimal for gather: directly iterate the target column's entries.
    std::vector<bool> is_spiking(nrows_, false);
    for (int s : spike_sources) {
        is_spiking[s] = true;
    }

    double sum = 0.0;
    const int start = col_ptr_[target];
    const int end   = col_ptr_[target + 1];
    for (int j = start; j < end; ++j) {
        if (is_spiking[row_idx_[j]]) {
            sum += val_[j];
        }
    }
    return sum;
}

size_t CSCMatrix::memory_bytes() const
{
    return col_ptr_.size()  * sizeof(int)
         + row_idx_.size()  * sizeof(int)
         + val_.size()      * sizeof(double);
}
