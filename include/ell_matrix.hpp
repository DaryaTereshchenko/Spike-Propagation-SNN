#pragma once
#include "sparse_matrix.hpp"
#include <vector>
#include <cstddef>
#include <limits>

// ELL (ELLPACK) sparse matrix format.
// Each row stores exactly max_cols_ entries; shorter rows are padded with
// a sentinel column index (cols_) and weight 0.
// Regular structure benefits GPU coalesced access and SIMD on CPU.
// Degrades on scale-free networks where max_degree >> mean_degree causes
// heavy padding waste.
class EllMatrix : public SparseMatrix {
public:
    // Build ELL from COO-style (row, col, val) triples.
    // max_cols is derived automatically as max row degree.
    static EllMatrix from_coo(std::size_t rows, std::size_t cols,
                               const std::vector<std::size_t>& row_idx,
                               const std::vector<std::size_t>& col_idx,
                               const std::vector<double>&      values);

    std::size_t rows() const override { return rows_; }
    std::size_t cols() const override { return cols_; }
    std::size_t nnz()  const override { return nnz_; }

    // Returns the allocated slots per row (includes padding).
    std::size_t max_cols() const { return max_cols_; }

    void scatter(const std::vector<bool>& spikes,
                 std::vector<double>& output) const override;

    void gather(const std::vector<bool>& spikes,
                std::vector<double>& output) const override;

    std::size_t memory_bytes() const override;

private:
    EllMatrix() = default;

    std::size_t rows_{0};
    std::size_t cols_{0};
    std::size_t nnz_{0};
    std::size_t max_cols_{0};                 // padded width per row

    // col_idx_[i * max_cols_ + k]: k-th column for row i (cols_ = padding sentinel)
    std::vector<std::size_t> col_idx_;
    // values_[i * max_cols_ + k]: corresponding weight
    std::vector<double>      values_;
};
