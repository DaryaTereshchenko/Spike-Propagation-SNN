#pragma once
#include "sparse_matrix.hpp"
#include <vector>
#include <cstddef>

// CSC (Compressed Sparse Column) format.
// colptr[j]..colptr[j+1] gives the range in rowidx/values for target neuron j.
// Excellent for pull (gather): contiguous row indices per column.
class CscMatrix : public SparseMatrix {
public:
    CscMatrix(std::size_t rows, std::size_t cols,
              std::vector<std::size_t> colptr,
              std::vector<std::size_t> rowidx,
              std::vector<double>      values);

    // Factory: build from COO-style (row, col, val) triples.
    static CscMatrix from_coo(std::size_t rows, std::size_t cols,
                               const std::vector<std::size_t>& row_idx,
                               const std::vector<std::size_t>& col_idx,
                               const std::vector<double>&      values);

    std::size_t rows() const override { return rows_; }
    std::size_t cols() const override { return cols_; }
    std::size_t nnz()  const override { return rowidx_.size(); }

    void scatter(const std::vector<bool>& spikes,
                 std::vector<double>& output) const override;

    void gather(const std::vector<bool>& spikes,
                std::vector<double>& output) const override;

    std::size_t memory_bytes() const override;

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<std::size_t> colptr_;   // size cols_+1
    std::vector<std::size_t> rowidx_;   // size nnz
    std::vector<double>      values_;   // size nnz
};
