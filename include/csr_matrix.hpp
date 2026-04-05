#pragma once
#include "sparse_matrix.hpp"
#include <vector>
#include <cstddef>

// CSR (Compressed Sparse Row) format.
// rowptr[i]..rowptr[i+1] gives the range in colidx/values for source neuron i.
// Excellent for push (scatter): contiguous column indices per row yield good cache utilisation.
class CsrMatrix : public SparseMatrix {
public:
    CsrMatrix(std::size_t rows, std::size_t cols,
              std::vector<std::size_t> rowptr,
              std::vector<std::size_t> colidx,
              std::vector<double>      values);

    // Factory: build from COO-style (row, col, val) triples (need not be sorted).
    static CsrMatrix from_coo(std::size_t rows, std::size_t cols,
                               const std::vector<std::size_t>& row_idx,
                               const std::vector<std::size_t>& col_idx,
                               const std::vector<double>&      values);

    std::size_t rows() const override { return rows_; }
    std::size_t cols() const override { return cols_; }
    std::size_t nnz()  const override { return colidx_.size(); }

    void scatter(const std::vector<bool>& spikes,
                 std::vector<double>& output) const override;

    void gather(const std::vector<bool>& spikes,
                std::vector<double>& output) const override;

    std::size_t memory_bytes() const override;

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<std::size_t> rowptr_;   // size rows_+1
    std::vector<std::size_t> colidx_;   // size nnz
    std::vector<double>      values_;   // size nnz
};
