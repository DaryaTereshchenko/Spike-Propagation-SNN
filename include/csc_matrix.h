#pragma once

#include "sparse_matrix.h"
#include <vector>

/// CSC (Compressed Sparse Column) sparse matrix format.
///
/// Stores non-zeros grouped by column.  For an N-column matrix with nnz entries:
///   col_ptr[N+1]  — col_ptr[c] is the index into row_idx/val where column c begins.
///   row_idx[nnz]  — row indices of the non-zeros.
///   val[nnz]      — values of the non-zeros.
///
/// Gather is efficient: for target column c, iterate
/// row_idx[col_ptr[c] .. col_ptr[c+1]) and sum weights from spiking sources.
/// Complexity: O(degree(target)).
class CSCMatrix : public SparseMatrix {
public:
    /// Construct from COO triplets.  Internally sorts by (col, row).
    explicit CSCMatrix(const COOTriplets& triplets);

    void   scatter(const std::vector<int>& spike_sources,
                   std::vector<double>&    out_buffer) const override;
    double gather(int target,
                  const std::vector<int>& spike_sources) const override;
    void   gather_all(const std::vector<int>& spike_sources,
                      std::vector<double>&    out_buffer) const override;

    size_t memory_bytes() const override;
    int    num_rows()     const override { return nrows_; }
    int    num_cols()     const override { return ncols_; }
    size_t num_nonzeros() const override { return row_idx_.size(); }

    void save_sparsity_pattern(const std::string& filename,
                               int resolution = 256) const override;

    void save_storage_layout(const std::string& filename,
                             int resolution = 512) const override;

    void dump() const override;

private:
    int nrows_, ncols_;
    std::vector<int>    col_ptr_;   // size N+1
    std::vector<int>    row_idx_;   // size nnz
    std::vector<double> val_;       // size nnz
};
