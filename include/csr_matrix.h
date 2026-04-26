#pragma once

#include "sparse_matrix.h"
#include <vector>

/// CSR (Compressed Sparse Row) sparse matrix format.
///
/// Stores non-zeros grouped by row.  For an N-row matrix with nnz entries:
///   row_ptr[N+1]  — row_ptr[r] is the index into col_idx/val where row r begins.
///   col_idx[nnz]  — column indices of the non-zeros.
///   val[nnz]      — values of the non-zeros.
///
/// Scatter is efficient: for each spiking row r, iterate
/// col_idx[row_ptr[r] .. row_ptr[r+1]) and accumulate weights.
/// Complexity: O(Σ degree(spiking rows)).
class CSRMatrix : public SparseMatrix {
public:
    /// Construct from COO triplets.  Internally sorts by (row, col).
    explicit CSRMatrix(const COOTriplets& triplets);

    void   scatter(const std::vector<int>& spike_sources,
                   std::vector<double>&    out_buffer) const override;
    double gather(int target,
                  const std::vector<int>& spike_sources) const override;
    void   gather_all(const std::vector<int>& spike_sources,
                      std::vector<double>&    out_buffer) const override;

    size_t memory_bytes() const override;
    int    num_rows()     const override { return nrows_; }
    int    num_cols()     const override { return ncols_; }
    size_t num_nonzeros() const override { return col_idx_.size(); }

    void save_sparsity_pattern(const std::string& filename,
                               int resolution = 256) const override;

    void save_storage_layout(const std::string& filename,
                             int resolution = 512) const override;

    void dump() const override;

private:
    int nrows_, ncols_;
    std::vector<int>    row_ptr_;   // size N+1
    std::vector<int>    col_idx_;   // size nnz
    std::vector<double> val_;       // size nnz
};
