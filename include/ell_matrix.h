#pragma once

#include "sparse_matrix.h"
#include <vector>

/// ELLPACK (ELL) sparse matrix format.
///
/// Stores non-zeros in a dense N × max_nnz_per_row matrix, where each row is
/// padded with sentinel values (-1 index) for rows with fewer non-zeros than
/// the maximum.  This yields a regular, cache-friendly memory access pattern
/// for scatter.
///
/// Storage (row-major):
///   indices_[r * max_nnz_ + k] — column index of k-th non-zero in row r
///   values_ [r * max_nnz_ + k] — weight of k-th non-zero in row r
///   sentinel: indices_ entry == -1 means "no entry".
///
/// max_nnz_per_row is computed automatically from the input data.
class ELLMatrix : public SparseMatrix {
public:
    /// Construct from COO triplets.  Determines max_nnz_per_row from the data.
    explicit ELLMatrix(const COOTriplets& triplets);

    void   scatter(const std::vector<int>& spike_sources,
                   std::vector<double>&    out_buffer) const override;
    double gather(int target,
                  const std::vector<int>& spike_sources) const override;
    void   gather_all(const std::vector<int>& spike_sources,
                      std::vector<double>&    out_buffer) const override;

    size_t memory_bytes() const override;
    int    num_rows()     const override { return nrows_; }
    int    num_cols()     const override { return ncols_; }
    size_t num_nonzeros() const override { return nnz_; }

    void save_sparsity_pattern(const std::string& filename,
                               int resolution = 256) const override;

    void save_storage_layout(const std::string& filename,
                             int resolution = 512) const override;

    void dump() const override;

    /// Maximum non-zeros per row (determines padding).
    int max_nnz_per_row() const { return max_nnz_; }

private:
    int    nrows_, ncols_;
    size_t nnz_;
    int    max_nnz_;
    std::vector<int>    indices_;   // size N * max_nnz_
    std::vector<double> values_;    // size N * max_nnz_
};
