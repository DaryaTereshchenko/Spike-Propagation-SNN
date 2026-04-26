#pragma once

#include "sparse_matrix.h"
#include <vector>

/// COO (Coordinate) sparse matrix format.
///
/// Stores every non-zero as an explicit (row, col, val) triplet.
/// Scatter and gather both require a full scan of all nnz entries → O(nnz).
class COOMatrix : public SparseMatrix {
public:
    /// Construct from COO triplets.  The triplets are copied and the internal
    /// arrays are stored in the order given (no sorting).
    explicit COOMatrix(const COOTriplets& triplets);

    void   scatter(const std::vector<int>& spike_sources,
                   std::vector<double>&    out_buffer) const override;
    double gather(int target,
                  const std::vector<int>& spike_sources) const override;
    void   gather_all(const std::vector<int>& spike_sources,
                      std::vector<double>&    out_buffer) const override;

    size_t memory_bytes() const override;
    int    num_rows()     const override { return nrows_; }
    int    num_cols()     const override { return ncols_; }
    size_t num_nonzeros() const override { return row_.size(); }

    void save_sparsity_pattern(const std::string& filename,
                               int resolution = 256) const override;

    void save_storage_layout(const std::string& filename,
                             int resolution = 512) const override;

    void dump() const override;

private:
    int nrows_, ncols_;
    std::vector<int>    row_;
    std::vector<int>    col_;
    std::vector<double> val_;
};
