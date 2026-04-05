#pragma once
#include "sparse_matrix.hpp"
#include <vector>
#include <cstddef>

// COO (Coordinate) sparse matrix format.
// Stores each non-zero as a (row, col, value) triple.
// Simple to construct but gives irregular memory access during traversal.
class CooMatrix : public SparseMatrix {
public:
    CooMatrix(std::size_t rows, std::size_t cols);

    // Add a synapse (row i -> col j) with the given weight.
    void add_entry(std::size_t i, std::size_t j, double weight);

    std::size_t rows() const override { return rows_; }
    std::size_t cols() const override { return cols_; }
    std::size_t nnz()  const override { return row_idx_.size(); }

    void scatter(const std::vector<bool>& spikes,
                 std::vector<double>& output) const override;

    void gather(const std::vector<bool>& spikes,
                std::vector<double>& output) const override;

    std::size_t memory_bytes() const override;

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<std::size_t> row_idx_;  // source neuron index
    std::vector<std::size_t> col_idx_;  // target neuron index
    std::vector<double>      values_;   // synaptic weight
};
