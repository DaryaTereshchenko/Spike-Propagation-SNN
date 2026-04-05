#include "coo_matrix.hpp"
#include <stdexcept>

CooMatrix::CooMatrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols) {}

void CooMatrix::add_entry(std::size_t i, std::size_t j, double weight) {
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("CooMatrix::add_entry: index out of range");
    }
    row_idx_.push_back(i);
    col_idx_.push_back(j);
    values_.push_back(weight);
}

void CooMatrix::scatter(const std::vector<bool>& spikes,
                        std::vector<double>& output) const {
    for (std::size_t k = 0; k < row_idx_.size(); ++k) {
        if (spikes[row_idx_[k]]) {
            output[col_idx_[k]] += values_[k];
        }
    }
}

void CooMatrix::gather(const std::vector<bool>& spikes,
                       std::vector<double>& output) const {
    // Pull: for each entry, if source fires, add to target's accumulator.
    // Identical computation to scatter for COO since there is no
    // columnar index structure.
    for (std::size_t k = 0; k < row_idx_.size(); ++k) {
        if (spikes[row_idx_[k]]) {
            output[col_idx_[k]] += values_[k];
        }
    }
}

std::size_t CooMatrix::memory_bytes() const {
    return row_idx_.size() * sizeof(std::size_t)  // row_idx
         + col_idx_.size() * sizeof(std::size_t)  // col_idx
         + values_.size()  * sizeof(double);       // values
}
