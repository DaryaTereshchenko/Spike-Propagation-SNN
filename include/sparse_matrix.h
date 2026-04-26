#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

/// Common triplet representation used to construct any sparse format.
struct COOTriplets {
    std::vector<int>    rows;
    std::vector<int>    cols;
    std::vector<double> vals;
    int N = 0;  // matrix dimension (N×N)

    size_t nnz() const { return rows.size(); }
};

/// Abstract base class for all sparse matrix formats.
///
/// Convention: row = presynaptic (source) neuron index
///             col = postsynaptic (target) neuron index
/// A non-zero at (r, c) with weight w means "neuron r projects to neuron c
/// with synaptic weight w".
class SparseMatrix {
public:
    virtual ~SparseMatrix() = default;

    /// Push-based spike delivery (primary benchmark operation).
    /// For every spiking source neuron listed in @p spike_sources, add its
    /// outgoing synaptic weights to the corresponding entries in @p out_buffer.
    /// @p out_buffer must be pre-allocated to size num_cols() and zeroed.
    virtual void scatter(const std::vector<int>& spike_sources,
                         std::vector<double>&    out_buffer) const = 0;

    /// Pull-based synaptic input (secondary operation, single target).
    /// Return the total synaptic input arriving at neuron @p target from the
    /// neurons listed in @p spike_sources.
    virtual double gather(int target,
                          const std::vector<int>& spike_sources) const = 0;

    /// Pull-based synaptic input for ALL target neurons (gather benchmark).
    /// For every target neuron, sum incoming weights from @p spike_sources
    /// and write the result to @p out_buffer.
    /// @p out_buffer must be pre-allocated to size num_cols() and zeroed.
    virtual void gather_all(const std::vector<int>& spike_sources,
                            std::vector<double>&    out_buffer) const = 0;

    /// Number of bytes used by the internal storage arrays (excluding this
    /// object's overhead).
    virtual size_t memory_bytes() const = 0;

    virtual int    num_rows()     const = 0;
    virtual int    num_cols()     const = 0;
    virtual size_t num_nonzeros() const = 0;

    /// Save a PPM image visualizing the sparsity pattern of the matrix.
    /// The matrix is binned into a resolution × resolution grid; each pixel
    /// intensity reflects the number of non-zeros in that block.
    /// @param filename  Output path (should end in .ppm).
    /// @param resolution  Width/height of the output image in pixels.
    virtual void save_sparsity_pattern(const std::string& filename,
                                       int resolution = 256) const = 0;
    /// Save an image showing the memory storage order of non-zeros.
    /// Each non-zero is colored by its position in the internal storage array
    /// (blue=early/index 0, red=late/index nnz-1), revealing how the format
    /// physically lays out data in memory.
    virtual void save_storage_layout(const std::string& filename,
                                     int resolution = 512) const = 0;

    /// Print the internal storage arrays to stdout for inspection.
    virtual void dump() const = 0;
};
