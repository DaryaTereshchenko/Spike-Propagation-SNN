#include "csc_matrix.h"
#include "stb_image_write.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

CSCMatrix::CSCMatrix(const COOTriplets& t)
    : nrows_(t.N), ncols_(t.N)
{
    const size_t nnz = t.nnz();

    // Sort indices by (col, row).
    std::vector<size_t> perm(nnz);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
        if (t.cols[a] != t.cols[b]) return t.cols[a] < t.cols[b];
        return t.rows[a] < t.rows[b];
    });

    // Build CSC arrays.
    col_ptr_.assign(ncols_ + 1, 0);
    row_idx_.resize(nnz);
    val_.resize(nnz);

    for (size_t i = 0; i < nnz; ++i) {
        size_t idx  = perm[i];
        row_idx_[i] = t.rows[idx];
        val_[i]     = t.vals[idx];
        col_ptr_[t.cols[idx] + 1]++;
    }

    // Prefix sum to get column pointers.
    for (int c = 0; c < ncols_; ++c) {
        col_ptr_[c + 1] += col_ptr_[c];
    }
}

void CSCMatrix::scatter(const std::vector<int>& spike_sources,
                        std::vector<double>&    out_buffer) const
{
    // CSC is not optimal for scatter – we must scan every column and check
    // whether each entry's row is a spiking source.
    std::vector<bool> is_spiking(nrows_, false);
    for (int s : spike_sources) {
        is_spiking[s] = true;
    }

    for (int c = 0; c < ncols_; ++c) {
        const int start = col_ptr_[c];
        const int end   = col_ptr_[c + 1];
        for (int j = start; j < end; ++j) {
            if (is_spiking[row_idx_[j]]) {
                out_buffer[c] += val_[j];
            }
        }
    }
}

double CSCMatrix::gather(int target,
                         const std::vector<int>& spike_sources) const
{
    // Optimal for gather: directly iterate the target column's entries.
    std::vector<bool> is_spiking(nrows_, false);
    for (int s : spike_sources) {
        is_spiking[s] = true;
    }

    double sum = 0.0;
    const int start = col_ptr_[target];
    const int end   = col_ptr_[target + 1];
    for (int j = start; j < end; ++j) {
        if (is_spiking[row_idx_[j]]) {
            sum += val_[j];
        }
    }
    return sum;
}

void CSCMatrix::gather_all(const std::vector<int>& spike_sources,
                           std::vector<double>&    out_buffer) const
{
    // CSC is optimal for gather_all: iterate each column directly.
    std::vector<bool> is_spiking(nrows_, false);
    for (int s : spike_sources) {
        is_spiking[s] = true;
    }

    for (int c = 0; c < ncols_; ++c) {
        const int start = col_ptr_[c];
        const int end   = col_ptr_[c + 1];
        for (int j = start; j < end; ++j) {
            if (is_spiking[row_idx_[j]]) {
                out_buffer[c] += val_[j];
            }
        }
    }
}

size_t CSCMatrix::memory_bytes() const
{
    return col_ptr_.size()  * sizeof(int)
         + row_idx_.size()  * sizeof(int)
         + val_.size()      * sizeof(double);
}

void CSCMatrix::save_sparsity_pattern(const std::string& filename,
                                      int resolution) const
{
    const int res = std::min(resolution, std::max(nrows_, ncols_));
    std::vector<int> grid(res * res, 0);

    for (int c = 0; c < ncols_; ++c) {
        int bc = static_cast<int>((long long)c * res / ncols_);
        if (bc >= res) bc = res - 1;
        for (int j = col_ptr_[c]; j < col_ptr_[c + 1]; ++j) {
            int br = static_cast<int>((long long)row_idx_[j] * res / nrows_);
            if (br >= res) br = res - 1;
            grid[br * res + bc]++;
        }
    }

    int max_count = *std::max_element(grid.begin(), grid.end());
    if (max_count == 0) max_count = 1;

    std::vector<unsigned char> pixels(res * res * 3);
    for (int i = 0; i < res * res; ++i) {
        double t = std::sqrt(static_cast<double>(grid[i]) / max_count);
        if (grid[i] == 0) {
            pixels[i*3] = pixels[i*3+1] = pixels[i*3+2] = 255;
        } else {
            pixels[i*3]   = static_cast<unsigned char>(std::min(255.0, 255.0 * t * 2.0));
            pixels[i*3+1] = static_cast<unsigned char>(std::min(255.0, 255.0 * (1.0 - t)));
            pixels[i*3+2] = static_cast<unsigned char>(std::min(255.0, 255.0 * (1.0 - t) * 0.8));
        }
    }

    if (!stbi_write_png(filename.c_str(), res, res, 3, pixels.data(), res * 3)) {
        std::cerr << "Failed to write " << filename << "\n";
        return;
    }
    std::cout << "Sparsity pattern saved to " << filename
              << " (" << res << "x" << res << ")\n";
}

static void jet_color_csc(double t, unsigned char& r, unsigned char& g, unsigned char& b)
{
    if (t < 0.25) {
        r = 0;
        g = static_cast<unsigned char>(255 * (t / 0.25));
        b = 255;
    } else if (t < 0.5) {
        r = 0;
        g = 255;
        b = static_cast<unsigned char>(255 * (1.0 - (t - 0.25) / 0.25));
    } else if (t < 0.75) {
        r = static_cast<unsigned char>(255 * ((t - 0.5) / 0.25));
        g = 255;
        b = 0;
    } else {
        r = 255;
        g = static_cast<unsigned char>(255 * (1.0 - (t - 0.75) / 0.25));
        b = 0;
    }
}

void CSCMatrix::save_storage_layout(const std::string& filename,
                                    int resolution) const
{
    const int res = std::min(resolution, std::max(nrows_, ncols_));
    std::vector<int> pixel_order(res * res, -1);

    int idx = 0;
    for (int c = 0; c < ncols_; ++c) {
        int bc = static_cast<int>((long long)c * res / ncols_);
        if (bc >= res) bc = res - 1;
        for (int j = col_ptr_[c]; j < col_ptr_[c + 1]; ++j) {
            int br = static_cast<int>((long long)row_idx_[j] * res / nrows_);
            if (br >= res) br = res - 1;
            int px = br * res + bc;
            if (pixel_order[px] < 0)
                pixel_order[px] = idx;
            idx++;
        }
    }

    const size_t nnz = row_idx_.size();
    std::vector<unsigned char> pixels(res * res * 3);
    for (int i = 0; i < res * res; ++i) {
        if (pixel_order[i] < 0) {
            pixels[i*3] = pixels[i*3+1] = pixels[i*3+2] = 245;
        } else {
            double t = (nnz > 1) ? static_cast<double>(pixel_order[i]) / (nnz - 1) : 0.0;
            jet_color_csc(t, pixels[i*3], pixels[i*3+1], pixels[i*3+2]);
        }
    }

    if (!stbi_write_png(filename.c_str(), res, res, 3, pixels.data(), res * 3)) {
        std::cerr << "Failed to write " << filename << "\n";
        return;
    }
    std::cout << "Storage layout saved to " << filename
              << " (" << res << "x" << res << ")\n";
}

void CSCMatrix::dump() const
{
    std::cout << "=== CSC Matrix (" << nrows_ << "x" << ncols_
              << ", nnz=" << row_idx_.size() << ") ===\n";
    std::cout << "  col_ptr[]: ";
    for (size_t i = 0; i < col_ptr_.size(); ++i) std::cout << col_ptr_[i] << " ";
    std::cout << "\n  row_idx[]: ";
    for (size_t i = 0; i < row_idx_.size(); ++i) std::cout << row_idx_[i] << " ";
    std::cout << "\n  val[]:     ";
    for (size_t i = 0; i < val_.size(); ++i)
        std::cout << (static_cast<int>(val_[i] * 1000) / 1000.0) << " ";
    std::cout << "\n\n  Per-column breakdown:\n";
    for (int c = 0; c < ncols_; ++c) {
        std::cout << "    col " << c << ":  rows=[";
        for (int j = col_ptr_[c]; j < col_ptr_[c + 1]; ++j) {
            if (j > col_ptr_[c]) std::cout << ", ";
            std::cout << row_idx_[j];
        }
        std::cout << "]  (" << (col_ptr_[c+1] - col_ptr_[c]) << " entries)\n";
    }
    std::cout << "\n  Dense view:\n      ";
    for (int c = 0; c < ncols_; ++c) std::cout << c << "  ";
    std::cout << "\n";
    for (int r = 0; r < nrows_; ++r) {
        std::cout << "  " << r << " [ ";
        for (int c = 0; c < ncols_; ++c) {
            bool found = false;
            for (int j = col_ptr_[c]; j < col_ptr_[c + 1]; ++j) {
                if (row_idx_[j] == r) { found = true; break; }
            }
            std::cout << (found ? " x " : " . ");
        }
        std::cout << "]\n";
    }
    std::cout << std::endl;
}
