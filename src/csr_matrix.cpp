#include "csr_matrix.h"
#include "stb_image_write.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

CSRMatrix::CSRMatrix(const COOTriplets& t)
    : nrows_(t.N), ncols_(t.N)
{
    const size_t nnz = t.nnz();

    // Sort indices by (row, col).
    std::vector<size_t> perm(nnz);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
        if (t.rows[a] != t.rows[b]) return t.rows[a] < t.rows[b];
        return t.cols[a] < t.cols[b];
    });

    // Build CSR arrays.
    row_ptr_.assign(nrows_ + 1, 0);
    col_idx_.resize(nnz);
    val_.resize(nnz);

    for (size_t i = 0; i < nnz; ++i) {
        size_t idx  = perm[i];
        col_idx_[i] = t.cols[idx];
        val_[i]     = t.vals[idx];
        row_ptr_[t.rows[idx] + 1]++;
    }

    // Prefix sum to get row pointers.
    for (int r = 0; r < nrows_; ++r) {
        row_ptr_[r + 1] += row_ptr_[r];
    }
}

void CSRMatrix::scatter(const std::vector<int>& spike_sources,
                        std::vector<double>&    out_buffer) const
{
    for (int r : spike_sources) {
        const int start = row_ptr_[r];
        const int end   = row_ptr_[r + 1];
        for (int j = start; j < end; ++j) {
            out_buffer[col_idx_[j]] += val_[j];
        }
    }
}

double CSRMatrix::gather(int target,
                         const std::vector<int>& spike_sources) const
{
    // CSR is not optimal for gather – we must scan each spiking row's entries
    // looking for the target column.
    double sum = 0.0;
    for (int r : spike_sources) {
        const int start = row_ptr_[r];
        const int end   = row_ptr_[r + 1];
        for (int j = start; j < end; ++j) {
            if (col_idx_[j] == target) {
                sum += val_[j];
                break;  // at most one entry per (row, col) pair
            }
        }
    }
    return sum;
}

void CSRMatrix::gather_all(const std::vector<int>& spike_sources,
                           std::vector<double>&    out_buffer) const
{
    // CSR gather: for each spiking row, iterate its entries and accumulate
    // into out_buffer by column.  Same work as scatter but different semantics.
    for (int r : spike_sources) {
        const int start = row_ptr_[r];
        const int end   = row_ptr_[r + 1];
        for (int j = start; j < end; ++j) {
            out_buffer[col_idx_[j]] += val_[j];
        }
    }
}

size_t CSRMatrix::memory_bytes() const
{
    return row_ptr_.size()  * sizeof(int)
         + col_idx_.size()  * sizeof(int)
         + val_.size()      * sizeof(double);
}

void CSRMatrix::save_sparsity_pattern(const std::string& filename,
                                      int resolution) const
{
    const int res = std::min(resolution, std::max(nrows_, ncols_));
    std::vector<int> grid(res * res, 0);

    for (int r = 0; r < nrows_; ++r) {
        int br = static_cast<int>((long long)r * res / nrows_);
        if (br >= res) br = res - 1;
        for (int j = row_ptr_[r]; j < row_ptr_[r + 1]; ++j) {
            int bc = static_cast<int>((long long)col_idx_[j] * res / ncols_);
            if (bc >= res) bc = res - 1;
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

static void jet_color_csr(double t, unsigned char& r, unsigned char& g, unsigned char& b)
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

void CSRMatrix::save_storage_layout(const std::string& filename,
                                    int resolution) const
{
    const int res = std::min(resolution, std::max(nrows_, ncols_));
    std::vector<int> pixel_order(res * res, -1);

    int idx = 0;
    for (int r = 0; r < nrows_; ++r) {
        int br = static_cast<int>((long long)r * res / nrows_);
        if (br >= res) br = res - 1;
        for (int j = row_ptr_[r]; j < row_ptr_[r + 1]; ++j) {
            int bc = static_cast<int>((long long)col_idx_[j] * res / ncols_);
            if (bc >= res) bc = res - 1;
            int px = br * res + bc;
            if (pixel_order[px] < 0)
                pixel_order[px] = idx;
            idx++;
        }
    }

    const size_t nnz = col_idx_.size();
    std::vector<unsigned char> pixels(res * res * 3);
    for (int i = 0; i < res * res; ++i) {
        if (pixel_order[i] < 0) {
            pixels[i*3] = pixels[i*3+1] = pixels[i*3+2] = 245;
        } else {
            double t = (nnz > 1) ? static_cast<double>(pixel_order[i]) / (nnz - 1) : 0.0;
            jet_color_csr(t, pixels[i*3], pixels[i*3+1], pixels[i*3+2]);
        }
    }

    if (!stbi_write_png(filename.c_str(), res, res, 3, pixels.data(), res * 3)) {
        std::cerr << "Failed to write " << filename << "\n";
        return;
    }
    std::cout << "Storage layout saved to " << filename
              << " (" << res << "x" << res << ")\n";
}

void CSRMatrix::dump() const
{
    std::cout << "=== CSR Matrix (" << nrows_ << "x" << ncols_
              << ", nnz=" << col_idx_.size() << ") ===\n";
    std::cout << "  row_ptr[]: ";
    for (size_t i = 0; i < row_ptr_.size(); ++i) std::cout << row_ptr_[i] << " ";
    std::cout << "\n  col_idx[]: ";
    for (size_t i = 0; i < col_idx_.size(); ++i) std::cout << col_idx_[i] << " ";
    std::cout << "\n  val[]:     ";
    for (size_t i = 0; i < val_.size(); ++i)
        std::cout << (static_cast<int>(val_[i] * 1000) / 1000.0) << " ";
    std::cout << "\n\n  Per-row breakdown:\n";
    for (int r = 0; r < nrows_; ++r) {
        std::cout << "    row " << r << ":  cols=[";
        for (int j = row_ptr_[r]; j < row_ptr_[r + 1]; ++j) {
            if (j > row_ptr_[r]) std::cout << ", ";
            std::cout << col_idx_[j];
        }
        std::cout << "]  (" << (row_ptr_[r+1] - row_ptr_[r]) << " entries)\n";
    }
    std::cout << "\n  Dense view:\n      ";
    for (int c = 0; c < ncols_; ++c) std::cout << c << "  ";
    std::cout << "\n";
    for (int r = 0; r < nrows_; ++r) {
        std::cout << "  " << r << " [ ";
        for (int c = 0; c < ncols_; ++c) {
            bool found = false;
            for (int j = row_ptr_[r]; j < row_ptr_[r + 1]; ++j) {
                if (col_idx_[j] == c) { found = true; break; }
            }
            std::cout << (found ? " x " : " . ");
        }
        std::cout << "]\n";
    }
    std::cout << std::endl;
}
