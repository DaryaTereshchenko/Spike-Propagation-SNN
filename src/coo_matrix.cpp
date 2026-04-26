#include "coo_matrix.h"
#include "stb_image_write.h"
#include <algorithm>
#include <cmath>
#include <iostream>

COOMatrix::COOMatrix(const COOTriplets& t)
    : nrows_(t.N), ncols_(t.N), row_(t.rows), col_(t.cols), val_(t.vals)
{
}

void COOMatrix::scatter(const std::vector<int>& spike_sources,
                        std::vector<double>&    out_buffer) const
{
    // Build a fast boolean lookup for spiking neurons.
    std::vector<bool> is_spiking(nrows_, false);
    for (int s : spike_sources) {
        is_spiking[s] = true;
    }

    // Scan every stored non-zero.
    const size_t nnz = row_.size();
    for (size_t i = 0; i < nnz; ++i) {
        if (is_spiking[row_[i]]) {
            out_buffer[col_[i]] += val_[i];
        }
    }
}

double COOMatrix::gather(int target,
                         const std::vector<int>& spike_sources) const
{
    std::vector<bool> is_spiking(nrows_, false);
    for (int s : spike_sources) {
        is_spiking[s] = true;
    }

    double sum = 0.0;
    const size_t nnz = row_.size();
    for (size_t i = 0; i < nnz; ++i) {
        if (col_[i] == target && is_spiking[row_[i]]) {
            sum += val_[i];
        }
    }
    return sum;
}

void COOMatrix::gather_all(const std::vector<int>& spike_sources,
                           std::vector<double>&    out_buffer) const
{
    std::vector<bool> is_spiking(nrows_, false);
    for (int s : spike_sources) {
        is_spiking[s] = true;
    }

    const size_t nnz = row_.size();
    for (size_t i = 0; i < nnz; ++i) {
        if (is_spiking[row_[i]]) {
            out_buffer[col_[i]] += val_[i];
        }
    }
}

size_t COOMatrix::memory_bytes() const
{
    return row_.size() * sizeof(int)
         + col_.size() * sizeof(int)
         + val_.size() * sizeof(double);
}

void COOMatrix::save_sparsity_pattern(const std::string& filename,
                                      int resolution) const
{
    const int res = std::min(resolution, std::max(nrows_, ncols_));
    std::vector<int> grid(res * res, 0);

    const size_t nnz = row_.size();
    for (size_t i = 0; i < nnz; ++i) {
        int br = static_cast<int>((long long)row_[i] * res / nrows_);
        int bc = static_cast<int>((long long)col_[i] * res / ncols_);
        if (br >= res) br = res - 1;
        if (bc >= res) bc = res - 1;
        grid[br * res + bc]++;
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

// Jet-like colormap: index 0..nnz-1 → blue→cyan→green→yellow→red
static void jet_color(double t, unsigned char& r, unsigned char& g, unsigned char& b)
{
    // t in [0,1]
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

void COOMatrix::save_storage_layout(const std::string& filename,
                                    int resolution) const
{
    const int res = std::min(resolution, std::max(nrows_, ncols_));
    // Store the earliest storage index that maps to each pixel (-1 = empty).
    std::vector<int> pixel_order(res * res, -1);

    const size_t nnz = row_.size();
    for (size_t i = 0; i < nnz; ++i) {
        int br = static_cast<int>((long long)row_[i] * res / nrows_);
        int bc = static_cast<int>((long long)col_[i] * res / ncols_);
        if (br >= res) br = res - 1;
        if (bc >= res) bc = res - 1;
        int px = br * res + bc;
        if (pixel_order[px] < 0)
            pixel_order[px] = static_cast<int>(i);
    }

    std::vector<unsigned char> pixels(res * res * 3);
    for (int i = 0; i < res * res; ++i) {
        if (pixel_order[i] < 0) {
            pixels[i*3] = pixels[i*3+1] = pixels[i*3+2] = 245; // light gray
        } else {
            double t = (nnz > 1) ? static_cast<double>(pixel_order[i]) / (nnz - 1) : 0.0;
            jet_color(t, pixels[i*3], pixels[i*3+1], pixels[i*3+2]);
        }
    }

    if (!stbi_write_png(filename.c_str(), res, res, 3, pixels.data(), res * 3)) {
        std::cerr << "Failed to write " << filename << "\n";
        return;
    }
    std::cout << "Storage layout saved to " << filename
              << " (" << res << "x" << res << ")\n";
}

void COOMatrix::dump() const
{
    std::cout << "=== COO Matrix (" << nrows_ << "x" << ncols_
              << ", nnz=" << row_.size() << ") ===\n";
    std::cout << "  row[]: ";
    for (size_t i = 0; i < row_.size(); ++i) std::cout << row_[i] << " ";
    std::cout << "\n  col[]: ";
    for (size_t i = 0; i < col_.size(); ++i) std::cout << col_[i] << " ";
    std::cout << "\n  val[]: ";
    for (size_t i = 0; i < val_.size(); ++i)
        std::cout << (static_cast<int>(val_[i] * 1000) / 1000.0) << " ";
    std::cout << "\n\n  Dense view:\n      ";
    for (int c = 0; c < ncols_; ++c) std::cout << c << "  ";
    std::cout << "\n";
    for (int r = 0; r < nrows_; ++r) {
        std::cout << "  " << r << " [ ";
        for (int c = 0; c < ncols_; ++c) {
            bool found = false;
            for (size_t k = 0; k < row_.size(); ++k) {
                if (row_[k] == r && col_[k] == c) { found = true; break; }
            }
            std::cout << (found ? " x " : " . ");
        }
        std::cout << "]\n";
    }
    std::cout << std::endl;
}
