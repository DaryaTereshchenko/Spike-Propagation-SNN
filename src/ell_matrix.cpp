#include "ell_matrix.h"
#include "stb_image_write.h"
#include <algorithm>
#include <cmath>
#include <iostream>

ELLMatrix::ELLMatrix(const COOTriplets& t)
    : nrows_(t.N), ncols_(t.N), nnz_(t.nnz()), max_nnz_(0)
{
    // Count non-zeros per row to determine max_nnz_per_row.
    std::vector<int> row_counts(nrows_, 0);
    for (size_t i = 0; i < nnz_; ++i) {
        row_counts[t.rows[i]]++;
    }
    max_nnz_ = *std::max_element(row_counts.begin(), row_counts.end());

    // Allocate padded arrays.
    const size_t total = static_cast<size_t>(nrows_) * max_nnz_;
    indices_.assign(total, -1);       // sentinel
    values_.assign(total, 0.0);

    // Fill in entries row by row.
    std::vector<int> cursor(nrows_, 0);   // next free slot per row
    for (size_t i = 0; i < nnz_; ++i) {
        int r   = t.rows[i];
        int pos = r * max_nnz_ + cursor[r];
        indices_[pos] = t.cols[i];
        values_[pos]  = t.vals[i];
        cursor[r]++;
    }
}

void ELLMatrix::scatter(const std::vector<int>& spike_sources,
                        std::vector<double>&    out_buffer) const
{
    for (int r : spike_sources) {
        const int base = r * max_nnz_;
        for (int k = 0; k < max_nnz_; ++k) {
            int c = indices_[base + k];
            if (c < 0) break;           // sentinel → no more entries
            out_buffer[c] += values_[base + k];
        }
    }
}

double ELLMatrix::gather(int target,
                         const std::vector<int>& spike_sources) const
{
    // ELL is row-oriented → gather requires scanning spiking rows.
    std::vector<bool> is_spiking(nrows_, false);
    for (int s : spike_sources) {
        is_spiking[s] = true;
    }

    double sum = 0.0;
    for (int r = 0; r < nrows_; ++r) {
        if (!is_spiking[r]) continue;
        const int base = r * max_nnz_;
        for (int k = 0; k < max_nnz_; ++k) {
            int c = indices_[base + k];
            if (c < 0) break;
            if (c == target) {
                sum += values_[base + k];
            }
        }
    }
    return sum;
}

void ELLMatrix::gather_all(const std::vector<int>& spike_sources,
                           std::vector<double>&    out_buffer) const
{
    // ELL gather_all: iterate spiking rows and distribute to columns.
    for (int r : spike_sources) {
        const int base = r * max_nnz_;
        for (int k = 0; k < max_nnz_; ++k) {
            int c = indices_[base + k];
            if (c < 0) break;
            out_buffer[c] += values_[base + k];
        }
    }
}

size_t ELLMatrix::memory_bytes() const
{
    return indices_.size() * sizeof(int)
         + values_.size()  * sizeof(double);
}

void ELLMatrix::save_sparsity_pattern(const std::string& filename,
                                      int resolution) const
{
    const int res = std::min(resolution, std::max(nrows_, ncols_));
    std::vector<int> grid(res * res, 0);

    for (int r = 0; r < nrows_; ++r) {
        int br = static_cast<int>((long long)r * res / nrows_);
        if (br >= res) br = res - 1;
        const int base = r * max_nnz_;
        for (int k = 0; k < max_nnz_; ++k) {
            int c = indices_[base + k];
            if (c < 0) break;
            int bc = static_cast<int>((long long)c * res / ncols_);
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

static void jet_color_ell(double t, unsigned char& r, unsigned char& g, unsigned char& b)
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

void ELLMatrix::save_storage_layout(const std::string& filename,
                                    int resolution) const
{
    const int res = std::min(resolution, std::max(nrows_, ncols_));
    std::vector<int> pixel_order(res * res, -1);

    int idx = 0;
    for (int r = 0; r < nrows_; ++r) {
        int br = static_cast<int>((long long)r * res / nrows_);
        if (br >= res) br = res - 1;
        const int base = r * max_nnz_;
        for (int k = 0; k < max_nnz_; ++k) {
            int c = indices_[base + k];
            if (c < 0) break;
            int bc = static_cast<int>((long long)c * res / ncols_);
            if (bc >= res) bc = res - 1;
            int px = br * res + bc;
            if (pixel_order[px] < 0)
                pixel_order[px] = idx;
            idx++;
        }
    }

    const size_t total = static_cast<size_t>(idx);
    std::vector<unsigned char> pixels(res * res * 3);
    for (int i = 0; i < res * res; ++i) {
        if (pixel_order[i] < 0) {
            pixels[i*3] = pixels[i*3+1] = pixels[i*3+2] = 245;
        } else {
            double t = (total > 1) ? static_cast<double>(pixel_order[i]) / (total - 1) : 0.0;
            jet_color_ell(t, pixels[i*3], pixels[i*3+1], pixels[i*3+2]);
        }
    }

    if (!stbi_write_png(filename.c_str(), res, res, 3, pixels.data(), res * 3)) {
        std::cerr << "Failed to write " << filename << "\n";
        return;
    }
    std::cout << "Storage layout saved to " << filename
              << " (" << res << "x" << res << ")\n";
}

void ELLMatrix::dump() const
{
    std::cout << "=== ELL Matrix (" << nrows_ << "x" << ncols_
              << ", nnz=" << nnz_ << ", max_nnz_per_row=" << max_nnz_ << ") ===\n";
    std::cout << "  indices[] (row-major, -1=padding):\n";
    for (int r = 0; r < nrows_; ++r) {
        std::cout << "    row " << r << ": [";
        for (int k = 0; k < max_nnz_; ++k) {
            if (k > 0) std::cout << ", ";
            int c = indices_[r * max_nnz_ + k];
            if (c < 0) std::cout << " _";
            else       std::cout << " " << c;
        }
        std::cout << " ]\n";
    }
    std::cout << "  values[] (row-major, 0=padding):\n";
    for (int r = 0; r < nrows_; ++r) {
        std::cout << "    row " << r << ": [";
        for (int k = 0; k < max_nnz_; ++k) {
            if (k > 0) std::cout << ", ";
            double v = values_[r * max_nnz_ + k];
            std::cout << " " << (static_cast<int>(v * 1000) / 1000.0);
        }
        std::cout << " ]\n";
    }
    std::cout << "\n  Dense view:\n      ";
    for (int c = 0; c < ncols_; ++c) std::cout << c << "  ";
    std::cout << "\n";
    for (int r = 0; r < nrows_; ++r) {
        std::cout << "  " << r << " [ ";
        for (int c = 0; c < ncols_; ++c) {
            bool found = false;
            const int base = r * max_nnz_;
            for (int k = 0; k < max_nnz_; ++k) {
                if (indices_[base + k] < 0) break;
                if (indices_[base + k] == c) { found = true; break; }
            }
            std::cout << (found ? " x " : " . ");
        }
        std::cout << "]\n";
    }
    std::cout << std::endl;
}
