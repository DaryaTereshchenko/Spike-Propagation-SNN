#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sparse_matrix.h"
#include "coo_matrix.h"
#include "csr_matrix.h"
#include "csc_matrix.h"
#include "ell_matrix.h"

#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Shared test fixture: a known 5×5 matrix with 8 nonzeros.
//
//      0   1   2   3   4
// 0 [  .  1.0  .  2.0  . ]
// 1 [  .   .  3.0  .   . ]
// 2 [ 4.0  .   .   .  5.0]
// 3 [  .   .   .   .   . ]
// 4 [  .  6.0  .  7.0 8.0]
// ---------------------------------------------------------------------------
static COOTriplets make_test_matrix()
{
    COOTriplets t;
    t.N = 5;
    // (row, col, val) triplets
    t.rows = {0, 0, 1, 2, 2, 4, 4, 4};
    t.cols = {1, 3, 2, 0, 4, 1, 3, 4};
    t.vals = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    return t;
}

// Convenience: build all four formats from the same COO.
struct AllFormats {
    COOTriplets          triplets;
    std::unique_ptr<COOMatrix> coo;
    std::unique_ptr<CSRMatrix> csr;
    std::unique_ptr<CSCMatrix> csc;
    std::unique_ptr<ELLMatrix> ell;

    AllFormats()
        : triplets(make_test_matrix()),
          coo(std::make_unique<COOMatrix>(triplets)),
          csr(std::make_unique<CSRMatrix>(triplets)),
          csc(std::make_unique<CSCMatrix>(triplets)),
          ell(std::make_unique<ELLMatrix>(triplets))
    {}
};

// ---------------------------------------------------------------------------
// Test: dimensions and nnz
// ---------------------------------------------------------------------------
TEST_CASE("Dimensions and nnz are consistent across formats", "[formats]")
{
    AllFormats f;
    for (auto* m : std::vector<SparseMatrix*>{f.coo.get(), f.csr.get(),
                                                f.csc.get(), f.ell.get()}) {
        REQUIRE(m->num_rows()     == 5);
        REQUIRE(m->num_cols()     == 5);
        REQUIRE(m->num_nonzeros() == 8);
    }
}

// ---------------------------------------------------------------------------
// Test: scatter produces identical results across all formats
// ---------------------------------------------------------------------------
TEST_CASE("Scatter is identical across formats", "[formats][scatter]")
{
    AllFormats f;

    // Spike sources: neurons 0 and 4 fire.
    // Row 0 contributes: col 1 += 1.0, col 3 += 2.0
    // Row 4 contributes: col 1 += 6.0, col 3 += 7.0, col 4 += 8.0
    // Expected out: [0, 7, 0, 9, 8]
    std::vector<int> spikes = {0, 4};
    std::vector<double> expected = {0.0, 7.0, 0.0, 9.0, 8.0};

    auto run_scatter = [&](SparseMatrix& m) {
        std::vector<double> out(5, 0.0);
        m.scatter(spikes, out);
        return out;
    };

    auto out_coo = run_scatter(*f.coo);
    auto out_csr = run_scatter(*f.csr);
    auto out_csc = run_scatter(*f.csc);
    auto out_ell = run_scatter(*f.ell);

    for (int i = 0; i < 5; ++i) {
        INFO("index " << i);
        REQUIRE(out_coo[i] == expected[i]);
        REQUIRE(out_csr[i] == expected[i]);
        REQUIRE(out_csc[i] == expected[i]);
        REQUIRE(out_ell[i] == expected[i]);
    }
}

// ---------------------------------------------------------------------------
// Test: gather produces identical results across all formats
// ---------------------------------------------------------------------------
TEST_CASE("Gather is identical across formats", "[formats][gather]")
{
    AllFormats f;

    // Spike sources: neurons 0, 2.
    // Gather for target = 4:
    //   Column 4 has entries from rows {2: 5.0, 4: 8.0}.
    //   Only row 2 is spiking → sum = 5.0.
    std::vector<int> spikes = {0, 2};
    int target = 4;
    double expected = 5.0;

    REQUIRE(f.coo->gather(target, spikes) == expected);
    REQUIRE(f.csr->gather(target, spikes) == expected);
    REQUIRE(f.csc->gather(target, spikes) == expected);
    REQUIRE(f.ell->gather(target, spikes) == expected);
}

// ---------------------------------------------------------------------------
// Test: gather with no spiking sources returns 0.
// ---------------------------------------------------------------------------
TEST_CASE("Gather with no spikes returns zero", "[formats][gather]")
{
    AllFormats f;
    std::vector<int> spikes = {};

    for (int target = 0; target < 5; ++target) {
        REQUIRE(f.coo->gather(target, spikes) == 0.0);
        REQUIRE(f.csr->gather(target, spikes) == 0.0);
        REQUIRE(f.csc->gather(target, spikes) == 0.0);
        REQUIRE(f.ell->gather(target, spikes) == 0.0);
    }
}

// ---------------------------------------------------------------------------
// Test: gather_all produces identical results to scatter (same operation)
// ---------------------------------------------------------------------------
TEST_CASE("Gather_all matches scatter across formats", "[formats][gather]")
{
    AllFormats f;
    std::vector<int> spikes = {0, 4};

    // scatter and gather_all should produce identical output since both
    // compute the same SpMV product for the spiking rows.
    auto run_scatter = [&](SparseMatrix& m) {
        std::vector<double> out(5, 0.0);
        m.scatter(spikes, out);
        return out;
    };

    auto run_gather_all = [&](SparseMatrix& m) {
        std::vector<double> out(5, 0.0);
        m.gather_all(spikes, out);
        return out;
    };

    auto scatter_ref = run_scatter(*f.coo);

    for (auto* m : std::vector<SparseMatrix*>{f.coo.get(), f.csr.get(),
                                                f.csc.get(), f.ell.get()}) {
        auto g = run_gather_all(*m);
        for (int i = 0; i < 5; ++i) {
            INFO("index " << i);
            REQUIRE(g[i] == scatter_ref[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Test: scatter with all neurons spiking
// ---------------------------------------------------------------------------
TEST_CASE("Scatter with all spikes", "[formats][scatter]")
{
    AllFormats f;
    std::vector<int> spikes = {0, 1, 2, 3, 4};

    // Expected: sum of each column.
    // col 0: 4.0 (from row 2)
    // col 1: 1.0 + 6.0 = 7.0
    // col 2: 3.0
    // col 3: 2.0 + 7.0 = 9.0
    // col 4: 5.0 + 8.0 = 13.0
    std::vector<double> expected = {4.0, 7.0, 3.0, 9.0, 13.0};

    auto run_scatter = [&](SparseMatrix& m) {
        std::vector<double> out(5, 0.0);
        m.scatter(spikes, out);
        return out;
    };

    auto out_coo = run_scatter(*f.coo);
    auto out_csr = run_scatter(*f.csr);
    auto out_csc = run_scatter(*f.csc);
    auto out_ell = run_scatter(*f.ell);

    for (int i = 0; i < 5; ++i) {
        INFO("index " << i);
        REQUIRE(out_coo[i] == expected[i]);
        REQUIRE(out_csr[i] == expected[i]);
        REQUIRE(out_csc[i] == expected[i]);
        REQUIRE(out_ell[i] == expected[i]);
    }
}

// ---------------------------------------------------------------------------
// Test: memory_bytes() returns positive values with expected relationships
// ---------------------------------------------------------------------------
TEST_CASE("memory_bytes is positive and sensible", "[formats][memory]")
{
    AllFormats f;

    size_t coo_mem = f.coo->memory_bytes();
    size_t csr_mem = f.csr->memory_bytes();
    size_t csc_mem = f.csc->memory_bytes();
    size_t ell_mem = f.ell->memory_bytes();

    // All should be positive.
    REQUIRE(coo_mem > 0);
    REQUIRE(csr_mem > 0);
    REQUIRE(csc_mem > 0);
    REQUIRE(ell_mem > 0);

    // COO stores 3 arrays × 8 entries: 8*(4+4+8) = 128 bytes
    REQUIRE(coo_mem == 8 * (sizeof(int) + sizeof(int) + sizeof(double)));

    // CSR: row_ptr(6 ints) + col_idx(8 ints) + val(8 doubles) = 24 + 32 + 64 = 120
    REQUIRE(csr_mem == 6 * sizeof(int) + 8 * sizeof(int) + 8 * sizeof(double));

    // ELL: max_nnz_per_row = 3 (rows 0 and 4 have 2 and 3 entries respectively,
    //       so max is 3).  5 * 3 * (sizeof(int) + sizeof(double)) = 180
    REQUIRE(f.ell->max_nnz_per_row() == 3);
    REQUIRE(ell_mem == 5u * 3 * (sizeof(int) + sizeof(double)));
}

// ---------------------------------------------------------------------------
// Test: empty spike source
// ---------------------------------------------------------------------------
TEST_CASE("Scatter with no spikes produces zero buffer", "[formats][scatter]")
{
    AllFormats f;
    std::vector<int> spikes = {};
    std::vector<double> out(5, 0.0);

    f.coo->scatter(spikes, out);
    for (double v : out) REQUIRE(v == 0.0);

    f.csr->scatter(spikes, out);
    for (double v : out) REQUIRE(v == 0.0);

    f.csc->scatter(spikes, out);
    for (double v : out) REQUIRE(v == 0.0);

    f.ell->scatter(spikes, out);
    for (double v : out) REQUIRE(v == 0.0);
}
