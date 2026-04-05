// Minimal unit tests for sparse matrix formats and LIF neuron.
// Uses no external test framework – each TEST() macro counts failures and
// the program exits with 1 if any test failed.

#include "coo_matrix.hpp"
#include "csr_matrix.hpp"
#include "csc_matrix.hpp"
#include "ell_matrix.hpp"
#include "lif_neuron.hpp"
#include "network.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int failures = 0;

// ── Test helpers ──────────────────────────────────────────────────────────────

#define EXPECT_EQ(a, b)                                                       \
    do {                                                                       \
        if ((a) != (b)) {                                                     \
            std::printf("FAIL %s:%d  %s != %s  (%zu vs %zu)\n",              \
                        __FILE__, __LINE__, #a, #b,                           \
                        static_cast<std::size_t>(a),                          \
                        static_cast<std::size_t>(b));                         \
            ++failures;                                                        \
        }                                                                      \
    } while (false)

#define EXPECT_NEAR(a, b, tol)                                                \
    do {                                                                       \
        double _a = (a), _b = (b), _t = (tol);                               \
        if (std::fabs(_a - _b) > _t) {                                        \
            std::printf("FAIL %s:%d  |%s - %s| = %.6f > %.6f\n",             \
                        __FILE__, __LINE__, #a, #b,                           \
                        std::fabs(_a - _b), _t);                              \
            ++failures;                                                        \
        }                                                                      \
    } while (false)

#define EXPECT_TRUE(cond)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d  %s is false\n",                          \
                        __FILE__, __LINE__, #cond);                            \
            ++failures;                                                        \
        }                                                                      \
    } while (false)

// ── Shared test fixture: 3×3 matrix ──────────────────────────────────────────
//
// Connectivity:   0->1 (w=2), 0->2 (w=3), 1->2 (w=5)
//
// Scatter with neuron 0 firing:
//   out[1] += 2, out[2] += 3
// Scatter with neurons 0 and 1 firing:
//   out[1] += 2, out[2] += 3+5 = 8

static const std::size_t ROWS = 3;
static const std::size_t COLS = 3;
static const std::vector<std::size_t> ROW_IDX = {0, 0, 1};
static const std::vector<std::size_t> COL_IDX = {1, 2, 2};
static const std::vector<double>      VALS    = {2.0, 3.0, 5.0};

static std::vector<double> zero_vec(std::size_t n) { return std::vector<double>(n, 0.0); }

// ── COO tests ────────────────────────────────────────────────────────────────

static void test_coo_nnz() {
    CooMatrix m(ROWS, COLS);
    for (std::size_t k = 0; k < ROW_IDX.size(); ++k)
        m.add_entry(ROW_IDX[k], COL_IDX[k], VALS[k]);
    EXPECT_EQ(m.nnz(), 3u);
    EXPECT_EQ(m.rows(), ROWS);
    EXPECT_EQ(m.cols(), COLS);
    std::printf("PASS test_coo_nnz\n");
}

static void test_coo_scatter() {
    CooMatrix m(ROWS, COLS);
    for (std::size_t k = 0; k < ROW_IDX.size(); ++k)
        m.add_entry(ROW_IDX[k], COL_IDX[k], VALS[k]);

    std::vector<bool> spikes = {true, false, false};
    auto out = zero_vec(COLS);
    m.scatter(spikes, out);
    EXPECT_NEAR(out[0], 0.0, 1e-9);
    EXPECT_NEAR(out[1], 2.0, 1e-9);
    EXPECT_NEAR(out[2], 3.0, 1e-9);
    std::printf("PASS test_coo_scatter\n");
}

static void test_coo_scatter_two_spikes() {
    CooMatrix m(ROWS, COLS);
    for (std::size_t k = 0; k < ROW_IDX.size(); ++k)
        m.add_entry(ROW_IDX[k], COL_IDX[k], VALS[k]);

    std::vector<bool> spikes = {true, true, false};
    auto out = zero_vec(COLS);
    m.scatter(spikes, out);
    EXPECT_NEAR(out[1], 2.0, 1e-9);
    EXPECT_NEAR(out[2], 8.0, 1e-9);
    std::printf("PASS test_coo_scatter_two_spikes\n");
}

// ── CSR tests ────────────────────────────────────────────────────────────────

static void test_csr_scatter() {
    auto m = CsrMatrix::from_coo(ROWS, COLS, ROW_IDX, COL_IDX, VALS);
    EXPECT_EQ(m.nnz(), 3u);

    std::vector<bool> spikes = {true, false, false};
    auto out = zero_vec(COLS);
    m.scatter(spikes, out);
    EXPECT_NEAR(out[1], 2.0, 1e-9);
    EXPECT_NEAR(out[2], 3.0, 1e-9);
    std::printf("PASS test_csr_scatter\n");
}

static void test_csr_gather() {
    auto m = CsrMatrix::from_coo(ROWS, COLS, ROW_IDX, COL_IDX, VALS);

    std::vector<bool> spikes = {false, true, false};
    auto out = zero_vec(COLS);
    m.gather(spikes, out);
    EXPECT_NEAR(out[2], 5.0, 1e-9);
    EXPECT_NEAR(out[0], 0.0, 1e-9);
    std::printf("PASS test_csr_gather\n");
}

static void test_csr_memory() {
    auto m = CsrMatrix::from_coo(ROWS, COLS, ROW_IDX, COL_IDX, VALS);
    // rowptr: (ROWS+1) * sizeof(size_t)
    // colidx: 3 * sizeof(size_t)
    // values: 3 * sizeof(double)
    std::size_t expected = (ROWS + 1) * sizeof(std::size_t)
                         + 3 * sizeof(std::size_t)
                         + 3 * sizeof(double);
    EXPECT_EQ(m.memory_bytes(), expected);
    std::printf("PASS test_csr_memory\n");
}

// ── CSC tests ────────────────────────────────────────────────────────────────

static void test_csc_gather() {
    auto m = CscMatrix::from_coo(ROWS, COLS, ROW_IDX, COL_IDX, VALS);
    EXPECT_EQ(m.nnz(), 3u);

    std::vector<bool> spikes = {true, true, false};
    auto out = zero_vec(COLS);
    m.gather(spikes, out);
    EXPECT_NEAR(out[1], 2.0, 1e-9);
    EXPECT_NEAR(out[2], 8.0, 1e-9);
    std::printf("PASS test_csc_gather\n");
}

static void test_csc_scatter() {
    auto m = CscMatrix::from_coo(ROWS, COLS, ROW_IDX, COL_IDX, VALS);

    std::vector<bool> spikes = {true, false, false};
    auto out = zero_vec(COLS);
    m.scatter(spikes, out);
    EXPECT_NEAR(out[1], 2.0, 1e-9);
    EXPECT_NEAR(out[2], 3.0, 1e-9);
    std::printf("PASS test_csc_scatter\n");
}

// ── ELL tests ────────────────────────────────────────────────────────────────

static void test_ell_scatter() {
    auto m = EllMatrix::from_coo(ROWS, COLS, ROW_IDX, COL_IDX, VALS);
    EXPECT_EQ(m.nnz(), 3u);

    std::vector<bool> spikes = {true, false, false};
    auto out = zero_vec(COLS);
    m.scatter(spikes, out);
    EXPECT_NEAR(out[1], 2.0, 1e-9);
    EXPECT_NEAR(out[2], 3.0, 1e-9);
    std::printf("PASS test_ell_scatter\n");
}

static void test_ell_max_cols() {
    auto m = EllMatrix::from_coo(ROWS, COLS, ROW_IDX, COL_IDX, VALS);
    // Row 0 has degree 2, row 1 has degree 1, row 2 has degree 0 -> max_cols = 2
    EXPECT_EQ(m.max_cols(), 2u);
    std::printf("PASS test_ell_max_cols\n");
}

static void test_ell_memory() {
    auto m = EllMatrix::from_coo(ROWS, COLS, ROW_IDX, COL_IDX, VALS);
    std::size_t slots = ROWS * m.max_cols();
    std::size_t expected = slots * sizeof(std::size_t) + slots * sizeof(double);
    EXPECT_EQ(m.memory_bytes(), expected);
    std::printf("PASS test_ell_memory\n");
}

// ── Cross-format consistency ──────────────────────────────────────────────────

static void test_all_formats_agree() {
    CooMatrix coo(ROWS, COLS);
    for (std::size_t k = 0; k < ROW_IDX.size(); ++k)
        coo.add_entry(ROW_IDX[k], COL_IDX[k], VALS[k]);
    auto csr = CsrMatrix::from_coo(ROWS, COLS, ROW_IDX, COL_IDX, VALS);
    auto csc = CscMatrix::from_coo(ROWS, COLS, ROW_IDX, COL_IDX, VALS);
    auto ell = EllMatrix::from_coo(ROWS, COLS, ROW_IDX, COL_IDX, VALS);

    std::vector<bool> spikes = {true, true, false};
    auto ref = zero_vec(COLS);
    csr.scatter(spikes, ref);

    for (SparseMatrix* m : {static_cast<SparseMatrix*>(&coo),
                            static_cast<SparseMatrix*>(&csc),
                            static_cast<SparseMatrix*>(&ell)}) {
        auto out = zero_vec(COLS);
        m->scatter(spikes, out);
        for (std::size_t j = 0; j < COLS; ++j) {
            EXPECT_NEAR(out[j], ref[j], 1e-9);
        }
    }
    std::printf("PASS test_all_formats_agree\n");
}

// ── LIF neuron tests ──────────────────────────────────────────────────────────

static void test_lif_fires() {
    LifNeuronPop::Params p;
    p.V_thresh    = -50.0;
    p.V_rest      = -65.0;
    p.R           = 10.0;
    p.dt          = 0.5;
    p.refrac_steps = 2;
    LifNeuronPop pop(1, p);

    // Drive a large current to force a spike within a few steps.
    std::vector<double> I = {10.0};
    bool fired = false;
    for (int t = 0; t < 200 && !fired; ++t) {
        auto spk = pop.step(I);
        if (spk[0]) fired = true;
    }
    EXPECT_TRUE(fired);
    std::printf("PASS test_lif_fires\n");
}

static void test_lif_refractory() {
    LifNeuronPop::Params p;
    p.refrac_steps = 5;
    p.R     = 20.0;
    p.dt    = 1.0;
    LifNeuronPop pop(1, p);

    std::vector<double> I = {100.0};
    int spk_count = 0;
    for (int t = 0; t < 10; ++t) {
        auto spk = pop.step(I);
        if (spk[0]) ++spk_count;
    }
    // With refrac=5 and 10 steps, at most 2 spikes should occur.
    EXPECT_TRUE(spk_count <= 2);
    std::printf("PASS test_lif_refractory\n");
}

// ── Network topology tests ────────────────────────────────────────────────────

static void test_erdos_renyi_density() {
    const std::size_t N = 200;
    const double p = 0.05;
    auto edges = erdos_renyi(N, p, 1.0, 42);
    double actual_p = static_cast<double>(edges.size()) / (N * (N - 1));
    // Should be within 2 percentage points of target density.
    EXPECT_TRUE(std::fabs(actual_p - p) < 0.02);
    std::printf("PASS test_erdos_renyi_density (actual=%.4f)\n", actual_p);
}

static void test_fixed_indegree() {
    const std::size_t N = 50, k = 5;
    auto edges = fixed_indegree(N, k, 1.0, 0);
    EXPECT_EQ(edges.size(), N * k);
    std::printf("PASS test_fixed_indegree\n");
}

static void test_barabasi_albert_no_self_loops() {
    const std::size_t N = 50, m = 2;
    auto edges = barabasi_albert(N, m, 1.0, 0);
    bool has_self_loop = false;
    for (const auto& e : edges) {
        if (e.src == e.dst) has_self_loop = true;
    }
    EXPECT_TRUE(!has_self_loop);
    std::printf("PASS test_barabasi_albert_no_self_loops\n");
}

static void test_watts_strogatz_size() {
    const std::size_t N = 40, k = 4;
    auto edges = watts_strogatz(N, k, 0.1, 1.0, 0);
    // Ring lattice has N*k directed edges; rewiring preserves count.
    EXPECT_EQ(edges.size(), N * k);
    std::printf("PASS test_watts_strogatz_size\n");
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    std::printf("--- COO ---\n");
    test_coo_nnz();
    test_coo_scatter();
    test_coo_scatter_two_spikes();

    std::printf("--- CSR ---\n");
    test_csr_scatter();
    test_csr_gather();
    test_csr_memory();

    std::printf("--- CSC ---\n");
    test_csc_gather();
    test_csc_scatter();

    std::printf("--- ELL ---\n");
    test_ell_scatter();
    test_ell_max_cols();
    test_ell_memory();

    std::printf("--- Cross-format ---\n");
    test_all_formats_agree();

    std::printf("--- LIF neuron ---\n");
    test_lif_fires();
    test_lif_refractory();

    std::printf("--- Network topologies ---\n");
    test_erdos_renyi_density();
    test_fixed_indegree();
    test_barabasi_albert_no_self_loops();
    test_watts_strogatz_size();

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) FAILED.\n", failures);
    return 1;
}
