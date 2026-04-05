#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "lif_neuron.h"

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Test: zero input → voltage decays toward V_rest
// ---------------------------------------------------------------------------
TEST_CASE("Zero input: voltage decays toward V_rest", "[lif]")
{
    LIFParams params;
    params.tau_m    = 20.0;
    params.V_rest   = -65.0;
    params.V_thresh = -50.0;
    params.dt       = 1.0;

    LIFPopulation pop(1, params);

    // Manually set voltage above rest to test decay.
    // We can't directly set V, so we inject a small current to raise it
    // first, then let it decay.

    // Inject current to raise voltage to ~ -60 mV.
    // dV = (dt/tau_m) * (-(V - Vrest) + R * I)
    // At V = -65: dV = (1/20) * (0 + 1 * I) = I/20
    // To get V = -60: need dV = 5 → I = 100.
    std::vector<double> strong_input = {100.0};
    pop.step(strong_input);
    double V_after_kick = pop.voltages()[0];
    REQUIRE(V_after_kick > params.V_rest);

    // Now apply zero input and verify exponential decay.
    std::vector<double> zero_input = {0.0};
    double V_prev = V_after_kick;
    for (int t = 0; t < 100; ++t) {
        pop.step(zero_input);
        double V_now = pop.voltages()[0];
        // V should move toward V_rest.
        REQUIRE(std::abs(V_now - params.V_rest) <
                std::abs(V_prev - params.V_rest) + 1e-12);
        V_prev = V_now;
    }

    // After 100 steps (100 ms with tau=20ms = 5 time constants),
    // should be very close to V_rest.
    REQUIRE_THAT(pop.voltages()[0],
                 Catch::Matchers::WithinAbs(params.V_rest, 0.1));
}

// ---------------------------------------------------------------------------
// Test: constant strong input → regular spiking
// ---------------------------------------------------------------------------
TEST_CASE("Constant input produces regular spiking", "[lif]")
{
    LIFParams params;
    params.tau_m    = 20.0;
    params.V_rest   = -65.0;
    params.V_thresh = -50.0;
    params.V_reset  = -65.0;
    params.R        = 1.0;
    params.dt       = 1.0;
    params.t_ref    = 2.0;

    LIFPopulation pop(1, params);

    // Strong constant input: I = 500.
    // dV per step ≈ (1/20) * (15 + 500) at threshold ≈ 25.75 mV/step
    // So neuron should fire frequently.
    std::vector<double> input = {500.0};

    int total_spikes = 0;
    for (int t = 0; t < 1000; ++t) {
        auto spikes = pop.step(input);
        total_spikes += static_cast<int>(spikes.size());
    }

    // Should spike many times with strong input.
    REQUIRE(total_spikes > 100);

    // Given t_ref = 2 steps, maximum possible rate = 1000 / (1 + 2) ≈ 333.
    // Should be below that.
    REQUIRE(total_spikes <= 334);
}

// ---------------------------------------------------------------------------
// Test: refractory period → no spike for t_ref/dt steps after a spike
// ---------------------------------------------------------------------------
TEST_CASE("Refractory period prevents immediate re-firing", "[lif]")
{
    LIFParams params;
    params.tau_m    = 20.0;
    params.V_rest   = -65.0;
    params.V_thresh = -50.0;
    params.V_reset  = -65.0;
    params.R        = 1.0;
    params.dt       = 1.0;
    params.t_ref    = 5.0;   // 5 ms refractory

    LIFPopulation pop(1, params);

    // Very strong input to guarantee spiking as soon as refractory ends.
    std::vector<double> input = {10000.0};

    // Find the first spike.
    int first_spike_time = -1;
    for (int t = 0; t < 100; ++t) {
        auto spikes = pop.step(input);
        if (!spikes.empty() && first_spike_time < 0) {
            first_spike_time = t;
        }
    }
    REQUIRE(first_spike_time >= 0);

    // Reset and run again, recording spike times.
    pop.reset();
    std::vector<int> spike_times;
    for (int t = 0; t < 50; ++t) {
        auto spikes = pop.step(input);
        if (!spikes.empty()) {
            spike_times.push_back(t);
        }
    }

    // Check that inter-spike intervals are >= t_ref/dt + 1.
    int ref_steps = static_cast<int>(params.t_ref / params.dt);
    for (size_t i = 1; i < spike_times.size(); ++i) {
        int isi = spike_times[i] - spike_times[i - 1];
        REQUIRE(isi >= ref_steps + 1);
    }
}

// ---------------------------------------------------------------------------
// Test: reset() restores initial state
// ---------------------------------------------------------------------------
TEST_CASE("Reset restores initial state", "[lif]")
{
    LIFParams params;
    LIFPopulation pop(10, params);

    // Drive some activity.
    std::vector<double> input(10, 500.0);
    for (int t = 0; t < 100; ++t) {
        pop.step(input);
    }

    pop.reset();

    for (int i = 0; i < 10; ++i) {
        REQUIRE(pop.voltages()[i] == params.V_rest);
    }
}

// ---------------------------------------------------------------------------
// Test: subthreshold input → no spikes
// ---------------------------------------------------------------------------
TEST_CASE("Subthreshold input produces no spikes", "[lif]")
{
    LIFParams params;
    params.tau_m    = 20.0;
    params.V_rest   = -65.0;
    params.V_thresh = -50.0;
    params.R        = 1.0;
    params.dt       = 1.0;

    LIFPopulation pop(1, params);

    // Equilibrium voltage: V_eq = V_rest + R * I = -65 + 1 * 14 = -51.
    // Just below threshold (-50).  Actually that's -51 which is below -50.
    // The neuron should NOT spike because steady state is V_rest + R*I
    // and the dynamics approach it asymptotically from below.
    // V_eq = V_rest + R * I when dV/dt = 0: -(V-Vrest) + R*I = 0 → V = Vrest + R*I.
    // For I = 14: V_eq = -65 + 14 = -51 < -50.  So no spike ever.
    std::vector<double> input = {14.0};

    int total_spikes = 0;
    for (int t = 0; t < 500; ++t) {
        auto spikes = pop.step(input);
        total_spikes += static_cast<int>(spikes.size());
    }

    REQUIRE(total_spikes == 0);
}
