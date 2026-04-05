#!/usr/bin/env python3
"""
nest_export.py — Export NEST cortical-column connectivity to CSV.

Builds a balanced excitatory/inhibitory network using PyNEST:
  - 8000 excitatory iaf_psc_alpha neurons
  - 2000 inhibitory iaf_psc_alpha neurons
  - Fixed in-degree connectivity (CE=800, CI=200)
  - Excitatory weight J = 0.1 mV, inhibitory weight -g*J (g=5)
  - Delay 1.5 ms

Extracts the full connectivity matrix via nest.GetConnections() and
writes it to CSV format: source,target,weight
"""

import argparse
import sys

try:
    import nest
except ImportError:
    print("ERROR: PyNEST is not installed. Install NEST simulator first.")
    print("       See: https://nest-simulator.readthedocs.io/")
    sys.exit(1)


def build_network(seed=42):
    """Build the balanced E/I cortical column and return connection data."""
    nest.ResetKernel()
    nest.rng_seed = seed

    # --- Network parameters ---
    N_exc = 8000       # excitatory neurons
    N_inh = 2000       # inhibitory neurons
    CE    = 800        # excitatory in-degree
    CI    = 200        # inhibitory in-degree
    J     = 0.1        # excitatory weight (mV)
    g     = 5.0        # relative inhibitory weight
    delay = 1.5        # synaptic delay (ms)

    # --- Neuron populations ---
    neuron_params = {
        "tau_m":    20.0,     # membrane time constant (ms)
        "V_th":    -50.0,     # spike threshold (mV)
        "V_reset": -65.0,     # reset potential (mV)
        "E_L":     -65.0,     # resting potential (mV)
        "t_ref":     2.0,     # refractory period (ms)
    }

    exc_pop = nest.Create("iaf_psc_alpha", N_exc, params=neuron_params)
    inh_pop = nest.Create("iaf_psc_alpha", N_inh, params=neuron_params)

    # --- Connections ---
    exc_syn = {"synapse_model": "static_synapse", "weight": J, "delay": delay}
    inh_syn = {"synapse_model": "static_synapse", "weight": -g * J, "delay": delay}
    conn_rule = {"rule": "fixed_indegree", "indegree": CE}

    # E → E and E → I
    nest.Connect(exc_pop, exc_pop, conn_rule, exc_syn)
    nest.Connect(exc_pop, inh_pop, conn_rule, exc_syn)

    # I → E and I → I
    conn_rule_inh = {"rule": "fixed_indegree", "indegree": CI}
    nest.Connect(inh_pop, exc_pop, conn_rule_inh, inh_syn)
    nest.Connect(inh_pop, inh_pop, conn_rule_inh, inh_syn)

    # --- Extract connectivity ---
    conns = nest.GetConnections()
    sources = conns.source
    targets = conns.target
    weights = conns.weight

    # Convert NEST node IDs (1-based) to 0-based indices.
    all_nodes = sorted(list(exc_pop.tolist()) + list(inh_pop.tolist()))
    node_to_idx = {nid: i for i, nid in enumerate(all_nodes)}

    rows = [node_to_idx[s] for s in sources]
    cols = [node_to_idx[t] for t in targets]
    vals = list(weights)

    return rows, cols, vals, N_exc + N_inh


def export_csv(rows, cols, vals, output_path):
    """Write connectivity to CSV."""
    with open(output_path, "w") as f:
        f.write("# source,target,weight\n")
        for r, c, v in zip(rows, cols, vals):
            f.write(f"{r},{c},{v}\n")
    print(f"Exported {len(rows)} connections to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export NEST cortical-column connectivity to CSV")
    parser.add_argument("--output", "-o", default="results/nest_connectivity.csv",
                        help="Output CSV file path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for NEST")
    args = parser.parse_args()

    print(f"Building NEST network (seed={args.seed})...")
    rows, cols, vals, N = build_network(args.seed)
    print(f"  N = {N}, connections = {len(rows)}")
    export_csv(rows, cols, vals, args.output)


if __name__ == "__main__":
    main()
