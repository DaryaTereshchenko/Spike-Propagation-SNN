# CSV I/O and NEST Integration

> **Implementation Status (April 2026):** CSV I/O is fully implemented in
> `src/csv_io.cpp` / `include/csv_io.h`. The NEST export script
> (`scripts/nest_export.py`) defines the Brunel-style balanced E/I column
> model. Cross-validation with NEST-exported connectivity is supported via
> the `--nest-csv` CLI flag.

## Overview

This module provides CSV import/export for sparse connectivity matrices in
COO (Coordinate) format.  Its primary purpose is to enable **cross-validation**
between our C++ sparse-format benchmark and the NEST simulator: a network
can be defined once in PyNEST, exported to CSV, and then loaded into the
C++ benchmark to propagate spikes through the exact same connectivity.

---

## CSV Format Specification

```
# Comment lines start with '#' and are skipped.
# source_id  target_id  weight
0  5  0.1
0  12 0.1
1  3  0.1
...
```

| Field | Type | Description |
|-------|------|-------------|
| Column 0 | int | Source (pre-synaptic) neuron index (0-based) |
| Column 1 | int | Target (post-synaptic) neuron index (0-based) |
| Column 2 | double | Synaptic weight |

- Whitespace-delimited (spaces or tabs).
- Lines starting with `#` are skipped.
- The matrix dimension $N$ is inferred as $\max(\text{all indices}) + 1$.

---

## NEST Reference Model

The script `scripts/nest_export.py` defines a balanced E/I cortical column:

| Parameter | Value |
|-----------|-------|
| Excitatory neurons ($N_E$) | 8 000 |
| Inhibitory neurons ($N_I$) | 2 000 |
| Total $N$ | 10 000 |
| Neuron model | `iaf_psc_alpha` (LIF with alpha-shaped PSCs) |
| Excitatory in-degree ($C_E$) | 800 |
| Inhibitory in-degree ($C_I$) | 200 |
| Excitatory weight ($J$) | 0.1 mV |
| Inhibitory weight | $-g \cdot J = -0.5$ mV |
| Inhibition factor ($g$) | 5.0 |
| Synaptic delay | 1.5 ms |
| Connection rule | `fixed_indegree` (no autapses) |

This model is based on the cortical microcircuit by Brunel (2000) and
the NEST reference implementation by Potjans & Diesmann (2014).

### Export Workflow

```bash
# 1. Run the PyNEST script to export connectivity.
python3 scripts/nest_export.py

# 2. Load into the C++ benchmark.
./spike_benchmark --format csr --size 10000 \
    --nest-csv results/nest_connectivity.csv
```

The export script converts NEST's 1-based NodeIDs to 0-based indices
before writing the CSV.

---

## API Reference

### Loading

```cpp
COOTriplets load_coo_from_csv(const std::string& filename);
```

Returns a `COOTriplets` struct.  The number of neurons $N$ is determined
from the maximum index found in the file.

### Saving

```cpp
void save_coo_to_csv(const std::string& filename, const COOTriplets& coo);
```

Writes the COO data in the format described above, including a header
comment line.

### Files

| File | Purpose |
|------|---------|
| `include/csv_io.h` | Function declarations |
| `src/csv_io.cpp`   | CSV parsing and writing |
| `scripts/nest_export.py` | PyNEST balanced E/I column export |
