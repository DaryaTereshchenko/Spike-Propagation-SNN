#include "coo_matrix.h"
#include "csr_matrix.h"
#include "csc_matrix.h"
#include "ell_matrix.h"
#include "topology.h"
#include <iostream>

int main()
{
    // Generate a tiny 8-neuron Erdős–Rényi network so you can see everything.
    std::cout << "Generating ER topology: N=8, p=0.3, seed=42\n\n";
    COOTriplets triplets = generate_erdos_renyi(8, 0.3, 42);

    // Set all weights to 1.0 for readability.
    for (auto& v : triplets.vals) v = 1.0;

    // Build all four formats from the SAME triplets.
    COOMatrix coo(triplets);
    CSRMatrix csr(triplets);
    CSCMatrix csc(triplets);
    ELLMatrix ell(triplets);

    // Dump each one.
    coo.dump();
    csr.dump();
    csc.dump();
    ell.dump();

    return 0;
}
