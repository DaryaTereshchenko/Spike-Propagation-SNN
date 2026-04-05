#pragma once

#include "sparse_matrix.h"   // COOTriplets
#include <string>

/// Load a COO sparse matrix from a CSV file.
/// Expected format: one line per non-zero entry, comma-separated:
///   source,target,weight
/// Lines starting with '#' are treated as comments and skipped.
/// The matrix dimension N is inferred as max(source, target) + 1.
COOTriplets load_coo_from_csv(const std::string& filename);

/// Save a COO sparse matrix to a CSV file.
/// Writes a header "# source,target,weight" followed by one entry per line.
void save_coo_to_csv(const std::string& filename, const COOTriplets& triplets);
