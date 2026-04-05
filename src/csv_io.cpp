#include "csv_io.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

COOTriplets load_coo_from_csv(const std::string& filename)
{
    std::ifstream f(filename);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + filename);
    }

    COOTriplets coo;
    int max_id = 0;
    std::string line;

    while (std::getline(f, line)) {
        // Skip empty lines and comments.
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        int src, tgt;
        double weight;
        char comma1, comma2;

        if (!(iss >> src >> comma1 >> tgt >> comma2 >> weight) ||
            comma1 != ',' || comma2 != ',') {
            continue;   // skip malformed lines
        }

        coo.rows.push_back(src);
        coo.cols.push_back(tgt);
        coo.vals.push_back(weight);

        max_id = std::max(max_id, std::max(src, tgt));
    }

    coo.N = max_id + 1;
    return coo;
}

void save_coo_to_csv(const std::string& filename, const COOTriplets& triplets)
{
    std::ofstream f(filename);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open CSV file for writing: " + filename);
    }

    f << "# source,target,weight\n";
    for (size_t i = 0; i < triplets.nnz(); ++i) {
        f << triplets.rows[i] << ","
          << triplets.cols[i] << ","
          << triplets.vals[i] << "\n";
    }
}
