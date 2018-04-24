/*BHEADER**********************************************************************
 *
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * LLNL-CODE-745247. All Rights reserved. See file COPYRIGHT for details.
 *
 * This file is part of smoothG. For more information and source code
 * availability, see https://www.github.com/llnl/smoothG.
 *
 * smoothG is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 ***********************************************************************EHEADER*/

/**
   Test metis partitioning, just make sure it runs.
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "smoothG.hpp"

using namespace smoothg;
using linalgcpp::ReadCSR;

int main(int argc, char* argv[])
{
    // load the graph
    std::string graph_filename = "../../graphdata/vertex_edge_tiny.txt";
    SparseMatrix vertex_edge = ReadCSR(graph_filename);

    // partition
    int num_parts = 2;
    double ubal_tol = 2.0;

    SparseMatrix edge_vertex = vertex_edge.Transpose();
    SparseMatrix vertex_vertex = vertex_edge.Mult(edge_vertex);

    std::vector<int> partition = Partition(vertex_vertex, num_parts, ubal_tol);

    // check partition
    int size = partition.size();
    int result = 0;

    for (int i = 0; i < size / 2; ++i)
    {
        std::cout << "partition[" << i << "] = " << partition[i] << "\n";

        if (partition[i] != 1)
        {
            result++;
        }
    }

    for (int i = size / 2; i < size; ++i)
    {
        std::cout << "partition[" << i << "] = " << partition[i] << "\n";

        if (partition[i] != 0)
        {
            result++;
        }
    }

    if (result > 0)
    {
        std::cerr << "Unexpected partitioning from metis!" << std::endl;
    }

    return result;
}
