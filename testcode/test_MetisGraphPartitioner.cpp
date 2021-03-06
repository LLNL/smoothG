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

#include "mfem.hpp"
#include "../src/MetisGraphPartitioner.hpp"
#include "../src/utilities.hpp"

int main(int argc, char* argv[])
{
    // initialize MPI
    smoothg::mpi_session session(argc, argv);

    int result = 0;

    // parse command line options
    mfem::OptionsParser args(argc, argv);
    const char* graphFileName = "../../graphdata/vertex_edge_tiny.txt";
    args.AddOption(&graphFileName, "-g", "--graph",
                   "Graph connection data.");

    // load the graph
    mfem::SparseMatrix vertex_edge_global;
    {
        std::ifstream graphFile(graphFileName);
        smoothg::ReadVertexEdge(graphFile, vertex_edge_global);
    }

    // partition
    int num_partitions = 2;
    mfem::Array<int> global_partitioning;
    {
        smoothg::MetisGraphPartitioner partitioner;
        partitioner.setUnbalanceTol(2);
        mfem::SparseMatrix* edge_vertex = Transpose(vertex_edge_global);
        mfem::SparseMatrix* vertex_vertex = Mult(vertex_edge_global,
                                                 *edge_vertex);
        delete edge_vertex;
        partitioner.doPartition(*vertex_vertex, num_partitions,
                                global_partitioning);
        delete vertex_vertex;
    }

    for (int i = 0; i < global_partitioning.Size() / 2; ++i)
    {
        std::cout << "partition[" << i << "] = " << global_partitioning[i] << std::endl;
        if (global_partitioning[i] != 1) result++;
    }
    for (int i = global_partitioning.Size() / 2; i < global_partitioning.Size(); ++i)
    {
        std::cout << "partition[" << i << "] = " << global_partitioning[i] << std::endl;
        if (global_partitioning[i] != 0) result++;
    }

    if (result > 0)
        std::cerr << "Unexpected partitioning from metis!" << std::endl;
    return result;
}
