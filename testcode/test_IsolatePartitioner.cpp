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
   Test metis partitioning with isolated vertices, just make sure it runs.
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
    mfem::SparseMatrix vertex_edge = smoothg::ReadVertexEdge(graphFileName);

    // partition
    int num_partitions = 2;
    mfem::Array<int> global_partitioning;
    std::vector<int> isolated_vertices;
    {
        auto type = smoothg::MetisGraphPartitioner::PartType::RECURSIVE;
        smoothg::MetisGraphPartitioner metis_partitioner(type);
        metis_partitioner.setUnbalanceTol(1);
        mfem::SparseMatrix edge_vertex = smoothg::Transpose(vertex_edge);
        auto vertex_vertex = smoothg::Mult(vertex_edge, edge_vertex);

        std::vector<int> pre_isolated_vertices(2);
        pre_isolated_vertices[0] = 0;
        pre_isolated_vertices[1] = 1;
        isolated_vertices.push_back(pre_isolated_vertices[0]);
        isolated_vertices.push_back(pre_isolated_vertices[1]);

        int post_isolated_vertex = 2;
        isolated_vertices.push_back(post_isolated_vertex);

        metis_partitioner.SetPreIsolateVertices(pre_isolated_vertices);
        metis_partitioner.SetPostIsolateVertices(post_isolated_vertex);

        metis_partitioner.doPartition(vertex_vertex, num_partitions,
                                      global_partitioning);
    }

    for (int i = 0; i < global_partitioning.Size(); ++i)
    {
        std::cout << "partition[" << i << "] = " << global_partitioning[i] << std::endl;
    }

    // Check if each isolated vertex does form a partition itself
    for (auto isolated_vertex : isolated_vertices)
    {
        int singleton_partition = global_partitioning[isolated_vertex];
        for (int i = 0; i < global_partitioning.Size(); i++)
        {
            if ( (i != isolated_vertex) && (global_partitioning[i] == singleton_partition) )
                result++;
        }
    }

    if (result > 0)
        std::cerr << "Unexpected partitioning from IsolatePartitioner!" << std::endl;
    return result;
}
