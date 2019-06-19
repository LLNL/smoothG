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
    mfem::SparseMatrix vertex_edge_global;
    {
        std::ifstream graphFile(graphFileName);
        smoothg::ReadVertexEdge(graphFile, vertex_edge_global);
    }

    // partition
    int num_partitions = 2;
    mfem::Array<int> global_partitioning;
    mfem::Array<int> isolated_vertices;
    {
        smoothg::MetisGraphPartitioner metis_partitioner;
        metis_partitioner.setUnbalanceTol(1);
        mfem::SparseMatrix edge_vertex = smoothg::Transpose(vertex_edge_global);
        mfem::SparseMatrix vertex_vertex = smoothg::Mult(vertex_edge_global,
                                                         edge_vertex);

        mfem::Array<int> pre_isolated_vertices(2);
        pre_isolated_vertices[0] = 0;
        pre_isolated_vertices[1] = 5;
        isolated_vertices.Append(pre_isolated_vertices);

        mfem::Array<int> post_isolated_vertices(1);
        post_isolated_vertices[0] = 2;
        isolated_vertices.Append(post_isolated_vertices);

        metis_partitioner.SetPreIsolateVertices(pre_isolated_vertices[0]);
        metis_partitioner.SetPreIsolateVertices(pre_isolated_vertices[1]);
        metis_partitioner.SetPostIsolateVertices(post_isolated_vertices);

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
