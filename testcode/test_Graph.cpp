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
   Test Graph object, distributed vs global

   edge maps should be the same

   vertex maps don't have to be the same,
   but should assign each vertex a unique global id
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "smoothG.hpp"

using namespace smoothg;
using linalgcpp::ReadCSR;
using linalgcpp::operator<<;

int main(int argc, char* argv[])
{
    MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm_;
    int myid = mpi_info.myid_;
    int num_procs = mpi_info.num_procs_;

    assert(num_procs == 1 || num_procs == 2);

    // load the graph
    std::string graph_filename = "../../graphdata/vertex_edge_tiny.txt";
    SparseMatrix vertex_edge = ReadCSR(graph_filename);

    std::vector<int> partition {0, 0, 0, 1, 1, 1};

    Graph global_graph(comm, vertex_edge, partition);
    Graph dist_graph(global_graph.vertex_edge_local_,
                     global_graph.edge_true_edge_,
                     global_graph.part_local_);

    const auto& global_map = global_graph.edge_map_;
    const auto& dist_map = dist_graph.edge_map_;
    int num_edges = dist_map.size();

    assert(global_map.size() == dist_map.size());

    bool failed = false;

    for (int i = 0; i < num_edges; ++i)
    {
        if (global_map[i] != dist_map[i])
        {
            failed = true;
        }
    }

    if (myid == 0)
    {
        std::cout << "Global Vertex Map: " << global_graph.vertex_map_;
        std::cout << "Global Edge Map:   " << global_graph.edge_map_;

        std::cout << "Dist.  Vertex Map: " << dist_graph.vertex_map_;
        std::cout << "Dist.  Edge Map:   " << dist_graph.edge_map_;
    }

    return failed;
}

