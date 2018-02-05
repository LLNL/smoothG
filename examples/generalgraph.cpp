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
   @example
   @file generalgraph.cpp
   @brief Compares a graph upscaled solution to the fine solution.
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "linalgcpp.hpp"
#include "parlinalgcpp.hpp"
#include "partition.hpp"

//using namespace smoothg;
using namespace linalgcpp;
using namespace parlinalgcpp;

std::vector<int> MetisPart(const SparseMatrix<int>& vertex_edge, int num_parts);

int main(int argc, char* argv[])
{
    // 1. Initialize MPI
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // program options from command line
    int num_partitions = 12;
    std::string graphFileName = "../../graphdata/vertex_edge_sample.txt";
    std::string FiedlerFileName = "../../graphdata/fiedler_sample.txt";
    std::string partition_filename = "../../graphdata/partition_sample.txt";
    std::string weight_filename = "";
    std::string w_block_filename = "";
    bool metis_agglomeration = false;
    int max_evects = 4;
    double spect_tol = 1.e-3;
    bool hybridization = false;
    int isolate = -1;

    assert(num_partitions >= num_procs);

    /// [Load graph from file or generate one]
    SparseMatrix<int> vertex_edge_global = ReadCSR<int>(graphFileName);

    const int nvertices_global = vertex_edge_global.Rows();
    const int nedges_global = vertex_edge_global.Cols();
    /// [Load graph from file or generate one]

    /// [Partitioning]
    std::vector<int> global_partitioning;
    if (metis_agglomeration)
    {
        global_partitioning = MetisPart(vertex_edge_global, num_partitions);
    }
    else
    {
        global_partitioning = ReadText<int>(partition_filename);
    }
    /// [Partitioning]

    /// [Load the edge weights]
    std::vector<double> weight;
    if (weight_filename.size() > 0)
    {
        weight = ReadText(weight_filename);
    }
    else
    {
        weight = std::vector<double>(nedges_global, 1.0);
    }
    /// [Load the edge weights]

    // Set up GraphUpscale
    /*
    {
        /// [Upscale]
        GraphUpscale upscale(comm, vertex_edge_global, global_partitioning,
                             spect_tol, max_evects, weight);

        upscale.PrintInfo();
        upscale.ShowSetupTime();
        /// [Upscale]

        /// [Right Hand Side]
        mfem::Vector rhs_u_fine = upscale.ReadVertexVector(FiedlerFileName);

        mfem::BlockVector fine_rhs(upscale.GetFineBlockVector());
        fine_rhs.GetBlock(0) = 0.0;
        fine_rhs.GetBlock(1) = rhs_u_fine;
        /// [Right Hand Side]

        /// [Solve]
        mfem::BlockVector upscaled_sol = upscale.Solve(fine_rhs);
        upscale.ShowCoarseSolveInfo();

        mfem::BlockVector fine_sol = upscale.SolveFine(fine_rhs);
        upscale.ShowFineSolveInfo();
        /// [Solve]

        /// [Check Error]
        upscale.ShowErrors(upscaled_sol, fine_sol);
        /// [Check Error]
    }
    */

    MPI_Finalize();
    return 0;
}

std::vector<int> MetisPart(const SparseMatrix<int>& vertex_edge, int num_parts)
{
    SparseMatrix<int> edge_vertex = vertex_edge.Transpose();
    SparseMatrix<int> vertex_vertex = vertex_edge.Mult(edge_vertex);

    return Partition(vertex_vertex, num_parts);
}
