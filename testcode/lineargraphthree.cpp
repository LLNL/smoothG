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
   Test code for linear graph.

   do three levels
*/

#include "mfem.hpp"

#include "../src/picojson.h"

#include "../src/smoothG.hpp"

using namespace smoothg;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

class LinearGraph
{
public:
    /// n here is number of vertices, we will have n-1 edges in the graph
    LinearGraph(int n);

    int GetN() const { return n_; }

    const Graph& GetGraph() const { return graph_; }

private:
    int n_;
    Graph graph_;
};

LinearGraph::LinearGraph(int n) :
    n_(n)
{
    mfem::Vector edge_weight(n - 1);
    mfem::SparseMatrix vertex_edge(n, n - 1);

    for (int i = 0; i < n - 1; ++i)
    {
        edge_weight[i] = 1.0;
        vertex_edge.Add(i, i, 1.0);
        vertex_edge.Add(i + 1, i, 1.0);
    }
    vertex_edge.Finalize();

    graph_ = Graph(MPI_COMM_WORLD, vertex_edge, edge_weight);
}

int main(int argc, char* argv[])
{
    int num_procs, myid;
    int result = 0;

    // 1. Initialize MPI
    mpi_session session(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    UpscaleParameters upscale_param;
    mfem::OptionsParser args(argc, argv);
    int global_size = 4;
    args.AddOption(&global_size, "-s", "--size", "Size of fine linear graph.");
    int partitions = 2;
    args.AddOption(&partitions, "--partitions", "--partitions",
                   "Number of partitions to use at first coarse level.");
    upscale_param.RegisterInOptionsParser(args);
    // const int num_partitions = 2;
    args.Parse();
    // force three levels for simplicity
    upscale_param.max_levels = 3;
    // upscale_param.hybridization = true;
    if (myid == 0)
    {
        args.PrintOptions(std::cout);
    }

    LinearGraph graph(global_size);
    mfem::Array<int> global_partitioning(global_size);
    global_partitioning = 0;
    for (int p = 0; p < partitions; ++p)
    {
        for (int i = p * (global_size / partitions); i < (p + 1) * (global_size / partitions); ++i)
        {
            global_partitioning[i] = p;
        }
    }

    mfem::Vector weight(global_size - 1);
    weight = 1.0;

    std::cout << "Finished constructing Graph." << std::endl;

    Upscale upscale(graph.GetGraph(), upscale_param, &global_partitioning);
    std::cout << "Finished constructing Upscale." << std::endl;
    upscale.PrintInfo();

    upscale.GetHierarchy().DumpDebug("debug/");

    mfem::BlockVector fine_rhs(upscale.GetBlockVector(0));
    fine_rhs.GetBlock(0) = 0.0;

    // setup average zero right hand side (block 1).
    fine_rhs.GetBlock(1) = 1.0;
    for (int i = 0; i < fine_rhs.BlockSize(1) / 2; ++i)
    {
        fine_rhs.GetBlock(1)[i] = -1.0;
    }

    mfem::BlockVector sol0(fine_rhs);
    upscale.Solve(0, fine_rhs, sol0);
    upscale.ShowSolveInfo(0);

    mfem::BlockVector sol1(fine_rhs);
    upscale.Solve(1, fine_rhs, sol1);
    upscale.ShowSolveInfo(1);
    auto error_info_1 = upscale.ComputeErrors(sol1, sol0, 1);
    std::cout << "Level 1 errors:" << std::endl;
    std::cout << "  vertex error: " << error_info_1[0] << std::endl;
    std::cout << "  edge error: " << error_info_1[1] << std::endl;
    std::cout << "  div error: " << error_info_1[2] << std::endl;
    double h1 = 2.0 / ((double) global_size);
    double expected_error1 = 1.0 * h1;
    if (error_info_1[0] > expected_error1)
        result++;

    mfem::BlockVector sol2(fine_rhs);
    upscale.Solve(2, fine_rhs, sol2);
    upscale.ShowSolveInfo(2);
    auto error_info_2 = upscale.ComputeErrors(sol2, sol0, 2);
    std::cout << "Level 2 errors:" << std::endl;
    std::cout << "  vertex error: " << error_info_2[0] << std::endl;
    std::cout << "  edge error: " << error_info_2[1] << std::endl;
    std::cout << "  div error: " << error_info_2[2] << std::endl;
    double h2 = 8.0 / ((double) global_size);
    double expected_error2 = 1.0 * h2;
    if (error_info_2[0] > expected_error2)
        result++;

    const bool verbose = false;
    if (verbose)
    {
        for (int i = 0; i < sol0.GetBlock(1).Size(); ++i)
        {
            std::cout << i << ": "
                      << sol0.GetBlock(1)(i) << "   "
                      << sol1.GetBlock(1)(i) << "   "
                      << sol2.GetBlock(1)(i) << std::endl;
        }
    }

    return result;
}
