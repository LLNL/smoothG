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

    int GetN() const {return n_;}

    const mfem::SparseMatrix& GetM() const {return M_;}
    const mfem::SparseMatrix& GetD() const {return D_;}
    const mfem::SparseMatrix& GetVertexEdge() const {return vertex_edge_;}

private:
    int n_;
    mfem::SparseMatrix M_;
    mfem::SparseMatrix D_;
    mfem::SparseMatrix vertex_edge_;
};

LinearGraph::LinearGraph(int n) :
    n_(n),
    M_(n - 1, n - 1),
    D_(n, n - 1),
    vertex_edge_(n, n - 1)
{
    for (int i = 0; i < n - 1; ++i)
    {
        M_.Add(i, i, 1.0);
        D_.Add(i, i, -1.0);
        D_.Add(i + 1, i, 1.0);
        vertex_edge_.Add(i, i, 1.0);
        vertex_edge_.Add(i + 1, i, 1.0);
    }
    M_.Finalize();
    D_.Finalize();
    vertex_edge_.Finalize();
}

int main(int argc, char* argv[])
{
    int num_procs, myid;
    int result = 0;

    // 1. Initialize MPI
    MPI_Init(&argc, &argv);
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
    // upscale_param.hybridization = true; // tempted to do this, may make more sense for three level
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

    GraphUpscale upscale(comm, graph.GetVertexEdge(), global_partitioning,
                         upscale_param, weight);
    std::cout << "Finished constructing GraphUpscale." << std::endl;
    upscale.PrintInfo();

    mfem::BlockVector fine_rhs(upscale.GetFineBlockVector());
    fine_rhs.GetBlock(0) = 0.0;
    fine_rhs.GetBlock(1) = 1.0; // ?

    mfem::BlockVector sol0(fine_rhs);
    upscale.Solve(0, fine_rhs, sol0);
    upscale.ShowSolveInfo(0);

    mfem::BlockVector sol1(fine_rhs);
    upscale.Solve(1, fine_rhs, sol1);
    upscale.ShowSolveInfo(1);

    mfem::BlockVector sol2(fine_rhs);
    upscale.Solve(2, fine_rhs, sol2);
    upscale.ShowSolveInfo(2);

    MPI_Finalize();
    return result;
}
