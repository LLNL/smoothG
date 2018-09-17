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

private:
    int n_;
    mfem::SparseMatrix M_;
    mfem::SparseMatrix D_;
};

LinearGraph::LinearGraph(int n) :
    n_(n),
    M_(n - 1, n - 1),
    D_(n, n - 1)
{
    for (int i = 0; i < n - 1; ++i)
    {
        M_.Add(i, i, 1.0);
        D_.Add(i, i, -1.0);
        D_.Add(i + 1, i, 1.0);
    }
    M_.Finalize();
    D_.Finalize();
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
    upscale_param.RegisterInOptionsParser(args);
    // const int num_partitions = 2;
    args.Parse();
    // force for simplicity
    upscale_param.max_levels = 3;
    if (myid == 0)
    {
        args.PrintOptions(std::cout);
    }

    LinearGraph graph(global_size);
    mfem::Array<int> global_partitioning(global_size);
    global_partitioning = 1;
    for (int i = 0; i < global_size / 2; ++i)
    {
        global_partitioning[i] = 0;
    }
    mfem::Vector weight(global_size - 1);
    weight = 1.0;
    
    GraphUpscale upscale(comm, graph.GetD(), global_partitioning,
                         upscale_param, weight);

    MPI_Finalize();
    return result;
}