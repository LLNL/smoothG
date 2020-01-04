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
   Do what tinygraphsolver.cpp does, but in parallel.

   Loads a tiny graph and solves the graph Laplacian problem.

   Does not do any upscaling.
*/

#include "mfem.hpp"

#include "../src/GraphCoarsen.hpp"
#include "../src/utilities.hpp"
#include "../src/MinresBlockSolver.hpp"
#include "../src/MatrixUtilities.hpp"

using namespace smoothg;

/// @todo return some kind of smart pointer?
mfem::HypreParMatrix* build_tiny_graph()
{
    int num_procs, myid;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    MFEM_ASSERT(num_procs == 2, "This example for 2 processors!");
    MFEM_ASSERT(HYPRE_AssumedPartitionCheck(),
                "Only implemented for assumed partition!");

    int nrows = 3;
    HYPRE_Int glob_nrows = 6;
    HYPRE_Int glob_ncols = 7;
    int local_nnz = 7;

    HYPRE_Int* I = new HYPRE_Int[nrows + 1];
    HYPRE_Int* J = new HYPRE_Int[local_nnz];
    double* data = new double[local_nnz];
    HYPRE_Int* rows = new HYPRE_Int[2];
    HYPRE_Int* cols = new HYPRE_Int[2];
    if (myid == 0)
    {
        I[0] = 0;
        I[1] = 2;
        I[2] = 4;
        I[3] = 7;
        J[0] = 0;
        data[0] = 1.0;
        J[1] = 1;
        data[1] = 1.0;
        J[2] = 0;
        data[2] = -1.0;
        J[3] = 2;
        data[3] = 1.0;
        J[4] = 1;
        data[4] = -1.0;
        J[5] = 2;
        data[5] = -1.0;
        J[6] = 3;
        data[6] = 1.0;

        rows[0] = 0;
        rows[1] = 3;
        cols[0] = 0;
        cols[1] = 4;
    }
    else
    {
        MFEM_ASSERT(myid == 1, "Something is wrong with MPI ids!");

        I[0] = 0;
        I[1] = 3;
        I[2] = 5;
        I[3] = 7;
        J[0] = 3;
        data[0] = -1.0;
        J[1] = 4;
        data[1] = 1.0;
        J[2] = 5;
        data[2] = 1.0;
        J[3] = 4;
        data[3] = -1.0;
        J[4] = 6;
        data[4] = 1.0;
        J[5] = 5;
        data[5] = -1.0;
        J[6] = 6;
        data[6] = -1.0;

        rows[0] = 3;
        rows[1] = 6;
        cols[0] = 4;
        cols[1] = 7;
    }

    mfem::HypreParMatrix* out =  new mfem::HypreParMatrix(
        comm, nrows, glob_nrows, glob_ncols, I, J, data, rows, cols);

    delete [] I;
    delete [] J;
    delete [] data;
    delete [] rows;
    delete [] cols;

    return out;
}

mfem::SparseMatrix* build_tiny_w_block()
{
    int num_procs, myid;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    MFEM_ASSERT(num_procs == 2, "This example for 2 processors!");
    MFEM_ASSERT(HYPRE_AssumedPartitionCheck(),
                "Only implemented for assumed partition!");

    int nrows = 3;
    int local_nnz = 3;

    HYPRE_Int* I = new HYPRE_Int[nrows + 1];
    HYPRE_Int* J = new HYPRE_Int[local_nnz];
    double* data = new double[local_nnz];
    if (myid == 0)
    {
        I[0] = 0;
        I[1] = 1;
        I[2] = 2;
        I[3] = 3;

        J[0] = 0;
        data[0] = 1.0;
        J[1] = 1;
        data[1] = 2.0;
        J[2] = 2;
        data[2] = 3.0;
    }
    else
    {
        MFEM_ASSERT(myid == 1, "Something is wrong with MPI ids!");

        I[0] = 0;
        I[1] = 1;
        I[2] = 2;
        I[3] = 3;

        J[0] = 0;
        data[0] = 4.0;
        J[1] = 1;
        data[1] = 5.0;
        J[2] = 2;
        data[2] = 6.0;
    }

    return new mfem::SparseMatrix(I, J, data, nrows, nrows);
}

mfem::HypreParMatrix* build_tiny_graph_weights(bool weighted = false)
{
    int num_procs, myid;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    MFEM_ASSERT(num_procs == 2, "This example for 2 processors!");
    MFEM_ASSERT(HYPRE_AssumedPartitionCheck(),
                "Only implemented for assumed partition!");

    HYPRE_Int glob_nrows = 7;
    HYPRE_Int glob_ncols = 7;
    HYPRE_Int* rows = new HYPRE_Int[2];
    // HYPRE_Int *cols = new HYPRE_Int[2];
    int nrows, local_nnz;
    if (myid == 0)
    {
        nrows = 4;
        local_nnz = 4;
        rows[0] = 0;
        rows[1] = 4;
    }
    else
    {
        MFEM_ASSERT(myid == 1, "Something is wrong with MPI ids!");
        nrows = 3;
        local_nnz = 3;
        rows[0] = 4;
        rows[1] = 7;
    }
    HYPRE_Int* I = new HYPRE_Int[nrows + 1];
    HYPRE_Int* J = new HYPRE_Int[local_nnz];
    double* data = new double[local_nnz];
    I[0] = 0;
    for (int i = 0; i < nrows; ++i)
    {
        I[i + 1] = i + 1;
        J[i] = (myid * 4) + i;
        if (weighted)
        {
            data[i] = 1.0 / (1 + i + myid * (glob_nrows - nrows));
        }
        else
        {
            data[i] = 1.0;
        }
    }

    mfem::HypreParMatrix* out =  new mfem::HypreParMatrix(
        comm, nrows, glob_nrows, glob_ncols, I, J, data, rows, rows);

    delete [] I;
    delete [] J;
    delete [] data;
    delete [] rows;

    return out;
}

int main(int argc, char* argv[])
{
    // this is basically the Minres tolerance
    const double equality_tolerance = 1.e-9;

    // initialize MPI
    mpi_session session(argc, argv);

    int num_procs, myid;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // parse command line options
    mfem::OptionsParser args(argc, argv);
    bool weighted = false;
    args.AddOption(&weighted, "-w", "--weighted", "-no-w",
                   "--no-weighted", "Use weighted graph.");
    bool w_block = false;
    args.AddOption(&w_block, "-m", "--w_block", "-no-m",
                   "--no-w_block", "Use W block.");

    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(std::cout);
        }
        MPI_Finalize();
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(std::cout);
    }

    const int nvertices = 6;

    // generate the graph
    if (myid == 0)
        std::cout << "Building parallel graph..." << std::endl;
    mfem::HypreParMatrix* D = build_tiny_graph();
    mfem::HypreParMatrix* M = build_tiny_graph_weights(weighted);
    mfem::SparseMatrix* W = w_block ? build_tiny_w_block() : nullptr;

    // setup mixed problem
    if (myid == 0)
        std::cout << "Setting up mixed problem..." << std::endl;
    const int num_blocks = 2;
    mfem::Array<int> block_true_offsets(num_blocks + 1);
    block_true_offsets[0] = 0;
    block_true_offsets[1] = M->Height();
    block_true_offsets[2] = D->Height();
    block_true_offsets.PartialSum();

    // set the appropriate right hand side
    mfem::BlockVector rhs(block_true_offsets);
    rhs.GetBlock(0) = 0.0;
    rhs.GetBlock(1) = 1.0;

    // make rhs average zero so the problem is well defined when W block is zero
    if (!w_block && myid == 0)
        rhs.GetBlock(1)[0] = -5.0;

    // solve
    mfem::BlockVector sol(block_true_offsets);
    sol = 0.0;
    if (myid == 0)
        std::cout << "Solving graph problem..." << std::endl;
    BlockSolver mgp(M, D, W, block_true_offsets);
    mgp.Mult(rhs, sol);
    int iter = mgp.GetNumIterations();
    // int nnz = mgp.GetNNZ();
    // std::cout << "Global system has " << nnz << " nonzeros." << std::endl;
    if (myid == 0)
        std::cout << "Minres converged in " << iter << " iterations."
                  << std::endl;

    if (!w_block)
    {
        par_orthogonalize_from_constant(sol.GetBlock(1), D->M());
    }

    // truesol was found "independently" with python: testcode/tinygraph.py
    mfem::Vector global_truesol(nvertices);
    if (weighted && w_block)
    {
        global_truesol[0] = -5.46521374685666e-01;
        global_truesol[1] = -4.43419949706622e-01;
        global_truesol[2] = -3.71332774518022e-01;
        global_truesol[3] = -2.58172673931266e-01;
        global_truesol[4] = -2.23976847914538e-01;
        global_truesol[5] = -2.16677577841545e-01;
    }
    else if (weighted)
    {
        global_truesol[0] = 1.84483857264231e+00;
        global_truesol[1] = 2.99384027187765e-01;
        global_truesol[2] = 1.17565845369583e-01;
        global_truesol[3] = -6.32434154630417e-01;
        global_truesol[4] = -8.19350042480884e-01;
        global_truesol[5] = -8.10004248088361e-01;
    }
    else if (w_block)
    {
        global_truesol[0] = -6.36443964459280e-01;
        global_truesol[1] = -5.09155171567424e-01;
        global_truesol[2] = -4.00176721810417e-01;
        global_truesol[3] = -2.55461194835796e-01;
        global_truesol[4] = -2.05439104609494e-01;
        global_truesol[5] = -1.82612537430661e-01;
    }
    else
    {
        global_truesol[0] = 4.16666666666667e+00;
        global_truesol[1] = 2.16666666666667e+00;
        global_truesol[2] = 1.16666666666667e+00;
        global_truesol[3] = -1.83333333333333e+00;
        global_truesol[4] = -2.83333333333333e+00;
        global_truesol[5] = -2.83333333333333e+00;
    }

    mfem::Vector truesol(3);
    for (int i = 0; i < 3; ++i)
    {
        truesol[i] = global_truesol[i + (myid * 3)];
    }

    truesol -= sol.GetBlock(1);
    double norm = truesol.Norml2();
    if (myid == 0)
        std::cout << "Error norm: " << norm << std::endl;

    delete D;
    delete M;
    delete W;

    if (norm < equality_tolerance)
        return 0;
    else
        return 1;
}
