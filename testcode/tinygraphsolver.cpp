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
   Loads a tiny graph and solves the graph Laplacian problem.

   Does not do any upscaling.
*/

#include <fstream>

#include "mfem.hpp"

#include "../src/GraphCoarsen.hpp"
#include "../src/utilities.hpp"
#include "../src/MixedMatrix.hpp"
#include "../src/MinresBlockSolver.hpp"
#include "../src/MatrixUtilities.hpp"

using namespace smoothg;
using std::make_shared;

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
    const char* graphFileName = "../../graphdata/vertex_edge_tiny.txt";
    args.AddOption(&graphFileName, "-g", "--graph",
                   "Graph connection data.");

    bool weighted = false;
    args.AddOption(&weighted, "-w", "--weighted", "-no-w",
                   "--no-weighted", "Use weighted graph.");
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

    // load the graph
    mfem::SparseMatrix vertex_edge;
    {
        std::ifstream graphFile(graphFileName);
        ReadVertexEdge(graphFile, vertex_edge);
    }
    int nvertices = vertex_edge.Height();
    int nedges = vertex_edge.Width();

    mfem::Vector weight(nedges);
    if (weighted)
    {
        for (int i = 0; i < nedges; ++i)
        {
            weight[i] = i + 1;
        }
    }
    else
    {
        weight = 1.0;
    }

    // set the appropriate right hand side and weights for graph problem
    mfem::Vector rhs_u_fine;
    mfem::Vector rhs_p_fine;
    rhs_u_fine.SetSize(nedges);
    rhs_u_fine = 0.0;
    rhs_p_fine.SetSize(nvertices);
    rhs_p_fine = 1.0;

    // setup mixed problem
    auto edge_d_td_diag = SparseIdentity(nedges);
    mfem::Array<HYPRE_Int> edge_start(2);
    edge_start[0] = 0;
    edge_start[1] = nedges;
    auto edge_d_td = make_shared<mfem::HypreParMatrix>(comm, nedges, edge_start,
                                                       edge_d_td_diag.get());

    MixedMatrix mixed_graph_laplacian(vertex_edge, weight, edge_d_td);
    mfem::SparseMatrix& Mref = mixed_graph_laplacian.getWeight();
    mfem::SparseMatrix& Dref = mixed_graph_laplacian.getD();
    const int num_blocks = 2;
    mfem::Array<int> blockOffsets(num_blocks + 1);
    blockOffsets[0] = 0;
    blockOffsets[1] = Mref.Height();
    blockOffsets[2] = blockOffsets[1] + Dref.Height();
    mfem::BlockVector rhs(blockOffsets);
    rhs.GetBlock(0) = rhs_u_fine;
    rhs.GetBlock(1) = rhs_p_fine;
    mfem::BlockVector sol(blockOffsets);
    sol = 0.0;

    // solve
    HYPRE_Int row_starts_M[2];
    row_starts_M[0] = 0;
    row_starts_M[1] = Mref.Height();
    mfem::HypreParMatrix hM(comm, Mref.Height(), row_starts_M, &Mref);
    HYPRE_Int row_starts_D[2];
    row_starts_D[0] = 0;
    row_starts_D[1] = Dref.Height();
    HYPRE_Int col_starts_D[2];
    col_starts_D[0] = 0;
    col_starts_D[1] = Dref.Width();
    mfem::HypreParMatrix hD(comm, Dref.Height(), Dref.Width(),
                            row_starts_D, col_starts_D, &Dref);
    MinresBlockSolver mgp(comm, &hM, &hD, blockOffsets);
    mgp.Mult(rhs, sol);
    int iter = mgp.GetNumIterations();
    int nnz = mgp.GetNNZ();
    std::cout << "Global system has " << nnz << " nonzeros." << std::endl;
    std::cout << "Minres converged in " << iter << " iterations." << std::endl;
    orthogonalize_from_constant(sol.GetBlock(1));

    // truesol was found "independently" with python: testcode/tinygraph.py
    mfem::Vector truesol(nvertices);
    // [1.84483857 0.29938403 0.11756585  -0.63243415  -0.81935004  -0.81000425]
    if (weighted)
    {
        truesol[0] = 1.84483857264231e+00;
        truesol[1] = 2.99384027187765e-01;
        truesol[2] = 1.17565845369583e-01;
        truesol[3] = -6.32434154630417e-01;
        truesol[4] = -8.19350042480884e-01;
        truesol[5] = -8.10004248088361e-01;
    }
    else
        // [ 4.16666667  2.16666667  1.16666667 -1.83333333 -2.83333333 -2.83333333]
    {
        truesol[0] = 4.16666666666667e+00;
        truesol[1] = 2.16666666666667e+00;
        truesol[2] = 1.16666666666667e+00;
        truesol[3] = -1.83333333333333e+00;
        truesol[4] = -2.83333333333333e+00;
        truesol[5] = -2.83333333333333e+00;
    }

    sol.GetBlock(1).Print();

    truesol -= sol.GetBlock(1);
    double norm = truesol.Norml2();
    std::cout << "Error norm: " << norm << std::endl;

    if (norm < equality_tolerance)
        return 0;
    else
        return 1;
}
