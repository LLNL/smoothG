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
#include "../src/HybridSolver.hpp"
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

    // load the graph
    mfem::SparseMatrix vertex_edge;
    {
        std::ifstream graphFile(graphFileName);
        ReadVertexEdge(graphFile, vertex_edge);
    }
    const int nvertices = vertex_edge.Height();
    const int nedges = vertex_edge.Width();

    assert(nvertices == 6 && nedges == 7);

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

    mfem::Vector w(nvertices);
    if (w_block)
    {
        for (int i = 0; i < nvertices; ++i)
        {
            w[i] = i + 1;
        }
    }
    else
    {
        w = 0.0;
    }

    // set the appropriate right hand side and weights for graph problem
    mfem::Vector rhs_u_fine;
    mfem::Vector rhs_p_fine;
    rhs_u_fine.SetSize(nedges);
    rhs_u_fine = 0.0;
    rhs_p_fine.SetSize(nvertices);
    rhs_p_fine = 1.0;

    // make rhs average zero so the problem is well defined when W block is zero
    if (!w_block && myid == 0)
        rhs_p_fine(0) = -5.0;

    // setup mixed problem
    auto edge_d_td_diag = SparseIdentity(nedges);
    mfem::Array<HYPRE_Int> edge_start(2);
    edge_start[0] = 0;
    edge_start[1] = nedges;

    mfem::HypreParMatrix edge_d_td(comm, nedges, edge_start,
                                   &edge_d_td_diag);

    MixedMatrix mixed_graph_laplacian(vertex_edge, weight, w, edge_d_td);

    mfem::Array<int>& blockOffsets(mixed_graph_laplacian.get_blockoffsets());
    mfem::BlockVector rhs = *mixed_graph_laplacian.subvecs_to_blockvector(rhs_u_fine, rhs_p_fine);
    mfem::BlockVector sol(blockOffsets);

    // setup solvers
    std::map<MixedLaplacianSolver*, std::string> solver_to_name;

    MinresBlockSolver minres(comm, mixed_graph_laplacian);
    solver_to_name[&minres] = "Minres + block preconditioner";

    HybridSolver hb_bamg(comm, mixed_graph_laplacian);
    solver_to_name[&hb_bamg] = "Hybridization + BoomerAMG";

#if SMOOTHG_USE_SAAMGE
    SAAMGeParam sa_param;
    HybridSolver hb_saamge(comm, mixed_graph_laplacian, nullptr, nullptr, 0, &sa_param);
    solver_to_name[&hb_saamge] = "Hybridization + SA-AMGe";
#endif

    // truesol was found "independently" with python: testcode/tinygraph.py
    mfem::Vector truesol(nvertices);
    if (weighted && w_block)
    {
        truesol[0] = -5.46521374685666e-01;
        truesol[1] = -4.43419949706622e-01;
        truesol[2] = -3.71332774518022e-01;
        truesol[3] = -2.58172673931266e-01;
        truesol[4] = -2.23976847914538e-01;
        truesol[5] = -2.16677577841545e-01;
    }
    else if (weighted)
    {
        truesol[0] = 1.84483857264231e+00;
        truesol[1] = 2.99384027187765e-01;
        truesol[2] = 1.17565845369583e-01;
        truesol[3] = -6.32434154630417e-01;
        truesol[4] = -8.19350042480884e-01;
        truesol[5] = -8.10004248088361e-01;
    }
    else if (w_block)
    {
        truesol[0] = -6.36443964459280e-01;
        truesol[1] = -5.09155171567424e-01;
        truesol[2] = -4.00176721810417e-01;
        truesol[3] = -2.55461194835796e-01;
        truesol[4] = -2.05439104609494e-01;
        truesol[5] = -1.82612537430661e-01;
    }
    else
    {
        truesol[0] = 4.16666666666667e+00;
        truesol[1] = 2.16666666666667e+00;
        truesol[2] = 1.16666666666667e+00;
        truesol[3] = -1.83333333333333e+00;
        truesol[4] = -2.83333333333333e+00;
        truesol[5] = -2.83333333333333e+00;
    }

    bool some_solver_fails = false;
    for (auto& solver_pair : solver_to_name)
    {
        auto& solver = solver_pair.first;
        auto& name = solver_pair.second;

        std::cout << "\nSolving mixed graph Laplacian problem with "
                  << name << " ..." << std::endl;

        // solve
        sol = 0.0;
        solver->SetPrintLevel(1);
        solver->Solve(rhs, sol);

        int iter = solver->GetNumIterations();
        int nnz = solver->GetNNZ();
        std::cout << "Global system has " << nnz << " nonzeros." << std::endl;
        std::cout << name << " converged in " << iter << " iterations." << std::endl;

        if (!w_block)
        {
            orthogonalize_from_constant(sol.GetBlock(1));
        }

        std::cout.precision(16);
        sol.GetBlock(1).Print(std::cout, 1);

        sol.GetBlock(1) -= truesol;
        double norm = sol.GetBlock(1).Norml2();
        std::cout << "Error norm: " << norm << std::endl;

        if (norm > equality_tolerance)
        {
            some_solver_fails = true;
        }
    }

    if (some_solver_fails)
        return 1;
    else
        return 0;
}
