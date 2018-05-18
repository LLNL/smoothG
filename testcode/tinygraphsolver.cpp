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
#include <sstream>
#include <mpi.h>

#include "smoothG.hpp"

using namespace smoothg;
using linalgcpp::ReadCSR;

int main(int argc, char* argv[])
{
    // Initialize MPI
    MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm_;
    int myid = mpi_info.myid_;
    int num_procs = mpi_info.num_procs_;

    assert(num_procs == 1 || num_procs == 2);

    // parse command line options
    std::string graph_filename = "../../graphdata/vertex_edge_tiny.txt";
    bool weighted = false;
    bool w_block = false;

    linalgcpp::ArgParser arg_parser(argc, argv);

    arg_parser.Parse(graph_filename, "--g", "Graph connection data.");
    arg_parser.Parse(weighted, "--wg", "Use weighted graph.");
    arg_parser.Parse(w_block, "--wb", "Use w block.");

    if (!arg_parser.IsGood())
    {
        ParPrint(myid, arg_parser.ShowHelp());
        ParPrint(myid, arg_parser.ShowErrors());

        return EXIT_FAILURE;
    }

    ParPrint(myid, arg_parser.ShowOptions());

    // load the graph
    SparseMatrix vertex_edge = ReadCSR(graph_filename);

    const int nvertices = vertex_edge.Rows();
    const int nedges = vertex_edge.Cols();

    assert(nvertices == 6 && nedges == 7);

    std::vector<double> weight(nedges, 1.0);
    if (weighted)
    {
        for (int i = 0; i < nedges; ++i)
        {
            weight[i] = i + 1;
        }
    }

    CooMatrix W_coo(nvertices, nvertices);

    if (w_block)
    {
        for (int i = 0; i < nvertices; ++i)
        {
            W_coo.Add(i, i, i + 1);
        }
    }
    
    SparseMatrix W_block = W_coo.ToSparse();

    std::vector<int> partition {0, 0, 0, 1, 1, 1};

    Graph graph(comm, vertex_edge, partition, weight, W_block);
    MixedMatrix mgl(graph);
    mgl.AssembleM();

    BlockVector sol(mgl.Offsets());
    BlockVector rhs(mgl.Offsets());
    rhs.GetBlock(0) = 0.0;
    rhs.GetBlock(1) = 1.0;

    if (!w_block && myid == 0)
    {
        rhs.GetBlock(1)[0] = -5.0;
    }

    // truesol was found "independently" with python: testcode/tinygraph.py
    Vector global_truesol(nvertices);
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

    int local_size = 6 / num_procs;
    Vector local_truesol(local_size);

    for (int i = 0; i < local_size; ++i)
    {
        local_truesol[i] = global_truesol[i + (myid * 3)];
    }

    MinresBlockSolver minres(mgl);
    HybridSolver hb(mgl);

    std::map<MGLSolver*, std::string> solver_to_name;
    solver_to_name[&minres] = "Minres + block preconditioner";
    solver_to_name[&hb] = "Hybridizaiton + BoomerAMG";

    const double equality_tolerance = 1.e-9;
    bool some_solver_fails = false;

    for (auto& solver_pair : solver_to_name)
    {
        auto& solver = solver_pair.first;
        auto& name = solver_pair.second;

        sol = 0.0;
        solver->SetPrintLevel(1);
        solver->Solve(rhs, sol);

        if (!w_block)
        {
            OrthoConstant(comm, sol.GetBlock(1), nvertices);
        }

        double error = CompareError(comm, sol.GetBlock(1), local_truesol);

        if (error > equality_tolerance)
        {
            some_solver_fails = true;
        }

        if (myid == 0)
        {
            int iter = solver->GetNumIterations();
            int nnz = solver->GetNNZ();

            std::cout << "\n" << name << " Solver:\n";
            std::cout << "Global system has " << nnz << " nonzeros.\n";
            std::cout << name << " converged in " << iter << " iterations.\n";
            std::cout << "Error norm: " << error << "\n";
            sol.GetBlock(1).Print("Sol:");
        }
    }

    return some_solver_fails;
}
