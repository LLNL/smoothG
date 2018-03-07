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

#include "mfem.hpp"

#include "../src/picojson.h"
#include "../src/smoothG.hpp"

using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

using namespace smoothg;

void MetisPart(const mfem::SparseMatrix& vertex_edge, mfem::Array<int>& part, int num_parts);
unique_ptr<mfem::HypreParMatrix> GraphLaplacian(const MixedMatrix& mixed_laplacian);
mfem::Vector ComputeFiedlerVector(const MixedMatrix& mixed_laplacian);

int main(int argc, char* argv[])
{
    // 1. Initialize MPI
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    mfem::StopWatch chrono;

    // program options from command line
    mfem::OptionsParser args(argc, argv);
    int agg_size = 12;
    args.AddOption(&agg_size, "-as", "--agg-size",
                   "Number of vertices in an aggregated in hybridization.");
    const char* graphFileName = "../../graphdata/vertex_edge_sample.txt";
    args.AddOption(&graphFileName, "-g", "--graph",
                   "File to load for graph connection data.");
    const char* weight_filename = "";
    args.AddOption(&weight_filename, "-w", "--weight",
                   "File to load for graph edge weights.");
    const char* w_block_filename = "";
    args.AddOption(&w_block_filename, "-wb", "--w_block",
                   "File to load for w block.");
    bool generate_graph = false;
    args.AddOption(&generate_graph, "-gg", "--generate-graph", "-no-gg",
                   "--no-generate-graph", "Generate a graph at runtime.");
    int gen_vertices = 1000;
    args.AddOption(&gen_vertices, "-nv", "--num-vert",
                   "Number of vertices of the graph to be generated.");
    int mean_degree = 40;
    args.AddOption(&mean_degree, "-md", "--mean-degree",
                   "Average vertex degree of the graph to be generated.");
    double beta = 0.15;
    args.AddOption(&beta, "-b", "--beta",
                   "Probability of rewiring in the Watts-Strogatz model.");
    int seed = 0;
    args.AddOption(&seed, "-s", "--seed",
                   "Seed (unsigned integer) for the random number generator.");
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

    /// [Load graph from file or generate one]
    mfem::SparseMatrix vertex_edge_global;
    if (generate_graph)
    {
        mfem::SparseMatrix tmp = GenerateGraph(comm, gen_vertices, mean_degree, beta, seed);
        vertex_edge_global.Swap(tmp);
    }
    else
    {
        mfem::SparseMatrix tmp = ReadVertexEdge(graphFileName);
        vertex_edge_global.Swap(tmp);
    }
    /// [Load graph from file or generate one]

    /// [Load the edge weights]
    const int nedges_global = vertex_edge_global.Width();
    mfem::Vector weight(nedges_global);
    if (std::strlen(weight_filename))
    {
        std::ifstream weight_file(weight_filename);
        weight.Load(weight_file, nedges_global);
    }
    else
    {
        weight = 1.0;
    }
    /// [Load the edge weights]

    /// [Set up parallel graph and Laplacian]
    mfem::Array<int> global_partitioning;
    MetisPart(vertex_edge_global, global_partitioning, num_procs);
    smoothg::ParGraph pgraph(comm, vertex_edge_global, global_partitioning);
    auto& vertex_edge = pgraph.GetLocalVertexToEdge();
    const auto& edge_trueedge = pgraph.GetEdgeToTrueEdge();
    mfem::Vector local_weight(vertex_edge.Width());
    weight.GetSubVector(pgraph.GetEdgeLocalToGlobalMap(), local_weight);

    MixedMatrix mixed_laplacian(vertex_edge, local_weight, edge_trueedge);
    unique_ptr<mfem::HypreParMatrix> gL = GraphLaplacian(mixed_laplacian);
    /// [Set up parallel graph and Laplacian]

    /// [Right Hand Side]
    mfem::Vector rhs_u_fine = ComputeFiedlerVector(mixed_laplacian);
    mfem::BlockVector fine_rhs(mixed_laplacian.get_blockoffsets());
    fine_rhs.GetBlock(0) = 0.0;
    fine_rhs.GetBlock(1) = rhs_u_fine;
    fine_rhs *= -1.0;
    /// [Right Hand Side]

    /// [Solve primal problem by CG + BoomerAMG]
    mfem::Vector primal_sol(rhs_u_fine);
    {
        if (myid == 0)
        {
            std::cout << "\nSolving primal problem by CG + BoomerAMG ...\n";
        }

        chrono.Clear();
        chrono.Start();
        mfem::CGSolver cg(comm);
        cg.SetPrintLevel(0);
        cg.SetMaxIter(5000);
        cg.SetRelTol(1e-9);
        cg.SetAbsTol(1e-12);
        cg.SetOperator(*gL);

        mfem::Array<int> ess_dof(1);
        ess_dof = 0;
        gL->EliminateRowsCols(ess_dof);
        rhs_u_fine(0) = 0.0;

        mfem::HypreBoomerAMG prec(*gL);
        prec.SetPrintLevel(0);
        cg.SetPreconditioner(prec);
        if (myid == 0)
        {
            std::cout << "System size: " << gL->N() <<"\n";
            std::cout << "System NNZ: " << gL->NNZ() <<"\n";
            std::cout << "Setup time: " << chrono.RealTime() <<"s. \n";
        }

        chrono.Clear();
        chrono.Start();

        primal_sol = 0.0;
        cg.Mult(rhs_u_fine, primal_sol);
        par_orthogonalize_from_constant(primal_sol, vertex_edge_global.Height());
        if (myid == 0)
        {
            std::cout << "Solve time: " << chrono.RealTime() <<"s. \n";
            std::cout << "Number of iterations: " << cg.GetNumIterations() <<"\n";
        }
    }
    /// [Solve primal problem by CG + BoomerAMG]

    /// [Solve mixed problem by generalized hybridization]
    mfem::BlockVector mixed_sol(fine_rhs);
    {
        if (myid == 0)
        {
            std::cout << "\nSolving mixed problem by generalized hybridization ...\n";
        }

        chrono.Clear();
        chrono.Start();
        HybridSolver hb_solver(comm, mixed_laplacian, agg_size);
        if (myid == 0)
        {
            std::cout << "System size: " << hb_solver.GetHybridSystemSize() <<"\n";
            std::cout << "System NNZ: " << hb_solver.GetNNZ() <<"\n";
            std::cout << "Setup time: " << chrono.RealTime() <<"s. \n";
        }

        chrono.Clear();
        chrono.Start();
        mixed_sol = 0.0;
        hb_solver.Solve(fine_rhs, mixed_sol);
        par_orthogonalize_from_constant(mixed_sol.GetBlock(1), vertex_edge_global.Height());
        if (myid == 0)
        {
            std::cout << "Solve time: " << chrono.RealTime() <<"s. \n";
            std::cout << "Number of iterations: " << hb_solver.GetNumIterations() <<"\n\n";
        }
    }
    /// [Solve mixed problem by generalized hybridization]

    /// [Check solution difference]
    primal_sol -= mixed_sol.GetBlock(1);
    double diff = mfem::InnerProduct(comm, primal_sol, primal_sol);
    if (myid == 0)
    {
        std::cout << "|| primal_sol - mixed_sol || = " << std::sqrt(diff) <<" \n";
    }
    /// [Check solution difference]

    MPI_Finalize();
    return 0;
}

void MetisPart(const mfem::SparseMatrix& vertex_edge, mfem::Array<int>& part, int num_parts)
{
    smoothg::MetisGraphPartitioner partitioner;
    partitioner.setUnbalanceTol(2);
    mfem::SparseMatrix edge_vertex = smoothg::Transpose(vertex_edge);
    mfem::SparseMatrix vertex_vertex = smoothg::Mult(vertex_edge, edge_vertex);

    partitioner.doPartition(vertex_vertex, num_parts, part);
}

unique_ptr<mfem::HypreParMatrix> GraphLaplacian(const MixedMatrix& mixed_laplacian)
{
    auto& pM = mixed_laplacian.get_pM();
    auto& pD = mixed_laplacian.get_pD();
    auto* pW = mixed_laplacian.get_pW();

    unique_ptr<mfem::HypreParMatrix> MinvDT(pD.Transpose());

    mfem::HypreParVector M_inv(pM.GetComm(), pM.GetGlobalNumRows(), pM.GetRowStarts());
    pM.GetDiag(M_inv);
    MinvDT->InvScaleRows(M_inv);

    unique_ptr<mfem::HypreParMatrix> A(mfem::ParMult(&pD, MinvDT.get()));

    const bool use_w = mixed_laplacian.CheckW();

    if (use_w)
    {
        (*pW) *= -1.0;
        // TODO(gelever1): define ParSub lol
        A.reset(ParAdd(*A, *pW));
        (*pW) *= -1.0;
    }

    A->CopyRowStarts();
    A->CopyColStarts();

    return A;
}

mfem::Vector ComputeFiedlerVector(const MixedMatrix& mixed_laplacian)
{
    auto& pM = mixed_laplacian.get_pM();
    auto& pD = mixed_laplacian.get_pD();
    auto* pW = mixed_laplacian.get_pW();
    const bool use_w = mixed_laplacian.CheckW();

    unique_ptr<mfem::HypreParMatrix> MinvDT(pD.Transpose());

    mfem::HypreParVector M_inv(pM.GetComm(), pM.GetGlobalNumRows(), pM.GetRowStarts());
    pM.GetDiag(M_inv);
    MinvDT->InvScaleRows(M_inv);

    unique_ptr<mfem::HypreParMatrix> A(mfem::ParMult(&pD, MinvDT.get()));

    if (use_w)
    {
        (*pW) *= -1.0;
        // TODO(gelever1): define ParSub lol
        A.reset(ParAdd(*A, *pW));
        (*pW) *= -1.0;
    }
    else
    {
        // Adding identity to A so that it is non-singular
        mfem::SparseMatrix diag;
        A->GetDiag(diag);
        for (int i = 0; i < diag.Width(); i++)
            diag(i, i) += 1.0;
    }

    mfem::HypreBoomerAMG prec(*A);
    prec.SetPrintLevel(0);

    mfem::HypreLOBPCG lobpcg(A->GetComm());
    lobpcg.SetMaxIter(5000);
    lobpcg.SetTol(1e-8);
    lobpcg.SetPrintLevel(0);
    lobpcg.SetNumModes(2);
    lobpcg.SetOperator(*A);
    lobpcg.SetPreconditioner(prec);
    lobpcg.Solve();

    mfem::Array<double> evals;
    lobpcg.GetEigenvalues(evals);

    bool converged = true;

    // First eigenvalue of A+I should be 1 (graph Laplacian has a 1D null space)
    if (!use_w)
    {
        converged &= std::abs(evals[0] - 1.0) < 1e-8;
    }

    // Second eigenvalue of A+I should be greater than 1 for connected graphs
    converged &= std::abs(evals[1] - 1.0) > 1e-8;

    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (!converged && myid == 0)
    {
        std::cout << "LOBPCG Failed to converge: \n";
        std::cout << evals[0] << "\n";
        std::cout << evals[1] << "\n";
    }

    return lobpcg.GetEigenvector(1);
}
