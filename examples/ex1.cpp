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
#include "hybridprecond.hpp"

using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

using namespace smoothg;
using namespace mfem;

void MetisPart(const mfem::SparseMatrix& vertex_edge, mfem::Array<int>& part, int num_parts);
void MetisPartCF(const mfem::SparseMatrix& vertex_edge, mfem::Array<int>& part, int coarse_factor);
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
    int parts_per_proc = 2;
    args.AddOption(&parts_per_proc, "-ppc", "--parts-per-proc",
            "Parts per processor to partition hybrid solver.");


    // MFEM Options
    const char* mesh_file = "star.mesh";
    args.AddOption(&mesh_file, "-mesh", "--mesh",
            "Mesh file to use.");
    int order = 1;
    args.AddOption(&order, "-o", "--order",
            "Finite element order (polynomial degree) or -1 for"
            " isoparametric space.");
    int ref_levels = 1;
    args.AddOption(&ref_levels, "-nr", "--num-refine",
            "Number of times to refine mesh.");
    bool static_cond = false;
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
            "--no-static-condensation", "Enable static condensation.");
    bool visualization = 1;
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
            "--no-visualization",
            "Enable or disable GLVis visualization.");

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

    Mesh* mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    for (int l = 0; l < ref_levels; l++)
    {
        mesh->UniformRefinement();
    }

    FiniteElementCollection* fec;
    if (order > 0)
    {
        fec = new H1_FECollection(order, dim);
    }
    else if (mesh->GetNodes())
    {
        fec = mesh->GetNodes()->OwnFEC();
    }
    else
    {
        fec = new H1_FECollection(order = 1, dim);
    }
    FiniteElementSpace* fespace = new FiniteElementSpace(mesh, fec);

    Array<int> ess_tdof_list;
    if (mesh->bdr_attributes.Size())
    {
        Array<int> ess_bdr(mesh->bdr_attributes.Max());
        ess_bdr = 0;
        fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    LinearForm* b = new LinearForm(fespace);
    ConstantCoefficient one(1.0);
    b->AddDomainIntegrator(new DomainLFIntegrator(one));
    b->Assemble();

    GridFunction x(fespace);
    x = 0.0;

    BilinearForm* a = new BilinearForm(fespace);
    a->AddDomainIntegrator(new DiffusionIntegrator(one));

    if (static_cond) { a->EnableStaticCondensation(); }
    a->Assemble();

    SparseMatrix A;
    Vector B, X;
    a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);


    /// [Load graph from file or generate one]
    mfem::Vector weight;
    mfem::SparseMatrix vertex_edge_global;

    Split(A, vertex_edge_global, weight);

    int negative_count = 0;

    for (int i = 0; i < weight.Size(); ++i)
    {
        if (weight[i] < 0)
        {
            negative_count++;
        }
    }

    if (myid == 0)
    {
        printf("Mesh: %ld VE: %d %d A: %d\n", mesh->GetGlobalNE(), vertex_edge_global.Height(), vertex_edge_global.Width(),
                A.NumNonZeroElems());
        printf("NegativeWeights: %d / %d\n", negative_count, weight.Size());
    }

    /// [Set up parallel graph and Laplacian]
    mfem::Array<int> global_partitioning;
    //MetisPart(vertex_edge_global, global_partitioning, num_procs * parts_per_proc);
    MetisPartCF(vertex_edge_global, global_partitioning, agg_size);
    assert(global_partitioning.Max() + 1 >= num_procs);

    smoothg::ParGraph pgraph(comm, vertex_edge_global, global_partitioning);
    auto& vertex_edge = pgraph.GetLocalVertexToEdge();
    const auto& edge_trueedge = pgraph.GetEdgeToTrueEdge();

    mfem::Vector local_weight(vertex_edge.Width());
    weight.GetSubVector(pgraph.GetEdgeLocalToGlobalMap(), local_weight);

    MixedMatrix mixed_laplacian(vertex_edge, local_weight, edge_trueedge);
    unique_ptr<mfem::HypreParMatrix> gL = GraphLaplacian(mixed_laplacian);
    unique_ptr<mfem::HypreParMatrix> gL_unelim = GraphLaplacian(mixed_laplacian);
    unique_ptr<mfem::HypreParMatrix> gL_hybrid = GraphLaplacian(mixed_laplacian);

    /// [Set up parallel graph and Laplacian]

    /// [Right Hand Side]
    mfem::Vector rhs_u_fine(vertex_edge.Width());
    B.GetSubVector(pgraph.GetVertexLocalToGlobalMap(), rhs_u_fine);
    //mfem::Vector rhs_u_fine = ComputeFiedlerVector(mixed_laplacian);

    if (myid == 0)
    {
        rhs_u_fine(0) -= B.Sum();
    }

    mfem::BlockVector fine_rhs(mixed_laplacian.get_blockoffsets());
    fine_rhs.GetBlock(0) = 0.0;
    fine_rhs.GetBlock(1) = rhs_u_fine;
    fine_rhs *= -1.0;
    /// [Right Hand Side]

    /// [Solve mixed problem by generalized hybridization]
    /*
       mfem::BlockVector mixed_sol(fine_rhs);
       {
       if (myid == 0)
       {
       std::cout << "\nSolving mixed problem by generalized hybridization ...\n";
       }

       chrono.Clear();
       chrono.Start();
    //HybridSolver hb_solver(comm, mixed_laplacian, agg_size);
    HybridSolver hb_solver(comm, mixed_laplacian, 1);
    if (myid == 0)
    {
    std::cout << "System size: " << hb_solver.GetHybridSystemSize() <<"\n";
    std::cout << "System NNZ: " << hb_solver.GetNNZ() <<"\n";
    std::cout << "Setup time: " << chrono.RealTime() <<"s. \n";
    }

    hb_solver.SetPrintLevel(0);
    hb_solver.SetMaxIter(5000);
    hb_solver.SetRelTol(1e-12);
    hb_solver.SetAbsTol(1e-12);

    if (myid == 0)
    {
    fine_rhs.GetBlock(1)[0] += B.Sum();
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
    */
    /// [Solve mixed problem by generalized hybridization]

    /// [Solve mixed problem by hybridization preconditioner]
    mfem::Vector hybrid_prec_sol(rhs_u_fine);
    {
        if (myid == 0)
        {
            std::cout << "\nSolving mixed problem by hybridization preconditioner ...\n";
        }

        chrono.Clear();
        chrono.Start();
        mfem::CGSolver cg(comm);
        HybridPrec prec(comm, *gL_hybrid, vertex_edge_global, weight, global_partitioning);


        cg.SetPrintLevel(0);
        cg.SetMaxIter(5000);
        cg.SetRelTol(1e-9);
        cg.SetAbsTol(1e-12);
        cg.iterative_mode = false;

        mfem::Array<int> ess_dof;
        if (myid == 0)
        {
            ess_dof.Append(0);
        }

        auto raw_memory = gL_hybrid->EliminateRowsCols(ess_dof);
        delete raw_memory;

        cg.SetPreconditioner(prec);
        cg.SetOperator(*gL_hybrid);


        if (myid == 0)
        {
            std::cout << "System size: " << gL_hybrid->N() <<"\n";
            std::cout << "System NNZ: " << gL_hybrid->NNZ() <<"\n";
            std::cout << "Setup time: " << chrono.RealTime() <<"s. \n";
            printf("Num Parts: %d\n", global_partitioning.Max() + 1);
        }

        chrono.Clear();
        chrono.Start();

        hybrid_prec_sol = 0.0;
        cg.Mult(rhs_u_fine, hybrid_prec_sol);
        par_orthogonalize_from_constant(hybrid_prec_sol, vertex_edge_global.Height());
        if (myid == 0)
        {
            std::cout << "Solve time: " << chrono.RealTime() <<"s. \n";
            std::cout << "Number of iterations: " << cg.GetNumIterations() <<"\n";
        }
    }
    /// [Solve mixed problem by hybridization preconditioner]

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
        cg.SetRelTol(1e-12);
        cg.SetAbsTol(1e-12);
        cg.SetOperator(*gL);


        mfem::Array<int> ess_dof;
        if (myid == 0)
        {
            ess_dof.Append(0);
        }

        auto raw_memory = gL->EliminateRowsCols(ess_dof);
        delete raw_memory;


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

    /// [Check solution difference]

    mfem::Vector diff = primal_sol;
    diff -= hybrid_prec_sol;
    double error = mfem::ParNormlp(diff, 2, comm) / mfem::ParNormlp(primal_sol, 2, comm);
    if (myid == 0)
    {
        std::cout << "|| primal_sol - hybrid_prec_sol || = " << error <<" \n";
    }

    /*
       diff = primal_sol;
       diff -= mixed_sol.GetBlock(1);
       error = mfem::ParNormlp(diff, 2, comm) / mfem::ParNormlp(primal_sol, 2, comm);
       if (myid == 0)
       {
       std::cout << "|| primal_sol - mixed_sol || = " << error <<" \n";
       }
       */
    /// [Check solution difference]

    MPI_Finalize();

    delete b;
    delete a;
    delete fec;
    delete fespace;
    delete mesh;

    return 0;
}


void MetisPartCF(const mfem::SparseMatrix& vertex_edge, mfem::Array<int>& part, int coarse_factor)
{
    smoothg::MetisGraphPartitioner partitioner;
    partitioner.setUnbalanceTol(1.001);
    mfem::SparseMatrix edge_vertex = smoothg::Transpose(vertex_edge);
    mfem::SparseMatrix vertex_vertex = smoothg::Mult(vertex_edge, edge_vertex);

    int num_parts = std::max(1, (int)((vertex_vertex.Height() / (double)coarse_factor) + 0.5));

    partitioner.doPartition(vertex_vertex, num_parts, part);
}

void MetisPart(const mfem::SparseMatrix& vertex_edge, mfem::Array<int>& part, int num_parts)
{
    smoothg::MetisGraphPartitioner partitioner;
    partitioner.setUnbalanceTol(1.001);
    mfem::SparseMatrix edge_vertex = smoothg::Transpose(vertex_edge);
    mfem::SparseMatrix vertex_vertex = smoothg::Mult(vertex_edge, edge_vertex);

    int total_parts = num_parts;
    partitioner.doPartition(vertex_vertex, total_parts, part);

    assert(total_parts == num_parts);
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
