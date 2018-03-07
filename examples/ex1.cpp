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
using namespace mfem;

void MetisPart(const mfem::SparseMatrix& vertex_edge, mfem::Array<int>& part, int num_parts);
unique_ptr<mfem::HypreParMatrix> GraphLaplacian(const MixedMatrix& mixed_laplacian);
void Split(const mfem::SparseMatrix& A, mfem::SparseMatrix& vertex_edge, mfem::Vector& weight);

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

    //if (myid == 0)
    {
        printf("Mesh: %ld VE: %d %d A: %d\n", mesh->GetGlobalNE(), vertex_edge_global.Height(), vertex_edge_global.Width(),
                A.NumNonZeroElems());
    }

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
    mfem::Vector rhs_u_fine(vertex_edge.Width());
    B.GetSubVector(pgraph.GetVertexLocalToGlobalMap(), rhs_u_fine);
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

void Split(const mfem::SparseMatrix& A, mfem::SparseMatrix& vertex_edge, mfem::Vector& weight)
{
    int num_vertices = A.Height();
    int nnz_total = A.NumNonZeroElems();
    assert((nnz_total - num_vertices) % 2 == 0);
    int num_edges = (nnz_total - num_vertices) / 2;

    SparseMatrix ev(num_edges, num_vertices);
    weight.SetSize(num_edges);

    auto A_i = A.GetI();
    auto A_j = A.GetJ();
    auto A_a = A.GetData();

    int count = 0;

    for (int i = 0; i < num_vertices; ++i)
    {
        for (int j = A_i[i]; j < A_i[i + 1]; ++j)
        {
            int col = A_j[j];

            if (col > i)
            {
                weight[count] = -1.0 * A_a[j];
                ev.Add(count, i, 1.0);
                ev.Add(count, col, 1.0);
                count++;
            }
        }
    }

    ev.Finalize();

    assert(count == num_edges);

    SparseMatrix ve = smoothg::Transpose(ev);
    vertex_edge.Swap(ve);
}
