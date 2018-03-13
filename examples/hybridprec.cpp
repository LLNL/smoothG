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

class HybridPrec : public mfem::Solver
{
    public:
        HybridPrec(MPI_Comm comm, mfem::HypreParMatrix& gL, const ParGraph& pgraph, const mfem::Vector& weight);

        void Mult(const mfem::Vector& input, mfem::Vector& output) const;
    private:
        void SetOperator(const mfem::Operator&) {}

        mfem::HypreParMatrix& gL_;
        mfem::HypreSmoother smoother_;
        mfem::SparseMatrix P_vertex_;
        unique_ptr<HybridSolver> solver_;
        unique_ptr<MixedMatrix> mgl_;
        unique_ptr<GraphTopology> topo_;
        unique_ptr<mfem::HypreParMatrix> A_c_;

        mutable unique_ptr<mfem::BlockVector> tmp_coarse_;
        mutable unique_ptr<mfem::BlockVector> sol_coarse_;

        mutable mfem::Vector tmp_fine_;
};

HybridPrec::HybridPrec(MPI_Comm comm, mfem::HypreParMatrix& gL, const ParGraph& pgraph, const mfem::Vector& weight)
    : gL_(gL), smoother_(gL_)
{
    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    const auto& vertex_edge = pgraph.GetLocalVertexToEdge();
    const auto& edge_trueedge = pgraph.GetEdgeToTrueEdge();
    const auto& local_part = pgraph.GetLocalPartition();

    int num_parts = local_part.Max() + 1;

    auto agg_vertex = PartitionToMatrix(local_part, num_parts);

    mfem::Array<HYPRE_Int> agg_starts;
    mfem::Array<HYPRE_Int> vertex_starts;
    mfem::Array<HYPRE_Int> edge_starts;

    GenerateOffsets(comm, agg_vertex.Height(), agg_starts);
    GenerateOffsets(comm, agg_vertex.Width(), vertex_starts);
    GenerateOffsets(comm, vertex_edge.Width(), edge_starts);

    mfem::HypreParMatrix agg_vertex_d(comm, agg_starts.Last(), vertex_starts.Last(),
                                      agg_starts, vertex_starts, &agg_vertex);

    auto ve_copy = vertex_edge;
    mfem::HypreParMatrix ve_d(comm, vertex_starts.Last(), edge_starts.Last(),
                              vertex_starts, edge_starts, &ve_copy);

    auto ve = unique_ptr<mfem::HypreParMatrix>(ParMult(&ve_d, &edge_trueedge));
    auto ev = unique_ptr<mfem::HypreParMatrix>(ve->Transpose());
    auto vertex_vertex = unique_ptr<mfem::HypreParMatrix>(ParMult(ve.get(), ev.get()));
    auto agg_ext_vertex = unique_ptr<mfem::HypreParMatrix>(ParMult(&agg_vertex_d, vertex_vertex.get()));
    auto vertex_agg_ext = unique_ptr<mfem::HypreParMatrix>(agg_ext_vertex->Transpose());

    mfem::SparseMatrix vertex_agg_diag;
    mfem::SparseMatrix vertex_agg_offd;
    HYPRE_Int* junk;
    vertex_agg_ext->GetDiag(vertex_agg_diag);
    vertex_agg_ext->GetOffd(vertex_agg_offd, junk);

    int num_aggs = agg_vertex.Height();
    int num_vertices = vertex_agg_ext->Height();

    mfem::Array<int> vertex_marker(num_vertices);
    vertex_marker = 0;

    for (int i = 0; i < num_vertices; ++i)
    {
        if (vertex_agg_diag.RowSize(i) > 1 ||
            vertex_agg_offd.RowSize(i) > 0)
        {
            vertex_marker[i]++;
        }
    }

    mfem::SparseMatrix P_vertex(num_vertices);

    mfem::Array<int> vertices;
    int counter = 0;

    for (int i = 0; i < num_aggs; ++i)
    {
        int boundary = -1;

        GetTableRow(agg_vertex, i, vertices);

        for (auto vertex : vertices)
        {
            if (vertex_marker[vertex] > 0) // Boundary Vertex
            {
                if (boundary < 0)
                {
                    boundary = counter;
                    counter++;
                }

                P_vertex.Add(vertex, boundary, 1.0);
            }
            else                    // Internal Vertex
            {
                P_vertex.Add(vertex, counter, 1.0);
                counter++;
            }
        }
    }

    P_vertex.SetWidth(counter);
    P_vertex.Finalize();
    P_vertex_.Swap(P_vertex);

    mfem::SparseMatrix P_vertex_T = smoothg::Transpose(P_vertex_);
    mfem::SparseMatrix agg_cdof = smoothg::Mult(agg_vertex, P_vertex_);
    mfem::SparseMatrix cdof_agg = smoothg::Transpose(agg_cdof);
    mfem::SparseMatrix ve_c = smoothg::Mult(P_vertex_T, vertex_edge);


    mfem::Array<int> part_c(cdof_agg.GetJ(), cdof_agg.Height());

    mfem::Vector local_weight(vertex_edge.Width());
    weight.GetSubVector(pgraph.GetEdgeLocalToGlobalMap(), local_weight);

    mgl_ = make_unique<MixedMatrix>(ve_c, local_weight, edge_trueedge);
    topo_ = make_unique<GraphTopology>(ve_c, edge_trueedge, part_c);
    solver_ = make_unique<HybridSolver>(comm, *mgl_, *topo_);

    mfem::Array<int> coarse_starts;
    GenerateOffsets(comm, P_vertex_.Width(), coarse_starts);

    mfem::HypreParMatrix P_d(comm, vertex_starts.Last(), coarse_starts.Last(),
                             vertex_starts, coarse_starts, &P_vertex_);

    A_c_ = unique_ptr<mfem::HypreParMatrix>(mfem::RAP(&gL_, &P_d));

    if (myid == 0)
    {
        printf("Coarsen: %d -> %d\n", P_vertex_.Height(), P_vertex_.Width());
    }

    tmp_fine_.SetSize(vertex_edge.Height());

    tmp_coarse_ = make_unique<mfem::BlockVector>(mgl_->get_blockoffsets());
    sol_coarse_ = make_unique<mfem::BlockVector>(mgl_->get_blockoffsets());

    tmp_coarse_->GetBlock(0) = 0.0;
    sol_coarse_->GetBlock(0) = 0.0;
}

void HybridPrec::Mult(const mfem::Vector& b, mfem::Vector& x) const
{
    // PreSmooth
    x = 0.0;
    smoother_.Mult(b, x);

    // Residual
    gL_.Mult(x, tmp_fine_);
    tmp_fine_ *= -1.0;
    tmp_fine_ += b;

    // Restrict Residual
    P_vertex_.MultTranspose(tmp_fine_, tmp_coarse_->GetBlock(1));

    // Solve Coarse Level w/ HB solver
    solver_->Mult(*tmp_coarse_, *sol_coarse_);

    // Coarse Grid Correction
    P_vertex_.Mult(sol_coarse_->GetBlock(1), tmp_fine_);
    x += tmp_fine_;

    // Post
    smoother_.Mult(b, x);
}

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
    int parts_per_proc = 2;
    args.AddOption(&parts_per_proc, "-ppc", "--parts-per-proc",
                   "Parts per processor to partition hybrid solver.");
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
    MetisPart(vertex_edge_global, global_partitioning, num_procs * parts_per_proc);

    smoothg::ParGraph pgraph(comm, vertex_edge_global, global_partitioning);
    auto& vertex_edge = pgraph.GetLocalVertexToEdge();
    const auto& edge_trueedge = pgraph.GetEdgeToTrueEdge();
    mfem::Vector local_weight(vertex_edge.Width());
    weight.GetSubVector(pgraph.GetEdgeLocalToGlobalMap(), local_weight);

    MixedMatrix mixed_laplacian(vertex_edge, local_weight, edge_trueedge);
    unique_ptr<mfem::HypreParMatrix> gL_cg_ = GraphLaplacian(mixed_laplacian);
    unique_ptr<mfem::HypreParMatrix> gL_hybrid_ = GraphLaplacian(mixed_laplacian);
    /// [Set up parallel graph and Laplacian]

    /// [Right Hand Side]
    mfem::Vector rhs_u_fine = ComputeFiedlerVector(mixed_laplacian);
    mfem::BlockVector fine_rhs(mixed_laplacian.get_blockoffsets());
    fine_rhs.GetBlock(0) = 0.0;
    fine_rhs.GetBlock(1) = rhs_u_fine;
    fine_rhs *= -1.0;
    /// [Right Hand Side]

    /// [Solve by CG + Hybrid Preconditioner]
    mfem::Vector hybrid_prec_sol(rhs_u_fine);
    {
        if (myid == 0)
        {
            std::cout << "\nSolving primal problem by CG + Hybrid Preconditioner...\n";
        }

        chrono.Clear();
        chrono.Start();
        mfem::CGSolver cg(comm);

        HybridPrec prec(comm, *gL_hybrid_, pgraph, weight);

        cg.SetPreconditioner(prec);
        cg.SetPrintLevel(1);
        cg.SetMaxIter(5000);
        cg.SetRelTol(1e-9);
        cg.SetAbsTol(1e-12);


        mfem::Array<int> ess_dof;
        if (myid == 0)
        {
            rhs_u_fine(0) = 0.0;
            ess_dof.Append(0);
        }
        auto raw_memory = gL_hybrid_->EliminateRowsCols(ess_dof);
        delete raw_memory;

        cg.SetOperator(*gL_hybrid_);
        if (myid == 0)
        {
            std::cout << "System size: " << gL_hybrid_->N() <<"\n";
            std::cout << "System NNZ: " << gL_hybrid_->NNZ() <<"\n";
            std::cout << "Setup time: " << chrono.RealTime() <<"s. \n";
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
    /// [Solve by CG + Hybrid Preconditioner]

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

        mfem::HypreBoomerAMG prec(*gL_cg_);
        prec.SetPrintLevel(0);

        cg.SetPreconditioner(prec);
        cg.SetPrintLevel(0);
        cg.SetMaxIter(5000);
        cg.SetRelTol(1e-9);
        cg.SetAbsTol(1e-12);


        mfem::Array<int> ess_dof;
        if (myid == 0)
        {
            rhs_u_fine(0) = 0.0;
            ess_dof.Append(0);
        }
        auto raw_memory = gL_cg_->EliminateRowsCols(ess_dof);
        delete raw_memory;

        cg.SetOperator(*gL_cg_);
        if (myid == 0)
        {
            std::cout << "System size: " << gL_cg_->N() <<"\n";
            std::cout << "System NNZ: " << gL_cg_->NNZ() <<"\n";
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
        std::cout << "|| primal_sol - mixed_sol || = " << error <<" \n";
        for (int i = 0; i < 10; ++i)
        {
            printf("%d: %.8f %.8f\n", i, primal_sol[i], hybrid_prec_sol[i]);
        }
    }
    /// [Check solution difference]

    MPI_Finalize();
    return 0;
}

void MetisPart(const mfem::SparseMatrix& vertex_edge, mfem::Array<int>& part, int num_parts)
{
    smoothg::MetisGraphPartitioner partitioner;
    partitioner.setUnbalanceTol(1.001);
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
