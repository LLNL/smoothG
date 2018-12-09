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
mfem::Vector ComputeFiedlerVector(const MixedMatrix& mixed_laplacian);
void Split(const mfem::SparseMatrix& A, mfem::SparseMatrix& vertex_edge, mfem::Vector& weight);
double fFun(const mfem::Vector& x)
{
    return x(0) + x (1) + 1.0;
}

class Multigrid : public mfem::Solver
{
public:
    Multigrid(mfem::HypreParMatrix& Operator, const mfem::Operator& CoarseSolver)
        : mfem::Solver(Operator.Height()), Operator_(Operator), Smoother_(Operator),
          CoarseSolver_(CoarseSolver), residual_(height), help_vec_(height)
    {}

    virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const;
    virtual void SetOperator(const mfem::Operator& op) {}
    ~Multigrid() {}
private:
    void MG_Cycle() const;

    const mfem::HypreParMatrix& Operator_;
    const mfem::HypreSmoother Smoother_;
    const Operator& CoarseSolver_;

    mutable mfem::Vector correction_;
    mutable mfem::Vector residual_;
    mutable mfem::Vector help_vec_;
};

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
    int parts_per_proc = 1;
    args.AddOption(&parts_per_proc, "-ppc", "--parts-per-proc",
                   "Parts per processor to partition hybrid solver.");

    // MFEM Options
    const char* mesh_file = "/Users/lee1029/Codes/mfem-3.2/data/star.mesh";
    args.AddOption(&mesh_file, "-mesh", "--mesh", "Mesh file to use.");
    int order = 1;
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) or -1 for"
                   " isoparametric space.");
    int ref_levels = 1;
    args.AddOption(&ref_levels, "-nr", "--num-refine",
                   "Number of times to refine mesh.");
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
    FunctionCoefficient fcoef(fFun);
    b->AddDomainIntegrator(new DomainLFIntegrator(fcoef));
    b->Assemble();

    GridFunction x(fespace);
    x = 0.0;

    BilinearForm* a = new BilinearForm(fespace);
    a->AddDomainIntegrator(new DiffusionIntegrator);
    a->Assemble();

    SparseMatrix A;
    Vector B, X;
    a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

    B -= (B.Sum() / B.Size());

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
        printf("Mesh: %ld VE: %d %d A: %d\n", mesh->GetGlobalNE(), vertex_edge_global.Height(),
               vertex_edge_global.Width(),
               A.NumNonZeroElems());
        printf("NegativeWeights: %d / %d\n", negative_count, weight.Size());
    }

    /// [Set up parallel graph and Laplacian]
    mfem::Array<int> global_partitioning;
    MetisPart(vertex_edge_global, global_partitioning, num_procs * parts_per_proc);

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
    //par_orthogonalize_from_constant(rhs_u_fine, vertex_edge_global.Height());
    //mfem::Vector rhs_u_fine = ComputeFiedlerVector(mixed_laplacian);
    /// [Right Hand Side]

    /// [Solve mixed problem by generalized hybridization]
    mfem::Vector mixed_sol(rhs_u_fine);
    if (false)
    {
        if (myid == 0)
        {
            std::cout << "\nSolving mixed problem by generalized hybridization ...\n";
        }

        chrono.Clear();
        chrono.Start();
        HybridSolver hb_solver(comm, mixed_laplacian, agg_size, nullptr, nullptr, 30);
        if (myid == 0)
        {
            std::cout << "System size: " << hb_solver.GetHybridSystemSize() << "\n";
            std::cout << "System NNZ: " << hb_solver.GetNNZ() << "\n";
            std::cout << "Setup time: " << chrono.RealTime() << "s. \n";
        }

        hb_solver.SetPrintLevel(1);
        hb_solver.SetMaxIter(5000);
        hb_solver.SetRelTol(1e-9);
        hb_solver.SetAbsTol(1e-12);

        chrono.Clear();
        chrono.Start();
        mixed_sol = 0.0;
        hb_solver.Mult(rhs_u_fine, mixed_sol);
        par_orthogonalize_from_constant(mixed_sol, gL->N());
        if (myid == 0)
        {
            std::cout << "Solve time: " << chrono.RealTime() << "s. \n";
            std::cout << "Number of iterations: " << hb_solver.GetNumIterations() << "\n";
        }
    }
    /// [Solve mixed problem by generalized hybridization]

    /// [Solve mixed problem by hybridization preconditioner (face coarsening approach)]
    mfem::Vector hybrid_prec_sol(rhs_u_fine);//if (false)
    {
        if (myid == 0)
        {
            std::cout << "\nSolving mixed problem by hybridization preconditioner ...\n";
        }

        chrono.Clear();
        chrono.Start();

        HybridSolver hb_solver(comm, mixed_laplacian, agg_size,
                               nullptr, nullptr, 30, nullptr, true);
        //        hb_solver.SetMaxIter(4);
        hb_solver.SetPrintLevel(-1);

        //        mfem::Array<int> partition;
        //        PartitionAAT(vertex_edge, partition, agg_size);
        //        mfem::SparseMatrix bdr(vertex_edge.Width(), 0);
        //        bdr.Finalize();
        //        mfem::Array<int> ess_edof(vertex_edge.Width());
        //        ess_edof = 0;
        //        FiniteVolumeUpscale fvup(comm, vertex_edge, local_weight, partition,
        //                                 edge_trueedge, bdr, ess_edof, 1.0, 1, 0, 0, 0, 1);

        mfem::CGSolver cg(comm);
        cg.SetPrintLevel(0);
        cg.SetMaxIter(500);
        cg.SetRelTol(1e-9);
        cg.SetAbsTol(1e-12);
        cg.SetOperator(*gL);

        Multigrid prec(*gL, hb_solver);
        cg.SetPreconditioner(prec);

        if (myid == 0)
        {
            std::cout << "Coarse System size: " << hb_solver.GetHybridSystemSize() << "\n";
            std::cout << "Coarse System NNZ: " << hb_solver.GetNNZ() << "\n";
            std::cout << "Setup time: " << chrono.RealTime() << "s. \n";
        }

        chrono.Clear();
        chrono.Start();
        hybrid_prec_sol = 0.0;
        cg.Mult(rhs_u_fine, hybrid_prec_sol);
        par_orthogonalize_from_constant(hybrid_prec_sol, gL->N());
        if (myid == 0)
        {
            std::cout << "Solve time: " << chrono.RealTime() << "s. \n";
            std::cout << "Number of iterations: " << cg.GetNumIterations() << "\n";
        }
    }
    /// [Solve mixed problem by hybridization preconditioner]

    mfem::Array<int> ess_dof;
    if (myid == 0)
    {
        rhs_u_fine(0) = 0.0;
        ess_dof.Append(0);
    }
    auto raw_memory = gL->EliminateRowsCols(ess_dof);
    delete raw_memory;

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

        mfem::HypreBoomerAMG prec(*gL);
        prec.SetPrintLevel(0);
        cg.SetPreconditioner(prec);
        if (myid == 0)
        {
            std::cout << "System size: " << gL->N() << "\n";
            std::cout << "System NNZ: " << gL->NNZ() << "\n";
            std::cout << "Setup time: " << chrono.RealTime() << "s. \n";
        }

        chrono.Clear();
        chrono.Start();

        primal_sol = 0.0;
        cg.Mult(rhs_u_fine, primal_sol);
        par_orthogonalize_from_constant(primal_sol, vertex_edge_global.Height());
        if (myid == 0)
        {
            std::cout << "Solve time: " << chrono.RealTime() << "s. \n";
            std::cout << "Number of iterations: " << cg.GetNumIterations() << "\n";
        }
    }
    /// [Solve primal problem by CG + BoomerAMG]

    /// [Check solution difference]

    mfem::Vector diff = primal_sol;
    diff -= hybrid_prec_sol;
    double error = mfem::ParNormlp(diff, 2, comm) / mfem::ParNormlp(primal_sol, 2, comm);
    if (myid == 0)
    {
        std::cout << "|| primal_sol - hybrid_prec_sol || = " << error << " \n";
    }

    diff = primal_sol;
    diff -= mixed_sol;
    error = mfem::ParNormlp(diff, 2, comm) / mfem::ParNormlp(primal_sol, 2, comm);
    if (myid == 0)
    {
        std::cout << "|| primal_sol - mixed_sol || = " << error << " \n";
    }
    /// [Check solution difference]

    MPI_Finalize();

    delete b;
    delete a;
    delete fec;
    delete fespace;
    delete mesh;

    return 0;
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

    mfem::Vector M_inv;
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

            if (col > i && std::fabs(A_a[j]) > 1e-8)
            {
                weight[count] = -1.0 * A_a[j];
                ev.Add(count, i, 1.0);
                ev.Add(count, col, 1.0);
                count++;
            }
        }
    }

    ev.Finalize();
    weight.SetSize(count);

    //    assert(count == num_edges);

    SparseMatrix ve = smoothg::Transpose(ev);
    ve.SetWidth(count);
    vertex_edge.Swap(ve);
}

void Multigrid::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    residual_ = x;
    correction_.SetDataAndSize(y.GetData(), y.Size());
    MG_Cycle();
}

void Multigrid::MG_Cycle() const
{
    // PreSmoothing
    Smoother_.Mult(residual_, correction_);
    //    par_orthogonalize_from_constant(correction_, Operator_.N());
    Operator_.Mult(-1.0, correction_, 1.0, residual_);
    //    mfem::Vector ones(residual_);
    //    ones = 1.0;
    //    mfem::Vector result(residual_);
    //    result = 0.0;
    //    Operator_.Mult(ones, result);

    //    std::cout << std::fabs(mfem::InnerProduct(Operator_.GetComm(), residual_, ones))
    //              << " "<< mfem::ParNormlp(result, 2, Operator_.GetComm()) <<"\n";
    //assert(std::fabs(mfem::InnerProduct(Operator_.GetComm(), residual_, ones)) < 1e-8);
    // Coarse grid correction
    CoarseSolver_.Mult(residual_, help_vec_);
    //    par_orthogonalize_from_constant(help_vec_, Operator_.N());
    correction_ += help_vec_;
    Operator_.Mult(-1.0, help_vec_, 1.0, residual_);

    // PostSmoothing
    Smoother_.Mult(residual_, help_vec_);
    //    par_orthogonalize_from_constant(help_vec_, Operator_.N());
    correction_ += help_vec_;
}
