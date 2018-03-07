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
#include "spe10.hpp"

using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

using namespace smoothg;

void MetisPart(const mfem::SparseMatrix& vertex_edge, mfem::Array<int>& part, int num_parts);
unique_ptr<mfem::HypreParMatrix> GraphLaplacian(const MixedMatrix& mixed_laplacian,
                                                const mfem::SparseMatrix* bdr = nullptr);
mfem::Vector ComputeFiedlerVector(const MixedMatrix& mixed_laplacian);
void CartPart(mfem::Array<int>& partitioning, std::vector<int>& num_procs_xyz,
              mfem::ParMesh& pmesh, mfem::Array<int>& coarsening_factor);

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

    int nDimensions = 2;
    mfem::Array<int> coarseningFactor(nDimensions);
    coarseningFactor[0] = 5;
    coarseningFactor[1] = 5;
    if (nDimensions == 3)
        coarseningFactor[2] = 5;

    int nbdr;
    if (nDimensions == 3)
        nbdr = 6;
    else
        nbdr = 4;
    mfem::Array<int> ess_zeros(nbdr);
    mfem::Array<int> nat_one(nbdr);
    mfem::Array<int> nat_zeros(nbdr);
    ess_zeros = 1;
    nat_one = 0;
    nat_zeros = 0;

    mfem::Array<int> ess_attr;
    mfem::Vector local_weight;
    mfem::Vector rhs_u_fine;

    // Setting up finite volume discretization problem
    SPE10Problem spe10problem("spe_perm.dat", nDimensions, 5, 0,
                              false, coarseningFactor);

    mfem::ParMesh* pmesh = spe10problem.GetParMesh();

    if (myid == 0)
    {
        std::cout << pmesh->GetNEdges() << " fine edges, " <<
                  pmesh->GetNFaces() << " fine faces, " <<
                  pmesh->GetNE() << " fine elements\n";
    }

    ess_attr.SetSize(nbdr);
    for (int i(0); i < nbdr; ++i)
        ess_attr[i] = ess_zeros[i];

    // Construct "finite volume mass" matrix using mfem instead of parelag
    mfem::RT_FECollection sigmafec(0, nDimensions);
    mfem::ParFiniteElementSpace sigmafespace(pmesh, &sigmafec);

    mfem::ParBilinearForm a(&sigmafespace);
    a.AddDomainIntegrator(
        new FiniteVolumeMassIntegrator(*spe10problem.GetKInv()) );
    a.Assemble();
    a.Finalize();
    a.SpMat().GetDiag(local_weight);

    for (int i = 0; i < local_weight.Size(); ++i)
    {
        local_weight[i] = 1.0 / local_weight[i];
    }

    mfem::L2_FECollection ufec(0, nDimensions);
    mfem::ParFiniteElementSpace ufespace(pmesh, &ufec);

    mfem::LinearForm q(&ufespace);
    q.AddDomainIntegrator(new mfem::DomainLFIntegrator(*spe10problem.GetForceCoeff()));
    q.Assemble();
    rhs_u_fine = q;

    // Construct vertex_edge table in mfem::SparseMatrix format
    auto& vertex_edge_table = nDimensions == 2 ? pmesh->ElementToEdgeTable()
                              : pmesh->ElementToFaceTable();
    mfem::SparseMatrix vertex_edge = TableToMatrix(vertex_edge_table);

    const auto& edge_trueedge(sigmafespace.Dof_TrueDof_Matrix());
    auto edge_boundary_att = GenerateBoundaryAttributeTable(pmesh);


    MixedMatrix mixed_laplacian(vertex_edge, local_weight, *edge_trueedge);
    unique_ptr<mfem::HypreParMatrix> gL = GraphLaplacian(mixed_laplacian, &edge_boundary_att);
    /// [Set up parallel graph and Laplacian]

    /// [Right Hand Side]
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

        mfem::Array<int> ess_dof;
        if (myid == 0)
        {
            ess_dof.SetSize(1);
            ess_dof = 0;
        }
        gL->EliminateRowsCols(ess_dof);
        rhs_u_fine(0) = 0.0;

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
        par_orthogonalize_from_constant(primal_sol, gL->N());
        if (myid == 0)
        {
            std::cout << "Solve time: " << chrono.RealTime() << "s. \n";
            std::cout << "Number of iterations: " << cg.GetNumIterations() << "\n";
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
        mfem::Array<int> marker(mixed_laplacian.getD().Width());
        marker = 0;
        sigmafespace.GetEssentialVDofs(ess_attr, marker);
        mfem::Array<int> partitioning;
        auto num_procs_xyz = spe10problem.GetNumProcsXYZ();
        CartPart(partitioning, num_procs_xyz, *pmesh, coarseningFactor);
        HybridSolver hb_solver(comm, mixed_laplacian, partitioning, &edge_boundary_att, &marker);
        if (myid == 0)
        {
            std::cout << "System size: " << hb_solver.GetHybridSystemSize() << "\n";
            std::cout << "System NNZ: " << hb_solver.GetNNZ() << "\n";
            std::cout << "Setup time: " << chrono.RealTime() << "s. \n";
        }

        chrono.Clear();
        chrono.Start();
        mixed_sol = 0.0;
        hb_solver.Solve(fine_rhs, mixed_sol);
        par_orthogonalize_from_constant(mixed_sol.GetBlock(1), gL->N());
        if (myid == 0)
        {
            std::cout << "Solve time: " << chrono.RealTime() << "s. \n";
            std::cout << "Number of iterations: " << hb_solver.GetNumIterations() << "\n\n";
        }
    }
    /// [Solve mixed problem by generalized hybridization]

    /// [Check solution difference]
    primal_sol -= mixed_sol.GetBlock(1);
    double diff = mfem::InnerProduct(comm, primal_sol, primal_sol);
    if (myid == 0)
    {
        std::cout << "|| primal_sol - mixed_sol || = " << std::sqrt(diff) << " \n";
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

unique_ptr<mfem::HypreParMatrix> GraphLaplacian(const MixedMatrix& mixed_laplacian,
                                                const mfem::SparseMatrix* bdr)
{
    auto& pM = mixed_laplacian.get_pM();
    auto& pD = mixed_laplacian.get_pD();
    auto* pW = mixed_laplacian.get_pW();

    unique_ptr<mfem::HypreParMatrix> MinvDT(pD.Transpose());

    mfem::Vector M_inv;
    pM.GetDiag(M_inv);

    // elimination of essential edge dofs
    auto& te_e = mixed_laplacian.get_edge_td_d();
    mfem::SparseMatrix te_e_diag;
    te_e.GetDiag(te_e_diag);
    for (int te = 0; te < M_inv.Size(); te++)
    {
        int e = te_e_diag.GetRowColumns(te)[0];
        if (bdr->RowSize(e) > 0)
        {
            M_inv(te) = 0.0;
        }
        else
        {
            M_inv(te) = 1.0 / M_inv(te);
        }
    }

    MinvDT->ScaleRows(M_inv);

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


void CartPart(mfem::Array<int>& partitioning, std::vector<int>& num_procs_xyz,
              mfem::ParMesh& pmesh, mfem::Array<int>& coarsening_factor)
{
    const int nDimensions = num_procs_xyz.size();

    mfem::Array<int> nxyz(nDimensions);
    nxyz[0] = 60 / num_procs_xyz[0] / coarsening_factor[0];
    nxyz[1] = 220 / num_procs_xyz[1] / coarsening_factor[1];
    if (nDimensions == 3)
        nxyz[2] = 85 / num_procs_xyz[2] / coarsening_factor[2];

    for (int& i : nxyz)
    {
        i = std::max(1, i);
    }

    mfem::Array<int> cart_part(pmesh.CartesianPartitioning(nxyz.GetData()), pmesh.GetNE());
    partitioning.Append(cart_part);

    cart_part.MakeDataOwner();
}