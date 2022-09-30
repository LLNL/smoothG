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
   @file

   @brief Implements HybridSolver object
*/

#include "sharedentitycommunication.hpp"
#include "HybridSolver.hpp"
#include "utilities.hpp"
#include "MatrixUtilities.hpp"
#include "MetisGraphPartitioner.hpp"

using std::unique_ptr;

namespace smoothg
{

HybridSolver::HybridSolver(const MixedMatrix& mgL,
                           const mfem::Array<int>* ess_attr,
                           const int rescale_iter,
                           const SAAMGeParam* saamge_param)
    :
    MixedLaplacianSolver(mgL.GetComm(), mgL.BlockOffsets(), mgL.CheckW()),
    mgL_(mgL),
    rescale_iter_(rescale_iter),
    saamge_param_(saamge_param)
{
    MixedLaplacianSolver::Init(mgL, ess_attr);

    auto mbuilder = dynamic_cast<const ElementMBuilder*>(&(mgL.GetMBuilder()));
    if (!mbuilder)
    {
        std::cout << "HybridSolver requires fine level M builder to be FineMBuilder!\n";
        std::abort();
    }

    const GraphSpace& graph_space = mgL.GetGraphSpace();

    Init(graph_space.EdgeToEDof(), mbuilder->GetElementMatrices(),
         graph_space.EDofToTrueEDof(), graph_space.EDofToBdrAtt());
}

HybridSolver::HybridSolver(const MixedMatrix& mgL,
                           const std::vector<mfem::DenseMatrix>& dMdS,
                           const std::vector<mfem::DenseMatrix>& dTdsigma,
                           const mfem::Array<int>* ess_attr,
                           const int rescale_iter)
    :
    MixedLaplacianSolver(mgL.GetComm(), mgL.BlockOffsets(), mgL.CheckW()),
    mgL_(mgL),
    rescale_iter_(rescale_iter)
{

}


HybridSolver::~HybridSolver()
{
#if SMOOTHG_USE_SAAMGE
    if (saamge_param_)
    {
        saamge::ml_free_data(sa_ml_data_);
        saamge::agg_free_partitioning(sa_apr_);
    }
#endif
}

void HybridSolver::Init(
    const mfem::SparseMatrix& face_edgedof,
    const std::vector<mfem::DenseMatrix>& M_el,
    const mfem::HypreParMatrix& edgedof_d_td,
    const mfem::SparseMatrix& edgedof_bdrattr)
{
    mfem::StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    nAggs_ = mgL_.GetGraph().NumVertices();

    // Set the size of the Hybrid_el_, AinvCT, Ainv_f_, these are all local
    // matrices and vector for each element
    DMinv_.resize(nAggs_);
    MinvCT_.resize(nAggs_);
    AinvDMinvCT_.resize(nAggs_);
    Ainv_.resize(nAggs_);
    Minv_.resize(nAggs_);
    Minv_ref_.resize(nAggs_);
    Hybrid_el_.resize(nAggs_);
    C_.resize(nAggs_);
    CM_.resize(nAggs_);
    CDT_.resize(nAggs_);
    Minv_g_.resize(nAggs_);
    local_rhs_.resize(nAggs_);

    elem_scaling_.SetSize(nAggs_);
    elem_scaling_ = 1.0;

    edof_needs_averaging_ = MakeAveragingIndicators();

    CreateMultiplierRelations(face_edgedof, edgedof_d_td);

    CollectEssentialDofs(edgedof_bdrattr);

    // Assemble the hybridized system on each processor
    mfem::SparseMatrix H_proc = AssembleHybridSystem(M_el);
    if (myid_ == 0 && print_level_ > 0)
        std::cout << "  Timing: Hybridized system built in "
                  << chrono.RealTime() << "s. \n";

    solver_ = InitKrylovSolver(KrylovMethod::CG);
    BuildParallelSystemAndSolver(H_proc);

    Hrhs_.SetSize(num_multiplier_dofs_);
    trueHrhs_.SetSize(multiplier_d_td_->GetNumCols());
    Mu_.SetSize(num_multiplier_dofs_);
    trueMu_.SetSize(trueHrhs_.Size());
    trueMu_ = 0.0;

    MakeDarcyInverses(M_el);
}

void HybridSolver::MakeDarcyInverses(const std::vector<mfem::DenseMatrix>& M_el)
{
    const int num_hatdofs = mgL_.GetGraphSpace().VertexToEDof().NumNonZeroElems();
    const int num_injectors = mgL_.NumInjectors();
    auto& vert_edof = mgL_.GetGraphSpace().VertexToEDof();
    auto& vert_vdof = mgL_.GetGraphSpace().VertexToVDof();

    mfem::Array<int> local_edofs, local_vdofs, local_mult, local_hat;
    mfem::DenseMatrix DenseDloc, DenseCloc;

    C_proc_ = make_unique<mfem::SparseMatrix>(mgL_.NumEDofs(), num_hatdofs);
    darcy_inv_00_.resize(vert_edof.NumRows()-num_injectors);
    darcy_inv_01_.resize(vert_edof.NumRows()-num_injectors);
    darcy_inv_10_.resize(vert_edof.NumRows()-num_injectors);
    darcy_inv_11_.resize(vert_edof.NumRows()-num_injectors);
    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(Agg_multiplier_, i, local_mult);
        GetTableRow(vert_edof, i, local_edofs);
        local_hat.SetSize(local_edofs.Size());
        std::iota(local_hat.begin(), local_hat.end(), vert_edof.GetI()[i]);
        Full(C_[i], DenseCloc);
        C_proc_->AddSubMatrix(local_mult, local_hat, DenseCloc);

        if (i >= vert_edof.NumRows() - num_injectors) { continue; }

        GetTableRow(vert_vdof, i, local_vdofs);
        auto Dloc = ExtractRowAndColumns(mgL_.GetD(), local_vdofs, local_edofs);
        Full(Dloc, DenseDloc);

        mfem::DenseMatrix darcy_el(DenseDloc.NumCols() + DenseDloc.NumRows());
        darcy_el.CopyMN(M_el[i], 0, 0);
        darcy_el.CopyMN(DenseDloc, DenseDloc.NumCols(), 0);
        darcy_el.CopyMNt(DenseDloc, 0, DenseDloc.NumCols());

        mfem::DenseMatrixInverse darcy_el_solver(darcy_el);
        mfem::DenseMatrix inv;
        darcy_el_solver.GetInverseMatrix(inv);

        darcy_inv_00_[i].CopyMN(inv, local_hat.Size(), local_hat.Size(), 0, 0);
        darcy_inv_01_[i].CopyMN(inv, local_hat.Size(), local_vdofs.Size(),
                                0, local_hat.Size());
        darcy_inv_10_[i].CopyMN(inv, local_vdofs.Size(), local_hat.Size(),
                                local_hat.Size(), 0);
        darcy_inv_11_[i].CopyMN(inv, local_vdofs.Size(), local_vdofs.Size(),
                                local_hat.Size(), local_hat.Size());
    }
    C_proc_->Finalize();
}

void HybridSolver::CreateMultiplierRelations(
    const mfem::SparseMatrix& face_edgedof,
    const mfem::HypreParMatrix& edgedof_d_td)
{
    // Constructing the relation table (in SparseMatrix format) between edge
    // dof and multiplier dof. For every edge dof that is associated with a
    // face, a Lagrange multiplier dof associated with the edge dof is created
    num_multiplier_dofs_ = face_edgedof.Width();

    const auto& Agg_edgedof = mgL_.GetGraphSpace().VertexToEDof();
    const int num_edge_dofs = Agg_edgedof.Width();

    int* i_edgedof_multiplier = new int[num_edge_dofs + 1];
    std::iota(i_edgedof_multiplier,
              i_edgedof_multiplier + num_multiplier_dofs_ + 1, 0);
    std::fill_n(i_edgedof_multiplier + num_multiplier_dofs_ + 1,
                num_edge_dofs - num_multiplier_dofs_,
                i_edgedof_multiplier[num_multiplier_dofs_]);

    int* j_edgedof_multiplier = new int[num_multiplier_dofs_];
    std::iota(j_edgedof_multiplier,
              j_edgedof_multiplier + num_multiplier_dofs_, 0);
    double* data_edgedof_multiplier = new double[num_multiplier_dofs_];
    std::fill_n(data_edgedof_multiplier, num_multiplier_dofs_, 1.0);
    mfem::SparseMatrix edgedof_multiplier(
        i_edgedof_multiplier, j_edgedof_multiplier,
        data_edgedof_multiplier, num_edge_dofs, num_multiplier_dofs_);
    mfem::SparseMatrix mult_edof(smoothg::Transpose(edgedof_multiplier) );

    mfem::Array<int> j_array(mult_edof.GetJ(), mult_edof.NumNonZeroElems());
    j_array.Copy(multiplier_to_edof_);

    mfem::SparseMatrix Agg_m_tmp(smoothg::Mult(Agg_edgedof, edgedof_multiplier));
    Agg_multiplier_.Swap(Agg_m_tmp);

    GenerateOffsets(comm_, num_multiplier_dofs_, multiplier_start_);

    auto mult_trueedof = ParMult(mult_edof, edgedof_d_td, multiplier_start_);
    unique_ptr<mfem::HypreParMatrix> multiplier_d_td_d(AAt(*mult_trueedof));

    // Construct multiplier "dof to true dof" table
    multiplier_d_td_ = BuildEntityToTrueEntity(*multiplier_d_td_d);
    multiplier_td_d_.reset(multiplier_d_td_->Transpose());

    {
        edof_shared_mean_.reset(edgedof_d_td.Transpose());
        auto edof_shared_mean_offd = GetOffd(*edof_shared_mean_);
        auto edof_shared_mean_diag = GetDiag(*edof_shared_mean_);
        for (int i = 0; i < edof_shared_mean_offd.NumRows(); ++i)
        {
            if (edof_shared_mean_offd.RowSize(i))
            {
                edof_shared_mean_offd.GetRowEntries(i)[0] = 0.5;
                edof_shared_mean_diag.GetRowEntries(i)[0] = 0.5;
            }
        }
    }
}

std::vector<bool> HybridSolver::MakeAveragingIndicators()
{
    auto edof_vert = smoothg::Transpose(mgL_.GetGraphSpace().VertexToEDof());
    std::vector<bool> out(edof_vert.NumRows());
    for (int i = 0; i < edof_vert.NumRows(); ++i)
    {
        out[i] = (edof_vert.RowSize(i) > 1);
    }
    return out;
}

mfem::SparseMatrix HybridSolver::AssembleHybridSystem(
    const std::vector<mfem::DenseMatrix>& M_el)
{
    mfem::SparseMatrix H_proc(num_multiplier_dofs_);

    const auto& Agg_vertexdof = mgL_.GetGraphSpace().VertexToVDof();
    const auto& Agg_edgedof = mgL_.GetGraphSpace().VertexToEDof();

    const int map_size = std::max(Agg_edgedof.Width(), Agg_vertexdof.Width());
    mfem::Array<int> edof_global_to_local_map(map_size);
    edof_global_to_local_map = -1;
    std::vector<bool> edge_marker(Agg_edgedof.Width(), true);

    mfem::SparseMatrix edof_IsOwned = GetDiag(*multiplier_d_td_);

    const int scaling_size = rescale_iter_ < 0 ? H_proc.NumRows() : 0;
    mfem::Vector CCT_diag(scaling_size), CDT1(scaling_size);
    CCT_diag = 0.0;
    CDT1 = 0.0;

    mfem::DenseMatrix DlocT, ClocT, Aloc, CMinvDT, DMinvCT, CMDADMC;
    mfem::Vector one;
    mfem::DenseMatrixInverse Mloc_solver, Aloc_solver;
    for (int iAgg = 0; iAgg < nAggs_; ++iAgg)
    {
        // Extracting the size and global numbering of local dof
        mfem::Array<int> local_vertexdof, local_edgedof, local_multiplier;
        GetTableRow(Agg_vertexdof, iAgg, local_vertexdof);
        GetTableRow(Agg_edgedof, iAgg, local_edgedof);
        GetTableRow(Agg_multiplier_, iAgg, local_multiplier);

        const int nlocal_vertexdof = local_vertexdof.Size();
        const int nlocal_edgedof = local_edgedof.Size();
        const int nlocal_multiplier = local_multiplier.Size();

        // Build the edge dof global to local map which will be used
        // later for mapping local multiplier dof to local edge dof
        for (int i = 0; i < nlocal_edgedof; ++i)
            edof_global_to_local_map[local_edgedof[i]] = i;

        // Extract Dloc as a sparse submatrix of D_
        auto Dloc = ExtractRowAndColumns(mgL_.GetD(), local_vertexdof, local_edgedof,
                                         edof_global_to_local_map, false);

        // Fill DlocT as a dense matrix of Dloc^T
        FullTranspose(Dloc, DlocT);

        // Construct the constraint matrix C which enforces the continuity of
        // the broken edge space
        ClocT.SetSize(nlocal_edgedof, nlocal_multiplier);
        ClocT = 0.;

        int* Cloc_i = new int[nlocal_multiplier + 1];
        std::iota(Cloc_i, Cloc_i + nlocal_multiplier + 1, 0);
        int* Cloc_j = new int[nlocal_multiplier];
        double* Cloc_data = new double[nlocal_multiplier];
        for (int i = 0; i < nlocal_multiplier; ++i)
        {
            const int edof_global_id = multiplier_to_edof_[local_multiplier[i]];
            const int edof_local_id = edof_global_to_local_map[edof_global_id];
            Cloc_j[i] = edof_local_id;
            if (edof_IsOwned.RowSize(edof_global_id) && edge_marker[edof_global_id])
            {
                edge_marker[edof_global_id] = false;
                ClocT(edof_local_id, i) = 1.;
                Cloc_data[i] = 1.;
            }
            else
            {
                ClocT(edof_local_id, i) = -1.;
                Cloc_data[i] = -1.;
            }
        }

        mfem::SparseMatrix Cloc(Cloc_i, Cloc_j, Cloc_data,
                                nlocal_multiplier, nlocal_edgedof);

        for (int i = 0; i < nlocal_edgedof; ++i)
            edof_global_to_local_map[local_edgedof[i]] = -1;

        // for initial guess
        {
            C_[iAgg].Swap(Cloc);
            CM_[iAgg] = smoothg::Mult(C_[iAgg], M_el[iAgg]);
            auto DT = smoothg::Transpose(Dloc);
            auto CDT = smoothg::Mult(C_[iAgg], DT);
            CDT_[iAgg].Swap(CDT);
        }

        Mloc_solver.SetOperator(M_el[iAgg]);
        Mloc_solver.GetInverseMatrix(Minv_ref_[iAgg]);

        mfem::DenseMatrix& MinvCT_i(MinvCT_[iAgg]);
        mfem::DenseMatrix& AinvDMinvCT_i(AinvDMinvCT_[iAgg]);
        mfem::DenseMatrix& Ainv_i(Ainv_[iAgg]);

        MinvCT_i.SetSize(nlocal_edgedof, nlocal_multiplier);
        AinvDMinvCT_i.SetSize(nlocal_vertexdof, nlocal_multiplier);

        mfem::DenseMatrix MinvDT_i(nlocal_edgedof, nlocal_vertexdof);
        mfem::Mult(Minv_ref_[iAgg], DlocT, MinvDT_i);
        mfem::Mult(Minv_ref_[iAgg], ClocT, MinvCT_i);

        DMinv_[iAgg].Transpose(MinvDT_i);

        // Compute CMinvCT = Cloc * MinvCT
        Hybrid_el_[iAgg] = smoothg::Mult(C_[iAgg], MinvCT_i);

        // Compute Aloc = DMinvDT = Dloc * MinvDT
        Aloc = smoothg::Mult(Dloc, MinvDT_i);

        if (mgL_.GetW().Width())
        {
            mfem::DenseMatrix Wloc(nlocal_vertexdof, nlocal_vertexdof);
            auto& W_ref = const_cast<mfem::SparseMatrix&>(mgL_.GetW());
            W_ref.GetSubMatrix(local_vertexdof, local_vertexdof, Wloc);
            Aloc += Wloc;
        }

        // Compute DMinvCT = Dloc * MinvCT
        DMinvCT = smoothg::Mult(Dloc, MinvCT_i);

        // Compute the LU factorization of Aloc and Ainv_ * DMinvCT
        Aloc_solver.SetOperator(Aloc);
        Aloc_solver.GetInverseMatrix(Ainv_i);
        mfem::Mult(Ainv_i, DMinvCT, AinvDMinvCT_i);

        // Compute CMinvDTAinvDMinvCT = CMinvDT * AinvDMinvCT_
        CMinvDT.Transpose(DMinvCT);
        CMDADMC.SetSize(nlocal_multiplier, nlocal_multiplier);

        if (CMinvDT.Height() > 0 && CMinvDT.Width() > 0)
        {
            mfem::Mult(CMinvDT, AinvDMinvCT_i, CMDADMC);
        }
        else
        {
            CMDADMC = 0.0;
        }

        // Hybrid_el_ = CMinvCT - CMinvDTAinvDMinvCT
        Hybrid_el_[iAgg] -= CMDADMC;

        // Add contribution of the element matrix to the global system
        H_proc.AddSubMatrix(local_multiplier, local_multiplier, Hybrid_el_[iAgg]);

        // Save CCT and CDT1
        if (scaling_size > 0)
        {
            mfem::DenseMatrix CCT(nlocal_multiplier);
            CCT = smoothg::Mult(C_[iAgg], ClocT);
            mfem::Vector CCT_diag_local;
            CCT.GetDiag(CCT_diag_local);

            const_rep_->GetSubVector(local_vertexdof, one);
            mfem::Vector DTone(nlocal_edgedof);
            Dloc.MultTranspose(one, DTone);

            mfem::Vector CDT1_local(nlocal_multiplier);
            C_[iAgg].Mult(DTone, CDT1_local);

            for (int i = 0; i < nlocal_multiplier; ++i)
            {
                CCT_diag[local_multiplier[i]] += CCT_diag_local[i];
                CDT1[local_multiplier[i]] += CDT1_local[i];
            }
        }
    }

    // Assemble global rescaling vector (CC^T)^{-1}CD^T 1
    if (scaling_size)
    {
        mfem::Vector CCT_diag_global(multiplier_d_td_->NumCols());
        multiplier_td_d_->Mult(CCT_diag, CCT_diag_global);

        mfem::Vector CDT1_global(multiplier_d_td_->NumCols());
        multiplier_td_d_->Mult(CDT1, CDT1_global);

        diagonal_scaling_.SetSize(multiplier_d_td_->NumCols());
        for (int i = 0; i < diagonal_scaling_.Size(); ++i)
        {
            diagonal_scaling_[i] = CDT1_global[i] / CCT_diag_global[i];
            diagonal_scaling_[i] = 1.0;
        }
    }

    return H_proc;
}

mfem::SparseMatrix HybridSolver::AssembleHybridSystem(
    const mfem::Vector& elem_scaling_inverse,
    const std::vector<mfem::DenseMatrix>& N_el)
{
    const auto& Agg_vertexdof = mgL_.GetGraphSpace().VertexToVDof();
    const auto& Agg_edgedof = mgL_.GetGraphSpace().VertexToEDof();

    mfem::SparseMatrix H_proc(num_multiplier_dofs_);

    MinvN_.resize(Minv_.size());
    CMinvNAinv_.resize(Minv_.size());

    mfem::Array<int> col_marker(mgL_.GetD().NumCols());
    col_marker = -1;

    const int scaling_size = rescale_iter_ < 0 ? H_proc.NumRows() : 0;
    mfem::Vector CCT_diag(scaling_size), CDT1(scaling_size);
    CCT_diag = 0.0;
    CDT1 = 0.0;

    mfem::DenseMatrix DlocT, Aloc, CMinv, CMinvN, DMinvCT, CMDADMC;
    mfem::Array<int> local_vertexdof, local_edgedof, local_multiplier;
    mfem::Vector one;

    mfem::DenseMatrixInverse Aloc_solver;
    for (int iAgg = 0; iAgg < nAggs_; ++iAgg)
    {
        // Extracting the size and global numbering of local dof
        GetTableRow(Agg_vertexdof, iAgg, local_vertexdof);
        GetTableRow(Agg_edgedof, iAgg, local_edgedof);
        GetTableRow(Agg_multiplier_, iAgg, local_multiplier);

        const int nlocal_vertexdof = local_vertexdof.Size();
        const int nlocal_edgedof = local_edgedof.Size();
        const int nlocal_multiplier = local_multiplier.Size();

        // Extract Dloc as a sparse submatrix of D_
        auto Dloc = ExtractRowAndColumns(mgL_.GetD(), local_vertexdof,
                                         local_edgedof, col_marker);

        // Fill DlocT as a dense matrix of Dloc^T
        FullTranspose(Dloc, DlocT);
        DlocT += N_el[iAgg];

        Minv_[iAgg] = Minv_ref_[iAgg];
        Minv_[iAgg] *= elem_scaling_inverse[iAgg];

        mfem::DenseMatrix& MinvCT_i(MinvCT_[iAgg]);
        mfem::DenseMatrix& MinvN_i(MinvN_[iAgg]);
        mfem::DenseMatrix& AinvDMinvCT_i(AinvDMinvCT_[iAgg]);
        mfem::DenseMatrix& CMinvNAinv_i(CMinvNAinv_[iAgg]);
        mfem::DenseMatrix& Ainv_i(Ainv_[iAgg]);

        MinvCT_i.SetSize(nlocal_edgedof, nlocal_multiplier);
        MinvN_i.SetSize(nlocal_edgedof, nlocal_vertexdof);
        AinvDMinvCT_i.SetSize(nlocal_vertexdof, nlocal_multiplier);
        CMinvNAinv_i.SetSize(nlocal_multiplier, nlocal_vertexdof);

        mfem::Mult(Minv_[iAgg], DlocT, MinvN_i);
        MultSparseDenseTranspose(C_[iAgg], Minv_[iAgg], MinvCT_i);
        DMinv_[iAgg] = smoothg::Mult(Dloc, Minv_[iAgg]);

        // Compute CMinvCT = Cloc * MinvCT
        Hybrid_el_[iAgg] = smoothg::Mult(C_[iAgg], MinvCT_i);

        // Compute Aloc = DMinvN = Dloc * Minv * N
        Aloc = smoothg::Mult(Dloc, MinvN_i);

        if (mgL_.GetW().NumCols())
        {
            mfem::DenseMatrix Wloc(nlocal_vertexdof, nlocal_vertexdof);
            auto& W_ref = const_cast<mfem::SparseMatrix&>(mgL_.GetW());
            W_ref.GetSubMatrix(local_vertexdof, local_vertexdof, Wloc);
            Aloc += Wloc;
        }

        // Compute DMinvCT = Dloc * MinvCT
        DMinvCT = smoothg::Mult(Dloc, MinvCT_i);

        // Compute the LU factorization of Aloc and Ainv_ * DMinvCT
        Aloc_solver.SetOperator(Aloc);
        Aloc_solver.GetInverseMatrix(Ainv_i);
        mfem::Mult(Ainv_i, DMinvCT, AinvDMinvCT_i);

        // Compute CMinvNAinvDMinvCT = CMinvN * AinvDMinvCT_
        CMinv.Transpose(MinvCT_i);
        CMinvN.SetSize(nlocal_multiplier, nlocal_vertexdof);
        mfem::Mult(CMinv, DlocT, CMinvN);

        mfem::Mult(CMinvN, Ainv_i, CMinvNAinv_i);

        CMDADMC.SetSize(nlocal_multiplier, nlocal_multiplier);

        if (CMinvN.Height() > 0 && CMinvN.Width() > 0)
        {
            mfem::Mult(CMinvN, AinvDMinvCT_i, CMDADMC);
        }
        else
        {
            CMDADMC = 0.0;
        }

        // Hybrid_el_ = CMinvCT - CMinvDTAinvDMinvCT
        Hybrid_el_[iAgg] -= CMDADMC;

        // Add contribution of the element matrix to the global system
        H_proc.AddSubMatrix(local_multiplier, local_multiplier, Hybrid_el_[iAgg]);

        // Save CCT and CDT1
        if (scaling_size > 0)
        {
            mfem::SparseMatrix ClocT = smoothg::Transpose(C_[iAgg]);
            mfem::SparseMatrix CCT = smoothg::Mult(C_[iAgg], ClocT);
            mfem::Vector CCT_diag_local;
            CCT.GetDiag(CCT_diag_local);

            const_rep_->GetSubVector(local_vertexdof, one);
            mfem::Vector DTone(nlocal_edgedof);
            Dloc.MultTranspose(one, DTone);

            mfem::Vector CDT1_local(nlocal_multiplier);
            C_[iAgg].Mult(DTone, CDT1_local);

            for (int i = 0; i < nlocal_multiplier; ++i)
            {
                CCT_diag[local_multiplier[i]] += CCT_diag_local[i];
                CDT1[local_multiplier[i]] += CDT1_local[i];
            }
        }
    }

    // Assemble global rescaling vector (CC^T)^{-1}CD^T 1
    if (scaling_size)
    {
        mfem::Vector CCT_diag_global(multiplier_d_td_->NumCols());
        multiplier_td_d_->Mult(CCT_diag, CCT_diag_global);

        mfem::Vector CDT1_global(multiplier_d_td_->NumCols());
        multiplier_td_d_->Mult(CDT1, CDT1_global);

        diagonal_scaling_.SetSize(multiplier_d_td_->NumCols());
        for (int i = 0; i < diagonal_scaling_.Size(); ++i)
        {
            diagonal_scaling_[i] = CDT1_global[i] / CCT_diag_global[i];
        }
    }

    return H_proc;
}

/// @todo nonzero Neumann BC (edge unknown), solve on true dof (original system)
void HybridSolver::Mult(const mfem::BlockVector& Rhs, mfem::BlockVector& Sol) const
{
    rhs_ = Rhs;

//    if (is_symmetric_)
//    {
//        trueMu_ = MakeInitialGuess(Sol, Rhs);
//    }
//    else
    {
        trueMu_ = 0.0;
    }

    // TODO: nonzero b.c.
    // correct right hand side due to boundary condition
    for (int m = 0; m < ess_true_multipliers_.Size(); ++m)
    {
        trueMu_(ess_true_multipliers_[m]) = -Rhs(ess_true_mult_to_edof_[m]);
        rhs_(ess_true_mult_to_edof_[m]) = 0.0;
    }

    RHSTransform(rhs_, Hrhs_);

    // assemble true right hand side
    multiplier_d_td_->MultTranspose(Hrhs_, trueHrhs_);

    H_elim_->Mult(-1.0, trueMu_, 1.0, trueHrhs_);
    for (int ess_true_mult : const_cast<mfem::Array<int>&>(ess_true_multipliers_))
    {
        trueHrhs_(ess_true_mult) = trueMu_(ess_true_mult);
    }

    if (diagonal_scaling_.Size() > 0)
    {
        RescaleVector(diagonal_scaling_, trueHrhs_);
        InvRescaleVector(diagonal_scaling_, trueMu_);
    }

    // solve the parallel global hybridized system
    mfem::StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    solver_->Mult(trueHrhs_, trueMu_);

    chrono.Stop();
    timing_ = chrono.RealTime();

    std::string solver_name = is_symmetric_ ? "CG" : "GMRES";

    if (myid_ == 0 && print_level_ > 0)
    {
        std::cout << "  Timing: " + solver_name + " done in "
                  << timing_ << "s. \n";
    }

    // TODO: decide to use = or += here and in timing_ update (MinresBlockSolver uses +=)
    num_iterations_ = solver_->GetNumIterations();

    if (myid_ == 0 && print_level_ > 0)
    {
        if (solver_->GetConverged())
        {
            std::cout << "  " + solver_name + " converged in "
                      << num_iterations_
                      << " with a final residual norm "
                      << solver_->GetFinalNorm() << "\n";
        }
        else
        {
            std::cout << "  " + solver_name + " did not converge in "
                      << num_iterations_
                      << ". Final residual norm is "
                      << solver_->GetFinalNorm() << "\n";
        }
    }

    // distribute true dofs to dofs and recover solution of the original system
    chrono.Clear();
    chrono.Start();

    if (diagonal_scaling_.Size() > 0)
        RescaleVector(diagonal_scaling_, trueMu_);

    multiplier_d_td_->Mult(trueMu_, Mu_);
    RecoverOriginalSolution(Mu_, Sol);

    mfem::Vector mean_correction(edof_shared_mean_->NumRows());
    edof_shared_mean_->Mult(Sol.GetBlock(0), mean_correction);
    mgL_.GetGraphSpace().EDofToTrueEDof().Mult(mean_correction, Sol.GetBlock(0));

    if (!W_is_nonzero_ && remove_one_dof_)
    {
        Orthogonalize(Sol.GetBlock(1));
    }

    chrono.Stop();

    if (myid_ == 0 && print_level_ > 0)
    {
        std::cout << "  Timing: original solution recovered in "
                  << chrono.RealTime() << "s. \n";
    }
}

/// @todo impose nonzero boundary condition for u.n
void HybridSolver::RHSTransform(const mfem::BlockVector& OriginalRHS,
                                mfem::Vector& HybridRHS) const
{
    const mfem::Vector& OriginalRHS_block1(OriginalRHS.GetBlock(1));
    const auto& Agg_vertexdof = mgL_.GetGraphSpace().VertexToVDof();
    const auto& Agg_edgedof = mgL_.GetGraphSpace().VertexToEDof();

    mfem::Vector mean_correction(edof_shared_mean_->NumRows());
    edof_shared_mean_->Mult(OriginalRHS.GetBlock(0), mean_correction);

    mfem::Vector CorrectedRHS_block0(edof_shared_mean_->NumCols());
    mgL_.GetGraphSpace().EDofToTrueEDof().Mult(mean_correction, CorrectedRHS_block0);

    HybridRHS = 0.;

    mfem::Vector f_loc, g_loc, CMinv_g_loc, DMinv_g_loc, rhs_loc_help;
    mfem::Array<int> local_vertexdof, local_edgedof, local_multiplier;
    for (int iAgg = 0; iAgg < nAggs_; ++iAgg)
    {
        // Extracting the size and global numbering of local dof
        GetTableRow(Agg_vertexdof, iAgg, local_vertexdof);
        GetTableRow(Agg_edgedof, iAgg, local_edgedof);
        GetTableRow(Agg_multiplier_, iAgg, local_multiplier);

        int nlocal_vertexdof = local_vertexdof.Size();
        int nlocal_edgedof = local_edgedof.Size();
        int nlocal_multiplier = local_multiplier.Size();

        // Compute local contribution to the RHS of the hybrid system
        // CM^{-1} g - CM^{-1}D^T A^{-1} (DM^{-1} g - f)
        OriginalRHS_block1.GetSubVector(local_vertexdof, f_loc);
        CorrectedRHS_block0.GetSubVector(local_edgedof, g_loc);
        for (int i = 0; i < nlocal_edgedof; ++i)
        {
            if (edof_needs_averaging_[local_edgedof[i]])
            {
                g_loc[i] /= 2.0;
            }
        }

        DMinv_g_loc.SetSize(nlocal_vertexdof);
        DMinv_[iAgg].Mult(g_loc, DMinv_g_loc);

        if (is_symmetric_)
        {
            DMinv_g_loc /= elem_scaling_[iAgg];
        }

        DMinv_g_loc -= f_loc;

        rhs_loc_help.SetSize(nlocal_multiplier);

        if (is_symmetric_)
        {
            AinvDMinvCT_[iAgg].MultTranspose(DMinv_g_loc, rhs_loc_help);
        }
        else
        {
            CMinvNAinv_[iAgg].Mult(DMinv_g_loc, rhs_loc_help);
        }

        CMinv_g_loc.SetSize(nlocal_multiplier);
        MinvCT_[iAgg].MultTranspose(g_loc, CMinv_g_loc);

        if (is_symmetric_)
        {
            CMinv_g_loc /= elem_scaling_[iAgg];
        }

        CMinv_g_loc -= rhs_loc_help;

        for (int i = 0; i < nlocal_multiplier; ++i)
        {
            HybridRHS[local_multiplier[i]] += CMinv_g_loc[i];
        }

        // Save M^{-1}g, A^{-1} (DM^{-1} g - f) for solution recovery
        Minv_g_[iAgg].SetSize(nlocal_edgedof);
        if (is_symmetric_)
        {
            Minv_ref_[iAgg].Mult(g_loc, Minv_g_[iAgg]);
        }
        else
        {
            Minv_[iAgg].Mult(g_loc, Minv_g_[iAgg]);
        }

        local_rhs_[iAgg].SetSize(nlocal_vertexdof);
        Ainv_[iAgg].Mult(DMinv_g_loc, local_rhs_[iAgg]);
        if (is_symmetric_)
        {
            local_rhs_[iAgg] *= elem_scaling_[iAgg];
        }
    }
}

void HybridSolver::RecoverOriginalSolution(const mfem::Vector& HybridSol,
                                           mfem::BlockVector& RecoveredSol) const
{
    const auto& Agg_vertexdof = mgL_.GetGraphSpace().VertexToVDof();
    const auto& Agg_edgedof = mgL_.GetGraphSpace().VertexToEDof();

    RecoveredSol = 0.;

    mfem::Vector& RecoveredSol_block2(RecoveredSol.GetBlock(1));

    mfem::Array<int> local_vertexdof, local_edgedof, local_multiplier;
    mfem::Vector mu_loc, tmp;
    for (int iAgg = 0; iAgg < nAggs_; ++iAgg)
    {
        // Extracting the size and global numbering of local dof
        GetTableRow(Agg_vertexdof, iAgg, local_vertexdof);
        GetTableRow(Agg_edgedof, iAgg, local_edgedof);
        GetTableRow(Agg_multiplier_, iAgg, local_multiplier);

        int nlocal_vertexdof = local_vertexdof.Size();
        int nlocal_edgedof = local_edgedof.Size();
        int nlocal_multiplier = local_multiplier.Size();

        // Initialize a vector which will store the local contribution of Hdiv
        // and L2 space
        mfem::Vector& u_loc(local_rhs_[iAgg]);
        mfem::Vector& sigma_loc(Minv_g_[iAgg]);

        // This check is just for the case when there is only one element for
        // the global problem, then there will be no Lagrange multipliers
        if (nlocal_multiplier > 0)
        {
            // Extract the local portion of the Lagrange multiplier solution
            HybridSol.GetSubVector(local_multiplier, mu_loc);

            // Compute u = A^{-1} (DM^{-1} (g - C^T mu) - f)
            tmp.SetSize(u_loc.Size());
            AinvDMinvCT_[iAgg].Mult(mu_loc, tmp);
            u_loc -= tmp;

            // Compute sigma = M^{-1} (g - D^T u - C^T mu)
            tmp.SetSize(nlocal_edgedof);
            if (is_symmetric_)
            {
                DMinv_[iAgg].MultTranspose(u_loc, tmp);
            }
            else
            {
                MinvN_[iAgg].Mult(u_loc, tmp);
            }

            sigma_loc -= tmp;
            tmp.SetSize(nlocal_edgedof);
            MinvCT_[iAgg].Mult(mu_loc, tmp);
            sigma_loc -= tmp;

            if (is_symmetric_)
            {
                sigma_loc /= elem_scaling_[iAgg];
            }
        }

        // Save local solution to the global solution vector
        for (int i = 0; i < nlocal_vertexdof; ++i)
            RecoveredSol_block2(local_vertexdof[i]) = u_loc(i);

        for (int i = 0; i < nlocal_edgedof; ++i)
        {
            if (edof_needs_averaging_[local_edgedof[i]])
            {
                RecoveredSol(local_edgedof[i]) += sigma_loc(i) * 0.5;
            }
            else
            {
                RecoveredSol(local_edgedof[i]) = sigma_loc(i);
            }
        }
    }
}

void HybridSolver::ComputeScaledHybridSystem(const mfem::HypreParMatrix& H)
{
    if (rescale_iter_ > 0)
    {
        mfem::HypreSmoother prec_scale(const_cast<mfem::HypreParMatrix&>(H));

        mfem::Vector zeros(H.Height());
        zeros = 0.0;
        diagonal_scaling_.SetSize(H.Height());
        diagonal_scaling_ = 1.0;

        mfem::SLISolver sli(comm_);
        sli.SetMaxIter(rescale_iter_);
        sli.SetPreconditioner(prec_scale);
        sli.SetOperator(H);
        sli.Mult(zeros, diagonal_scaling_);

        for (auto ess_true_mult : ess_true_multipliers_)
        {
            diagonal_scaling_[ess_true_mult] = 1.0;
        }
    }

    if (H.N() == mgL_.GetGraph().EdgeToTrueEdge().N())
    {
        auto Scale = VectorToMatrix(diagonal_scaling_);
        mfem::HypreParMatrix pScale(comm_, H.N(), H.GetColStarts(), &Scale);
        H_.reset(smoothg::Mult(pScale, H, pScale));
    }
}

void HybridSolver::BuildSpectralAMGePreconditioner()
{
#if SMOOTHG_USE_SAAMGE
    saamge::proc_init(comm_);

    mfem::Table elem_dof_tmp = MatrixToTable(Agg_multiplier_);
    mfem::Table* elem_dof = new mfem::Table;
    elem_dof->Swap(elem_dof_tmp);
    mfem::SparseMatrix multiplier_Agg = smoothg::Transpose(Agg_multiplier_);
    mfem::SparseMatrix Agg_Agg = smoothg::Mult(Agg_multiplier_, multiplier_Agg);
    auto elem_adjacency_matrix = MetisGraphPartitioner::getAdjacency(Agg_Agg);
    mfem::Table elem_elem_tmp = MatrixToTable(elem_adjacency_matrix);
    mfem::Table* elem_elem = new mfem::Table;
    elem_elem->Swap(elem_elem_tmp);

    // Mark dofs that are shared by more than one processor
    saamge::SharedEntityCommunication<mfem::Vector> sec(comm_, *multiplier_d_td_);

    std::vector<saamge::agg_dof_status_t> bdr_dofs(elem_dof->Width(), 0);
    for (unsigned int i = 0; i < bdr_dofs.size(); i++)
    {
        if (sec.NumNeighbors(i) > 1)
            SA_SET_FLAGS(bdr_dofs[i], AGG_ON_PROC_IFACE_FLAG);
    }

    int num_elems = elem_elem->Size();
    sa_nparts_.resize(saamge_param_->num_levels - 1);
    sa_nparts_[0] = (num_elems / saamge_param_->first_coarsen_factor) + 1;

    bool first_do_aggregates = (saamge_param_->num_levels <= 2 && saamge_param_->do_aggregates);
    sa_apr_ = saamge::agg_create_partitioning_fine(
                  *H_, num_elems, elem_dof, elem_elem, nullptr, bdr_dofs.data(),
                  sa_nparts_.data(), multiplier_d_td_.get(), first_do_aggregates);

    // FIXME (CSL): I suspect agg_create_partitioning_fine may change the value
    // of sa_nparts_[0] in some cases, so I define the rest of the array here
    for (int i = 1; i < saamge_param_->num_levels - 1; i++)
        sa_nparts_[i] = sa_nparts_[i - 1] / saamge_param_->coarsen_factor + 1;

    mfem::Array<mfem::DenseMatrix*> elmats(Hybrid_el_.size());
    for (unsigned int i = 0; i < Hybrid_el_.size(); i++)
        elmats[i] = &(Hybrid_el_[i]);
    auto emp = new saamge::ElementMatrixDenseArray(*sa_apr_, elmats);

    int polynomial_coarse = -1; // we do not have geometric information
    saamge::MultilevelParameters mlp(
        saamge_param_->num_levels - 1, sa_nparts_.data(), saamge_param_->first_nu_pro,
        saamge_param_->nu_pro, saamge_param_->nu_relax, saamge_param_->first_theta,
        saamge_param_->theta, polynomial_coarse, saamge_param_->correct_nulspace,
        saamge_param_->use_arpack, saamge_param_->do_aggregates);
    sa_ml_data_ = saamge::ml_produce_data(*H_, sa_apr_, emp, mlp);
    auto level = saamge::levels_list_get_level(sa_ml_data_->levels_list, 0);

    prec_ = make_unique<saamge::VCycleSolver>(level->tg_data, false);
    prec_->SetOperator(*H_);
#else
    if (myid_ == 0)
        std::cout << "SAAMGE needs to be enabled! \n";
    std::abort();
#endif
}

void HybridSolver::BuildParallelSystemAndSolver(mfem::SparseMatrix& H_proc)
{
    H_proc.Finalize();
    {
        auto tmp = ParMult(*multiplier_td_d_, H_proc, multiplier_start_);
        H_.reset(mfem::ParMult(tmp.get(), multiplier_d_td_.get()));
    }

    H_elim_.reset(H_->EliminateRowsCols(ess_true_multipliers_));

    if (std::abs(rescale_iter_) > 0 && !saamge_param_)
    {
        ComputeScaledHybridSystem(*H_);
    }
    nnz_ = H_->NNZ();

    solver_->SetOperator(*H_);

    mfem::StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    // HypreBoomerAMG is broken if local size is zero
    int local_size = H_->Height();
    int min_size;
    MPI_Allreduce(&local_size, &min_size, 1, MPI_INT, MPI_MIN, comm_);

    const bool use_prec = min_size > 0;
    if (use_prec)
    {
        if (saamge_param_)
        {
            BuildSpectralAMGePreconditioner();
        }
        else if (H_->N() == mgL_.GetGraph().EdgeToTrueEdge().N())
        {
            auto temp_prec = make_unique<mfem::HypreBoomerAMG>(*H_);
            temp_prec->SetPrintLevel(0);
            prec_ = std::move(temp_prec);
        }
        else
        {
            std::unique_ptr<mfem::SparseMatrix> te_tm;
            {
                const auto& edge_trueedge = mgL_.GetGraph().EdgeToTrueEdge();
                const auto& edge_edof = mgL_.GetGraphSpace().EdgeToEDof();
                mfem::SparseMatrix e_te_diag = GetDiag(edge_trueedge);
                mfem::SparseMatrix m_tm_diag = GetDiag(*multiplier_d_td_);
                te_tm.reset(RAP(e_te_diag, edge_edof, m_tm_diag));
            }

            mfem::SparseMatrix PV_map(te_tm->NumCols(), te_tm->NumRows());

            std::vector<mfem::Array<int>> local_dofs(te_tm->NumRows());
            for (int i = 0; i < te_tm->NumRows(); ++i)
            {
                GetTableRow(*te_tm, i, local_dofs[i]);
                PV_map.Set(local_dofs[i][0], i, 1.0);
            }
            PV_map.Finalize();

            if (std::abs(rescale_iter_) > 0)
            {
                diagonal_scaling_.SetSize(multiplier_d_td_->NumCols());
                PV_map.ScaleRows(diagonal_scaling_);
                for (int i = 0; i < PV_map.NumCols(); ++i)
                {
                    assert(PV_map.GetData()[i] != 0.0);
                }
                diagonal_scaling_.SetSize(0);
            }

            prec_ = make_unique<AuxSpacePrec>(*H_, std::move(PV_map), local_dofs);
        }

        solver_->SetPreconditioner(*prec_);
    }
    if (myid_ == 0 && print_level_ > 0)
        std::cout << "  Timing: Preconditioner for hybridized system"
                  " constructed in " << chrono.RealTime() << "s. \n";
}

void HybridSolver::CollectEssentialDofs(const mfem::SparseMatrix& edof_bdrattr)
{
    mfem::SparseMatrix mult_truemult = GetDiag(*multiplier_d_td_);
    mfem::Array<int> true_multiplier;

    mult_on_bdr_.SetSize(num_multiplier_dofs_, false);

    // Note: there is a 1-1 map from multipliers to edge dofs on faces
    if (edof_bdrattr.Width())
    {
        ess_true_multipliers_.Reserve(edof_bdrattr.NumNonZeroElems());
        ess_true_mult_to_edof_.Reserve(edof_bdrattr.NumNonZeroElems());
        for (int i = 0; i < num_multiplier_dofs_; ++i)
        {
            // natural BC for H(div) dof <=> essential BC for multiplier dof
            if (edof_bdrattr.RowSize(i))
            {
                mult_on_bdr_[i] = true;
                if (ess_edofs_[i] == 0)
                {
                    GetTableRow(mult_truemult, i, true_multiplier);
                    assert(true_multiplier.Size() == 1);
                    ess_true_multipliers_.Append(true_multiplier);
                    ess_true_mult_to_edof_.Append(i);
                }
            }
        }
    }

    int num_local_ess_true_mult = ess_true_multipliers_.Size();
    int num_global_ess_true_mult;
    MPI_Allreduce(&num_local_ess_true_mult, &num_global_ess_true_mult,
                  1, MPI_INT, MPI_SUM, comm_);

    // In case of normal graph Laplacian, eliminate one multiplier
    if (!num_global_ess_true_mult && !W_is_nonzero_ && myid_ == 0)
    {
        GetTableRow(mult_truemult, 0, true_multiplier);
        assert(true_multiplier.Size() == 1);
        ess_true_multipliers_.Append(true_multiplier);
        ess_true_mult_to_edof_.Append(0);
    }
}

void HybridSolver::UpdateElemScaling(const mfem::Vector& elem_scaling_inverse)
{
    mfem::StopWatch chrono;
    chrono.Start();

    // This is for consistency, could simply work with elem_scaling_inverse
    elem_scaling_.SetSize(elem_scaling_inverse.Size());
    for (int i = 0; i < elem_scaling_.Size(); ++i)
    {
        elem_scaling_[i] = 1.0 / elem_scaling_inverse[i];
    }

    // TODO: this is not valid when W is nonzero
    assert(W_is_nonzero_ == false);

    mfem::SparseMatrix H_proc(num_multiplier_dofs_);
    mfem::Array<int> local_multiplier;
    for (int iAgg = 0; iAgg < nAggs_; ++iAgg)
    {
        GetTableRow(Agg_multiplier_, iAgg, local_multiplier);
        mfem::DenseMatrix H_el = Hybrid_el_[iAgg]; // deep copy
        H_el *= (1.0 / elem_scaling_(iAgg));
        H_proc.AddSubMatrix(local_multiplier, local_multiplier, H_el);

    }
    BuildParallelSystemAndSolver(H_proc);

    if (myid_ == 0 && print_level_ > 0)
    {
        std::cout << "  HybridSolver: rescaled system assembled in "
                  << chrono.RealTime() << "s. \n";
    }
}

void HybridSolver::UpdateJacobian(const mfem::Vector& elem_scaling_inverse,
                                  const std::vector<mfem::DenseMatrix>& N_el)
{
    mfem::StopWatch chrono;
    chrono.Start();

    is_symmetric_ = false;

    solver_ = InitKrylovSolver(KrylovMethod::GMRES);
    solver_->iterative_mode = false;

    auto H_proc = AssembleHybridSystem(elem_scaling_inverse, N_el);
    BuildParallelSystemAndSolver(H_proc);

    if (myid_ == 0 && print_level_ > 0)
    {
        std::cout << "  HybridSolver: rescaled system assembled in "
                  << chrono.RealTime() << "s. \n";
    }
}

mfem::Vector HybridSolver::MakeInitialGuess(const mfem::BlockVector& sol,
                                            const mfem::BlockVector& rhs) const
{
    const auto& Agg_vertexdof = mgL_.GetGraphSpace().VertexToVDof();
    const auto& Agg_edgedof = mgL_.GetGraphSpace().VertexToEDof();

    mfem::Vector mu(num_multiplier_dofs_);
    mu = 0.0;

    mfem::Array<int> local_vertexdof, local_edgedof, local_multiplier;
    mfem::Vector sigma_loc, u_loc, mu_loc, g_loc;
    for (int i = 0; i < nAggs_; ++i)
    {
        // Extracting the size and global numbering of local dof
        GetTableRow(Agg_vertexdof, i, local_vertexdof);
        GetTableRow(Agg_edgedof, i, local_edgedof);
        GetTableRow(Agg_multiplier_, i, local_multiplier);

        rhs.GetSubVector(local_edgedof, g_loc);
        for (int i = 0; i < local_edgedof.Size(); ++i)
        {
            if (edof_needs_averaging_[local_edgedof[i]])
            {
                g_loc[i] /= 2.0;
            }
        }
        mu_loc.SetSize(local_multiplier.Size());
        C_[i].Mult(g_loc, mu_loc);

        sol.GetSubVector(local_edgedof, sigma_loc);
        CM_[i].AddMult_a(-elem_scaling_[i], sigma_loc, mu_loc);

        sol.GetBlock(1).GetSubVector(local_vertexdof, u_loc);
        CDT_[i].AddMult(u_loc, mu_loc, -1.0);

        for (int j = 0; j < mu_loc.Size(); ++j)
        {
            int local_mult = local_multiplier[j];
            double CCT_i = mult_on_bdr_[local_mult] ? 1.0 : 2.0;
            mu[local_mult] += mu_loc[j] / CCT_i;
        }
    }

    mfem::Vector true_mu(H_->NumRows());
    multiplier_td_d_->Mult(mu, true_mu);
    return true_mu;
}

AuxSpacePrec::AuxSpacePrec(mfem::HypreParMatrix& op, mfem::SparseMatrix aux_map,
                           const std::vector<mfem::Array<int>>& loc_dofs)
    : mfem::Solver(op.NumRows(), false),
      local_dofs_(loc_dofs.size()),
      local_ops_(local_dofs_.size()),
      local_solvers_(local_dofs_.size()),
      op_(op),
      aux_map_(std::move(aux_map))
{
    op.GetDiag(op_diag_);
    for (unsigned int i = 0; i < local_dofs_.size(); ++i)
    {
        loc_dofs[i].Copy(local_dofs_[i]);
        local_ops_[i].SetSize(local_dofs_[i].Size());
        op_diag_.GetSubMatrix(local_dofs_[i], local_dofs_[i], local_ops_[i]);
        mfem::DenseMatrixInverse local_solver_i(local_ops_[i]);
        local_solver_i.GetInverseMatrix(local_solvers_[i]);
    }

    // Set up auxilary space solver
    mfem::Array<int> adof_starts;
    GenerateOffsets(op.GetComm(), aux_map_.NumCols(), adof_starts);
    int num_global_adofs = adof_starts.Last();
    mfem::HypreParMatrix par_aux_map(op.GetComm(), op.N(), num_global_adofs,
                                     op.ColPart(), adof_starts, &aux_map_);
    aux_op_.reset(smoothg::RAP(op, par_aux_map));
    aux_op_->CopyRowStarts();
    aux_op_->CopyColStarts();

    aux_solver_ = make_unique<mfem::HypreBoomerAMG>(*aux_op_);
    aux_solver_->SetPrintLevel(-1);
}

void AuxSpacePrec::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    y = 0.0;

    mfem::Vector x_aux, y_aux, residual(x), correction(y.Size());
    correction = 0.0;

    Smoothing(x, y);

    op_.Mult(-1.0, y, 1.0, residual);

    x_aux.SetSize(aux_map_.NumCols());
    y_aux.SetSize(aux_map_.NumCols());
    x_aux = 0.0;
    y_aux = 0.0;
    aux_map_.MultTranspose(residual, x_aux);
    aux_solver_->Mult(x_aux, y_aux);
    aux_map_.Mult(y_aux, correction);

    op_.Mult(-1.0, correction, 1.0, residual);
    y += correction;

    Smoothing(residual, correction);

    y += correction;
}

void AuxSpacePrec::Smoothing(const mfem::Vector& x, mfem::Vector& y) const
{
    // block Jacobi
    y = 0.0;
    mfem::Vector x_local, y_local;
    for (unsigned int i = 0; i < local_dofs_.size(); ++i)
    {
        x.GetSubVector(local_dofs_[i], x_local);
        y_local.SetSize(x_local.Size());
        local_solvers_[i].Mult(x_local, y_local);
        y.AddElementVector(local_dofs_[i], y_local);
    }
}



void TwoPhaseHybrid::Init()
{
    offsets_[0] = 0;
    offsets_[1] = multiplier_d_td_->NumCols();
    offsets_[2] = offsets_[1] + mgL_.NumVDofs();

    op_.reset(new mfem::BlockOperator(offsets_));
    op_->owns_blocks = true;

    stage1_prec_.reset(new mfem::BlockLowerTriangularPreconditioner(offsets_));

//    for (int agg = 0; agg < nAggs_; ++agg)
//    {
//        mfem::DenseMatrix AinvDMinv = smoothg::Mult(Ainv_[agg], DMinv_[agg]);
//        B01_[agg].Transpose(AinvDMinv);
//        B00_[agg] = smoothg::Mult(B01_[agg], DMinv_[agg]);
//        B00_[agg] -= Minv_ref_[agg];
//        B00_[agg] *= -1.0;
//    }

    solver_ = InitKrylovSolver(KrylovMethod::GMRES);
//    solver_->SetMaxIter(1);
    solver_->SetAbsTol(1e-8);
    solver_->SetRelTol(1e-6);
}

void TwoPhaseHybrid::AssembleSolver(mfem::Vector elem_scaling_inverse,
                                    const std::vector<mfem::DenseMatrix>& dMdS,
                                    const std::vector<mfem::DenseMatrix>& dTdsigma,
                                    const mfem::HypreParMatrix& dTdS,
                                    double dt_density)
{
//    dt_density_ = dt_density;

//    const auto& agg_vdof = mgL_.GetGraphSpace().VertexToVDof();

//    mfem::SparseMatrix A00(num_multiplier_dofs_);
//    mfem::SparseMatrix A01(num_multiplier_dofs_, mgL_.NumVDofs());
//    mfem::SparseMatrix A10(mgL_.NumVDofs(), num_multiplier_dofs_);
//    mfem::SparseMatrix A11_tmp(mgL_.NumVDofs());

//    mfem::DenseMatrix A00_el, A01_el, A10_el, A11_el, help;
//    mfem::Array<int> local_vdof, local_mult;

//    for (int agg = 0; agg < nAggs_; ++agg)
//    {
//        elem_scaling_[agg] = 1.0 / elem_scaling_inverse[agg];

//        GetTableRow(agg_vdof, agg, local_vdof);
//        GetTableRow(Agg_multiplier_, agg, local_mult);

//        A00_el = Hybrid_el_[agg];
//        A00_el *= elem_scaling_inverse[agg];
////        A00_el *= dt_density_;

//        help = smoothg::Mult(C_[agg], B00_[agg]);
//        help *= elem_scaling_inverse[agg];
//        A01_el = smoothg::Mult(help, dMdS[agg]);

//        help.Transpose();
//        A10_el = smoothg::Mult(dTdsigma[agg], help);
////        A10_el *= (dt_density_ * dt_density_);

//        help = smoothg::Mult(dTdsigma[agg], B00_[agg]);
//        A11_el = smoothg::Mult(help, dMdS[agg]);
//        A11_el *= elem_scaling_inverse[agg];
////        A11_el *= (dt_density_);

//        A00.AddSubMatrix(local_mult, local_mult, A00_el);
//        A01.AddSubMatrix(local_mult, local_vdof, A01_el);
//        A10.AddSubMatrix(local_vdof, local_mult, A10_el);
//        A11_tmp.AddSubMatrix(local_vdof, local_vdof, A11_el);
//    }

//    A00.Finalize();
//    A01.Finalize();
//    A10.Finalize();
//    A11_tmp.Finalize();

//    auto dTdS_diag = GetDiag(dTdS);
//    unique_ptr<mfem::SparseMatrix> A11(Add(1.0, A11_tmp, -1.0, dTdS_diag));
//    A11->MoveDiagonalFirst();
//    unique_ptr<mfem::HypreParMatrix> pA11(ToParMatrix(comm_, *A11));


//    BuildParallelSystemAndSolver(A00); // pA00 and A00_inv store in H_ and prec_

//    auto Scale = VectorToMatrix(diagonal_scaling_);
//    mfem::HypreParMatrix pScale(comm_, H_->N(), H_->GetColStarts(), &Scale);

//    A10_elim_.reset(new mfem::SparseMatrix(GetEliminatedCols(A10, ess_true_multipliers_)));
//    for (auto mult : ess_true_multipliers_)
//    {
//        A01.EliminateRow(mult);
//        A10.EliminateCol(mult);
//    }

//    auto pA01_tmp = ParMult(*multiplier_td_d_, A01, mgL_.GetGraph().VertexStarts());
//    auto pA10_tmp = ParMult(A10, *multiplier_d_td_, mgL_.GetGraph().VertexStarts());

//    auto pA01 = mfem::ParMult(&pScale, pA01_tmp.get());
//    auto pA10 = mfem::ParMult(pA10_tmp.get(), &pScale);

//    A11_inv_.reset(new mfem::HypreSmoother(*pA11, mfem::HypreSmoother::l1Jacobi));
//    A00_inv_.reset(prec_.release());
//    stage1_prec_->SetDiagonalBlock(0, A00_inv_.get());
//    stage1_prec_->SetDiagonalBlock(1, A11_inv_.get());
//    stage1_prec_->SetBlock(1, 0, pA10);

//    op_->SetBlock(0, 0, H_.release());
//    op_->SetBlock(0, 1, pA01);
//    op_->SetBlock(1, 0, pA10);
//    op_->SetBlock(1, 1, pA11.release());

//    mono_mat_ = BlockGetDiag(*op_);
//    monolithic_.reset(ToParMatrix(comm_, *mono_mat_));
//    stage2_prec_.reset(new HypreILU(*monolithic_, 0));

////    prec_ = std::move(stage1_prec_);//
//    prec_.reset(new TwoStageSolver(*stage1_prec_, *stage2_prec_, *op_));

//    solver_->SetPreconditioner(*prec_);
//    solver_->SetOperator(*op_);
//    dynamic_cast<mfem::GMRESSolver*>(solver_.get())->SetKDim(100);

    dTdsigma_ = &dTdsigma;
    dMdS_ = &dMdS;
    dTdS_ = &dTdS;
}

mfem::BlockVector TwoPhaseHybrid::MakeHybridRHS(const mfem::BlockVector& rhs) const
{
    const auto& agg_vdof = mgL_.GetGraphSpace().VertexToVDof();
    const auto& agg_edof = mgL_.GetGraphSpace().VertexToEDof();

    mfem::BlockVector out(offsets_);
    out.GetBlock(0) = 0.0;
    out.GetBlock(1).Set(-1.0, rhs.GetBlock(2));

    mfem::Array<int> local_vdof, local_edof, local_mult;
    mfem::Vector local_rhs, sub_vec, helper; // helper = B00 * rhs0 + B01 * rhs1
    for (int agg = 0; agg < nAggs_; ++agg)
    {
        GetTableRow(agg_vdof, agg, local_vdof);
        GetTableRow(agg_edof, agg, local_edof);
        GetTableRow(Agg_multiplier_, agg, local_mult);

        rhs.GetSubVector(local_edof, sub_vec);
        for (int i = 0; i < local_edof.Size(); ++i)
        {
            if (edof_needs_averaging_[local_edof[i]])
            {
                sub_vec[i] /= 2.0;
            }
        }

        helper.SetSize(local_edof.Size());
        B00_[agg].Mult(sub_vec, helper);
        helper /= elem_scaling_[agg];
//        helper *= dt_density_;

        rhs.GetBlock(1).GetSubVector(local_vdof, sub_vec);
        B01_[agg].AddMult_a(1.0 / dt_density_, sub_vec, helper);

        local_rhs.SetSize(local_mult.Size());
        C_[agg].Mult(helper, local_rhs);
        out.AddElementVector(local_mult, local_rhs);

        local_rhs.SetSize(local_vdof.Size());
        (*dTdsigma_)[agg].Mult(helper, local_rhs);
//        local_rhs *= dt_density_;
        out.GetBlock(1).AddElementVector(local_vdof, local_rhs);
    }

    return out;
}

void TwoPhaseHybrid::BackSubstitute(const mfem::BlockVector& rhs,
                                    const mfem::BlockVector& sol_hb,
                                    mfem::BlockVector& sol) const
{
    const auto& agg_vdof = mgL_.GetGraphSpace().VertexToVDof();
    const auto& agg_edof = mgL_.GetGraphSpace().VertexToEDof();

    sol.GetBlock(0) = 0.0;
    sol.GetBlock(1) = 0.0;
    sol.GetBlock(2) = sol_hb.GetBlock(1);

    mfem::Array<int> local_vdof, local_edof, local_mult;
    mfem::Vector local_sol0, local_sol1, sub_vec, helper;
    for (int agg = 0; agg < nAggs_; ++agg)
    {
        GetTableRow(agg_vdof, agg, local_vdof);
        GetTableRow(agg_edof, agg, local_edof);
        GetTableRow(Agg_multiplier_, agg, local_mult);

        local_sol0.SetSize(local_edof.Size());
        local_sol1.SetSize(local_vdof.Size());

        rhs.GetBlock(1).GetSubVector(local_vdof, sub_vec);
        B01_[agg].Mult(sub_vec, local_sol0);
        Ainv_[agg].Mult(sub_vec, local_sol1);
        local_sol1 *= elem_scaling_[agg];

//        local_sol0 /= dt_density_;
//        local_sol1 /= (-1.0 * dt_density_);

        rhs.GetSubVector(local_edof, helper);
        for (int i = 0; i < local_edof.Size(); ++i)
        {
            if (edof_needs_averaging_[local_edof[i]])
            {
                helper[i] /= 2.0;
            }
        }

        sol_hb.GetBlock(1).GetSubVector(local_vdof, sub_vec);
        (*dMdS_)[agg].AddMult_a(-1.0 / dt_density_, sub_vec, helper);

        sol_hb.GetSubVector(local_mult, sub_vec);
        C_[agg].AddMultTranspose(sub_vec, helper, -1.0);

        B00_[agg].AddMult_a(dt_density_* 1.0 / elem_scaling_[agg], helper, local_sol0);
        B01_[agg].AddMultTranspose_a(dt_density_, helper, local_sol1);

        for (int i = 0; i < local_edof.Size(); ++i)
        {
            if (edof_needs_averaging_[local_edof[i]])
            {
                local_sol0[i] /= 2.0;
            }
        }

        sol.AddElementVector(local_edof, local_sol0);
        sol.GetBlock(1).AddElementVector(local_vdof, local_sol1);
    }
}

void TwoPhaseHybrid::Mult(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const
{
    mfem::BlockVector sol_hb(offsets_);
    sol_hb = 0.0;

    mfem::BlockVector rhs_copy(rhs);
    for (int m = 0; m < ess_true_multipliers_.Size(); ++m)
    {
        sol_hb(ess_true_multipliers_[m]) = -rhs(ess_true_mult_to_edof_[m]);
        rhs_copy(ess_true_mult_to_edof_[m]) = 0.0;
    }

    mfem::BlockVector rhs_hb = MakeHybridRHS(rhs_copy);

    H_elim_->Mult(-1.0, sol_hb.GetBlock(0), 1.0, rhs_hb.GetBlock(0));
    A10_elim_->AddMult(sol_hb.GetBlock(0), rhs_hb.GetBlock(1), -1.0);

    for (int ess_true_mult : const_cast<mfem::Array<int>&>(ess_true_multipliers_))
    {
        rhs_hb(ess_true_mult) = sol_hb(ess_true_mult);
    }

    solver_->Mult(rhs_hb, sol_hb);
    BackSubstitute(rhs_copy, sol_hb, sol);
    num_iterations_ = solver_->GetNumIterations();
    resid_norm_ = solver_->GetFinalNorm();
}


void TwoPhaseHybrid::Mult2(const mfem::BlockVector& rhs, mfem::BlockVector& sol)
{
    int num_hatdofs = mgL_.GetGraphSpace().VertexToEDof().NumNonZeroElems();

    mfem::Array<int> bos(3), darcy_bos(3);
    bos[0] = 0;
    bos[1] = bos[0] + multiplier_d_td_->NumCols();
    bos[2] = bos[1] + mgL_.GetGraphSpace().VertexToVDof().NumCols();

    darcy_bos[0] = 0;
    darcy_bos[1] = darcy_bos[0] + num_hatdofs;
    darcy_bos[2] = darcy_bos[1] + mgL_.GetGraphSpace().VertexToVDof().NumCols();

    auto& vert_edof = mgL_.GetGraphSpace().VertexToEDof();
    auto& vert_vdof = mgL_.GetGraphSpace().VertexToVDof();
    auto mbuilder = dynamic_cast<const ElementMBuilder*>(&(mgL_.GetMBuilder()));
    auto& M_el = mbuilder->GetElementMatrices();

    mfem::BlockVector sol_hb(offsets_);
    sol_hb = 0.0;

    mfem::BlockVector rhs_copy(rhs);
    for (int m = 0; m < ess_true_multipliers_.Size(); ++m)
    {
        sol_hb(ess_true_multipliers_[m]) = -rhs(ess_true_mult_to_edof_[m]);
        rhs_copy(ess_true_mult_to_edof_[m]) = 0.0;
    }

    mfem::BlockVector darcy_rhs(darcy_bos), rhs_debug(bos);
    mfem::Vector helper;

    mfem::Array<int> local_edofs, local_vdofs, local_mult,
            local_hat, local_special_vdofs, local_dofs_helper;

    mfem::SparseMatrix dTdsigma_hat(mgL_.NumVDofs(), num_hatdofs);
    mfem::SparseMatrix dMdS_hat(num_hatdofs, mgL_.NumVDofs());
//    mfem::SparseMatrix C_hat(mgL_.NumEDofs(), num_hatdofs);

    mfem::SparseMatrix darcy_hat_inv_00(num_hatdofs);
    mfem::SparseMatrix darcy_hat_inv_01(num_hatdofs, mgL_.NumVDofs());
    mfem::SparseMatrix darcy_hat_inv_10(mgL_.NumVDofs(), num_hatdofs);
    mfem::SparseMatrix darcy_hat_inv_11(mgL_.NumVDofs());

    int num_injectors = mgL_.NumInjectors();
    auto& inj_cells = mgL_.GetInjectorCells();

    mfem::DenseMatrix DenseDloc, DenseCloc;
    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        GetTableRow(vert_vdof, i, local_vdofs);
        GetTableRow(Agg_multiplier_, i, local_mult);
        local_hat.SetSize(local_edofs.Size());
        std::iota(local_hat.begin(), local_hat.end(), vert_edof.GetI()[i]);

        auto Dloc = ExtractRowAndColumns(mgL_.GetD(), local_vdofs, local_edofs);
        Full(Dloc, DenseDloc);
//        Full(C_[i], DenseCloc);

        mfem::DenseMatrix M_el_i(M_el[i]);
        if (i < vert_edof.NumRows() - num_injectors)
        {
//            M_el_i *= (1.0 / sol.GetBlock(2)[i]);
            dMdS_hat.AddSubMatrix(local_hat, local_vdofs, (*dMdS_)[i]);
        }
        else
        {
            mfem::Vector scale(local_edofs.Size());
            auto& cells = inj_cells[i-vert_edof.NumRows()+num_injectors];
            local_special_vdofs.SetSize(0);
            for (int j = 0; j < scale.Size(); ++j)
            {
                scale[j] = 1.0 / sol.GetBlock(2)[cells[j]];
                GetTableRow(vert_vdof, cells[j], local_dofs_helper);
                local_special_vdofs.Append(local_dofs_helper);
            }
            M_el_i.LeftScaling(scale);
            dMdS_hat.AddSubMatrix(local_hat, local_special_vdofs, (*dMdS_)[i]);
        }

        dTdsigma_hat.AddSubMatrix(local_vdofs, local_hat, (*dTdsigma_)[i]);
//        C_hat.AddSubMatrix(local_mult, local_hat, DenseCloc);


        mfem::DenseMatrix darcy_el_inv, inv_00, inv_01, inv_10, inv_11;

        if (i < vert_edof.NumRows() - num_injectors)
        {
            inv_00 = darcy_inv_00_[i];
            inv_00 *= sol.GetBlock(2)[i];
            inv_11 = darcy_inv_11_[i];
            inv_11 *= (1.0 / sol.GetBlock(2)[i]);

            darcy_hat_inv_01.AddSubMatrix(local_hat, local_vdofs, darcy_inv_01_[i]);
            darcy_hat_inv_10.AddSubMatrix(local_vdofs, local_hat, darcy_inv_10_[i]);
        }
        else
        {
            mfem::DenseMatrix darcy_el(DenseDloc.NumCols() + DenseDloc.NumRows());
            darcy_el.CopyMN(M_el_i, 0, 0);
            darcy_el.CopyMN(DenseDloc, DenseDloc.NumCols(), 0);
            darcy_el.CopyMNt(DenseDloc, 0, DenseDloc.NumCols());

            mfem::DenseMatrixInverse darcy_el_solver(darcy_el);
            darcy_el_solver.GetInverseMatrix(darcy_el_inv);

            inv_00.CopyMN(darcy_el_inv, local_hat.Size(), local_hat.Size(), 0, 0);
            inv_01.CopyMN(darcy_el_inv, local_hat.Size(),
                          local_vdofs.Size(), 0, local_hat.Size());
            inv_10.CopyMN(darcy_el_inv, local_vdofs.Size(),
                          local_hat.Size(), local_hat.Size(), 0);
            inv_11.CopyMN(darcy_el_inv, local_vdofs.Size(), local_vdofs.Size(),
                          local_hat.Size(), local_hat.Size());

            darcy_hat_inv_01.AddSubMatrix(local_hat, local_vdofs, inv_01);
            darcy_hat_inv_10.AddSubMatrix(local_vdofs, local_hat, inv_10);
        }

        darcy_hat_inv_00.AddSubMatrix(local_hat, local_hat, inv_00);
        darcy_hat_inv_11.AddSubMatrix(local_vdofs, local_vdofs, inv_11);

        rhs_copy.GetSubVector(local_edofs, helper);

        for (int j =0; j < local_edofs.Size(); ++j)
        {
            if (edof_needs_averaging_[local_edofs[j]])
            {
                helper[j] /= 2.0;
            }
        }
        darcy_rhs.GetBlock(0).SetSubVector(local_hat, helper);
    }
    dTdsigma_hat.Finalize();
    dMdS_hat.Finalize();
//    C_hat.Finalize();

    mfem::BlockMatrix darcy_hat_inv(darcy_bos);
    darcy_hat_inv.SetBlock(0, 0, &darcy_hat_inv_00);
    darcy_hat_inv.SetBlock(0, 1, &darcy_hat_inv_01);
    darcy_hat_inv.SetBlock(1, 0, &darcy_hat_inv_10);
    darcy_hat_inv.SetBlock(1, 1, &darcy_hat_inv_11);
    darcy_hat_inv.Finalize();

    mfem::SparseMatrix CT_proc = smoothg::Transpose(*C_proc_);

    mfem::BlockMatrix left_op(bos, darcy_bos);
    left_op.SetBlock(0, 0, C_proc_.get());
    left_op.SetBlock(1, 0, &dTdsigma_hat);

    mfem::BlockMatrix right_op(darcy_bos, bos);
    right_op.SetBlock(0, 0, &CT_proc);
    right_op.SetBlock(0, 1, &dMdS_hat);

    unique_ptr<mfem::BlockMatrix> left_tmp(mfem::Mult(left_op, darcy_hat_inv));
    unique_ptr<mfem::BlockMatrix> op_debug(mfem::Mult(*left_tmp, right_op));


    mfem::SparseMatrix dTdS = GetDiag(*dTdS_);
    mfem::SparseMatrix* op_debug_11 = &(op_debug->GetBlock(1, 1));
    mfem::SparseMatrix* op_11 = mfem::Add(1., *op_debug_11, -1., dTdS);
    op_debug->SetBlock(1, 1, op_11);
    delete op_debug_11;
    unique_ptr<mfem::SparseMatrix> op_sp(op_debug->CreateMonolithic());

    mfem::SparseMatrix op_elim = GetEliminatedCols(*op_sp, ess_true_multipliers_);

    for (int j = 0; j < ess_true_multipliers_.Size(); ++j)
    {
        op_debug->EliminateRowCol(ess_true_multipliers_[j]);
        op_sp->EliminateRow(ess_true_multipliers_[j]);
        op_sp->EliminateCol(ess_true_multipliers_[j]);
        op_sp->Set(ess_true_multipliers_[j], ess_true_multipliers_[j], 1.0);
    }


    darcy_rhs.GetBlock(1) = rhs_copy.GetBlock(1);


    rhs_debug.GetBlock(0) = 0.0;
    rhs_debug.GetBlock(1).Set(-1.0, rhs_copy.GetBlock(2));

    left_tmp->AddMult(darcy_rhs, rhs_debug);


    op_elim.AddMult(sol_hb, rhs_debug, -1.0);

    for (int ess_true_mult : const_cast<mfem::Array<int>&>(ess_true_multipliers_))
    {
        rhs_debug(ess_true_mult) = sol_hb(ess_true_mult);
    }


//    unique_ptr<mfem::HypreParMatrix> pA00(ToParMatrix(comm_, op_debug->GetBlock(0, 0)));
//    auto prec_00 = new mfem::HypreBoomerAMG(*pA00);
//    prec_00->SetPrintLevel(0);

    BuildParallelSystemAndSolver(op_debug->GetBlock(0, 0));
    A00_inv_.reset(prec_.release());
    stage1_prec_->SetDiagonalBlock(0, A00_inv_.get());

    unique_ptr<mfem::HypreParMatrix> pA11(ToParMatrix(comm_, *op_11));
    A11_inv_.reset(new mfem::HypreSmoother(*pA11, mfem::HypreSmoother::l1Jacobi));
    stage1_prec_->SetDiagonalBlock(1, A11_inv_.get());
    stage1_prec_->SetBlock(1, 0, &(op_debug->GetBlock(1, 0)));

    monolithic_.reset(ToParMatrix(comm_, *op_sp));
    stage2_prec_.reset(new HypreILU(*monolithic_, 0));

    prec_.reset(new TwoStageSolver(*stage1_prec_, *stage2_prec_, *op_sp));
    solver_->SetPreconditioner(*prec_);
    solver_->SetOperator(*op_sp);


    solver_->Mult(rhs_debug, sol_hb);
    num_iterations_ = solver_->GetNumIterations();
    resid_norm_ = solver_->GetFinalNorm();


    mfem::BlockVector darcy_sol_tmp(darcy_rhs), darcy_sol(darcy_bos);
    right_op.AddMult(sol_hb, darcy_sol_tmp, -1.0);
    darcy_hat_inv.Mult(darcy_sol_tmp, darcy_sol);

    sol.GetBlock(1) = darcy_sol.GetBlock(1);
    sol.GetBlock(2) = sol_hb.GetBlock(1);

    sol.GetBlock(0) = 0.0;
    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        local_hat.SetSize(local_edofs.Size());
        std::iota(local_hat.begin(), local_hat.end(), vert_edof.GetI()[i]);

        darcy_sol.GetSubVector(local_hat, helper);
        for (int j = 0; j < local_edofs.Size(); ++j)
        {
            if (edof_needs_averaging_[local_edofs[j]])
            {
                helper[j] /= 2.0;
            }
        }
        sol.AddElementVector(local_edofs, helper);
    }
}


void TwoPhaseHybrid::Mult3(const mfem::BlockVector& rhs, mfem::BlockVector& sol)
{
    auto& vert_edof = mgL_.GetGraphSpace().VertexToEDof();
    auto& vert_vdof = mgL_.GetGraphSpace().VertexToVDof();
    auto mbuilder = dynamic_cast<const ElementMBuilder*>(&(mgL_.GetMBuilder()));
    auto& M_el = mbuilder->GetElementMatrices();

    int num_hatdofs = mgL_.GetGraphSpace().VertexToEDof().NumNonZeroElems();
    int num_injectors = mgL_.NumInjectors();
    auto& inj_cells = mgL_.GetInjectorCells();
    int num_inj_flux = num_hatdofs - vert_edof.GetI()[vert_edof.NumRows()-num_injectors];

    mfem::Array<int> bos(5), darcy_bos(3);
    bos[0] = 0;
    bos[1] = bos[0] + multiplier_d_td_->NumCols();
    bos[2] = bos[1] + num_inj_flux;
    bos[3] = bos[2] + num_injectors;
    bos[4] = bos[3] + mgL_.GetGraphSpace().VertexToVDof().NumCols();

    darcy_bos[0] = 0;
    darcy_bos[1] = darcy_bos[0] + num_hatdofs;
    darcy_bos[2] = darcy_bos[1] + mgL_.GetGraphSpace().VertexToVDof().NumCols();

    mfem::BlockVector sol_hb(offsets_);
    sol_hb = 0.0;

    mfem::BlockVector rhs_copy(rhs);
    for (int m = 0; m < ess_true_multipliers_.Size(); ++m)
    {
        sol_hb(ess_true_multipliers_[m]) = -rhs(ess_true_mult_to_edof_[m]);
        rhs_copy(ess_true_mult_to_edof_[m]) = 0.0;
    }

    mfem::BlockVector darcy_rhs(darcy_bos), rhs_debug(bos);
    mfem::Vector helper;

    mfem::Array<int> local_edofs, local_vdofs, local_mult,
            local_hat, local_special_vdofs, local_dofs_helper;

    mfem::SparseMatrix dTdsigma_hat(mgL_.NumVDofs(), num_hatdofs);
    mfem::SparseMatrix dMdS_hat(num_hatdofs, mgL_.NumVDofs());
//    mfem::SparseMatrix C_hat(mgL_.NumEDofs(), num_hatdofs);

    mfem::SparseMatrix darcy_hat_inv_00(num_hatdofs);
    mfem::SparseMatrix darcy_hat_inv_01(num_hatdofs, mgL_.NumVDofs());
    mfem::SparseMatrix darcy_hat_inv_10(mgL_.NumVDofs(), num_hatdofs);
    mfem::SparseMatrix darcy_hat_inv_11(mgL_.NumVDofs());


    mfem::DenseMatrix DenseDloc, DenseCloc;
    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        GetTableRow(vert_vdof, i, local_vdofs);
        GetTableRow(Agg_multiplier_, i, local_mult);
        local_hat.SetSize(local_edofs.Size());
        std::iota(local_hat.begin(), local_hat.end(), vert_edof.GetI()[i]);

        auto Dloc = ExtractRowAndColumns(mgL_.GetD(), local_vdofs, local_edofs);
        Full(Dloc, DenseDloc);
//        Full(C_[i], DenseCloc);

        mfem::DenseMatrix M_el_i(M_el[i]);
        if (i < vert_edof.NumRows() - num_injectors)
        {
//            M_el_i *= (1.0 / sol.GetBlock(2)[i]);
            dMdS_hat.AddSubMatrix(local_hat, local_vdofs, (*dMdS_)[i]);
        }
        else
        {
            mfem::Vector scale(local_edofs.Size());
            auto& cells = inj_cells[i-vert_edof.NumRows()+num_injectors];
            local_special_vdofs.SetSize(0);
            for (int j = 0; j < scale.Size(); ++j)
            {
                scale[j] = 1.0 / sol.GetBlock(2)[cells[j]];
                GetTableRow(vert_vdof, cells[j], local_dofs_helper);
                local_special_vdofs.Append(local_dofs_helper);
            }
            M_el_i.LeftScaling(scale);
            dMdS_hat.AddSubMatrix(local_hat, local_special_vdofs, (*dMdS_)[i]);
        }

        dTdsigma_hat.AddSubMatrix(local_vdofs, local_hat, (*dTdsigma_)[i]);
//        C_hat.AddSubMatrix(local_mult, local_hat, DenseCloc);


        mfem::DenseMatrix darcy_el_inv, inv_00, inv_01, inv_10, inv_11;

        if (i < vert_edof.NumRows() - num_injectors)
        {
            inv_00 = darcy_inv_00_[i];
            inv_00 *= sol.GetBlock(2)[i];
            inv_11 = darcy_inv_11_[i];
            inv_11 *= (1.0 / sol.GetBlock(2)[i]);

            darcy_hat_inv_01.AddSubMatrix(local_hat, local_vdofs, darcy_inv_01_[i]);
            darcy_hat_inv_10.AddSubMatrix(local_vdofs, local_hat, darcy_inv_10_[i]);
        }
        else
        {
            mfem::DenseMatrix darcy_el(DenseDloc.NumCols() + DenseDloc.NumRows());
            darcy_el.CopyMN(M_el_i, 0, 0);
            darcy_el.CopyMN(DenseDloc, DenseDloc.NumCols(), 0);
            darcy_el.CopyMNt(DenseDloc, 0, DenseDloc.NumCols());

            mfem::DenseMatrixInverse darcy_el_solver(darcy_el);
            darcy_el_solver.GetInverseMatrix(darcy_el_inv);

            inv_00.CopyMN(darcy_el_inv, local_hat.Size(), local_hat.Size(), 0, 0);
            inv_01.CopyMN(darcy_el_inv, local_hat.Size(),
                          local_vdofs.Size(), 0, local_hat.Size());
            inv_10.CopyMN(darcy_el_inv, local_vdofs.Size(),
                          local_hat.Size(), local_hat.Size(), 0);
            inv_11.CopyMN(darcy_el_inv, local_vdofs.Size(), local_vdofs.Size(),
                          local_hat.Size(), local_hat.Size());

            darcy_hat_inv_01.AddSubMatrix(local_hat, local_vdofs, inv_01);
            darcy_hat_inv_10.AddSubMatrix(local_vdofs, local_hat, inv_10);
        }

        darcy_hat_inv_00.AddSubMatrix(local_hat, local_hat, inv_00);
        darcy_hat_inv_11.AddSubMatrix(local_vdofs, local_vdofs, inv_11);

        rhs_copy.GetSubVector(local_edofs, helper);

        for (int j =0; j < local_edofs.Size(); ++j)
        {
            if (edof_needs_averaging_[local_edofs[j]])
            {
                helper[j] /= 2.0;
            }
        }
        darcy_rhs.GetBlock(0).SetSubVector(local_hat, helper);
    }
    dTdsigma_hat.Finalize();
    dMdS_hat.Finalize();
//    C_hat.Finalize();

    mfem::BlockMatrix darcy_hat_inv(darcy_bos);
    darcy_hat_inv.SetBlock(0, 0, &darcy_hat_inv_00);
    darcy_hat_inv.SetBlock(0, 1, &darcy_hat_inv_01);
    darcy_hat_inv.SetBlock(1, 0, &darcy_hat_inv_10);
    darcy_hat_inv.SetBlock(1, 1, &darcy_hat_inv_11);
    darcy_hat_inv.Finalize();

    mfem::SparseMatrix CT_proc = smoothg::Transpose(*C_proc_);

    mfem::BlockMatrix left_op(bos, darcy_bos);
    left_op.SetBlock(0, 0, C_proc_.get());
    left_op.SetBlock(1, 0, &dTdsigma_hat);

    mfem::BlockMatrix right_op(darcy_bos, bos);
    right_op.SetBlock(0, 0, &CT_proc);
    right_op.SetBlock(0, 1, &dMdS_hat);

    unique_ptr<mfem::BlockMatrix> left_tmp(mfem::Mult(left_op, darcy_hat_inv));
    unique_ptr<mfem::BlockMatrix> op_debug(mfem::Mult(*left_tmp, right_op));


    mfem::SparseMatrix dTdS = GetDiag(*dTdS_);
    mfem::SparseMatrix* op_debug_11 = &(op_debug->GetBlock(1, 1));
    mfem::SparseMatrix* op_11 = mfem::Add(1., *op_debug_11, -1., dTdS);
    op_debug->SetBlock(1, 1, op_11);
    delete op_debug_11;
    unique_ptr<mfem::SparseMatrix> op_sp(op_debug->CreateMonolithic());

    mfem::SparseMatrix op_elim = GetEliminatedCols(*op_sp, ess_true_multipliers_);

    for (int j = 0; j < ess_true_multipliers_.Size(); ++j)
    {
        op_debug->EliminateRowCol(ess_true_multipliers_[j]);
        op_sp->EliminateRow(ess_true_multipliers_[j]);
        op_sp->EliminateCol(ess_true_multipliers_[j]);
        op_sp->Set(ess_true_multipliers_[j], ess_true_multipliers_[j], 1.0);
    }


    darcy_rhs.GetBlock(1) = rhs_copy.GetBlock(1);


    rhs_debug.GetBlock(0) = 0.0;
    rhs_debug.GetBlock(1).Set(-1.0, rhs_copy.GetBlock(2));

    left_tmp->AddMult(darcy_rhs, rhs_debug);


    op_elim.AddMult(sol_hb, rhs_debug, -1.0);

    for (int ess_true_mult : const_cast<mfem::Array<int>&>(ess_true_multipliers_))
    {
        rhs_debug(ess_true_mult) = sol_hb(ess_true_mult);
    }


//    unique_ptr<mfem::HypreParMatrix> pA00(ToParMatrix(comm_, op_debug->GetBlock(0, 0)));
//    auto prec_00 = new mfem::HypreBoomerAMG(*pA00);
//    prec_00->SetPrintLevel(0);

    BuildParallelSystemAndSolver(op_debug->GetBlock(0, 0));
    A00_inv_.reset(prec_.release());
    stage1_prec_->SetDiagonalBlock(0, A00_inv_.get());

    unique_ptr<mfem::HypreParMatrix> pA11(ToParMatrix(comm_, *op_11));
    A11_inv_.reset(new mfem::HypreSmoother(*pA11, mfem::HypreSmoother::l1Jacobi));
    stage1_prec_->SetDiagonalBlock(1, A11_inv_.get());
    stage1_prec_->SetBlock(1, 0, &(op_debug->GetBlock(1, 0)));

    monolithic_.reset(ToParMatrix(comm_, *op_sp));
    stage2_prec_.reset(new HypreILU(*monolithic_, 0));

    prec_.reset(new TwoStageSolver(*stage1_prec_, *stage2_prec_, *op_sp));
    solver_->SetPreconditioner(*prec_);
    solver_->SetOperator(*op_sp);


    solver_->Mult(rhs_debug, sol_hb);
    num_iterations_ = solver_->GetNumIterations();
    resid_norm_ = solver_->GetFinalNorm();


    mfem::BlockVector darcy_sol_tmp(darcy_rhs), darcy_sol(darcy_bos);
    right_op.AddMult(sol_hb, darcy_sol_tmp, -1.0);
    darcy_hat_inv.Mult(darcy_sol_tmp, darcy_sol);

    sol.GetBlock(1) = darcy_sol.GetBlock(1);
    sol.GetBlock(2) = sol_hb.GetBlock(1);

    sol.GetBlock(0) = 0.0;
    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        local_hat.SetSize(local_edofs.Size());
        std::iota(local_hat.begin(), local_hat.end(), vert_edof.GetI()[i]);

        darcy_sol.GetSubVector(local_hat, helper);
        for (int j = 0; j < local_edofs.Size(); ++j)
        {
            if (edof_needs_averaging_[local_edofs[j]])
            {
                helper[j] /= 2.0;
            }
        }
        sol.AddElementVector(local_edofs, helper);
    }
}

void TwoPhaseHybrid::DebugMult(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const
{
    int num_hatdofs = mgL_.GetGraphSpace().VertexToEDof().NumNonZeroElems();

    mfem::Array<int> bos(5);
    bos[0] = 0;
    bos[1] = bos[0] + num_hatdofs;
    bos[2] = bos[1] + mgL_.GetGraphSpace().VertexToVDof().NumCols();
    bos[3] = bos[2] + mgL_.GetGraphSpace().VertexToVDof().NumCols();
    bos[4] = bos[3] + multiplier_d_td_->NumCols();


    bos.Print();

    auto& vert_edof = mgL_.GetGraphSpace().VertexToEDof();
    auto& vert_vdof = mgL_.GetGraphSpace().VertexToVDof();
    auto mbuilder = dynamic_cast<const ElementMBuilder*>(&(mgL_.GetMBuilder()));
    auto M_el = mbuilder->GetElementMatrices();

    mfem::Vector rhs_blk0(rhs.GetBlock(0));
//    for (int m = 0; m < ess_true_multipliers_.Size(); ++m)
//    {
//        rhs_blk0(ess_true_mult_to_edof_[m]) = 0.0;
//    }

    mfem::BlockVector rhs_debug(bos), sol_debug(bos);
    mfem::Vector helper;

    mfem::Array<int> local_edofs, local_vdofs, local_mult, local_hat;
    mfem::Array<int> ess_hat(num_hatdofs);
    ess_hat = 0;

    mfem::SparseMatrix M_hat(num_hatdofs, num_hatdofs);
    mfem::SparseMatrix dTdsigma_hat(mgL_.NumVDofs(), num_hatdofs);
    mfem::SparseMatrix dMdS_hat(num_hatdofs, mgL_.NumVDofs());
    mfem::SparseMatrix D_hat(mgL_.NumVDofs(), num_hatdofs);
    mfem::SparseMatrix C_hat(mgL_.NumEDofs(), num_hatdofs);

    mfem::DenseMatrix DenseDloc, DenseCloc;
    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        GetTableRow(vert_vdof, i, local_vdofs);
        GetTableRow(Agg_multiplier_, i, local_mult);
        local_hat.SetSize(local_edofs.Size());
        std::iota(local_hat.begin(), local_hat.end(), vert_edof.GetI()[i]);

        auto Dloc = ExtractRowAndColumns(mgL_.GetD(), local_vdofs, local_edofs);
        Full(Dloc, DenseDloc);
        Full(C_[i], DenseCloc);

        mfem::DenseMatrix M_el_i(M_el[i]);
        M_el_i *= (1.0 / sol.GetBlock(2)[i]);
        M_hat.AddSubMatrix(local_hat, local_hat, M_el_i);
        dTdsigma_hat.AddSubMatrix(local_vdofs, local_hat, (*dTdsigma_)[i]);
        dMdS_hat.AddSubMatrix(local_hat, local_vdofs, (*dMdS_)[i]);
        D_hat.AddSubMatrix(local_vdofs, local_hat, DenseDloc);
        C_hat.AddSubMatrix(local_mult, local_hat, DenseCloc);

        rhs_blk0.GetSubVector(local_edofs, helper);

        for (int j =0; j < local_edofs.Size(); ++j)
        {
            if (edof_needs_averaging_[local_edofs[j]])
            {
                helper[j] /= 2.0;
            }
        }


        rhs_debug.GetBlock(0).SetSubVector(local_hat, helper);

        for (int j =0; j < local_edofs.Size(); ++j)
        {
            if (ess_edofs_[local_edofs[j]])
            {
                ess_hat[local_hat[j]] = 1;
                assert(rhs_debug[local_hat[j]] == 0.0);
            }
        }
    }
    M_hat.Finalize();
    dTdsigma_hat.Finalize();
    dMdS_hat.Finalize();
    D_hat.Finalize();
    C_hat.Finalize();



    rhs_debug.GetBlock(3) = 0.0;
    mfem::SparseMatrix mult_I(mgL_.NumEDofs(), mgL_.NumEDofs());
    for (int j = 0; j < ess_true_multipliers_.Size(); ++j)
    {
        C_hat.EliminateRow(ess_true_multipliers_[j]);
        mult_I.Add(ess_true_multipliers_[j], ess_true_multipliers_[j], 1.0);
        rhs_debug.GetBlock(3)[ess_true_multipliers_[j]] = -rhs(ess_true_mult_to_edof_[j]);
    }

    mfem::SparseMatrix CT_hat = smoothg::Transpose(C_hat);

//    for (int j =0; j < ess_edofs_.Size(); ++j)
//    {
//        if (ess_edofs_[j]) { mult_I.Add(j, j, 1.0); }
//    }
    mult_I.Finalize();

    mfem::SparseMatrix DT_hat = smoothg::Transpose(D_hat);
    mfem::SparseMatrix dTdS = GetDiag(*dTdS_);


    double dt_den = 8.85773e7;

//    M_hat *= 1./dt_den;
//    DT_hat *= 1./dt_den;
//    CT_hat *= 1./dt_den;
//    dMdS_hat *= 1./dt_den;
//    rhs_debug.GetBlock(0) *= 1./dt_den;

//    D_hat *= dt_den;
//    C_hat *= dt_den;
//    dTdS *= dt_den;
//    dTdsigma_hat *= dt_den;
//    mult_I *= dt_den;

    mfem::BlockMatrix op_debug(bos);
    op_debug.SetBlock(0, 0, &M_hat);
    op_debug.SetBlock(1, 0, &D_hat);
    op_debug.SetBlock(0, 1, &DT_hat);
    op_debug.SetBlock(2, 0, &dTdsigma_hat);
    op_debug.SetBlock(0, 2, &dMdS_hat);
    op_debug.SetBlock(3, 0, &C_hat);
    op_debug.SetBlock(0, 3, &CT_hat);
    op_debug.SetBlock(2, 2, &dTdS);
    op_debug.SetBlock(3, 3, &mult_I);

    unique_ptr<mfem::SparseMatrix> op_debug_sp(op_debug.CreateMonolithic());

    rhs_debug.GetBlock(1) = rhs.GetBlock(1);
    rhs_debug.GetBlock(2) = rhs.GetBlock(2);



//    rhs_debug.GetBlock(1) *= dt_den;
//    rhs_debug.GetBlock(2) *= dt_den;
//    rhs_debug.GetBlock(3) *= dt_den;


    sol_debug = 0.0;
//    for (int j =0; j < ess_hat.Size(); ++j)
//    {
//        if (ess_hat[j])
//        {
//            op_debug_sp->EliminateRowCol(j, mfem::Matrix::DIAG_KEEP);
//        }
//    }

//    op_debug_sp->EliminateRowCol(op_debug_sp->NumCols()-1, mfem::Matrix::DIAG_KEEP);

//    mfem::UMFPackSolver solver_debug(*op_debug_sp);
//    solver_debug.Mult(rhs_debug, sol_debug);

//    std::cout <<"|| M_hat || = "<<FrobeniusNorm(M_hat) << "\n";
//    std::cout <<"|| dTdS || = "<<FrobeniusNorm(dTdS) << "\n";

    std::ofstream  op_file("hb_block_mat_30_perf.txt");
    op_debug_sp->PrintMatlab(op_file);

//    D_hat.Print();
return;
    mfem::GMRESSolver solver_debug(comm_);
    solver_debug.SetOperator(*op_debug_sp);
    solver_debug.SetMaxIter(50000);
    solver_debug.SetRelTol(1e-15);
//    solver_debug.SetAbsTol(1e-18);

    std::unique_ptr<mfem::HypreParMatrix> op_par(ToParMatrix(comm_, *op_debug_sp));
    HypreILU prec_debug(*op_par);
    solver_debug.SetPreconditioner(prec_debug);

    mfem::BlockVector sol_hb2(sol_debug);
    sol_hb2 = 0.0;
    solver_debug.Mult(rhs_debug,  sol_hb2 );
//sol_hb2.GetBlock(1).Print();

    std::cout <<"flux sol = "<<sol_hb2[2187]<<" "<<sol_hb2[2186]<<" "<<sol_hb2[2185] << "\n";


    mfem::UMFPackSolver solver_debug2(*op_debug_sp);
    solver_debug2.Mult(rhs_debug, sol_debug );
//    sol_debug.GetBlock(1).Print();

    std::cout <<"flux sol2 = "<<sol_debug[2187]<<" "<<sol_debug[2186]<<" "<<sol_debug[2185] << "\n";

    mfem::BlockVector resid_debug(rhs_debug);
    resid_debug = 0.0;
    op_debug_sp->Mult(sol_hb2, resid_debug);
    resid_debug -= rhs_debug;

    std::cout <<"hb resid = "<<resid_debug.GetBlock(0).Norml2() / rhs_debug.GetBlock(0).Norml2() << "\n";
    std::cout <<"hb resid = "<<resid_debug.GetBlock(1).Norml2() / rhs_debug.GetBlock(1).Norml2() << "\n";
    std::cout <<"hb resid = "<<resid_debug.GetBlock(2).Norml2() / rhs_debug.GetBlock(2).Norml2() << "\n";
    std::cout <<"hb resid = "<<resid_debug.GetBlock(3).Norml2() / rhs_debug.GetBlock(3).Norml2() << "\n";
    std::cout <<"hb rhs = "<< rhs_debug.GetBlock(3).Norml2() << "\n";



    sol_hb2 -= sol_debug;
    std::cout <<"hb sol diff = "<<sol_hb2.GetBlock(0).Norml2() / sol_debug.GetBlock(0).Norml2() << "\n";
    std::cout <<"hb sol diff = "<<sol_hb2.GetBlock(1).Norml2() / sol_debug.GetBlock(1).Norml2() << "\n";
    std::cout <<"hb sol diff = "<<sol_hb2.GetBlock(2).Norml2() / sol_debug.GetBlock(2).Norml2() << "\n";
    std::cout <<"hb sol diff = "<<sol_hb2.GetBlock(3).Norml2() / sol_debug.GetBlock(3).Norml2() << "\n";
    std::cout <<"gmres iter = "<<solver_debug.GetNumIterations() << "\n";


    resid_debug = 0.0;
    op_debug_sp->Mult(sol_debug, resid_debug);
    resid_debug -= rhs_debug;

    std::cout <<"hb resid = "<<resid_debug.GetBlock(0).Norml2() / rhs_debug.GetBlock(0).Norml2() << "\n";
    std::cout <<"hb resid = "<<resid_debug.GetBlock(1).Norml2() / rhs_debug.GetBlock(1).Norml2() << "\n";
    std::cout <<"hb resid = "<<resid_debug.GetBlock(2).Norml2() / rhs_debug.GetBlock(2).Norml2() << "\n";
    std::cout <<"hb resid = "<<resid_debug.GetBlock(3).Norml2() / rhs_debug.GetBlock(3).Norml2() << "\n";



    sol = 0.0;
    sol.GetBlock(1) = sol_debug.GetBlock(1);
    sol.GetBlock(2) = sol_debug.GetBlock(2);

    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        local_hat.SetSize(local_edofs.Size());
        std::iota(local_hat.begin(), local_hat.end(), vert_edof.GetI()[i]);

        sol_debug.GetSubVector(local_hat, helper);
        for (int j = 0; j < local_edofs.Size(); ++j)
        {
            if (edof_needs_averaging_[local_edofs[j]])
            {
                helper[j] /= 2.0;
            }
        }
        sol.AddElementVector(local_edofs, helper);
    }
}

void TwoPhaseHybrid::DebugMult2(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const
{
    int num_hatdofs = mgL_.GetGraphSpace().VertexToEDof().NumNonZeroElems();

    mfem::Array<int> bos(3), darcy_bos(3);
    bos[0] = 0;
    bos[1] = bos[0] + multiplier_d_td_->NumCols();
    bos[2] = bos[1] + mgL_.GetGraphSpace().VertexToVDof().NumCols();

    darcy_bos[0] = 0;
    darcy_bos[1] = darcy_bos[0] + num_hatdofs;
    darcy_bos[2] = darcy_bos[1] + mgL_.GetGraphSpace().VertexToVDof().NumCols();

    auto& vert_edof = mgL_.GetGraphSpace().VertexToEDof();
    auto& vert_vdof = mgL_.GetGraphSpace().VertexToVDof();
    auto mbuilder = dynamic_cast<const ElementMBuilder*>(&(mgL_.GetMBuilder()));
    auto M_el = mbuilder->GetElementMatrices();


    mfem::BlockVector sol_hb(offsets_);
    sol_hb = 0.0;

    mfem::BlockVector rhs_copy(rhs);
    for (int m = 0; m < ess_true_multipliers_.Size(); ++m)
    {
        sol_hb(ess_true_multipliers_[m]) = -rhs(ess_true_mult_to_edof_[m]);
        rhs_copy(ess_true_mult_to_edof_[m]) = 0.0;
    }


    mfem::BlockVector darcy_rhs(darcy_bos), rhs_debug(bos);
    mfem::Vector helper;

    mfem::Array<int> local_edofs, local_vdofs, local_mult, local_hat, local_ddofs;
    mfem::Array<int> ess_hat(num_hatdofs);
    ess_hat = 0;

    mfem::SparseMatrix M_hat(num_hatdofs, num_hatdofs);
    mfem::SparseMatrix dTdsigma_hat(mgL_.NumVDofs(), num_hatdofs);
    mfem::SparseMatrix dMdS_hat(num_hatdofs, mgL_.NumVDofs());
    mfem::SparseMatrix D_hat(mgL_.NumVDofs(), num_hatdofs);
    mfem::SparseMatrix C_hat(mgL_.NumEDofs(), num_hatdofs);
    mfem::SparseMatrix darcy_hat_inv(num_hatdofs+D_hat.NumRows());

    int num_injectors = mgL_.NumInjectors();
    auto inj_cells = mgL_.GetInjectorCells();

    mfem::DenseMatrix DenseDloc, DenseCloc;
    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        GetTableRow(vert_vdof, i, local_vdofs);
        GetTableRow(Agg_multiplier_, i, local_mult);
        local_hat.SetSize(local_edofs.Size());
        std::iota(local_hat.begin(), local_hat.end(), vert_edof.GetI()[i]);

        auto Dloc = ExtractRowAndColumns(mgL_.GetD(), local_vdofs, local_edofs);
        Full(Dloc, DenseDloc);
        Full(C_[i], DenseCloc);

        mfem::DenseMatrix M_el_i(M_el[i]);
        if (i < vert_edof.NumRows() - num_injectors)
        {
            M_el_i *= (1.0 / sol.GetBlock(2)[i]);
        }
        else
        {
            mfem::Vector scale(local_edofs.Size());
            auto& cells = inj_cells[i-vert_edof.NumRows()+num_injectors];
            for (int j = 0; j < scale.Size(); ++j)
            {
                scale[j] = 1.0 / sol.GetBlock(2)[cells[j]];
            }
            M_el_i.LeftScaling(scale);
        }

//        M_hat.AddSubMatrix(local_hat, local_hat, M_el_i);
        dTdsigma_hat.AddSubMatrix(local_vdofs, local_hat, (*dTdsigma_)[i]);
        dMdS_hat.AddSubMatrix(local_hat, local_vdofs, (*dMdS_)[i]);
//        D_hat.AddSubMatrix(local_vdofs, local_hat, DenseDloc);
        C_hat.AddSubMatrix(local_mult, local_hat, DenseCloc);


        local_hat.Copy(local_ddofs);
        local_ddofs.Append(local_vdofs);
        for (int j = local_edofs.Size(); j < local_ddofs.Size(); ++j)
        {
            local_ddofs[j] = local_ddofs[j] + num_hatdofs;
        }

        mfem::DenseMatrix darcy_el(DenseDloc.NumCols() + DenseDloc.NumRows());
        darcy_el.CopyMN(M_el_i, 0, 0);
        darcy_el.CopyMN(DenseDloc, DenseDloc.NumCols(), 0);
        darcy_el.CopyMNt(DenseDloc, 0, DenseDloc.NumCols());

        mfem::DenseMatrixInverse darcy_el_solver(darcy_el);
        mfem::DenseMatrix darcy_el_inv;
        darcy_el_solver.GetInverseMatrix(darcy_el_inv);

        darcy_hat_inv.AddSubMatrix(local_ddofs, local_ddofs, darcy_el_inv);



        rhs_copy.GetSubVector(local_edofs, helper);

        for (int j =0; j < local_edofs.Size(); ++j)
        {
            if (edof_needs_averaging_[local_edofs[j]])
            {
                helper[j] /= 2.0;
            }
        }
        darcy_rhs.GetBlock(0).SetSubVector(local_hat, helper);

//        for (int j =0; j < local_edofs.Size(); ++j)
//        {
//            if (ess_edofs_[local_edofs[j]])
//            {
//                ess_hat[local_hat[j]] = 1;
////                assert(rhs_debug[local_hat[j]] == 0.0);
//            }
//        }
    }
//    M_hat.Finalize();
    dTdsigma_hat.Finalize();
    dMdS_hat.Finalize();
//    D_hat.Finalize();
    C_hat.Finalize();
    darcy_hat_inv.Finalize();

    mfem::SparseMatrix CT_hat = smoothg::Transpose(C_hat);
//    mfem::SparseMatrix DT_hat = smoothg::Transpose(D_hat);

//    mfem::BlockMatrix darcy_debug(darcy_bos);
//    darcy_debug.SetBlock(0, 0, &M_hat);
//    darcy_debug.SetBlock(1, 0, &D_hat);
//    darcy_debug.SetBlock(0, 1, &DT_hat);

    mfem::BlockMatrix left_op(bos, darcy_bos);
    left_op.SetBlock(0, 0, &C_hat);
    left_op.SetBlock(1, 0, &dTdsigma_hat);
    unique_ptr<mfem::SparseMatrix> left_sp(left_op.CreateMonolithic());

    mfem::BlockMatrix right_op(darcy_bos, bos);
    right_op.SetBlock(0, 0, &CT_hat);
    right_op.SetBlock(0, 1, &dMdS_hat);

//    mfem::DenseMatrix op_tmp;
//    {
//        {
//            mfem::DenseMatrix right_tmp, right_dense;
            unique_ptr<mfem::SparseMatrix> right_sp(right_op.CreateMonolithic());
//            Full(*right_sp, right_dense);

//            unique_ptr<mfem::SparseMatrix> darcy_sp(darcy_debug.CreateMonolithic());
//            mfem::UMFPackSolver darcy_inv(*darcy_sp);
//            right_tmp = smoothg::Mult(darcy_inv, right_dense);
//            op_tmp = smoothg::Mult(left_op, right_tmp);
//        }
//        {
//            mfem::DenseMatrix dtds_dense;
            mfem::SparseMatrix dTdS = GetDiag(*dTdS_);
            mfem::BlockMatrix dtds_op(bos, bos);
            dtds_op.SetBlock(1, 1, &dTdS);
            unique_ptr<mfem::SparseMatrix> dtds_sp(dtds_op.CreateMonolithic());
//            Full(*dtds_sp, dtds_dense);
//            op_tmp -= dtds_dense;
//        }
//    }

//    mfem::DenseMatrix op_elim(op_tmp);
//    op_elim = 0.0;
//    for (int j =0; j < ess_true_multipliers_.Size(); ++j)
//    {
//        for (int i = 0; i < op_tmp.NumRows(); ++i)
//        {
//            if (i != ess_true_multipliers_[j])
//            {
//                op_elim(i, ess_true_multipliers_[j]) = op_tmp(i, ess_true_multipliers_[j]);
//            }
//            op_tmp(i, ess_true_multipliers_[j]) = 0.0;
//        }
//        for (int i = 0; i < op_tmp.NumCols(); ++i)
//        {
//            op_tmp(ess_true_multipliers_[j], i) = 0.0;
//        }
//        op_tmp(ess_true_multipliers_[j], ess_true_multipliers_[j]) = 1.0;
//    }

//    for (int i = 0; i < op_tmp.NumRows(); ++i)
//    {
//        op_tmp(i, op_tmp.NumRows()-1) = 0.0;
//    }
//    for (int i = 0; i < op_tmp.NumCols(); ++i)
//    {
//        op_tmp(op_tmp.NumRows()-1, i) = 0.0;
//    }
//    op_tmp(op_tmp.NumRows()-1, op_tmp.NumRows()-1) = 1.0;

    auto left_tmp = smoothg::Mult(*left_sp, darcy_hat_inv);
    auto op_tmp = smoothg::Mult(left_tmp, *right_sp);

    unique_ptr<mfem::SparseMatrix> op_debug(mfem::Add(1.0, op_tmp, -1.0, *dtds_sp));

//    mfem::DenseMatrix mono_mat_dense;
//    Full(*mono_mat_, mono_mat_dense);
//    mono_mat_dense -= op_tmp;

//    for (int r = 0; r < mono_mat_->NumRows(); ++r)
//    {
//        for (int c_ptr = mono_mat_->GetI()[r]; c_ptr < mono_mat_->GetI()[r+1]; ++c_ptr)
//        {
//            mono_mat_->GetData()[c_ptr] -= op_tmp(r, mono_mat_->GetJ()[c_ptr]);
//        }
//    }

//    auto mono_mat_fix_small = DropSmall(*mono_mat_);
//    mono_mat_fix_small.Print();
////    mono_mat_->Print();
//    std::cout << "op_diff = " << FrobeniusNorm(*mono_mat_) / op_tmp.FNorm() << "\n";
//    std::cout << "diff_nnz = " << mono_mat_fix_small.NumNonZeroElems()<<"\n";
//return;
//    std::cout<<"op_diff = " << mono_mat_dense.FNorm()<<"\n";


    mfem::SparseMatrix op_elim = GetEliminatedCols(*op_debug, ess_true_multipliers_);

    for (int j = 0; j < ess_true_multipliers_.Size(); ++j)
    {
        assert((*op_debug)(ess_true_multipliers_[j], ess_true_multipliers_[j]) != 0.0);
//        op_debug->EliminateRowCol(ess_true_multipliers_[j]);
        op_debug->EliminateRow(ess_true_multipliers_[j]);
        op_debug->EliminateCol(ess_true_multipliers_[j]);
        op_debug->Set(ess_true_multipliers_[j], ess_true_multipliers_[j], 1.0);
    }

//    unique_ptr<mfem::SparseMatrix> op_diff(mfem::Add(1.0, *op_debug, -1.0, *mono_mat_));
//    op_diff->Print();
//    std::cout<<"op_diff = " << FrobeniusNorm(*op_diff) / FrobeniusNorm(*op_debug) <<"\n";

//    auto op_diff_small = DropSmall(*op_diff);
//    std::cout << "diff_nnz = " << op_diff_small.NumNonZeroElems()<<"\n";

//    return;



    darcy_rhs.GetBlock(1) = rhs_copy.GetBlock(1);

//    mfem::BlockVector rhs_hb = MakeHybridRHS(rhs_copy);

    rhs_debug.GetBlock(0) = 0.0;
    rhs_debug.GetBlock(1).Set(-1.0, rhs_copy.GetBlock(2));

    left_tmp.AddMult(darcy_rhs, rhs_debug);



    op_elim.AddMult(sol_hb, rhs_debug, -1.0);

    for (int ess_true_mult : const_cast<mfem::Array<int>&>(ess_true_multipliers_))
    {
        rhs_debug(ess_true_mult) = sol_hb(ess_true_mult);
    }

    std::ofstream  op_file("hb_schur_mat_30_perf.txt");
    op_debug->PrintMatlab(op_file);
return;
    mfem::UMFPackSolver solver_debug(*op_debug);
//    mfem::DenseMatrixInverse solver_debug(op_tmp);
//    mfem::DenseMatrix solver_debug;
//    solver_debug_tmp.GetInverseMatrix(solver_debug);
    solver_debug.Mult(rhs_debug, sol_hb);

//    solver_->Mult(rhs_hb, sol_hb);
    BackSubstitute(rhs_copy, sol_hb, sol);
    num_iterations_ = solver_->GetNumIterations();
    resid_norm_ = solver_->GetFinalNorm();
}

void TwoPhaseHybrid::DebugMult3(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const
{
    int num_hatdofs = mgL_.GetGraphSpace().VertexToEDof().NumNonZeroElems();

    mfem::Array<int> bos(3), darcy_bos(3);
    bos[0] = 0;
    bos[1] = bos[0] + multiplier_d_td_->NumCols();
    bos[2] = bos[1] + mgL_.GetGraphSpace().VertexToVDof().NumCols();

    darcy_bos[0] = 0;
    darcy_bos[1] = darcy_bos[0] + num_hatdofs;
    darcy_bos[2] = darcy_bos[1] + mgL_.GetGraphSpace().VertexToVDof().NumCols();

    auto& vert_edof = mgL_.GetGraphSpace().VertexToEDof();
    auto& vert_vdof = mgL_.GetGraphSpace().VertexToVDof();
    auto mbuilder = dynamic_cast<const ElementMBuilder*>(&(mgL_.GetMBuilder()));
    auto M_el = mbuilder->GetElementMatrices();


    mfem::BlockVector sol_hb(offsets_);
    sol_hb = 0.0;

    mfem::BlockVector darcy_rhs(darcy_bos), rhs_debug(bos);
    mfem::Vector helper;

    mfem::Array<int> local_edofs, local_vdofs, local_mult, local_hat, local_ddofs;
    mfem::Array<int> ess_hat(num_hatdofs);
    ess_hat = 0;

//    mfem::SparseMatrix M_hat(num_hatdofs, num_hatdofs);
    mfem::SparseMatrix dTdsigma_hat(mgL_.NumVDofs(), num_hatdofs);
    mfem::SparseMatrix dMdS_hat(num_hatdofs, mgL_.NumVDofs());
//    mfem::SparseMatrix D_hat(mgL_.NumVDofs(), num_hatdofs);
    mfem::SparseMatrix C_hat(mgL_.NumEDofs(), num_hatdofs);
    mfem::SparseMatrix darcy_hat_inv(num_hatdofs+mgL_.NumVDofs());

    mfem::DenseMatrix DenseDloc, DenseCloc;
    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        GetTableRow(vert_vdof, i, local_vdofs);
        GetTableRow(Agg_multiplier_, i, local_mult);
        local_hat.SetSize(local_edofs.Size());
        std::iota(local_hat.begin(), local_hat.end(), vert_edof.GetI()[i]);

        auto Dloc = ExtractRowAndColumns(mgL_.GetD(), local_vdofs, local_edofs);
        Full(Dloc, DenseDloc);
        Full(C_[i], DenseCloc);

        mfem::DenseMatrix M_el_i(M_el[i]);
        M_el_i *= (1.0 / sol.GetBlock(2)[i]);
        dTdsigma_hat.AddSubMatrix(local_vdofs, local_hat, (*dTdsigma_)[i]);
        dMdS_hat.AddSubMatrix(local_hat, local_vdofs, (*dMdS_)[i]);
        C_hat.AddSubMatrix(local_mult, local_hat, DenseCloc);


//        D_hat.AddSubMatrix(local_vdofs, local_hat, DenseDloc);
//        M_hat.AddSubMatrix(local_hat, local_hat, M_el_i);


        local_hat.Copy(local_ddofs);
        local_ddofs.Append(local_vdofs);
        for (int j = local_edofs.Size(); j < local_ddofs.Size(); ++j)
        {
            local_ddofs[j] = local_ddofs[j] + num_hatdofs;
        }

        mfem::DenseMatrix darcy_el(DenseDloc.NumCols() + DenseDloc.NumRows());
        darcy_el.CopyMN(M_el_i, 0, 0);
        darcy_el.CopyMN(DenseDloc, DenseDloc.NumCols(), 0);
        darcy_el.CopyMNt(DenseDloc, 0, DenseDloc.NumCols());

        mfem::DenseMatrixInverse darcy_el_solver(darcy_el);
        mfem::DenseMatrix darcy_el_inv;
        darcy_el_solver.GetInverseMatrix(darcy_el_inv);

        darcy_hat_inv.AddSubMatrix(local_ddofs, local_ddofs, darcy_el_inv);



        rhs.GetSubVector(local_edofs, helper);

        for (int j = 0; j < local_edofs.Size(); ++j)
        {
            if (edof_needs_averaging_[local_edofs[j]])
            {
                helper[j] /= 2.0;
            }
        }
        darcy_rhs.GetBlock(0).SetSubVector(local_hat, helper);

    }
    dTdsigma_hat.Finalize();
    dMdS_hat.Finalize();
    C_hat.Finalize();
    darcy_hat_inv.Finalize();


//    M_hat.Finalize();
//    D_hat.Finalize();
//    mfem::SparseMatrix DT_hat = smoothg::Transpose(D_hat);
//    mfem::BlockMatrix darcy_debug(darcy_bos);
//    darcy_debug.SetBlock(0, 0, &M_hat);
//    darcy_debug.SetBlock(1, 0, &D_hat);
//    darcy_debug.SetBlock(0, 1, &DT_hat);

//    unique_ptr<mfem::SparseMatrix> darcy_sp(darcy_debug.CreateMonolithic());
//    auto darcy_I = smoothg::Mult(*darcy_sp, darcy_hat_inv);
//    darcy_I.Print();



    rhs_debug.GetBlock(0) = 0.0;
    mfem::SparseMatrix mult_I(mgL_.NumEDofs(), mgL_.NumEDofs());
    for (int j = 0; j < ess_true_multipliers_.Size(); ++j)
    {
        C_hat.EliminateRow(ess_true_multipliers_[j]);
        mult_I.Add(ess_true_multipliers_[j], ess_true_multipliers_[j], 1.0);
        rhs_debug(ess_true_multipliers_[j]) = rhs(ess_true_mult_to_edof_[j]); // negative  of block version
    }
    mult_I.Finalize();

    mfem::SparseMatrix CT_hat = smoothg::Transpose(C_hat);

    mfem::BlockMatrix left_op(bos, darcy_bos);
    left_op.SetBlock(0, 0, &C_hat);
    left_op.SetBlock(1, 0, &dTdsigma_hat);
    unique_ptr<mfem::SparseMatrix> left_sp(left_op.CreateMonolithic());

    mfem::BlockMatrix right_op(darcy_bos, bos);
    right_op.SetBlock(0, 0, &CT_hat);
    right_op.SetBlock(0, 1, &dMdS_hat);
    unique_ptr<mfem::SparseMatrix> right_sp(right_op.CreateMonolithic());

    mfem::SparseMatrix dTdS = GetDiag(*dTdS_);
    mfem::BlockMatrix dtds_op(bos, bos);
    dtds_op.SetBlock(0, 0, &mult_I);
    dtds_op.SetBlock(1, 1, &dTdS);
    unique_ptr<mfem::SparseMatrix> dtds_sp(dtds_op.CreateMonolithic());

    auto left_tmp = smoothg::Mult(*left_sp, darcy_hat_inv);
    auto op_tmp = smoothg::Mult(left_tmp, *right_sp);

    unique_ptr<mfem::SparseMatrix> op_debug(mfem::Add(1.0, op_tmp, -1.0, *dtds_sp));


//    mfem::SparseMatrix op_elim = GetEliminatedCols(*op_debug, ess_true_multipliers_);

//    for (int j = 0; j < ess_true_multipliers_.Size(); ++j)
//    {
//        assert((*op_debug)(ess_true_multipliers_[j], ess_true_multipliers_[j]) != 0.0);
////        op_debug->EliminateRowCol(ess_true_multipliers_[j]);
//        op_debug->EliminateRow(ess_true_multipliers_[j]);
//        op_debug->EliminateCol(ess_true_multipliers_[j]);
//        op_debug->Set(ess_true_multipliers_[j], ess_true_multipliers_[j], 1.0);
//    }

//    unique_ptr<mfem::SparseMatrix> op_diff(mfem::Add(1.0, *op_debug, -1.0, *mono_mat_));
//    op_diff->Print();
//    std::cout<<"op_diff = " << FrobeniusNorm(*op_diff) / FrobeniusNorm(*op_debug) <<"\n";

//    auto op_diff_small = DropSmall(*op_diff);
//    std::cout << "diff_nnz = " << op_diff_small.NumNonZeroElems()<<"\n";

    darcy_rhs.GetBlock(1) = rhs.GetBlock(1);
    rhs_debug.GetBlock(1).Set(-1.0, rhs.GetBlock(2));

    left_tmp.AddMult(darcy_rhs, rhs_debug);



//    op_elim.AddMult(sol_hb, rhs_debug, -1.0);

    mfem::UMFPackSolver solver_debug2(*op_debug);

    mfem::GMRESSolver solver_debug(comm_);
    solver_debug.SetOperator(*op_debug);
    solver_debug.SetMaxIter(10000);
    solver_debug.SetRelTol(1e-12);
    solver_debug.SetAbsTol(1e-15);

    std::unique_ptr<mfem::HypreParMatrix> op_par(ToParMatrix(comm_, *op_debug));
    HypreILU prec_debug(*op_par);
    solver_debug.SetPreconditioner(prec_debug);



//    mfem::DenseMatrix op_dense;
//    Full(*op_debug, op_dense);

//    op_debug->Print();
//    mfem::DenseMatrixInverse solver_debug(op_dense);
//    mfem::DenseMatrix solver_debug;
//    solver_debug_tmp.GetInverseMatrix(solver_debug);
    solver_debug.Mult(rhs_debug, sol_hb);

    mfem::Vector sol_hb2(sol_hb);
    sol_hb2 = 0.0;
    solver_debug2.Mult(rhs_debug, sol_hb2);
    sol_hb2 -= sol_hb;
    std::cout <<"hb sol diff = "<<sol_hb2.Norml2() / sol_hb.Norml2() << "\n";
    std::cout <<"gmres iter = "<<solver_debug.GetNumIterations() << "\n";


    mfem::BlockVector darcy_sol_tmp(darcy_rhs), darcy_sol(darcy_bos);
    right_sp->AddMult(sol_hb, darcy_sol_tmp, -1.0);
    darcy_hat_inv.Mult(darcy_sol_tmp, darcy_sol);

    sol.GetBlock(1) = darcy_sol.GetBlock(1);
    sol.GetBlock(2) = sol_hb.GetBlock(1);

    sol.GetBlock(0) = 0.0;
    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        local_hat.SetSize(local_edofs.Size());
        std::iota(local_hat.begin(), local_hat.end(), vert_edof.GetI()[i]);

        darcy_sol.GetSubVector(local_hat, helper);
        for (int j = 0; j < local_edofs.Size(); ++j)
        {
            if (edof_needs_averaging_[local_edofs[j]])
            {
                helper[j] /= 2.0;
            }
        }
        sol.AddElementVector(local_edofs, helper);
    }

//    BackSubstitute(rhs, sol_hb, sol);
}

} // namespace smoothg
