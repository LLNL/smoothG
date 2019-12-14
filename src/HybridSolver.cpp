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
    cg_(comm_),
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

    cg_.SetPrintLevel(print_level_);
    cg_.SetMaxIter(max_num_iter_);
    cg_.SetRelTol(rtol_);
    cg_.SetAbsTol(atol_);
    cg_.iterative_mode = true;

    BuildParallelSystemAndSolver(H_proc);

    Hrhs_.SetSize(num_multiplier_dofs_);
    trueHrhs_.SetSize(multiplier_d_td_->GetNumCols());
    Mu_.SetSize(num_multiplier_dofs_);
    trueMu_.SetSize(trueHrhs_.Size());
    trueMu_ = 0.0;
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

    const int scaling_size = rescale_iter_ < 0 &&
            diagonal_scaling_.Capacity() == 0 ? H_proc.NumRows() : 0;
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

    if (is_symmetric_)
    {
//        trueMu_ = MakeInitialGuess(Sol, Rhs);
    }
    else
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
    for (int ess_true_mult : ess_true_multipliers_)
    {
        trueHrhs_(ess_true_mult) = trueMu_(ess_true_mult);
    }

    if (diagonal_scaling_.Size() > 0)
    {
        RescaleVector(diagonal_scaling_, trueHrhs_);
        InvRescaleVector(diagonal_scaling_, trueMu_);
    }

    auto solver = is_symmetric_ ? dynamic_cast<const mfem::IterativeSolver*>(&cg_)
                                : dynamic_cast<const mfem::IterativeSolver*>(&gmres_);

    // solve the parallel global hybridized system
    mfem::StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    const_cast<mfem::IterativeSolver*>(solver)->SetRelTol(1e-12);
    const_cast<mfem::IterativeSolver*>(solver)->SetAbsTol(1e-15);

    solver->Mult(trueHrhs_, trueMu_);

    chrono.Stop();
    timing_ = chrono.RealTime();

//        mfem::SparseMatrix H_diag(GetDiag(*H_), true);
//        mfem::UMFPackSolver H_solve(H_diag);
//std::cout<<"trueMu = " <<trueMu_.Norml2()<<"\n";
//    mfem::Vector trueMu2(trueMu_);trueMu2 = 0.0;
//        trueMu_ = 0.0;
//    solver->Mult(trueHrhs_, trueMu2);

//    if (solver->GetConverged() == false)
//        H_solve.Mult(trueHrhs_, trueMu_);
//        trueMu2-=trueMu_;
//    std::cout<<"rel trueMu diff = " <<trueMu2.Norml2()/ trueMu_.Norml2()<<"\n";


    std::string solver_name = is_symmetric_ ? "CG" : "GMRES";

    if (myid_ == 0 && print_level_ > 0)
    {
        std::cout << "  Timing: " + solver_name + " done in "
                  << timing_ << "s. \n";
    }

    // TODO: decide to use = or += here and in timing_ update (MinresBlockSolver uses +=)
    num_iterations_ = solver->GetNumIterations();

    if (myid_ == 0 && print_level_ > 0)
    {
        if (solver->GetConverged())
        {
            std::cout << "  " + solver_name + " converged in "
                      << num_iterations_
                      << " with a final residual norm "
                      << solver->GetFinalNorm() << "\n";
        }
        else
        {
            std::cout << "  " + solver_name + " did not converge in "
                      << num_iterations_
                      << ". Final residual norm is "
                      << solver->GetFinalNorm() << "\n";
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

    auto solver = is_symmetric_ ? dynamic_cast<mfem::IterativeSolver*>(&cg_)
                                : dynamic_cast<mfem::IterativeSolver*>(&gmres_);
    solver->SetOperator(*H_);

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

        solver->SetPreconditioner(*prec_);
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
//        H_el.Symmetrize();
        H_proc.AddSubMatrix(local_multiplier, local_multiplier, H_el);

    }
    BuildParallelSystemAndSolver(H_proc);

    gmres_.SetPrintLevel(print_level_);
    gmres_.SetMaxIter(max_num_iter_);
    gmres_.SetRelTol(rtol_);
    gmres_.SetAbsTol(atol_);
    gmres_.SetKDim(100);

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

    auto H_proc = AssembleHybridSystem(elem_scaling_inverse, N_el);
    BuildParallelSystemAndSolver(H_proc);

    gmres_.SetPrintLevel(print_level_);
    gmres_.SetMaxIter(max_num_iter_);
    gmres_.SetRelTol(rtol_);
    gmres_.SetAbsTol(atol_);
    gmres_.iterative_mode = false;
    gmres_.SetKDim(100);

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

void HybridSolver::SetPrintLevel(int print_level)
{
    MixedLaplacianSolver::SetPrintLevel(print_level);

    cg_.SetPrintLevel(print_level_);
}

void HybridSolver::SetMaxIter(int max_num_iter)
{
    MixedLaplacianSolver::SetMaxIter(max_num_iter);

    cg_.SetMaxIter(max_num_iter_);
}

void HybridSolver::SetRelTol(double rtol)
{
    MixedLaplacianSolver::SetRelTol(rtol);

    cg_.SetRelTol(rtol_);
}

void HybridSolver::SetAbsTol(double atol)
{
    MixedLaplacianSolver::SetAbsTol(atol);

    cg_.SetAbsTol(atol_);
}

AuxSpacePrec::AuxSpacePrec(mfem::HypreParMatrix& op, mfem::SparseMatrix aux_map,
                           const std::vector<mfem::Array<int>>& loc_dofs)
    : mfem::Solver(op.NumRows(), false),
      local_dofs_(loc_dofs.size()),
      local_ops_(local_dofs_.size()),
      local_solvers_(local_dofs_.size()),
      op_(op),
      aux_map_(std::move(aux_map)),
      gmres_(op.GetComm())
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
//    HYPRE_BoomerAMGSetMaxIter(*aux_solver_, 3);

    gmres_.SetPrintLevel(-1);
    gmres_.SetMaxIter(10);
    gmres_.SetRelTol(1e-6);
    gmres_.SetAbsTol(1e-8);
    gmres_.iterative_mode = true;
    gmres_.SetOperator(*aux_op_);
    gmres_.SetPreconditioner(*aux_solver_);
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

} // namespace smoothg
