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

#include "HybridSolver.hpp"
#include "utilities.hpp"
#include "MatrixUtilities.hpp"
#include "MetisGraphPartitioner.hpp"

using std::unique_ptr;

namespace smoothg
{

void HybridSolver::BuildFineLevelLocalMassMatrix(
    const mfem::SparseMatrix& vertex_edge,
    const mfem::SparseMatrix& M,
    std::vector<mfem::Vector>& M_el)
{
    const int nvertices = vertex_edge.Height();
    M_el.resize(nvertices);

    mfem::SparseMatrix edge_vertex(smoothg::Transpose(vertex_edge));
    mfem::Array<int> local_edgedof;

    const mfem::Vector M_data(M.GetData(), M.Height());
    for (int i = 0; i < nvertices; i++)
    {
        GetTableRow(vertex_edge, i, local_edgedof);
        const int nlocal_edgedof = local_edgedof.Size();

        mfem::Vector& Mloc(M_el[i]);
        Mloc.SetSize(nlocal_edgedof);

        for (int j = 0; j < nlocal_edgedof; j++)
        {
            const int edgedof = local_edgedof[j];

            if (edge_vertex.RowSize(edgedof) == 2)
                Mloc(j) = M_data[edgedof] / 2;
            else
                Mloc(j) = M_data[edgedof];
        }
    }
}

HybridSolver::HybridSolver(MPI_Comm comm,
                           const MixedMatrix& mgL,
                           const mfem::SparseMatrix* face_bdrattr,
                           const mfem::Array<int>* ess_edge_dofs,
                           const int rescale_iter,
                           const SAAMGeParam* saamge_param)
    :
    MixedLaplacianSolver(mgL.get_blockoffsets()),
    comm_(comm),
    D_(mgL.getD()),
    W_(mgL.getW()),
    use_spectralAMGe_((saamge_param != nullptr)),
    use_w_(mgL.CheckW()),
    rescale_iter_(rescale_iter),
    saamge_param_(saamge_param)
{
    MPI_Comm_rank(comm, &myid_);

    const int nvertices = D_.Height();

    // TODO(gelever1): use operator= when mfem version is updated
    mfem::SparseMatrix tmp = SparseIdentity(nvertices);
    Agg_vertexdof_.Swap(tmp);

    Agg_edgedof_.MakeRef(D_);
    const mfem::SparseMatrix edge_edgedof;

    std::vector<mfem::Vector> M_el;
    BuildFineLevelLocalMassMatrix(D_, mgL.getWeight(), M_el);

    Init(edge_edgedof, M_el, mgL.get_edge_d_td(),
         face_bdrattr, ess_edge_dofs);
}

HybridSolver::HybridSolver(MPI_Comm comm,
                           const MixedMatrix& mgL,
                           const Mixed_GL_Coarsener& mgLc,
                           const mfem::SparseMatrix* face_bdrattr,
                           const mfem::Array<int>* ess_edge_dofs,
                           const int rescale_iter,
                           const SAAMGeParam* saamge_param)
    :
    MixedLaplacianSolver(mgL.get_blockoffsets()),
    comm_(comm),
    D_(mgL.getD()),
    W_(mgL.getW()),
    use_spectralAMGe_((saamge_param != nullptr)),
    use_w_(mgL.CheckW()),
    rescale_iter_(rescale_iter),
    saamge_param_(saamge_param)
{
    MPI_Comm_rank(comm, &myid_);
    const mfem::SparseMatrix& face_edgedof(mgLc.construct_face_facedof_table());

    Agg_vertexdof_.MakeRef(mgLc.construct_Agg_cvertexdof_table());
    Agg_edgedof_.MakeRef(mgLc.construct_Agg_cedgedof_table());

    Init(face_edgedof, mgLc.get_CM_el(), mgL.get_edge_d_td(),
         face_bdrattr, ess_edge_dofs);
}

HybridSolver::~HybridSolver()
{
#if SMOOTHG_USE_SAAMGE
    if (use_spectralAMGe_)
    {
        saamge::ml_free_data(sa_ml_data_);
        saamge::agg_free_partitioning(sa_apr_);
    }
#endif
}

template<typename T>
void HybridSolver::Init(const mfem::SparseMatrix& face_edgedof,
                        const std::vector<T>& M_el,
                        const mfem::HypreParMatrix& edgedof_d_td,
                        const mfem::SparseMatrix* face_bdrattr,
                        const mfem::Array<int>* ess_edge_dofs)
{
    // Determine if we are solving fine level graph Laplacian problem
    bool fine_level = (typeid(T) == typeid(mfem::Vector)) ? true : false;

    mfem::StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    nAggs_ = Agg_edgedof_.Height();
    num_edge_dofs_ = Agg_edgedof_.Width();

    // Set the size of the Hybrid_el_, AinvCT, Ainv_f_, these are all local
    // matrices and vector for each element
    MinvDT_.resize(nAggs_);
    MinvCT_.resize(nAggs_);
    AinvDMinvCT_.resize(nAggs_);
    Ainv_f_.resize(nAggs_);
    Ainv_.resize(nAggs_);
    if (use_spectralAMGe_)
        Hybrid_el_.resize(nAggs_);

    mfem::SparseMatrix edgedof_bdrattr;
    if (face_bdrattr)
    {
        if (fine_level)
        {
            edgedof_bdrattr.MakeRef(*face_bdrattr);
        }
        else
        {
            mfem::SparseMatrix edgedof_face(smoothg::Transpose(face_edgedof));
            mfem::SparseMatrix tmp = smoothg::Mult(edgedof_face, *face_bdrattr);
            edgedof_bdrattr.Swap(tmp);
        }
    }

    mfem::HypreParMatrix edgedof_d_td_;
    edgedof_d_td_.MakeRef(edgedof_d_td);
    edgedof_d_td_.GetDiag(edgedof_IsOwned_);

    // Constructing the relation table (in SparseMatrix format) between edge
    // dof and multiplier dof. For every edge dof that is associated with a
    // face, a Lagrange multiplier dof associated with the edge dof is created
    mfem::Array<int> j_multiplier_edgedof;

    // construct multiplier dof to true dof table
    if (fine_level)
    {
        num_multiplier_dofs_ = num_edge_dofs_;
        j_multiplier_edgedof.SetSize(num_edge_dofs_);
        std::iota(j_multiplier_edgedof.GetData(), j_multiplier_edgedof.GetData() + num_edge_dofs_, 0);
        Agg_multiplier_.MakeRef(Agg_edgedof_);

        GenerateOffsets(comm_, num_multiplier_dofs_, multiplier_start_);

        multiplier_d_td_ = make_unique<mfem::HypreParMatrix>();
        multiplier_d_td_->MakeRef(edgedof_d_td_);
    }
    else
    {
        unique_ptr<mfem::HypreParMatrix> edgedof_td_d(edgedof_d_td_.Transpose());
        unique_ptr<mfem::HypreParMatrix> edgedof_d_td_d(ParMult(&edgedof_d_td_, edgedof_td_d.get()));

        num_multiplier_dofs_ = face_edgedof.Width();

        int* i_edgedof_multiplier = new int[num_edge_dofs_ + 1];
        std::iota(i_edgedof_multiplier,
                  i_edgedof_multiplier + num_multiplier_dofs_ + 1, 0);
        std::fill_n(i_edgedof_multiplier + num_multiplier_dofs_ + 1,
                    num_edge_dofs_ - num_multiplier_dofs_,
                    i_edgedof_multiplier[num_multiplier_dofs_]);

        int* j_edgedof_multiplier = new int[num_multiplier_dofs_];
        std::iota(j_edgedof_multiplier,
                  j_edgedof_multiplier + num_multiplier_dofs_, 0);
        double* data_edgedof_multiplier = new double[num_multiplier_dofs_];
        std::fill_n(data_edgedof_multiplier, num_multiplier_dofs_, 1.0);
        mfem::SparseMatrix edgedof_multiplier(
            i_edgedof_multiplier, j_edgedof_multiplier,
            data_edgedof_multiplier, num_edge_dofs_, num_multiplier_dofs_);
        mfem::SparseMatrix multiplier_edgedof(smoothg::Transpose(edgedof_multiplier) );

        mfem::Array<int> j_array(multiplier_edgedof.GetJ(), multiplier_edgedof.NumNonZeroElems());
        j_array.Copy(j_multiplier_edgedof);

        Agg_multiplier_.Clear();
        mfem::SparseMatrix Agg_m_tmp(smoothg::Mult(Agg_edgedof_, edgedof_multiplier));
        Agg_multiplier_.Swap(Agg_m_tmp);

        GenerateOffsets(comm_, num_multiplier_dofs_, multiplier_start_);

        auto edgedof_multiplier_d = make_unique<mfem::HypreParMatrix>(
                                        comm_, edgedof_d_td_.GetGlobalNumRows(),
                                        multiplier_start_.Last(), edgedof_d_td_.RowPart(),
                                        multiplier_start_, &edgedof_multiplier);

        assert(edgedof_d_td_d && edgedof_multiplier_d);
        unique_ptr<mfem::HypreParMatrix> multiplier_d_td_d(
            smoothg::RAP(*edgedof_d_td_d, *edgedof_multiplier_d) );

        // Construct multiplier "dof to true dof" table
        multiplier_d_td_ = BuildEntityToTrueEntity(*multiplier_d_td_d);
    }

    // Assemble the hybridized system
    HybridSystem_ = make_unique<mfem::SparseMatrix>(num_multiplier_dofs_);

    AssembleHybridSystem(M_el, j_multiplier_edgedof);
    HybridSystem_->Finalize();

    // Mark the multiplier dof with essential BC
    // Note again there is a 1-1 map from multipliers to edge dofs on faces
    ess_multiplier_bc_ = false;
    if (face_bdrattr && ess_edge_dofs)
    {
        ess_multiplier_dofs_.SetSize(num_multiplier_dofs_);
        for (int i = 0; i < num_multiplier_dofs_; i++)
        {
            // natural BC for H(div) dof <=> essential BC for multiplier dof
            //if (edgedof_bdrattr.RowSize(i) > 0 && ess_edge_dofs->operator[](i) != 0)
            if (edgedof_bdrattr.RowSize(i) && !ess_edge_dofs->operator[](i))
            {
                ess_multiplier_dofs_[i] = 1;
                ess_multiplier_bc_ = true;
            }
            else
                ess_multiplier_dofs_[i] = 0;
        }

    }

    HybridSystemElim_ = make_unique<mfem::SparseMatrix>(*HybridSystem_, false);
    if (ess_multiplier_bc_)
    {
        for (int mm = 0; mm < num_multiplier_dofs_; ++mm)
        {
            if (ess_multiplier_dofs_[mm])
            {
                HybridSystemElim_->EliminateRowCol(mm);
            }
        }
    }
    else if (!use_w_)
    {
        if (myid_ == 0)
            HybridSystemElim_->EliminateRowCol(0);
    }

    auto HybridSystem_d = make_unique<mfem::HypreParMatrix>(
                              comm_, multiplier_start_.Last(), multiplier_start_,
                              HybridSystemElim_.get());

    if (rescale_iter_ == 0 || use_spectralAMGe_)
    {
        pHybridSystem_.reset(smoothg::RAP(*HybridSystem_d, *multiplier_d_td_));
    }
    else
    {
        ComputeScaledHybridSystem(*HybridSystem_d);
    }
    nnz_ = pHybridSystem_->NNZ();

    if (myid_ == 0 && print_level_ > 0)
        std::cout << "  Timing: Hybridized system built in "
                  << chrono.RealTime() << "s. \n";

    chrono.Clear();
    chrono.Start();
    if (myid_ == 0 && print_level_ > 0)
        std::cout << "  Timing: Preconditioner for hybridized system"
                  " constructed in " << chrono.RealTime() << "s. \n";

    cg_ = make_unique<mfem::CGSolver>(comm_);
    cg_->SetPrintLevel(print_level_);
    cg_->SetMaxIter(max_num_iter_);
    cg_->SetRelTol(rtol_);
    cg_->SetAbsTol(atol_);
    cg_->SetOperator(*pHybridSystem_);
    cg_->iterative_mode = false;

    // HypreBoomerAMG is broken if local size is zero
    int local_size = pHybridSystem_->Height();
    int min_size;
    MPI_Allreduce(&local_size, &min_size, 1, MPI_INT, MPI_MIN, comm_);

    const bool use_prec = min_size > 0;
    if (use_prec)
    {
        if (use_spectralAMGe_)
        {
            BuildSpectralAMGePreconditioner();
        }
        else
        {
            prec_ = make_unique<mfem::HypreBoomerAMG>(*pHybridSystem_);
            ((mfem::HypreBoomerAMG&)*prec_).SetPrintLevel(0);
        }
        cg_->SetPreconditioner(*prec_);
    }

    trueHrhs_.SetSize(multiplier_d_td_->GetNumCols());
    trueMu_.SetSize(trueHrhs_.Size());
    Hrhs_.SetSize(num_multiplier_dofs_);
    Mu_.SetSize(num_multiplier_dofs_);
}

void HybridSolver::AssembleHybridSystem(
    const std::vector<mfem::DenseMatrix>& M_el,
    const mfem::Array<int>& j_multiplier_edgedof)
{
    const int map_size = std::max(num_edge_dofs_, Agg_vertexdof_.Width());
    mfem::Array<int> edgedof_global_to_local_map(map_size);
    edgedof_global_to_local_map = -1;
    mfem::Array<bool> edge_marker(num_edge_dofs_);
    edge_marker = true;

    mfem::DenseMatrix DlocT, ClocT, Aloc, CMinvDT, DMinvCT, CMDADMC;
    mfem::DenseMatrix tmpHybrid_el;

    mfem::DenseMatrixInverse Mloc_solver;
    for (int iAgg = 0; iAgg < nAggs_; ++iAgg)
    {
        // Extracting the size and global numbering of local dof
        mfem::Array<int> local_vertexdof, local_edgedof, local_multiplier;
        GetTableRow(Agg_vertexdof_, iAgg, local_vertexdof);
        GetTableRow(Agg_edgedof_, iAgg, local_edgedof);
        GetTableRow(Agg_multiplier_, iAgg, local_multiplier);

        const int nlocal_vertexdof = local_vertexdof.Size();
        const int nlocal_edgedof = local_edgedof.Size();
        const int nlocal_multiplier = local_multiplier.Size();

        // Build the edge dof global to local map which will be used
        // later for mapping local multiplier dof to local edge dof
        for (int i = 0; i < nlocal_edgedof; ++i)
            edgedof_global_to_local_map[local_edgedof[i]] = i;

        // Extract Dloc as a sparse submatrix of D_
        auto Dloc = ExtractRowAndColumns(D_, local_vertexdof, local_edgedof,
                                         edgedof_global_to_local_map, false);

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
            const int edgedof_global_id = j_multiplier_edgedof[local_multiplier[i]];
            const int edgedof_local_id = edgedof_global_to_local_map[edgedof_global_id];
            Cloc_j[i] = edgedof_local_id;
            if (edgedof_IsOwned_.RowSize(edgedof_global_id) &&
                edge_marker[edgedof_global_id])
            {
                edge_marker[edgedof_global_id] = false;
                ClocT(edgedof_local_id, i) = 1.;
                Cloc_data[i] = 1.;
            }
            else
            {
                ClocT(edgedof_local_id, i) = -1.;
                Cloc_data[i] = -1.;
            }
        }

        mfem::SparseMatrix Cloc(Cloc_i, Cloc_j, Cloc_data,
                                nlocal_multiplier, nlocal_edgedof);

        for (int i = 0; i < nlocal_edgedof; ++i)
            edgedof_global_to_local_map[local_edgedof[i]] = -1;

        Mloc_solver.SetOperator(M_el[iAgg]);

        mfem::DenseMatrix& MinvCT_i(MinvCT_[iAgg]);
        mfem::DenseMatrix& MinvDT_i(MinvDT_[iAgg]);
        mfem::DenseMatrix& AinvDMinvCT_i(AinvDMinvCT_[iAgg]);
        mfem::DenseMatrix& Ainv_i(Ainv_[iAgg]);

        MinvCT_i.SetSize(nlocal_edgedof, nlocal_multiplier);
        MinvDT_i.SetSize(nlocal_edgedof, nlocal_vertexdof);
        AinvDMinvCT_i.SetSize(nlocal_vertexdof, nlocal_multiplier);

        Mloc_solver.Mult(DlocT, MinvDT_i);
        Mloc_solver.Mult(ClocT, MinvCT_i);

        // Compute CMinvCT = Cloc * MinvCT
        MultSparseDense(Cloc, MinvCT_i, tmpHybrid_el);

        // Compute Aloc = DMinvDT = Dloc * MinvDT
        MultSparseDense(Dloc, MinvDT_i, Aloc);

        if (W_)
        {
            auto Wloc = ExtractRowAndColumns(*W_, local_vertexdof, local_vertexdof,
                                             edgedof_global_to_local_map);
            mfem::DenseMatrix tmpW;
            Full(Wloc, tmpW);

            Aloc -= tmpW;
        }

        // Compute DMinvCT Dloc * MinvCT
        MultSparseDense(Dloc, MinvCT_i, DMinvCT);

        // Compute the LU factorization of Aloc and Ainv_ * DMinvCT
        mfem::DenseMatrixInverse Ainv(Aloc);
        Ainv.GetInverseMatrix(Ainv_i);

        //Ainv_i.Mult(DMinvCT, AinvDMinvCT_i);
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
        tmpHybrid_el -= CMDADMC;

        // Add contribution of the element matrix to the global system
        HybridSystem_->AddSubMatrix(local_multiplier, local_multiplier,
                                    tmpHybrid_el);

        // Save element matrix [C 0][M B^T;B 0]^-1[C 0]^T (this is needed
        // only if one wants to construct H1 spectral AMGe preconditioner)
        if (use_spectralAMGe_)
            Hybrid_el_[iAgg] = tmpHybrid_el;
    }
}

void HybridSolver::AssembleHybridSystem(
    const std::vector<mfem::Vector>& M_el,
    const mfem::Array<int>& j_multiplier_edgedof)
{
    const int map_size = std::max(num_edge_dofs_, Agg_vertexdof_.Width());
    mfem::Array<int> edgedof_global_to_local_map(map_size);
    edgedof_global_to_local_map = -1;
    mfem::Array<bool> edge_marker(num_edge_dofs_);
    edge_marker = true;

    mfem::DenseMatrix Dloc, Aloc;
    mfem::DenseMatrix tmpHybrid_el;

    mfem::Vector column_in, CMinvCT, CMinvDT;

    for (int iAgg = 0; iAgg < nAggs_; ++iAgg)
    {
        // Extracting the size and global numbering of local dof
        mfem::Array<int> local_vertexdof, local_edgedof, local_multiplier;
        GetTableRow(Agg_vertexdof_, iAgg, local_vertexdof);
        GetTableRow(Agg_edgedof_, iAgg, local_edgedof);
        GetTableRow(Agg_multiplier_, iAgg, local_multiplier);

        const int nlocal_vertexdof = local_vertexdof.Size();
        const int nlocal_edgedof = local_edgedof.Size();
        const int nlocal_multiplier = local_multiplier.Size();

        // Build the edge dof global to local map which will be used
        // later for mapping local multiplier dof to local edge dof
        for (int i = 0; i < nlocal_edgedof; ++i)
            edgedof_global_to_local_map[local_edgedof[i]] = i;

        // Extract MinvDT as a dense submatrix of D^T
        ExtractSubMatrix(D_, local_vertexdof, local_edgedof,
                         edgedof_global_to_local_map, Dloc);

        mfem::DenseMatrix& MinvCT_i(MinvCT_[iAgg]);
        mfem::DenseMatrix& MinvDT_i(MinvDT_[iAgg]);
        mfem::DenseMatrix& AinvDMinvCT_i(AinvDMinvCT_[iAgg]);
        mfem::DenseMatrix& Ainv_i(Ainv_[iAgg]);

        MinvDT_i.Transpose(Dloc);
        MinvCT_i.SetSize(nlocal_edgedof, nlocal_multiplier);
        AinvDMinvCT_i.SetSize(nlocal_vertexdof, nlocal_multiplier);
        Ainv_i.SetSize(nlocal_vertexdof, nlocal_vertexdof);

        // Construct the constraint matrix C which enforces the continuity of
        // the broken edge space
        int* Cloc_i = new int[nlocal_multiplier + 1];
        std::iota(Cloc_i, Cloc_i + nlocal_multiplier + 1, 0);
        int* Cloc_j = new int[nlocal_multiplier];
        double* Cloc_data = new double[nlocal_multiplier];

        const mfem::Vector& M_diag(M_el[iAgg]);
        CMinvCT.SetSize(nlocal_multiplier);

        for (int i = 0; i < nlocal_multiplier; ++i)
        {
            const int edgedof_global_id = j_multiplier_edgedof[local_multiplier[i]];
            const int edgedof_local_id = edgedof_global_to_local_map[edgedof_global_id];
            Cloc_j[i] = edgedof_local_id;
            CMinvCT(i) = 1. / M_diag(edgedof_local_id);

            if (edgedof_IsOwned_.RowSize(edgedof_global_id) &&
                edge_marker[edgedof_global_id])
            {
                edge_marker[edgedof_global_id] = false;
                MinvCT_i.Elem(edgedof_local_id, i) = 1. / M_diag(edgedof_local_id);
                Cloc_data[i] = 1.;
            }
            else
            {
                MinvCT_i.Elem(edgedof_local_id, i) = -1. / M_diag(edgedof_local_id);
                Cloc_data[i] = -1.;
            }
        }

        mfem::SparseMatrix Cloc(Cloc_i, Cloc_j, Cloc_data,
                                nlocal_multiplier, nlocal_edgedof);

        for (int i = 0; i < nlocal_edgedof; ++i)
            edgedof_global_to_local_map[local_edgedof[i]] = -1;

        MinvDT_i.InvLeftScaling(M_diag);

        tmpHybrid_el.SetSize(nlocal_multiplier, nlocal_multiplier);

        Aloc.SetSize(nlocal_vertexdof, nlocal_vertexdof);

        if (Dloc.Height() > 0 && Dloc.Width() > 0)
        {
            mfem::Mult(Dloc, MinvDT_i, Aloc);
        }
        else
        {
            Aloc = 0.0;
        }

        // Compute CMinvDT  = Cloc * MinvDT
        CMinvDT.SetSize(nlocal_multiplier);
        column_in.SetDataAndSize(MinvDT_i.Data(), nlocal_multiplier);
        Cloc.Mult(column_in, CMinvDT);

        if (W_)
        {
            auto Wloc = ExtractRowAndColumns(*W_, local_vertexdof, local_vertexdof,
                                             edgedof_global_to_local_map);
            mfem::DenseMatrix tmpW;
            Full(Wloc, tmpW);

            Aloc -= tmpW;
        }

        // Compute the LU factorization of Aloc and Ainv_ * DMinvCT
        const double A00_inv = 1. / Aloc(0, 0);
        Ainv_i.Elem(0, 0) = A00_inv;

        for (int j = 0; j < nlocal_multiplier; ++j)
            AinvDMinvCT_i.Elem(0, j) = CMinvDT(j) * A00_inv;

        // Compute -CMinvDTAinvDMinvCT = -CMinvDT * Ainv_ * DMinvCT
        Mult_a_VVt(-A00_inv, CMinvDT, tmpHybrid_el);

        // Hybrid_el_ = CMinvCT - CMinvDTAinvDMinvCT
        for (int j = 0; j < nlocal_multiplier; ++j)
            tmpHybrid_el.Elem(j, j) += CMinvCT(j);

        // Add contribution of the element matrix to the golbal system
        HybridSystem_->AddSubMatrix(local_multiplier, local_multiplier,
                                    tmpHybrid_el);

        // Save element matrix [C 0][M B^T;B 0]^-1[C 0]^T (this is needed
        // only if one wants to construct H1 spectral AMGe preconditioner)
        if (use_spectralAMGe_)
            Hybrid_el_[iAgg] = tmpHybrid_el;
    }
}

/// @todo nonzero BC, solve on true dof
void HybridSolver::Mult(const mfem::BlockVector& Rhs, mfem::BlockVector& Sol) const
{
    RHSTransform(Rhs, Hrhs_);

    // TODO: nonzero b.c.
    // correct right hand side due to boundary condition
    // can this be calculated w/o copy of data on every mult?
    if (ess_multiplier_bc_)
    {
        mfem::SparseMatrix mat_hybrid(*HybridSystem_, false);
        for (int mm = 0; mm < mat_hybrid.Size(); ++mm)
        {
            if (ess_multiplier_dofs_[mm])
            {
                mat_hybrid.EliminateRowCol(mm, 0.0, Hrhs_);
                //mat_hybrid.EliminateRowCol(mm, Mu_(mm), Hrhs_);
            }
        }
    }
    else if (!use_w_)
    {
        if (myid_ == 0)
        {
            Hrhs_[0] = 0.0;
            //mat_hybrid.EliminateRowCol(0, 0., Hrhs_);
        }
    }

    // assemble true right hand side
    multiplier_d_td_->MultTranspose(Hrhs_, trueHrhs_);

    if (diagonal_scaling_.Size() > 0)
        RescaleVector(diagonal_scaling_, trueHrhs_);

    // solve the parallel global hybridized system
    mfem::StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    cg_->Mult(trueHrhs_, trueMu_);

    chrono.Stop();
    timing_ += chrono.RealTime();

    if (myid_ == 0 && print_level_ > 0)
    {
        std::cout << "  Timing: PCG done in "
                  << timing_ << "s. \n";
    }

    num_iterations_ += cg_->GetNumIterations();

    if (myid_ == 0 && print_level_ > 0)
    {
        if (cg_->GetConverged())
            std::cout << "  CG converged in "
                      << num_iterations_
                      << " with a final residual norm "
                      << cg_->GetFinalNorm() << "\n";
        else
            std::cout << "  CG did not converge in "
                      << num_iterations_
                      << ". Final residual norm is "
                      << cg_->GetFinalNorm() << "\n";
    }

    // distribute true dofs to dofs and recover solution of the original system
    chrono.Clear();
    chrono.Start();

    if (diagonal_scaling_.Size() > 0)
        RescaleVector(diagonal_scaling_, trueMu_);

    multiplier_d_td_->Mult(trueMu_, Mu_);
    RecoverOriginalSolution(Mu_, Sol);

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
    const mfem::Vector& OriginalRHS_block2(OriginalRHS.GetBlock(1));

    HybridRHS = 0.;

    mfem::Vector f_loc, CMinvDTAinv_f_loc;
    for (int iAgg = 0; iAgg < nAggs_; ++iAgg)
    {
        // Extracting the size and global numbering of local dof
        mfem::Array<int> local_vertexdof, local_multiplier;
        GetTableRow(Agg_vertexdof_, iAgg, local_vertexdof);
        GetTableRow(Agg_multiplier_, iAgg, local_multiplier);

        int nlocal_vertexdof = local_vertexdof.Size();
        int nlocal_multiplier = local_multiplier.Size();

        // Compute local contribution to the RHS of the hybrid system
        OriginalRHS_block2.GetSubVector(local_vertexdof, f_loc);
        f_loc *= -1.0;

        CMinvDTAinv_f_loc.SetSize(nlocal_multiplier);
        AinvDMinvCT_[iAgg].MultTranspose(f_loc, CMinvDTAinv_f_loc);

        for (int i = 0; i < nlocal_multiplier; ++i)
            HybridRHS(local_multiplier[i]) -= CMinvDTAinv_f_loc(i);

        // Save the element rhs [M B^T;B 0]^-1[f;g] for solution recovery
        Ainv_f_[iAgg].SetSize(nlocal_vertexdof);
        Ainv_[iAgg].Mult(f_loc, Ainv_f_[iAgg]);
    }
}

void HybridSolver::RecoverOriginalSolution(const mfem::Vector& HybridSol,
                                           mfem::BlockVector& RecoveredSol) const
{
    // Recover the solution of the original system from multiplier mu, i.e.,
    // [u;p] = [f;g] - [M B^T;B 0]^-1[C 0]^T * mu
    // This procedure is done locally in each element

    RecoveredSol = 0.;

    mfem::Vector& RecoveredSol_block2(RecoveredSol.GetBlock(1));

    mfem::Vector mu_loc, sigma_loc;
    for (int iAgg = 0; iAgg < nAggs_; ++iAgg)
    {
        // Extracting the size and global numbering of local dof
        mfem::Array<int> local_vertexdof, local_edgedof, local_multiplier;
        GetTableRow(Agg_vertexdof_, iAgg, local_vertexdof);
        GetTableRow(Agg_edgedof_, iAgg, local_edgedof);
        GetTableRow(Agg_multiplier_, iAgg, local_multiplier);

        int nlocal_vertexdof = local_vertexdof.Size();
        int nlocal_edgedof = local_edgedof.Size();
        int nlocal_multiplier = local_multiplier.Size();

        // Initialize a vector which will store the local contribution of Hdiv
        // and L2 space
        mfem::Vector& u_loc(Ainv_f_[iAgg]);

        // This check is just for the case when there is only one element for
        // the global problem, then there will be no Lagrange multipliers
        if (nlocal_multiplier > 0)
        {
            // Extract the local portion of the Lagrange multiplier solution
            HybridSol.GetSubVector(local_multiplier, mu_loc);

            // Compute u = (DMinvDT)^-1(f-DMinvC^T mu)
            AinvDMinvCT_[iAgg].AddMult_a(-1., mu_loc, u_loc);

            // Compute -sigma = Minv(DT u + DT mu)
            sigma_loc.SetSize(nlocal_edgedof);
            MinvDT_[iAgg].Mult(u_loc, sigma_loc);
            MinvCT_[iAgg].AddMult(mu_loc, sigma_loc);
        }

        // Save local solution to the global solution vector
        for (int i = 0; i < nlocal_vertexdof; ++i)
            RecoveredSol_block2(local_vertexdof[i]) = u_loc(i);

        for (int i = 0; i < nlocal_edgedof; ++i)
            RecoveredSol(local_edgedof[i]) = -sigma_loc(i);
    }
}

void HybridSolver::ComputeScaledHybridSystem(const mfem::HypreParMatrix& H_d)
{
    unique_ptr<mfem::HypreParMatrix> tmpH(smoothg::RAP(H_d, *multiplier_d_td_));
    mfem::HypreSmoother prec_scale(*tmpH);

    mfem::Vector zeros(tmpH->Height());
    zeros = 1e-8;
    diagonal_scaling_.SetSize(tmpH->Height());
    diagonal_scaling_ = 1.0;

    mfem::CGSolver cg_scale(comm_);
    cg_scale.SetMaxIter(rescale_iter_);
    cg_scale.SetPreconditioner(prec_scale);
    cg_scale.SetOperator(*tmpH);
    cg_scale.Mult(zeros, diagonal_scaling_);

    auto Scale = VectorToMatrix(diagonal_scaling_);
    mfem::HypreParMatrix pScale(comm_, tmpH->N(), tmpH->ColPart(), &Scale);
    pHybridSystem_.reset(smoothg::RAP(*tmpH, pScale));
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
                  *pHybridSystem_, num_elems, elem_dof, elem_elem, nullptr, bdr_dofs.data(),
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
    sa_ml_data_ = saamge::ml_produce_data(*pHybridSystem_, sa_apr_, emp, mlp);
    auto level = saamge::levels_list_get_level(sa_ml_data_->levels_list, 0);

    prec_ = make_unique<saamge::VCycleSolver>(level->tg_data, false);
    prec_->SetOperator(*pHybridSystem_);
#else
    if (myid_ == 0)
        std::cout << "SAAMGE needs to be enabled! \n";
    std::abort();
#endif
}

void HybridSolver::SetPrintLevel(int print_level)
{
    MixedLaplacianSolver::SetPrintLevel(print_level);

    cg_->SetPrintLevel(print_level_);
}

void HybridSolver::SetMaxIter(int max_num_iter)
{
    MixedLaplacianSolver::SetMaxIter(max_num_iter);

    cg_->SetMaxIter(max_num_iter_);
}

void HybridSolver::SetRelTol(double rtol)
{
    MixedLaplacianSolver::SetMaxIter(rtol);

    cg_->SetRelTol(rtol_);
}

void HybridSolver::SetAbsTol(double atol)
{
    MixedLaplacianSolver::SetAbsTol(atol);

    cg_->SetAbsTol(atol_);
}

} // namespace smoothg
