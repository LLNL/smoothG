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

using std::unique_ptr;
using std::shared_ptr;

namespace smoothg
{

void HybridSolver::BuildFineLevelLocalMassMatrix(
    const mfem::SparseMatrix& vertex_edge,
    const mfem::SparseMatrix& M,
    std::vector<unique_ptr<mfem::Vector> >& M_el)
{
    int nvertices = vertex_edge.Height();
    M_el.resize(nvertices);

    unique_ptr<mfem::SparseMatrix> edge_vertex( Transpose(vertex_edge) );
    mfem::Array<int> local_edgedof;

    mfem::Vector Mloc;
    double* M_data = M.GetData();
    int edgedof, nlocal_edgedof;
    for (int i = 0; i < nvertices; i++)
    {
        GetTableRow(vertex_edge, i, local_edgedof);
        nlocal_edgedof = local_edgedof.Size();

        Mloc.SetSize(nlocal_edgedof);
        for (int j = 0; j < nlocal_edgedof; j++)
        {
            edgedof = local_edgedof[j];
            if (edge_vertex->RowSize(edgedof) == 2)
                Mloc(j) = M_data[edgedof] / 2;
            else
                Mloc(j) = M_data[edgedof];
        }

        M_el[i] = make_unique<mfem::Vector>(Mloc);
    }
}

HybridSolver::HybridSolver(MPI_Comm comm,
                           const MixedMatrix& mgL,
                           shared_ptr<const mfem::SparseMatrix> face_bdrattr,
                           shared_ptr<const mfem::Array<int> > ess_edge_dofs,
                           bool spectralAMGe)
    :
    comm_(comm),
    D_(mgL.getD())
{
    MPI_Comm_rank(comm, &myid_);
    int nvertices = D_.Height();

    Agg_vertexdof_.Swap(*SparseIdentity(nvertices));
    Agg_edgedof_.MakeRef(D_);
    const mfem::SparseMatrix edge_edgedof;

    std::vector<unique_ptr<mfem::Vector> > M_el;
    BuildFineLevelLocalMassMatrix(D_, mgL.getWeight(), M_el);

    Init(edge_edgedof, M_el, mgL.get_edge_d_td(),
         face_bdrattr, ess_edge_dofs, spectralAMGe);
}

HybridSolver::HybridSolver(MPI_Comm comm,
                           const MixedMatrix& mgL,
                           const Mixed_GL_Coarsener& mgLc,
                           shared_ptr<const mfem::SparseMatrix> face_bdrattr,
                           shared_ptr<const mfem::Array<int> > ess_edge_dofs,
                           bool spectralAMGe)
    :
    comm_(comm),
    D_(mgL.getD())
{
    MPI_Comm_rank(comm, &myid_);
    const mfem::SparseMatrix& face_edgedof(mgLc.construct_face_facedof_table());

    Agg_vertexdof_.MakeRef(mgLc.construct_Agg_cvertexdof_table());
    Agg_edgedof_.MakeRef(mgLc.construct_Agg_cedgedof_table());

    Init(face_edgedof, mgLc.get_CM_el(), mgL.get_edge_d_td(),
         face_bdrattr, ess_edge_dofs, spectralAMGe);
}

template<typename T>
void HybridSolver::Init(const mfem::SparseMatrix& face_edgedof,
                        const std::vector<std::unique_ptr<T> >& M_el,
                        const mfem::HypreParMatrix& edgedof_d_td,
                        shared_ptr<const mfem::SparseMatrix> face_bdrattr,
                        shared_ptr<const mfem::Array<int> > ess_edge_dofs,
                        bool spectralAMGe)
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
    if (spectralAMGe)
        Hybrid_el_.resize(nAggs_);

    shared_ptr<const mfem::SparseMatrix> edgedof_bdrattr;
    if (face_bdrattr)
    {
        if (fine_level)
        {
            edgedof_bdrattr = face_bdrattr;
        }
        else
        {
            unique_ptr<mfem::SparseMatrix> edgedof_face(Transpose(face_edgedof));
            edgedof_bdrattr.reset( mfem::Mult(*edgedof_face, *face_bdrattr) );
        }
    }

    mfem::HypreParMatrix edgedof_d_td_;
    edgedof_d_td_.MakeRef(edgedof_d_td);
    edgedof_d_td_.GetDiag(edgedof_IsOwned_);

    // Constructing the relation table (in SparseMatrix format) between edge
    // dof and multiplier dof. For every edge dof that is associated with a
    // face, a Lagrange multiplier dof associated with the edge dof is created
    unique_ptr<mfem::SparseMatrix> edgedof_multiplier, multiplier_edgedof;
    unique_ptr<mfem::HypreParMatrix> edgedof_d_td_d;
    mfem::Array<int> j_multiplier_edgedof;
    if (fine_level)
    {
        num_multiplier_dofs_ = num_edge_dofs_;
        j_multiplier_edgedof.SetSize(num_edge_dofs_);
        std::iota(j_multiplier_edgedof.GetData(), j_multiplier_edgedof.GetData() + num_edge_dofs_, 0);
        Agg_multiplier_ = make_unique<mfem::SparseMatrix>();
        Agg_multiplier_->MakeRef(Agg_edgedof_);
    }
    else
    {
        unique_ptr<mfem::HypreParMatrix> edgedof_td_d(edgedof_d_td_.Transpose());
        edgedof_d_td_d.reset( ParMult(&edgedof_d_td_, edgedof_td_d.get()) );

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
        edgedof_multiplier = make_unique<mfem::SparseMatrix>(
                                 i_edgedof_multiplier, j_edgedof_multiplier,
                                 data_edgedof_multiplier, num_edge_dofs_, num_multiplier_dofs_);
        multiplier_edgedof.reset( Transpose(*edgedof_multiplier) );

        j_multiplier_edgedof.MakeRef(multiplier_edgedof->GetJ(), num_edge_dofs_);
        Agg_multiplier_.reset( mfem::Mult(Agg_edgedof_, *edgedof_multiplier) );
    }

    mfem::Array<int> edgedof_global_to_local_map(num_edge_dofs_);
    edgedof_global_to_local_map = -1;
    mfem::Array<bool> edge_marker(num_edge_dofs_);
    edge_marker = true;

    // Assemble the hybridized system
    HybridSystem_ = make_unique<mfem::SparseMatrix>(num_multiplier_dofs_);
    AssembleHybridSystem(M_el, edgedof_global_to_local_map, edge_marker,
                         j_multiplier_edgedof.GetData(), spectralAMGe);
    HybridSystem_->Finalize();

    // Mark the multiplier dof with essential BC
    // Note again there is a 1-1 map from multipliers to edge dofs on faces
    ess_multiplier_bc_ = false;
    if (face_bdrattr)
    {
        ess_multiplier_dofs_.SetSize(num_multiplier_dofs_);
        for (int i = 0; i < num_multiplier_dofs_; i++)
        {
            // natural BC for H(div) dof <=> essential BC for multiplier dof
            if (edgedof_bdrattr->RowSize(i) & !ess_edge_dofs->operator[](i))
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
            if (ess_multiplier_dofs_[mm])
                HybridSystemElim_->EliminateRowCol(mm);
    }
    else
    {
        if (myid_ == 0)
            HybridSystemElim_->EliminateRowCol(0);
    }

    // construct multiplier dof to true dof table
    mfem::Array<HYPRE_Int>* start[1] = {&multiplier_start_};
    GenerateOffsets(comm_, 1, &num_multiplier_dofs_, start);
    if (fine_level)
    {
        multiplier_d_td_ = make_unique<mfem::HypreParMatrix>();
        multiplier_d_td_->MakeRef(edgedof_d_td_);
    }
    else
    {
        auto edgedof_multiplier_d = make_unique<mfem::HypreParMatrix>(
                                        comm_, edgedof_d_td_.GetGlobalNumRows(),
                                        multiplier_start_.Last(), edgedof_d_td_.RowPart(),
                                        multiplier_start_, edgedof_multiplier.get());

        unique_ptr<mfem::HypreParMatrix> multiplier_d_td_d(
            RAP(edgedof_d_td_d.get(), edgedof_multiplier_d.get()) );

        // Create a selection matrix to set dofs on true faces to be true dofs
        hypre_ParCSRMatrix* multiplier_shared = *multiplier_d_td_d;
        HYPRE_Int* multiplier_shared_i = multiplier_shared->offd->i;
        HYPRE_Int* multiplier_shared_j = multiplier_shared->offd->j;
        HYPRE_Int* multiplier_shared_map = multiplier_shared->col_map_offd;
        HYPRE_Int maxmultiplier = multiplier_shared->last_row_index;

        // Create a selection matrix to pick one of the processors sharing a true
        // face to own the true face (we pick the processor with a smaller index)
        int* select_i = new int[num_multiplier_dofs_ + 1];
        int ntruemultipliers = 0;
        for (int i = 0; i < num_multiplier_dofs_; i++)
        {
            select_i[i] = ntruemultipliers;
            if (multiplier_shared_i[i + 1] == multiplier_shared_i[i])
                ntruemultipliers++;
            else if (multiplier_shared_map[ multiplier_shared_j[
                                                multiplier_shared_i[i] ] ] > maxmultiplier)
                ntruemultipliers++;
        }
        select_i[num_multiplier_dofs_] = ntruemultipliers;
        int* select_j = new int[ntruemultipliers];
        double* select_data = new double[ntruemultipliers];
        std::iota(select_j, select_j + ntruemultipliers, 0);
        std::fill_n(select_data, ntruemultipliers, 1.);
        mfem::SparseMatrix select(select_i, select_j, select_data,
                                  num_multiplier_dofs_, ntruemultipliers);

        // Construct a (block diagonal) global select matrix from local
        start[0] = &truemultiplier_start_;
        GenerateOffsets(comm_, 1, &ntruemultipliers, start);
        mfem::HypreParMatrix select_d(
            comm_, multiplier_start_.Last(), truemultiplier_start_.Last(),
            multiplier_start_, truemultiplier_start_, &select);

        // Construct face "dof to true dof" table
        multiplier_d_td_.reset( ParMult(multiplier_d_td_d.get(), &select_d) );
    }

    auto HybridSystem_d = make_unique<mfem::HypreParMatrix>(
                              comm_, multiplier_start_.Last(), multiplier_start_,
                              HybridSystemElim_.get());

    // This is just doing RAP, but for some reason for large systems this two-
    // step RAP is much faster than a direct RAP, so the two-step way is used
    unique_ptr<mfem::HypreParMatrix>multiplier_td_d(
        multiplier_d_td_->Transpose() );
    unique_ptr<mfem::HypreParMatrix> Htmp(
        ParMult(HybridSystem_d.get(), multiplier_d_td_.get()) );
    pHybridSystem_.reset(ParMult(multiplier_td_d.get(), Htmp.get()));
    pHybridSystem_->CopyRowStarts();
    hypre_ParCSRMatrixSetNumNonzeros(*pHybridSystem_);
    nnz_ = pHybridSystem_->NNZ();

    if (myid_ == 0)
        std::cout << "  Timing: Hybridized system built in "
                  << chrono.RealTime() << "s. \n";

    chrono.Clear();
    chrono.Start();
    prec_ = make_unique<mfem::HypreBoomerAMG>(*pHybridSystem_);
    prec_->SetPrintLevel(0);
    if (myid_ == 0)
        std::cout << "  Timing: Preconditioner for hybridized system"
                  " constructed in " << chrono.RealTime() << "s. \n";

    cg_ = make_unique<mfem::CGSolver>(comm_);
    cg_->SetPrintLevel(print_level_);
    cg_->SetMaxIter(max_num_iter_);
    cg_->SetRelTol(rtol_);
    cg_->SetAbsTol(atol_);
    cg_->SetOperator(*pHybridSystem_);
    cg_->SetPreconditioner(*prec_);
}

void HybridSolver::AssembleHybridSystem(
    const std::vector<std::unique_ptr<mfem::DenseMatrix> >& M_el,
    mfem::Array<int>& edgedof_global_to_local_map,
    mfem::Array<bool>& edge_marker,
    int* j_multiplier_edgedof,
    bool spectralAMGe)
{
    mfem::DenseMatrix DlocT, ClocT, Aloc, CMinvDT, DMinvCT, CMDADMC;
    unique_ptr<mfem::SparseMatrix> Dloc, Cloc;
    unique_ptr<mfem::DenseMatrix> tmpAinvCT, tmpMinvDT, tmpMinvCT, tmpAinvDMinvCT;
    unique_ptr<mfem::DenseMatrix> tmpHybrid_el;
    if (!spectralAMGe)
        tmpHybrid_el = make_unique<mfem::DenseMatrix>(0, 0);
    unique_ptr<mfem::DenseMatrixInverse> solver;

    mfem::DenseMatrixInverse Mloc_solver;
    for (int iAgg = 0; iAgg < nAggs_; ++iAgg)
    {
        // Extracting the size and global numbering of local dof
        mfem::Array<int> local_vertexdof, local_edgedof, local_multiplier;
        GetTableRow(Agg_vertexdof_, iAgg, local_vertexdof);
        GetTableRow(Agg_edgedof_, iAgg, local_edgedof);
        GetTableRow(*Agg_multiplier_, iAgg, local_multiplier);

        int nlocal_vertexdof = local_vertexdof.Size();
        int nlocal_edgedof = local_edgedof.Size();
        int nlocal_multiplier = local_multiplier.Size();

        // Build the edge dof global to local map which will be used
        // later for mapping local multiplier dof to local edge dof
        for (int i = 0; i < nlocal_edgedof; ++i)
            edgedof_global_to_local_map[local_edgedof[i]] = i;

        // Extract Dloc as a sparse submatrix of D_
        Dloc = ExtractRowAndColumns(D_, local_vertexdof, local_edgedof,
                                    edgedof_global_to_local_map, false);

        // Fill DlocT as a dense matrix of Dloc^T
        FullTranspose(*Dloc, DlocT);

        // Construct the constraint matrix C which enforces the continuity of
        // the broken edge space
        int edgedof_global_id, edgedof_local_id;
        ClocT.SetSize(nlocal_edgedof, nlocal_multiplier);
        ClocT = 0.;
        int* Cloc_i = new int[nlocal_multiplier + 1];
        std::iota(Cloc_i, Cloc_i + nlocal_multiplier + 1, 0);
        int* Cloc_j = new int[nlocal_multiplier];
        double* Cloc_data = new double[nlocal_multiplier];
        for (int i = 0; i < nlocal_multiplier; ++i)
        {
            edgedof_global_id = j_multiplier_edgedof[local_multiplier[i]];
            edgedof_local_id = edgedof_global_to_local_map[edgedof_global_id];
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
        Cloc = make_unique<mfem::SparseMatrix>(Cloc_i, Cloc_j, Cloc_data,
                                               nlocal_multiplier,
                                               nlocal_edgedof);
        for (int i = 0; i < nlocal_edgedof; ++i)
            edgedof_global_to_local_map[local_edgedof[i]] = -1;

        Mloc_solver.SetOperator(*M_el[iAgg]);
        tmpMinvDT = make_unique<mfem::DenseMatrix>(nlocal_edgedof,
                                                   nlocal_vertexdof);
        tmpMinvCT = make_unique<mfem::DenseMatrix>(nlocal_edgedof,
                                                   nlocal_multiplier);
        Mloc_solver.Mult(DlocT, *tmpMinvDT);
        Mloc_solver.Mult(ClocT, *tmpMinvCT);

        if (spectralAMGe)
            tmpHybrid_el = make_unique<mfem::DenseMatrix>(nlocal_multiplier,
                                                          nlocal_multiplier);
        else
            tmpHybrid_el->SetSize(nlocal_multiplier, nlocal_multiplier);

        // Compute CMinvCT = Cloc * tmpMinvCT
        MultSparseDense(*Cloc, *tmpMinvCT, *tmpHybrid_el);

        // Compute Aloc = DMinvDT = Dloc * tmpMinvDT
        Aloc.SetSize(nlocal_vertexdof, nlocal_vertexdof);
        MultSparseDense(*Dloc, *tmpMinvDT, Aloc);

        // Compute DMinvCT Dloc * tmpMinvCT
        DMinvCT.SetSize(nlocal_vertexdof, nlocal_multiplier);
        MultSparseDense(*Dloc, *tmpMinvCT, DMinvCT);

        // Compute the LU factorization of Aloc and Ainv_ * DMinvCT
        solver = make_unique<mfem::DenseMatrixInverse>(Aloc);
        tmpAinvDMinvCT = make_unique<mfem::DenseMatrix>();
        solver->Mult(DMinvCT, *tmpAinvDMinvCT);

        // Compute CMinvDTAinvDMinvCT = CMinvDT * AinvDMinvCT_
        CMinvDT.Transpose(DMinvCT);
        CMDADMC.SetSize(nlocal_multiplier, nlocal_multiplier);
        mfem::Mult(CMinvDT, *tmpAinvDMinvCT, CMDADMC);

        // Hybrid_el_ = CMinvCT - CMinvDTAinvDMinvCT
        *tmpHybrid_el -= CMDADMC;

        // Add contribution of the element matrix to the golbal system
        HybridSystem_->AddSubMatrix(local_multiplier, local_multiplier,
                                    *tmpHybrid_el);

        // Save the factorization of DMinvDT for solution recovery
        Ainv_[iAgg] = std::move(solver);

        // Save some element matrices for solution recover
        MinvDT_[iAgg] = std::move(tmpMinvDT);
        MinvCT_[iAgg] = std::move(tmpMinvCT);
        AinvDMinvCT_[iAgg] = std::move(tmpAinvDMinvCT);

        // Save element matrix [C 0][M B^T;B 0]^-1[C 0]^T (this is needed
        // only if one wants to construct H1 spectral AMGe preconditioner)
        if (spectralAMGe)
            Hybrid_el_[iAgg] = std::move(tmpHybrid_el);
    }
}

void HybridSolver::AssembleHybridSystem(
    const std::vector<std::unique_ptr<mfem::Vector> >& M_el,
    mfem::Array<int>& edgedof_global_to_local_map,
    mfem::Array<bool>& edge_marker,
    int* j_multiplier_edgedof,
    bool spectralAMGe)
{
    mfem::DenseMatrix Dloc, Aloc;
    unique_ptr<mfem::SparseMatrix> Cloc;
    unique_ptr<mfem::DenseMatrix> tmpAinvCT, tmpMinvDT, tmpMinvCT;
    unique_ptr<mfem::DenseMatrix> tmpAinvDMinvCT, tmpHybrid_el;
    if (!spectralAMGe)
        tmpHybrid_el = make_unique<mfem::DenseMatrix>(0, 0);
    unique_ptr<mfem::DenseMatrix> solver;

    mfem::Vector column_in, CMinvCT, CMinvDT;
    double A00_inv;
    for (int iAgg = 0; iAgg < nAggs_; ++iAgg)
    {
        // Extracting the size and global numbering of local dof
        mfem::Array<int> local_vertexdof, local_edgedof, local_multiplier;
        GetTableRow(Agg_vertexdof_, iAgg, local_vertexdof);
        GetTableRow(Agg_edgedof_, iAgg, local_edgedof);
        GetTableRow(*Agg_multiplier_, iAgg, local_multiplier);

        int nlocal_vertexdof = local_vertexdof.Size();
        int nlocal_edgedof = local_edgedof.Size();
        int nlocal_multiplier = local_multiplier.Size();

        // Build the edge dof global to local map which will be used
        // later for mapping local multiplier dof to local edge dof
        for (int i = 0; i < nlocal_edgedof; ++i)
            edgedof_global_to_local_map[local_edgedof[i]] = i;

        // Extract tmpMinvDT as a dense submatrix of D^T
        ExtractSubMatrix(D_, local_vertexdof, local_edgedof,
                         edgedof_global_to_local_map, Dloc);
        tmpMinvDT = make_unique<mfem::DenseMatrix>(Dloc, 't');

        // Construct the constraint matrix C which enforces the continuity of
        // the broken edge space
        int edgedof_global_id, edgedof_local_id;
        tmpMinvCT = make_unique<mfem::DenseMatrix>(nlocal_edgedof,
                                                   nlocal_multiplier);
        int* Cloc_i = new int[nlocal_multiplier + 1];
        std::iota(Cloc_i, Cloc_i + nlocal_multiplier + 1, 0);
        int* Cloc_j = new int[nlocal_multiplier];
        double* Cloc_data = new double[nlocal_multiplier];

        mfem::Vector& M_diag(*M_el[iAgg]);
        CMinvCT.SetSize(nlocal_multiplier);
        for (int i = 0; i < nlocal_multiplier; ++i)
        {
            edgedof_global_id = j_multiplier_edgedof[local_multiplier[i]];
            edgedof_local_id = edgedof_global_to_local_map[edgedof_global_id];
            Cloc_j[i] = edgedof_local_id;
            CMinvCT(i) = 1. / M_diag(edgedof_local_id);
            if (edgedof_IsOwned_.RowSize(edgedof_global_id) &&
                edge_marker[edgedof_global_id])
            {
                edge_marker[edgedof_global_id] = false;
                tmpMinvCT->Elem(edgedof_local_id, i) = 1. / M_diag(edgedof_local_id);
                Cloc_data[i] = 1.;
            }
            else
            {
                tmpMinvCT->Elem(edgedof_local_id, i) = -1. / M_diag(edgedof_local_id);
                Cloc_data[i] = -1.;
            }
        }
        Cloc = make_unique<mfem::SparseMatrix>(Cloc_i, Cloc_j, Cloc_data,
                                               nlocal_multiplier,
                                               nlocal_edgedof);
        for (int i = 0; i < nlocal_edgedof; ++i)
            edgedof_global_to_local_map[local_edgedof[i]] = -1;

        tmpMinvDT->InvLeftScaling(M_diag);

        if (spectralAMGe)
            tmpHybrid_el = make_unique<mfem::DenseMatrix>(nlocal_multiplier,
                                                          nlocal_multiplier);
        else
            tmpHybrid_el->SetSize(nlocal_multiplier, nlocal_multiplier);

        // Compute Aloc = DMinvDT = Dloc * tmpMinvDT
        Aloc.SetSize(nlocal_vertexdof, nlocal_vertexdof);
        mfem::Mult(Dloc, *tmpMinvDT, Aloc);

        // Compute CMinvDT  = Cloc * tmpMinvDT
        CMinvDT.SetSize(nlocal_multiplier);
        column_in.SetDataAndSize(tmpMinvDT->Data(), nlocal_multiplier);
        Cloc->Mult(column_in, CMinvDT);


        // Compute the LU factorization of Aloc and Ainv_ * DMinvCT
        solver = make_unique<mfem::DenseMatrix>(nlocal_vertexdof,
                                                nlocal_vertexdof);
        solver->Elem(0, 0) = A00_inv = 1. / Aloc(0, 0);
        tmpAinvDMinvCT = make_unique<mfem::DenseMatrix>(nlocal_vertexdof,
                                                        nlocal_multiplier);
        for (int j = 0; j < nlocal_multiplier; ++j)
            tmpAinvDMinvCT->Elem(0, j) = CMinvDT(j) * A00_inv;

        // Compute -CMinvDTAinvDMinvCT = -CMinvDT * Ainv_ * DMinvCT
        Mult_a_VVt(-A00_inv, CMinvDT, *tmpHybrid_el);

        // Hybrid_el_ = CMinvCT - CMinvDTAinvDMinvCT
        for (int j = 0; j < nlocal_multiplier; ++j)
            tmpHybrid_el->Elem(j, j) += CMinvCT(j);

        // Add contribution of the element matrix to the golbal system
        HybridSystem_->AddSubMatrix(local_multiplier, local_multiplier,
                                    *tmpHybrid_el);

        // Save the factorization of DMinvDT for solution recovery
        Ainv_[iAgg] = std::move(solver);

        // Save some element matrices for solution recover
        MinvDT_[iAgg] = std::move(tmpMinvDT);
        MinvCT_[iAgg] = std::move(tmpMinvCT);
        AinvDMinvCT_[iAgg] = std::move(tmpAinvDMinvCT);

        // Save element matrix [C 0][M B^T;B 0]^-1[C 0]^T (this is needed
        // only if one wants to construct H1 spectral AMGe preconditioner)
        if (spectralAMGe)
            Hybrid_el_[iAgg] = std::move(tmpHybrid_el);
    }
}

/// @todo nonzero BC
void HybridSolver::Mult(const mfem::BlockVector& Rhs, mfem::BlockVector& Sol)
{
    mfem::Vector Hrhs;
    RHSTransform(Rhs, Hrhs);

    mfem::Vector Mu(Hrhs.Size());
    Mu = 0.;

    // TODO: nonzero b.c.
    // correct right hand side due to boundary condition
    mfem::SparseMatrix mat_hybrid(*HybridSystem_, false);
    if (ess_multiplier_bc_)
    {
        for (int mm = 0; mm < mat_hybrid.Size(); ++mm)
            if (ess_multiplier_dofs_[mm])
                mat_hybrid.EliminateRowCol(mm, Mu(mm), Hrhs);
    }
    else
    {
        if (myid_ == 0 )
            mat_hybrid.EliminateRowCol(0, 0., Hrhs);
    }

    // assemble true right hand side
    mfem::Vector trueHrhs(multiplier_d_td_->GetNumCols());
    multiplier_d_td_->MultTranspose(Hrhs, trueHrhs);

    // solve the parallel global hybridized system
    mfem::Vector trueMu(trueHrhs.Size());
    trueMu = 0.;

    mfem::StopWatch chrono;
    chrono.Clear();
    chrono.Start();
    cg_->Mult(trueHrhs, trueMu);
    chrono.Stop();
    timing_ = chrono.RealTime();
    if (myid_ == 0)
        std::cout << "  Timing: PCG done in "
                  << timing_ << "s. \n";

    num_iterations_ = cg_->GetNumIterations();
    if (myid_ == 0)
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

    multiplier_d_td_->Mult(trueMu, Mu);
    RecoverOriginalSolution(Mu, Sol);

    chrono.Stop();
    if (myid_ == 0)
        std::cout << "  Timing: original solution recovered in "
                  << chrono.RealTime() << "s. \n";
}

/// @todo impose nonzero boundary condition for u.n
void HybridSolver::RHSTransform(const mfem::BlockVector& OriginalRHS,
                                mfem::Vector& HybridRHS)
{
    const mfem::Vector& OriginalRHS_block2(OriginalRHS.GetBlock(1));

    HybridRHS.SetSize(num_multiplier_dofs_);
    HybridRHS = 0.;

    mfem::Vector f_loc, CMinvDTAinv_f_loc;
    for (int iAgg = 0; iAgg < nAggs_; ++iAgg)
    {
        // Extracting the size and global numbering of local dof
        mfem::Array<int> local_vertexdof, local_multiplier;
        GetTableRow(Agg_vertexdof_, iAgg, local_vertexdof);
        GetTableRow(*Agg_multiplier_, iAgg, local_multiplier);

        int nlocal_vertexdof = local_vertexdof.Size();
        int nlocal_multiplier = local_multiplier.Size();

        // Compute local contribution to the RHS of the hybrid system
        OriginalRHS_block2.GetSubVector(local_vertexdof, f_loc);

        CMinvDTAinv_f_loc.SetSize(nlocal_multiplier);
        AinvDMinvCT_[iAgg]->MultTranspose(f_loc, CMinvDTAinv_f_loc);

        for (int i = 0; i < nlocal_multiplier; ++i)
            HybridRHS(local_multiplier[i]) -= CMinvDTAinv_f_loc(i);

        // Save the element rhs [M B^T;B 0]^-1[f;g] for solution recovery
        auto Ainv_f_loc = make_unique<mfem::Vector>(nlocal_vertexdof);
        Ainv_[iAgg]->Mult(f_loc, *Ainv_f_loc);
        Ainv_f_[iAgg] = std::move(Ainv_f_loc);
    }
}

void HybridSolver::RecoverOriginalSolution(const mfem::Vector& HybridSol,
                                           mfem::BlockVector& RecoveredSol)
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
        GetTableRow(*Agg_multiplier_, iAgg, local_multiplier);

        int nlocal_vertexdof = local_vertexdof.Size();
        int nlocal_edgedof = local_edgedof.Size();
        int nlocal_multiplier = local_multiplier.Size();

        // Initialize a vector which will store the local contribution of Hdiv
        // and L2 space
        mfem::Vector& u_loc(*Ainv_f_[iAgg]);

        // This check is just for the case when there is only one element for
        // the global problem, then there will be no Lagrange multipliers
        if (nlocal_multiplier > 0)
        {
            // Extract the local portion of the Lagrange multiplier solution
            HybridSol.GetSubVector(local_multiplier, mu_loc);

            // Compute u = (DMinvDT)^-1(f-DMinvC^T mu)
            AinvDMinvCT_[iAgg]->AddMult_a(-1., mu_loc, u_loc);

            // Compute -sigma = Minv(DT u + DT mu)
            sigma_loc.SetSize(nlocal_edgedof);
            MinvDT_[iAgg]->Mult(u_loc, sigma_loc);
            MinvCT_[iAgg]->AddMult(mu_loc, sigma_loc);
        }

        // Save local solution to the global solution vector
        for (int i = 0; i < nlocal_vertexdof; ++i)
            RecoveredSol_block2(local_vertexdof[i]) = u_loc(i);

        for (int i = 0; i < nlocal_edgedof; ++i)
            RecoveredSol(local_edgedof[i]) = -sigma_loc(i);
    }
}

} // namespace smoothg
