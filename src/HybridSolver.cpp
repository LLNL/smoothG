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

HybridSolver::HybridSolver(const MixedMatrix& mgL,
                           const mfem::Array<int>* ess_attr,
                           const int rescale_iter,
                           const SAAMGeParam* saamge_param)
    :
    MixedLaplacianSolver(mgL, ess_attr),
    D_(mgL.GetD()),
    W_(mgL.GetW()),
    rescale_iter_(rescale_iter),
    saamge_param_(saamge_param)
{
    MPI_Comm_rank(comm_, &myid_);

    auto mbuilder = dynamic_cast<const ElementMBuilder*>(&(mgL.GetMBuilder()));
    if (!mbuilder)
    {
        std::cout << "HybridSolver requires fine level M builder to be FineMBuilder!\n";
        std::abort();
    }

    Agg_vertexdof_.MakeRef(mgL.GetGraphSpace().VertexToVDof());
    Agg_edgedof_.MakeRef(mgL.GetGraphSpace().VertexToEDof());

    Init(mgL.GetGraphSpace().EdgeToEDof(), mbuilder->GetElementMatrices(),
         mgL.GetEdgeDofToTrueDof(), mgL.EDofToBdrAtt(), &ess_edofs_);
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
    const mfem::SparseMatrix& edgedof_bdrattr,
    const mfem::Array<int>* ess_edge_dofs)
{
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
    Hybrid_el_.resize(nAggs_);

    agg_weights_.SetSize(nAggs_);
    agg_weights_ = 1.0;

    edgedof_d_td.GetDiag(edgedof_IsOwned_);

    // Constructing the relation table (in SparseMatrix format) between edge
    // dof and multiplier dof. For every edge dof that is associated with a
    // face, a Lagrange multiplier dof associated with the edge dof is created
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
    j_array.Copy(multiplier_to_edgedof_);

    Agg_multiplier_.Clear();
    mfem::SparseMatrix Agg_m_tmp(smoothg::Mult(Agg_edgedof_, edgedof_multiplier));
    Agg_multiplier_.Swap(Agg_m_tmp);

    GenerateOffsets(comm_, num_multiplier_dofs_, multiplier_start_);

    unique_ptr<mfem::HypreParMatrix> multiplier_trueedgedof(
        edgedof_d_td.LeftDiagMult(multiplier_edgedof, multiplier_start_) );
    unique_ptr<mfem::HypreParMatrix> multiplier_d_td_d(AAt(*multiplier_trueedgedof));

    // Construct multiplier "dof to true dof" table
    multiplier_d_td_ = BuildEntityToTrueEntity(*multiplier_d_td_d);

    // Assemble the hybridized system
    HybridSystem_ = make_unique<mfem::SparseMatrix>(num_multiplier_dofs_);
    AssembleHybridSystem(M_el);
    if (myid_ == 0 && print_level_ > 0)
        std::cout << "  Timing: Hybridized system built in "
                  << chrono.RealTime() << "s. \n";

    // Mark the multiplier dof with essential BC
    // Note again there is a 1-1 map from multipliers to edge dofs on faces
    ess_multiplier_bc_ = false;
    if (edgedof_bdrattr.Width())
    {
        ess_multiplier_dofs_.SetSize(num_multiplier_dofs_, 0);
        for (int i = 0; i < num_multiplier_dofs_; i++)
        {
            // natural BC for H(div) dof <=> essential BC for multiplier dof
            if (edgedof_bdrattr.RowSize(i) && !(*ess_edge_dofs)[i])
            {
                ess_multiplier_dofs_[i] = 1;
                ess_multiplier_bc_ = true;
            }
        }
    }

    BuildParallelSystemAndSolver();

    trueHrhs_.SetSize(multiplier_d_td_->GetNumCols());
    trueMu_.SetSize(trueHrhs_.Size());
    Hrhs_.SetSize(num_multiplier_dofs_);
    Mu_.SetSize(num_multiplier_dofs_);
}

void HybridSolver::AssembleHybridSystem(const std::vector<mfem::DenseMatrix>& M_el)
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
            const int edgedof_global_id = multiplier_to_edgedof_[local_multiplier[i]];
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

        // Save element matrix [C 0][M B^T;B 0]^-1[C 0]^T
        Hybrid_el_[iAgg] = tmpHybrid_el;
    }
}


/// @todo nonzero BC, solve on true dof
void HybridSolver::Mult(const mfem::BlockVector& Rhs, mfem::BlockVector& Sol) const
{
    RHSTransform(Rhs, Hrhs_);

    // TODO: nonzero b.c.
    // correct right hand side due to boundary condition
    if (ess_multiplier_bc_)
    {
        for (int m = 0; m < ess_multiplier_dofs_.Size(); ++m)
        {
            if (ess_multiplier_dofs_[m])
            {
                Mu_(m) = -1.0 * Rhs(multiplier_to_edgedof_[m]);
                Hrhs_(m) = Mu_(m);
            }
            else
            {
                Mu_(m) = 0.0;
            }
        }
        HybridSystemElim_->AddMult(Mu_, Hrhs_, -1.0);
    }

    // assemble true right hand side
    multiplier_d_td_->MultTranspose(Hrhs_, trueHrhs_);

    if (!ess_multiplier_bc_ && !W_is_nonzero_ && myid_ == 0)
    {
        trueHrhs_[0] = 0.0;
    }

    if (diagonal_scaling_.Size() > 0)
        RescaleVector(diagonal_scaling_, trueHrhs_);

    // solve the parallel global hybridized system
    mfem::StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    cg_->Mult(trueHrhs_, trueMu_);

    chrono.Stop();
    timing_ = chrono.RealTime();

    if (myid_ == 0 && print_level_ > 0)
    {
        std::cout << "  Timing: PCG done in "
                  << timing_ << "s. \n";
    }

    // TODO: decide to use = or += here and in timing_ update (MinresBlockSolver uses +=)
    num_iterations_ = cg_->GetNumIterations();

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

    if (!W_is_nonzero_ && remove_one_dof_ )
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

        // Save the element rhs (DMinvDT)^-1 f for solution recovery
        Ainv_f_[iAgg].SetSize(nlocal_vertexdof);
        Ainv_[iAgg].Mult(f_loc, Ainv_f_[iAgg]);
        Ainv_f_[iAgg] *= agg_weights_(iAgg);
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
            sigma_loc /= agg_weights_(iAgg);
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

void HybridSolver::BuildParallelSystemAndSolver()
{
    HybridSystem_->Finalize();
    HybridSystemElim_ = make_unique<mfem::SparseMatrix>(*HybridSystem_, false);
    if (ess_multiplier_bc_)
    {
        // eliminate the essential dofs and save the eliminated part
        HybridSystemElim_ = make_unique<mfem::SparseMatrix>(num_multiplier_dofs_);
        for (int mm = 0; mm < num_multiplier_dofs_; ++mm)
        {
            if (ess_multiplier_dofs_[mm])
            {
                HybridSystem_->EliminateRowCol(mm, *HybridSystemElim_);
            }
        }

        // Only keep H_{ib} block for elimination later
        for (int mm = 0; mm < num_multiplier_dofs_; ++mm)
        {
            if (ess_multiplier_dofs_[mm])
            {
                HybridSystemElim_->EliminateRow(mm);
            }
        }
    }

    auto HybridSystem_d = make_unique<mfem::HypreParMatrix>(
                              comm_, multiplier_start_.Last(), multiplier_start_,
                              HybridSystemElim_.get());

    if (rescale_iter_ == 0 || saamge_param_)
    {
        pHybridSystem_.reset(smoothg::RAP(*HybridSystem_d, *multiplier_d_td_));
    }
    else
    {
        ComputeScaledHybridSystem(*HybridSystem_d);
    }
    nnz_ = pHybridSystem_->NNZ();

    mfem::Array<int> ess_dof;
    if (!ess_multiplier_bc_ && !W_is_nonzero_ && myid_ == 0)
    {
        ess_dof.Append(0);
    }
    mfem::HypreParVector junk_vec1(*pHybridSystem_);
    mfem::HypreParVector junk_vec2(*pHybridSystem_);
    pHybridSystem_->EliminateRowsCols(ess_dof, junk_vec1, junk_vec2);


    mfem::StopWatch chrono;
    chrono.Clear();
    chrono.Start();

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
        if (saamge_param_)
        {
            BuildSpectralAMGePreconditioner();
        }
        else
        {
            auto temp_prec = make_unique<mfem::HypreBoomerAMG>(*pHybridSystem_);
            temp_prec->SetPrintLevel(0);
            prec_ = std::move(temp_prec);
        }
        cg_->SetPreconditioner(*prec_);
    }
    if (myid_ == 0 && print_level_ > 0)
        std::cout << "  Timing: Preconditioner for hybridized system"
                  " constructed in " << chrono.RealTime() << "s. \n";
}

void HybridSolver::UpdateAggScaling(const mfem::Vector& agg_weights_inverse)
{
    // This is for consistency, could simply work with agg_weight_inverse
    agg_weights_.SetSize(agg_weights_inverse.Size());
    for (int i = 0; i < agg_weights_.Size(); ++i)
    {
        agg_weights_[i] = 1.0 / agg_weights_inverse[i];
    }

    // TODO: this is not valid when W is nonzero
    assert(W_is_nonzero_ == false);

    HybridSystem_ = make_unique<mfem::SparseMatrix>(num_multiplier_dofs_);
    mfem::Array<int> local_multiplier;
    for (int iAgg = 0; iAgg < nAggs_; ++iAgg)
    {
        GetTableRow(Agg_multiplier_, iAgg, local_multiplier);
        mfem::DenseMatrix H_el = Hybrid_el_[iAgg]; // deep copy
        H_el *= (1.0 / agg_weights_(iAgg));
        HybridSystem_->AddSubMatrix(local_multiplier, local_multiplier, H_el);
    }
    BuildParallelSystemAndSolver();
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
    MixedLaplacianSolver::SetRelTol(rtol);

    cg_->SetRelTol(rtol_);
}

void HybridSolver::SetAbsTol(double atol)
{
    MixedLaplacianSolver::SetAbsTol(atol);

    cg_->SetAbsTol(atol_);
}

} // namespace smoothg
