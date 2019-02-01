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
    MixedLaplacianSolver(mgL.GetComm(), mgL.BlockOffsets(), mgL.CheckW()),
    D_(mgL.GetD()),
    W_(mgL.GetW()),
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

    Agg_vertexdof_.MakeRef(graph_space.VertexToVDof());
    Agg_edgedof_.MakeRef(graph_space.VertexToEDof());

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

    nAggs_ = Agg_edgedof_.Height();
    num_edge_dofs_ = Agg_edgedof_.Width();

    // Set the size of the Hybrid_el_, AinvCT, Ainv_f_, these are all local
    // matrices and vector for each element
    MinvDT_.resize(nAggs_);
    MinvCT_.resize(nAggs_);
    AinvDMinvCT_.resize(nAggs_);
    Ainv_.resize(nAggs_);
    Minv_.resize(nAggs_);
    Hybrid_el_.resize(nAggs_);
    Minv_g_.resize(nAggs_);
    local_rhs_.resize(nAggs_);

    elem_scaling_.SetSize(nAggs_);
    elem_scaling_ = 1.0;

    CheckSharing();

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
    mfem::SparseMatrix mult_edof(smoothg::Transpose(edgedof_multiplier) );

    mfem::Array<int> j_array(mult_edof.GetJ(), mult_edof.NumNonZeroElems());
    j_array.Copy(multiplier_to_edof_);

    mfem::SparseMatrix Agg_m_tmp(smoothg::Mult(Agg_edgedof_, edgedof_multiplier));
    Agg_multiplier_.Swap(Agg_m_tmp);

    GenerateOffsets(comm_, num_multiplier_dofs_, multiplier_start_);

    auto mult_trueedof = ParMult(mult_edof, edgedof_d_td, multiplier_start_);
    unique_ptr<mfem::HypreParMatrix> multiplier_d_td_d(AAt(*mult_trueedof));

    // Construct multiplier "dof to true dof" table
    multiplier_d_td_ = BuildEntityToTrueEntity(*multiplier_d_td_d);
    multiplier_td_d_.reset(multiplier_d_td_->Transpose());
}

void HybridSolver::CheckSharing()
{
    auto edgedof_Agg = smoothg::Transpose(Agg_edgedof_);
    edgedof_is_shared_.SetSize(edgedof_Agg.NumRows());
    for (int i = 0; i < edgedof_Agg.NumRows(); ++i)
    {
        edgedof_is_shared_[i] = (edgedof_Agg.RowSize(i) > 1);
    }
}

mfem::SparseMatrix HybridSolver::AssembleHybridSystem(
    const std::vector<mfem::DenseMatrix>& M_el)
{
    mfem::SparseMatrix H_proc(num_multiplier_dofs_);

    const int map_size = std::max(num_edge_dofs_, Agg_vertexdof_.Width());
    mfem::Array<int> edof_global_to_local_map(map_size);
    edof_global_to_local_map = -1;
    mfem::Array<bool> edge_marker(num_edge_dofs_);
    edge_marker = true;

    mfem::SparseMatrix edof_IsOwned = GetDiag(*multiplier_d_td_);

    mfem::DenseMatrix DlocT, ClocT, Aloc, CMinvDT, DMinvCT, CMDADMC;

    mfem::DenseMatrixInverse Mloc_solver, Aloc_solver;
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
            edof_global_to_local_map[local_edgedof[i]] = i;

        // Extract Dloc as a sparse submatrix of D_
        auto Dloc = ExtractRowAndColumns(D_, local_vertexdof, local_edgedof,
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

        Mloc_solver.SetOperator(M_el[iAgg]);
        Mloc_solver.GetInverseMatrix(Minv_[iAgg]);

        mfem::DenseMatrix& MinvCT_i(MinvCT_[iAgg]);
        mfem::DenseMatrix& MinvDT_i(MinvDT_[iAgg]);
        mfem::DenseMatrix& AinvDMinvCT_i(AinvDMinvCT_[iAgg]);
        mfem::DenseMatrix& Ainv_i(Ainv_[iAgg]);

        MinvCT_i.SetSize(nlocal_edgedof, nlocal_multiplier);
        MinvDT_i.SetSize(nlocal_edgedof, nlocal_vertexdof);
        AinvDMinvCT_i.SetSize(nlocal_vertexdof, nlocal_multiplier);

        mfem::Mult(Minv_[iAgg], DlocT, MinvDT_i);
        mfem::Mult(Minv_[iAgg], ClocT, MinvCT_i);

        // Compute CMinvCT = Cloc * MinvCT
        MultSparseDense(Cloc, MinvCT_i, Hybrid_el_[iAgg]);

        // Compute Aloc = DMinvDT = Dloc * MinvDT
        MultSparseDense(Dloc, MinvDT_i, Aloc);

        if (W_.Width())
        {
            mfem::DenseMatrix Wloc(nlocal_vertexdof, nlocal_vertexdof);
            auto& W_ref = const_cast<mfem::SparseMatrix&>(W_);
            W_ref.GetSubMatrix(local_vertexdof, local_vertexdof, Wloc);
            Aloc += Wloc;
        }

        // Compute DMinvCT Dloc * MinvCT
        MultSparseDense(Dloc, MinvCT_i, DMinvCT);

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
    }

    return H_proc;
}


/// @todo nonzero Neumann BC (edge unknown), solve on true dof (original system)
void HybridSolver::Mult(const mfem::BlockVector& Rhs, mfem::BlockVector& Sol) const
{
    rhs_ = Rhs;
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
        RescaleVector(diagonal_scaling_, trueHrhs_);

    // solve the parallel global hybridized system
    mfem::StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    cg_.Mult(trueHrhs_, trueMu_);

    chrono.Stop();
    timing_ = chrono.RealTime();

    if (myid_ == 0 && print_level_ > 0)
    {
        std::cout << "  Timing: PCG done in "
                  << timing_ << "s. \n";
    }

    // TODO: decide to use = or += here and in timing_ update (MinresBlockSolver uses +=)
    num_iterations_ = cg_.GetNumIterations();

    if (myid_ == 0 && print_level_ > 0)
    {
        if (cg_.GetConverged())
            std::cout << "  CG converged in "
                      << num_iterations_
                      << " with a final residual norm "
                      << cg_.GetFinalNorm() << "\n";
        else
            std::cout << "  CG did not converge in "
                      << num_iterations_
                      << ". Final residual norm is "
                      << cg_.GetFinalNorm() << "\n";
    }

    // distribute true dofs to dofs and recover solution of the original system
    chrono.Clear();
    chrono.Start();

    if (diagonal_scaling_.Size() > 0)
        RescaleVector(diagonal_scaling_, trueMu_);

    multiplier_d_td_->Mult(trueMu_, Mu_);
    RecoverOriginalSolution(Mu_, Sol);

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
    const mfem::Vector& OriginalRHS_block2(OriginalRHS.GetBlock(1));

    HybridRHS = 0.;

    mfem::Vector f_loc, g_loc, CMinv_g_loc, DMinv_g_loc, rhs_loc_help;
    mfem::Array<int> local_vertexdof, local_edgedof, local_multiplier;
    for (int iAgg = 0; iAgg < nAggs_; ++iAgg)
    {
        // Extracting the size and global numbering of local dof
        GetTableRow(Agg_vertexdof_, iAgg, local_vertexdof);
        GetTableRow(Agg_edgedof_, iAgg, local_edgedof);
        GetTableRow(Agg_multiplier_, iAgg, local_multiplier);

        int nlocal_vertexdof = local_vertexdof.Size();
        int nlocal_edgedof = local_edgedof.Size();
        int nlocal_multiplier = local_multiplier.Size();

        // Compute local contribution to the RHS of the hybrid system
        // CM^{-1} g - CM^{-1}D^T A^{-1} (DM^{-1} g - f)
        OriginalRHS_block2.GetSubVector(local_vertexdof, f_loc);
        OriginalRHS.GetSubVector(local_edgedof, g_loc);
        for (int i = 0; i < nlocal_edgedof; ++i)
        {
            if (edgedof_is_shared_[local_edgedof[i]])
            {
                g_loc[i] /= 2.0;
            }
        }

        DMinv_g_loc.SetSize(nlocal_vertexdof);
        MinvDT_[iAgg].MultTranspose(g_loc, DMinv_g_loc);

        DMinv_g_loc /= elem_scaling_[iAgg];
        DMinv_g_loc -= f_loc;

        rhs_loc_help.SetSize(nlocal_multiplier);
        AinvDMinvCT_[iAgg].MultTranspose(DMinv_g_loc, rhs_loc_help);

        CMinv_g_loc.SetSize(nlocal_multiplier);
        MinvCT_[iAgg].MultTranspose(g_loc, CMinv_g_loc);
        CMinv_g_loc /= elem_scaling_[iAgg];
        CMinv_g_loc -= rhs_loc_help;

        for (int i = 0; i < nlocal_multiplier; ++i)
        {
            HybridRHS[local_multiplier[i]] += CMinv_g_loc[i];
        }

        // Save M^{-1}g, A^{-1} (DM^{-1} g - f) for solution recovery
        Minv_g_[iAgg].SetSize(nlocal_edgedof);
        Minv_[iAgg].Mult(g_loc, Minv_g_[iAgg]);

        local_rhs_[iAgg].SetSize(nlocal_vertexdof);
        Ainv_[iAgg].Mult(DMinv_g_loc, local_rhs_[iAgg]);
        local_rhs_[iAgg] *= elem_scaling_[iAgg];
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

    mfem::Vector mu_loc, tmp;
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
            MinvDT_[iAgg].Mult(u_loc, tmp);
            sigma_loc -= tmp;
            tmp.SetSize(nlocal_edgedof);
            MinvCT_[iAgg].Mult(mu_loc, tmp);
            sigma_loc -= tmp;
            sigma_loc /= elem_scaling_[iAgg];
        }

        // Save local solution to the global solution vector
        for (int i = 0; i < nlocal_vertexdof; ++i)
            RecoveredSol_block2(local_vertexdof[i]) = u_loc(i);

        for (int i = 0; i < nlocal_edgedof; ++i)
            RecoveredSol(local_edgedof[i]) = sigma_loc(i);
    }
}

void HybridSolver::ComputeScaledHybridSystem(const mfem::HypreParMatrix& H)
{
    mfem::HypreSmoother prec_scale(const_cast<mfem::HypreParMatrix&>(H));

    mfem::Vector zeros(H.Height());
    zeros = 1e-8;
    diagonal_scaling_.SetSize(H.Height());
    diagonal_scaling_ = 1.0;

    mfem::CGSolver cg_scale(comm_);
    cg_scale.SetMaxIter(rescale_iter_);
    cg_scale.SetPreconditioner(prec_scale);
    cg_scale.SetOperator(H);
    cg_scale.Mult(zeros, diagonal_scaling_);

    auto Scale = VectorToMatrix(diagonal_scaling_);
    mfem::HypreParMatrix pScale(comm_, H.N(), H.GetColStarts(), &Scale);
    H_.reset(smoothg::RAP(H, pScale));
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

    if (rescale_iter_ > 0 && !saamge_param_)
    {
        ComputeScaledHybridSystem(*H_);
    }
    nnz_ = H_->NNZ();

    mfem::StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    cg_.SetOperator(*H_);

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
        else
        {
            auto temp_prec = make_unique<mfem::HypreBoomerAMG>(*H_);
            temp_prec->SetPrintLevel(0);
            prec_ = std::move(temp_prec);
        }
        cg_.SetPreconditioner(*prec_);
    }
    if (myid_ == 0 && print_level_ > 0)
        std::cout << "  Timing: Preconditioner for hybridized system"
                  " constructed in " << chrono.RealTime() << "s. \n";
}

void HybridSolver::CollectEssentialDofs(const mfem::SparseMatrix& edof_bdrattr)
{
    mfem::SparseMatrix mult_truemult = GetDiag(*multiplier_d_td_);
    mfem::Array<int> true_multiplier;

    // Note: there is a 1-1 map from multipliers to edge dofs on faces
    if (edof_bdrattr.Width())
    {
        ess_true_multipliers_.Reserve(edof_bdrattr.NumNonZeroElems());
        ess_true_mult_to_edof_.Reserve(edof_bdrattr.NumNonZeroElems());
        for (int i = 0; i < num_multiplier_dofs_; ++i)
        {
            // natural BC for H(div) dof <=> essential BC for multiplier dof
            if (edof_bdrattr.RowSize(i) && !ess_edofs_[i])
            {
                GetTableRow(mult_truemult, i, true_multiplier);
                ess_true_multipliers_.Append(true_multiplier);
                ess_true_mult_to_edof_.Append(i);
            }
        }
    }

    // In case of normal graph Laplacian, eliminate one multiplier
    if (!ess_true_multipliers_.Size() && !W_is_nonzero_ && myid_ == 0)
    {
        GetTableRow(mult_truemult, 0, true_multiplier);
        assert(true_multiplier.Size() == 1);
        ess_true_multipliers_.Append(true_multiplier);
        ess_true_mult_to_edof_.Append(0);
    }
}

void HybridSolver::UpdateElemScaling(const mfem::Vector& elem_scaling_inverse)
{
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

} // namespace smoothg
