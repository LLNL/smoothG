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

namespace smoothg
{

HybridSolver::HybridSolver(const MixedMatrix& mgl)
    :
    MGLSolver(mgl),
    agg_vertexdof_(mgl.GetAggVertexDof()),
    agg_edgedof_(mgl.GetElemDof()),
    num_aggs_(agg_edgedof_.Rows()),
    num_edge_dofs_(agg_edgedof_.Cols()),
    num_multiplier_dofs_(mgl.GetFaceFaceDof().Cols()),
    MinvDT_(num_aggs_), MinvCT_(num_aggs_),
    AinvDMinvCT_(num_aggs_), Ainv_(num_aggs_),
    hybrid_elem_(num_aggs_), Ainv_f_(num_aggs_),
    agg_weights_(num_aggs_, 1.0)
{
    SparseMatrix edgedof_multiplier = MakeEdgeDofMultiplier();
    SparseMatrix multiplier_edgedof = edgedof_multiplier.Transpose();
    const std::vector<int>& j_multiplier_edgedof = multiplier_edgedof.GetIndices();

    agg_multiplier_ = agg_edgedof_.Mult(edgedof_multiplier);

    ParMatrix edge_td_d = mgl.EdgeTrueEdge().Transpose();
    ParMatrix edge_edge = mgl.EdgeTrueEdge().Mult(edge_td_d);
    ParMatrix edgedof_multiplier_d(comm_, std::move(edgedof_multiplier));
    ParMatrix multiplier_d_td_d = parlinalgcpp::RAP(edge_edge, edgedof_multiplier_d);

    multiplier_d_td_ = MakeEntityTrueEntity(multiplier_d_td_d);

    SparseMatrix local_hybrid = AssembleHybridSystem(mgl, j_multiplier_edgedof);

    InitSolver(std::move(local_hybrid));
}


void HybridSolver::InitSolver(SparseMatrix local_hybrid)
{
    if (myid_ == 0 && !use_w_)
    {
        local_hybrid.EliminateRowCol(0);
    }

    ParMatrix hybrid_d = ParMatrix(comm_, std::move(local_hybrid));

    pHybridSystem_ = parlinalgcpp::RAP(hybrid_d, multiplier_d_td_);
    nnz_ = pHybridSystem_.nnz();

    cg_ = linalgcpp::PCGSolver(pHybridSystem_, max_num_iter_, rtol_,
                               atol_, 0, parlinalgcpp::ParMult);
    if (myid_ == 0)
    {
        SetPrintLevel(print_level_);
    }

    // HypreBoomerAMG is broken if local size is zero
    int local_size = pHybridSystem_.Rows();
    int min_size;
    MPI_Allreduce(&local_size, &min_size, 1, MPI_INT, MPI_MIN, comm_);

    const bool use_prec = min_size > 0;
    if (use_prec)
    {
        prec_ = parlinalgcpp::BoomerAMG(pHybridSystem_);
        cg_.SetPreconditioner(prec_);
    }
    else
    {
        if (myid_ == 0)
        {
            // TODO(gelever1): create SMOOTHG_Warn or something of the sort
            // to make warnings optional
            printf("Warning: Not using preconditioner for Hybrid Solver!\n");
        }
    }

    trueHrhs_.SetSize(multiplier_d_td_.Cols());
    trueMu_.SetSize(trueHrhs_.size());
    Hrhs_.SetSize(num_multiplier_dofs_);
    Mu_.SetSize(num_multiplier_dofs_);
}

SparseMatrix HybridSolver::MakeEdgeDofMultiplier() const
{
    std::vector<int> indptr(num_edge_dofs_ + 1);
    std::iota(std::begin(indptr), std::begin(indptr) + num_multiplier_dofs_ + 1, 0);
    std::fill(std::begin(indptr) + num_multiplier_dofs_ + 1, std::end(indptr),
              indptr[num_multiplier_dofs_]);

    std::vector<int> indices(num_multiplier_dofs_);
    std::iota(std::begin(indices), std::end(indices), 0);

    std::vector<double> data(num_multiplier_dofs_, 1.0);

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                        num_edge_dofs_, num_multiplier_dofs_);
}

SparseMatrix HybridSolver::MakeLocalC(int agg, const ParMatrix& edge_true_edge,
                                      const std::vector<int>& j_multiplier_edgedof,
                                      std::vector<int>& edge_map,
                                      std::vector<bool>& edge_marker) const
{
    const auto& edgedof_IsOwned = edge_true_edge.GetDiag();

    std::vector<int> local_edgedof = agg_edgedof_.GetIndices(agg);
    std::vector<int> local_multiplier = agg_multiplier_.GetIndices(agg);

    const int nlocal_edgedof = local_edgedof.size();
    const int nlocal_multiplier = local_multiplier.size();

    SetMarker(edge_map, local_edgedof);

    std::vector<int> Cloc_i(nlocal_multiplier + 1);
    std::iota(std::begin(Cloc_i), std::end(Cloc_i), 0);

    std::vector<int> Cloc_j(nlocal_multiplier);
    std::vector<double> Cloc_data(nlocal_multiplier);

    for (int i = 0; i < nlocal_multiplier; ++i)
    {
        const int edgedof_global_id = j_multiplier_edgedof[local_multiplier[i]];
        const int edgedof_local_id = edge_map[edgedof_global_id];

        Cloc_j[i] = edgedof_local_id;

        if (edgedof_IsOwned.RowSize(edgedof_global_id) &&
            edge_marker[edgedof_global_id])
        {
            edge_marker[edgedof_global_id] = false;
            Cloc_data[i] = 1.;
        }
        else
        {
            Cloc_data[i] = -1.;
        }
    }

    ClearMarker(edge_map, local_edgedof);

    return SparseMatrix(std::move(Cloc_i), std::move(Cloc_j), std::move(Cloc_data),
                        nlocal_multiplier, nlocal_edgedof);
}

SparseMatrix HybridSolver::AssembleHybridSystem(
    const MixedMatrix& mgl,
    const std::vector<int>& j_multiplier_edgedof)
{
    const auto& M_el = mgl.GetElemM();

    const int map_size = std::max(num_edge_dofs_, agg_vertexdof_.Cols());
    std::vector<int> edge_map(map_size, -1);
    std::vector<bool> edge_marker(num_edge_dofs_, true);

    DenseMatrix Minv;

    DenseMatrix Aloc;
    DenseMatrix Wloc;
    DenseMatrix CMDADMC;
    DenseMatrix DMinvCT;

    CooMatrix hybrid_system(num_multiplier_dofs_);

    for (int agg = 0; agg < num_aggs_; ++agg)
    {
        // Extracting the size and global numbering of local dof
        std::vector<int> local_vertexdof = agg_vertexdof_.GetIndices(agg);
        std::vector<int> local_edgedof = agg_edgedof_.GetIndices(agg);
        std::vector<int> local_multiplier = agg_multiplier_.GetIndices(agg);

        const int nlocal_vertexdof = local_vertexdof.size();
        const int nlocal_multiplier = local_multiplier.size();

        SparseMatrix Dloc = mgl.LocalD().GetSubMatrix(local_vertexdof, local_edgedof,
                                                      edge_map);

        SparseMatrix Cloc = MakeLocalC(agg, mgl.EdgeTrueEdge(), j_multiplier_edgedof,
                                       edge_map, edge_marker);

        // Compute:
        //      CMinvCT = Cloc * MinvCT
        //      Aloc = DMinvDT = Dloc * MinvDT
        //      DMinvCT = Dloc * MinvCT
        //      CMinvDTAinvDMinvCT = CMinvDT * AinvDMinvCT_
        //      hybrid_elem = CMinvCT - CMinvDTAinvDMinvCT

        M_el[agg].Invert(Minv);

        DenseMatrix& MinvCT_i(MinvCT_[agg]);
        DenseMatrix& MinvDT_i(MinvDT_[agg]);
        DenseMatrix& AinvDMinvCT_i(AinvDMinvCT_[agg]);
        DenseMatrix& Ainv_i(Ainv_[agg]);
        DenseMatrix& hybrid_elem(hybrid_elem_[agg]);

        AinvDMinvCT_i.SetSize(nlocal_vertexdof, nlocal_multiplier);
        hybrid_elem.SetSize(nlocal_multiplier, nlocal_multiplier);
        Aloc.SetSize(nlocal_vertexdof, nlocal_vertexdof);
        DMinvCT.SetSize(nlocal_vertexdof, nlocal_multiplier);

        MinvDT_i.SetSize(Minv.Cols(), Dloc.Rows());
        MinvCT_i.SetSize(Minv.Cols(), Cloc.Rows());

        Dloc.MultCT(Minv, MinvDT_i);
        Cloc.MultCT(Minv, MinvCT_i);

        Cloc.Mult(MinvCT_i, hybrid_elem);
        Dloc.Mult(MinvCT_i, DMinvCT);
        Dloc.Mult(MinvDT_i, Aloc);

        if (use_w_)
        {
            auto Wloc_tmp = mgl.LocalW().GetSubMatrix(local_vertexdof, local_vertexdof, edge_map);
            Wloc_tmp.ToDense(Wloc);

            Aloc -= Wloc;
        }

        Aloc.Invert(Ainv_i);

        Ainv_i.Mult(DMinvCT, AinvDMinvCT_i);

        if (DMinvCT.Rows() > 0 && DMinvCT.Rows() > 0)
        {
            CMDADMC.SetSize(nlocal_multiplier, nlocal_multiplier);
            AinvDMinvCT_i.MultAT(DMinvCT, CMDADMC);
            hybrid_elem -= CMDADMC;
        }

        // Add contribution of the element matrix to the global system
        hybrid_system.Add(local_multiplier, hybrid_elem);
    }

    return hybrid_system.ToSparse();
}



void HybridSolver::Solve(const BlockVector& Rhs, BlockVector& Sol) const
{
    RHSTransform(Rhs, Hrhs_);

    if (!use_w_ && myid_ == 0)
    {
        Hrhs_[0] = 0.0;
    }

    // assemble true right hand side
    multiplier_d_td_.MultAT(Hrhs_, trueHrhs_);

    // solve the parallel global hybridized system
    Timer timer(Timer::Start::True);

    trueMu_ = 0.0;
    cg_.Mult(trueHrhs_, trueMu_);

    timer.Click();
    timing_ = timer.TotalTime();

    if (myid_ == 0 && print_level_ > 0)
    {
        std::cout << "  Timing: PCG done in " << timing_ << "s. \n";
    }

    num_iterations_ = cg_.GetNumIterations();

    if (myid_ == 0 && print_level_ > 0)
    {
        /*
        if (cg_.GetConverged())
            std::cout << "  CG converged in "
                      << num_iterations_
                      << " with a final residual norm "
                      << cg_->GetFinalNorm() << "\n";
        else
            std::cout << "  CG did not converge in "
                      << num_iterations_
                      << ". Final residual norm is "
                      << cg_->GetFinalNorm() << "\n";
        */
    }

    // distribute true dofs to dofs and recover solution of the original system
    multiplier_d_td_.Mult(trueMu_, Mu_);
    RecoverOriginalSolution(Mu_, Sol);
}

void HybridSolver::RHSTransform(const BlockVector& OriginalRHS,
                                VectorView HybridRHS) const
{
    HybridRHS = 0.;

    Vector f_loc;
    Vector CMinvDTAinv_f_loc;

    for (int iAgg = 0; iAgg < num_aggs_; ++iAgg)
    {
        // Extracting the size and global numbering of local dof
        std::vector<int> local_vertexdof = agg_vertexdof_.GetIndices(iAgg);
        std::vector<int> local_multiplier = agg_multiplier_.GetIndices(iAgg);

        int nlocal_vertexdof = local_vertexdof.size();
        int nlocal_multiplier = local_multiplier.size();

        // Compute local contribution to the RHS of the hybrid system
        OriginalRHS.GetBlock(1).GetSubVector(local_vertexdof, f_loc);
        f_loc *= -1.0;

        CMinvDTAinv_f_loc.SetSize(nlocal_multiplier);
        AinvDMinvCT_[iAgg].MultAT(f_loc, CMinvDTAinv_f_loc);

        for (int i = 0; i < nlocal_multiplier; ++i)
        {
            HybridRHS[local_multiplier[i]] -= CMinvDTAinv_f_loc[i];
        }

        // Save the element rhs [M B^T;B 0]^-1[f;g] for solution recovery
        Ainv_f_[iAgg].SetSize(nlocal_vertexdof);
        Ainv_[iAgg].Mult(f_loc, Ainv_f_[iAgg]);
        Ainv_f_[iAgg] /= agg_weights_[iAgg];
    }
}

void HybridSolver::RecoverOriginalSolution(const VectorView& HybridSol,
                                           BlockVector& RecoveredSol) const
{
    // Recover the solution of the original system from multiplier mu, i.e.,
    // [u;p] = [f;g] - [M B^T;B 0]^-1[C 0]^T * mu
    // This procedure is done locally in each element

    RecoveredSol = 0.;

    Vector mu_loc;
    Vector sigma_loc;
    Vector tmp;

    for (int iAgg = 0; iAgg < num_aggs_; ++iAgg)
    {
        // Extracting the size and global numbering of local dof
        std::vector<int> local_vertexdof = agg_vertexdof_.GetIndices(iAgg);
        std::vector<int> local_edgedof = agg_edgedof_.GetIndices(iAgg);
        std::vector<int> local_multiplier = agg_multiplier_.GetIndices(iAgg);

        int nlocal_vertexdof = local_vertexdof.size();
        int nlocal_edgedof = local_edgedof.size();
        int nlocal_multiplier = local_multiplier.size();

        // Initialize a vector which will store the local contribution of Hdiv
        // and L2 space
        Vector& u_loc(Ainv_f_[iAgg]);

        // This check is just for the case when there is only one element for
        // the global problem, then there will be no Lagrange multipliers
        if (nlocal_multiplier > 0)
        {
            // Extract the local portion of the Lagrange multiplier solution
            HybridSol.GetSubVector(local_multiplier, mu_loc);

            // Compute u = (DMinvDT)^-1(f-DMinvC^T mu)
            //AinvDMinvCT_[iAgg].AddMult_a(-1., mu_loc, u_loc);
            tmp.SetSize(u_loc.size());
            AinvDMinvCT_[iAgg].Mult(mu_loc, tmp);
            u_loc -= tmp;

            // Compute -sigma = Minv(DT u + DT mu)
            sigma_loc.SetSize(nlocal_edgedof);
            MinvDT_[iAgg].Mult(u_loc, sigma_loc);
            //MinvCT_[iAgg].AddMult(mu_loc, sigma_loc);
            tmp.SetSize(sigma_loc.size());
            MinvCT_[iAgg].Mult(mu_loc, tmp);
            sigma_loc += tmp;
            sigma_loc *= agg_weights_[iAgg];
        }

        // Save local solution to the global solution vector
        for (int i = 0; i < nlocal_edgedof; ++i)
        {
            RecoveredSol[local_edgedof[i]] = -sigma_loc[i];
        }

        for (int i = 0; i < nlocal_vertexdof; ++i)
        {
            RecoveredSol.GetBlock(1)[local_vertexdof[i]] = u_loc[i];
        }
    }
}

void HybridSolver::UpdateAggScaling(const std::vector<double>& agg_weight)
{
    assert(!use_w_);
    assert(static_cast<int>(agg_weight.size()) == num_aggs_);

    std::copy(std::begin(agg_weight), std::end(agg_weight), std::begin(agg_weights_));

    CooMatrix hybrid_system(num_multiplier_dofs_);

    for (int i = 0; i < num_aggs_; ++i)
    {
        std::vector<int> local_multipliers = agg_multiplier_.GetIndices(i);

        hybrid_system.Add(local_multipliers, agg_weight[i], hybrid_elem_[i]);
    }

    InitSolver(hybrid_system.ToSparse());
}

void HybridSolver::SetPrintLevel(int print_level)
{
    MGLSolver::SetPrintLevel(print_level);

    if (myid_ == 0)
    {
        cg_.SetVerbose(print_level_);
    }
}

void HybridSolver::SetMaxIter(int max_num_iter)
{
    MGLSolver::SetMaxIter(max_num_iter);

    cg_.SetMaxIter(max_num_iter_);
}

void HybridSolver::SetRelTol(double rtol)
{
    MGLSolver::SetRelTol(rtol);

    cg_.SetRelTol(rtol_);
}

void HybridSolver::SetAbsTol(double atol)
{
    MGLSolver::SetAbsTol(atol);

    cg_.SetAbsTol(atol_);
}

} // namespace smoothg
