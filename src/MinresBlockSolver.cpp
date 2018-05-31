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

   @brief Implements MinresBlockSolver object.
*/

#include "MinresBlockSolver.hpp"

namespace smoothg
{

MinresBlockSolver::MinresBlockSolver(const MixedMatrix& mgl)
    : MinresBlockSolver(mgl, {})
{
}

MinresBlockSolver::MinresBlockSolver(const MixedMatrix& mgl, const std::vector<int>& elim_dofs)
    : MGLSolver(mgl), M_(mgl.GlobalM()), D_(mgl.GlobalD()), W_(mgl.GlobalW()),
      edge_true_edge_(mgl.EdgeTrueEdge()),
      op_(mgl.TrueOffsets()), prec_(mgl.TrueOffsets()),
      true_rhs_(mgl.TrueOffsets()), true_sol_(mgl.TrueOffsets())
{
    std::vector<double> M_diag(M_.GetDiag().GetDiag());
    SparseMatrix D_elim = mgl.LocalD();

    if (!use_w_ && myid_ == 0)
    {
        D_elim.EliminateRow(0);
    }

    std::vector<int> marker(D_elim.Cols(), 0);

    for (auto&& dof : elim_dofs)
    {
        marker[dof] = 1;
    }

    D_elim.EliminateCol(marker);

    ParMatrix D_elim_g(comm_, D_elim);

    D_ = D_elim_g.Mult(mgl.EdgeTrueEdge());
    DT_ = D_.Transpose();

    ParMatrix MinvDT = DT_;
    MinvDT.InverseScaleRows(M_diag);
    ParMatrix schur_block = D_.Mult(MinvDT);

    if (!use_w_)
    {
        CooMatrix elim_dof(D_.Rows(), D_.Rows());

        if (myid_ == 0)
        {
            elim_dof.Add(0, 0, 1.0);
        }

        SparseMatrix W = elim_dof.ToSparse();
        W_ = ParMatrix(D_.GetComm(), D_.GetRowStarts(), std::move(W));
    }
    else
    {
        schur_block = parlinalgcpp::ParSub(schur_block, W_);
    }

    M_prec_ = parlinalgcpp::ParDiagScale(M_);
    schur_prec_ = parlinalgcpp::BoomerAMG(std::move(schur_block));

    op_.SetBlock(0, 0, M_);
    op_.SetBlock(0, 1, DT_);
    op_.SetBlock(1, 0, D_);
    op_.SetBlock(1, 1, W_);

    prec_.SetBlock(0, 0, M_prec_);
    prec_.SetBlock(1, 1, schur_prec_);

    pminres_ = linalgcpp::PMINRESSolver(op_, prec_, max_num_iter_, rtol_,
                                        atol_, 0, parlinalgcpp::ParMult);

    if (myid_ == 0)
    {
        SetPrintLevel(print_level_);
    }

    nnz_ = M_.nnz() + DT_.nnz() + D_.nnz() + W_.nnz();
}


MinresBlockSolver::MinresBlockSolver(const MinresBlockSolver& other) noexcept
    : MGLSolver(other), op_(other.op_), prec_(other.prec_),
      M_prec_(other.M_prec_), schur_prec_(other.schur_prec_),
      pminres_(other.pminres_),
      true_rhs_(other.true_rhs_), true_sol_(other.true_sol_)
{

}

MinresBlockSolver::MinresBlockSolver(MinresBlockSolver&& other) noexcept
{
    swap(*this, other);
}

MinresBlockSolver& MinresBlockSolver::operator=(MinresBlockSolver other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(MinresBlockSolver& lhs, MinresBlockSolver& rhs) noexcept
{
    swap(static_cast<MGLSolver&>(lhs),
         static_cast<MGLSolver&>(rhs));

    swap(lhs.op_, rhs.op_);
    swap(lhs.prec_, rhs.prec_);
    swap(lhs.M_prec_, rhs.M_prec_);
    swap(lhs.schur_prec_, rhs.schur_prec_);
    swap(lhs.pminres_, rhs.pminres_);
    swap(lhs.true_rhs_, rhs.true_rhs_);
    swap(lhs.true_sol_, rhs.true_sol_);
}

void MinresBlockSolver::Solve(const BlockVector& rhs, BlockVector& sol) const
{
    Timer timer(Timer::Start::True);

    edge_true_edge_.MultAT(rhs.GetBlock(0), true_rhs_.GetBlock(0));
    true_rhs_.GetBlock(1) = rhs.GetBlock(1);
    true_sol_ = 0.0;

    if (!use_w_ && myid_ == 0)
    {
        true_rhs_[0] = 0.0;
    }

    pminres_.Mult(true_rhs_, true_sol_);
    num_iterations_ = pminres_.GetNumIterations();

    edge_true_edge_.Mult(true_sol_.GetBlock(0), sol.GetBlock(0));
    sol.GetBlock(1) = true_sol_.GetBlock(1);

    timer.Click();
    timing_ = timer.TotalTime();
}

void MinresBlockSolver::SetPrintLevel(int print_level)
{
    MGLSolver::SetPrintLevel(print_level);

    if (myid_ == 0)
    {
        pminres_.SetVerbose(print_level_);
    }
}

void MinresBlockSolver::SetMaxIter(int max_num_iter)
{
    MGLSolver::SetMaxIter(max_num_iter);

    pminres_.SetMaxIter(max_num_iter);
}

void MinresBlockSolver::SetRelTol(double rtol)
{
    MGLSolver::SetRelTol(rtol);

    pminres_.SetRelTol(rtol);
}

void MinresBlockSolver::SetAbsTol(double atol)
{
    MGLSolver::SetAbsTol(atol);

    pminres_.SetAbsTol(atol);
}

} // namespace smoothg

