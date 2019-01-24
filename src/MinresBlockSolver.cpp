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
#include "utilities.hpp"
#include <assert.h>

namespace smoothg
{

MinresBlockSolver::MinresBlockSolver(mfem::HypreParMatrix* M,
                                     mfem::HypreParMatrix* D,
                                     mfem::HypreParMatrix* W,
                                     const mfem::Array<int>& block_true_offsets)
    :
    MixedLaplacianSolver(M->GetComm(), block_true_offsets, W),
    minres_(comm_), operator_(block_true_offsets), prec_(block_true_offsets)
{
    remove_one_dof_ = false;
    MPI_Comm_rank(comm_, &myid_);
    Init(M, D, W);
}

/// implementation largely lifted from ex5p.cpp
void MinresBlockSolver::Init(mfem::HypreParMatrix* M, mfem::HypreParMatrix* D,
                             mfem::HypreParMatrix* W)
{
    assert(M && D);

    hDt_.reset(D->Transpose());

    nnz_ = M->NNZ() + D->NNZ() + hDt_->NNZ();

    operator_.SetBlock(0, 0, M);
    operator_.SetBlock(0, 1, hDt_.get());
    operator_.SetBlock(1, 0, D);

    std::unique_ptr<mfem::HypreParMatrix> MinvDt(D->Transpose());
    mfem::Vector Md;
    M->GetDiag(Md);
    MinvDt->InvScaleRows(Md);
    schur_block_.reset(mfem::ParMult(D, MinvDt.get()));

    // D retains ownership of rows, but MinvDt will be deleted, so we need to
    // take ownership of col_starts here.
    hypre_ParCSRMatrixSetColStartsOwner(*schur_block_, 1);
    hypre_ParCSRMatrixSetColStartsOwner(*MinvDt, 0);

    if (W_is_nonzero_)
    {
        hypre_ParCSRMatrixSetColStartsOwner(*schur_block_, 0);
        hypre_ParCSRMatrixSetColStartsOwner(*MinvDt, 1);

        operator_.SetBlock(1, 1, W);
        nnz_ += W->NNZ();

        (*W) *= -1.0;
        schur_block_.reset(ParAdd(*W, *schur_block_));
        (*W) *= -1.0;

        hypre_ParCSRMatrixSetColStartsOwner(*schur_block_, 0);
    }
    else if (remove_one_dof_)
    {
        mfem::SparseMatrix W(D->Height());
        W_.Swap(W);

        if (myid_ == 0)
        {
            W_.Add(0, 0, 1.0);
        }
        W_.Finalize();

        hW_ = make_unique<mfem::HypreParMatrix>(comm_, D->M(), D->RowPart(), &W_);

        operator_.SetBlock(1, 1, hW_.get());

        nnz_ += 1;
    }

    mfem::HypreDiagScale* Mprec = new mfem::HypreDiagScale(*M);
    mfem::HypreBoomerAMG* Sprec = new mfem::HypreBoomerAMG(*schur_block_);
    Sprec->SetPrintLevel(0);

    prec_.owns_blocks = 1;
    prec_.SetDiagonalBlock(0, Mprec);
    prec_.SetDiagonalBlock(1, Sprec);

    minres_.SetPrintLevel(print_level_);
    minres_.SetMaxIter(max_num_iter_);
    minres_.SetRelTol(rtol_);
    minres_.SetAbsTol(atol_);
    minres_.SetPreconditioner(prec_);
    minres_.SetOperator(operator_);
    minres_.iterative_mode = false;
}

MinresBlockSolver::MinresBlockSolver(const MixedMatrix& mgL,
                                     const mfem::Array<int>* ess_attr)
    :
    MixedLaplacianSolver(mgL.GetComm(), mgL.GetBlockTrueOffsets(), mgL.CheckW()),
    minres_(comm_),
    operator_(mgL.GetBlockTrueOffsets()),
    prec_(mgL.GetBlockTrueOffsets())
{
    MixedLaplacianSolver::Init(mgL, ess_attr);

    mfem::SparseMatrix M_proc(mgL.GetM());
    for (int mm = 0; mm < ess_edofs_.Size(); ++mm)
    {
        // Assume M diagonal, no ess data
        if (ess_edofs_[mm])
            M_proc.EliminateRowCol(mm, true);
    }

    mfem::SparseMatrix D_proc(mgL.GetD());
    if (ess_edofs_.Size())
    {
        D_proc.EliminateCols(ess_edofs_);
    }

    mfem::Array<int>& D_row_start(mgL.GetDrowStarts());

    const mfem::HypreParMatrix& edge_d_td(mgL.GetEdgeDofToTrueDof());
    const mfem::HypreParMatrix& edge_td_d(mgL.GetEdgeTrueDofToDof());

    mfem::HypreParMatrix M(comm_, edge_d_td.M(), edge_d_td.GetRowStarts(), &M_proc);
    std::unique_ptr<mfem::HypreParMatrix> M_tmp(ParMult(&M, &edge_d_td));
    hM_.reset(ParMult(&edge_td_d, M_tmp.get()));

    hM_->CopyRowStarts();
    hM_->CopyColStarts();

    if (!W_is_nonzero_ && remove_one_dof_ && myid_ == 0)
    {
        D_proc.EliminateRow(0);
    }

    hD_ = ParMult(D_proc, edge_d_td, D_row_start);
    hDt_.reset(hD_->Transpose());

    hD_->CopyRowStarts();
    hDt_->CopyRowStarts();

    hD_->CopyColStarts();
    hDt_->CopyColStarts();

    if (W_is_nonzero_)
    {
        const mfem::SparseMatrix* W = mgL.GetW();
        assert(W);
        mfem::SparseMatrix W_copy(*W);

        W_.Swap(W_copy);
        W_.Finalize();

        hW_ = make_unique<mfem::HypreParMatrix>(comm_, hD_->M(), D_row_start, &W_);

        hW_->CopyRowStarts();
        hW_->CopyColStarts();
    }

    Init(hM_.get(), hD_.get(), hW_.get());
}

MinresBlockSolver::~MinresBlockSolver()
{
}

void MinresBlockSolver::Mult(const mfem::BlockVector& rhs,
                             mfem::BlockVector& sol) const
{
    mfem::StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    double rhs0 = rhs[0];

    if (!W_is_nonzero_ && remove_one_dof_ && myid_ == 0)
    {
        const_cast<mfem::Vector&>(rhs.GetBlock(1))(0) = 0.0;
    }

    minres_.Mult(rhs_, sol);

    const_cast<mfem::Vector&>(rhs.GetBlock(1))(0) = rhs0;

    if (!W_is_nonzero_ && remove_one_dof_)
    {
        Orthogonalize(sol.GetBlock(1));
    }

    chrono.Stop();
    timing_ = chrono.RealTime();

    if (myid_ == 0 && print_level_ > 0)
    {
        std::cout << "  Timing MINRES: Solver done in "
                  << timing_ << "s. \n";
        if (minres_.GetConverged())
            std::cout << "  Minres converged in "
                      << minres_.GetNumIterations()
                      << " with a final residual norm "
                      << minres_.GetFinalNorm() << "\n";
        else
            std::cout << "  Minres did not converge in "
                      << minres_.GetNumIterations()
                      << ". Final residual norm is "
                      << minres_.GetFinalNorm() << "\n";
    }

    num_iterations_ = minres_.GetNumIterations();
}

/**
   MinresBlockSolver acts on "true" dofs, this one does not.
*/
MinresBlockSolverFalse::MinresBlockSolverFalse(const MixedMatrix& mgL,
                                               const mfem::Array<int>* ess_attr)
    :
    MinresBlockSolver(mgL, ess_attr),
    mixed_matrix_(mgL)
{
}

MinresBlockSolverFalse::~MinresBlockSolverFalse()
{
}

void MinresBlockSolverFalse::Mult(const mfem::BlockVector& rhs,
                                  mfem::BlockVector& sol) const
{
    const auto& edof_trueedof = mixed_matrix_.GetEdgeDofToTrueDof();
    edof_trueedof.MultTranspose(rhs.GetBlock(0), rhs_.GetBlock(0));
    rhs_.GetBlock(1) = rhs.GetBlock(1);

    MinresBlockSolver::Mult(rhs_, sol_);

    edof_trueedof.Mult(sol_.GetBlock(0), sol.GetBlock(0));
    sol.GetBlock(1) = sol_.GetBlock(1);
}

void MinresBlockSolver::SetPrintLevel(int print_level)
{
    MixedLaplacianSolver::SetPrintLevel(print_level);

    minres_.SetPrintLevel(print_level_);
}

void MinresBlockSolver::SetMaxIter(int max_num_iter)
{
    MixedLaplacianSolver::SetMaxIter(max_num_iter);

    minres_.SetMaxIter(max_num_iter_);
}

void MinresBlockSolver::SetRelTol(double rtol)
{
    MixedLaplacianSolver::SetRelTol(rtol);

    minres_.SetRelTol(rtol_);
}

void MinresBlockSolver::SetAbsTol(double atol)
{
    MixedLaplacianSolver::SetAbsTol(atol);

    minres_.SetAbsTol(atol_);
}

} // namespace smoothg
