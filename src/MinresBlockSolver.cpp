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

/// implementation largely lifted from ex5p.cpp
MinresBlockSolver::MinresBlockSolver(MPI_Comm comm, mfem::HypreParMatrix* M, mfem::HypreParMatrix* D, mfem::HypreParMatrix* W,
    const mfem::Array<int>& block_true_offsets,
    bool remove_one_dof, bool use_W)
    :
    MixedLaplacianSolver(block_true_offsets),
    minres_(comm),
    comm_(comm),
    remove_one_dof_(remove_one_dof),
    use_W_(use_W),
    operator_(block_true_offsets),
    prec_(block_true_offsets)
{
    Init(M, D, W);
}

void MinresBlockSolver::Init(mfem::HypreParMatrix* M, mfem::HypreParMatrix* D,
                             mfem::HypreParMatrix* W)
{
    assert(M && D);

    MPI_Comm_rank(comm_, &myid_);

    hDt_.reset(D->Transpose());

    nnz_ = M->NNZ() + D->NNZ() + hDt_->NNZ();

    operator_.SetBlock(0, 0, M);
    operator_.SetBlock(0, 1, hDt_.get());
    operator_.SetBlock(1, 0, D);

    std::unique_ptr<mfem::HypreParMatrix> MinvDt(D->Transpose());
    mfem::Vector Md;
    M->GetDiag(Md);
    MinvDt->InvScaleRows(Md);
    schur_block_.reset(ParMult(D, MinvDt.get()));

    // D retains ownership of rows, but MinvDt will be deleted, so we need to
    // take ownership of col_starts here.
    hypre_ParCSRMatrixSetColStartsOwner(*schur_block_, 1);
    hypre_ParCSRMatrixSetColStartsOwner(*MinvDt, 0);

    if (use_W_ && W)
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

        hW_ = make_unique<mfem::HypreParMatrix>(comm_, D->GetGlobalNumRows(),
                                                D->RowPart(), &W_);

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

MinresBlockSolver::MinresBlockSolver(MPI_Comm comm, mfem::HypreParMatrix* M, mfem::HypreParMatrix* D,
    const mfem::Array<int>& block_true_offsets, bool remove_one_dof)
    : MinresBlockSolver(comm, M, D, nullptr, block_true_offsets, remove_one_dof)
{
}

MinresBlockSolver::MinresBlockSolver(MPI_Comm comm, const MixedMatrix& mgL, bool remove_one_dof)
    :
    MixedLaplacianSolver(mgL.get_blockoffsets()),
    minres_(comm),
    comm_(comm),
    remove_one_dof_(remove_one_dof),
    use_W_(mgL.CheckW()),
    operator_(mgL.get_blockTrueOffsets()),
    prec_(mgL.get_blockTrueOffsets()),
    M_(mgL.getWeight()),
    D_(mgL.getD())
{
    MPI_Comm_rank(comm_, &myid_);

    mfem::Array<int>& D_row_start(mgL.get_Drow_start());

    const mfem::HypreParMatrix& edge_d_td(mgL.get_edge_d_td());
    const mfem::HypreParMatrix& edge_td_d(mgL.get_edge_td_d());

    mfem::HypreParMatrix M(comm, edge_d_td.M(),
                           edge_d_td.GetRowStarts(), &M_);

    std::unique_ptr<mfem::HypreParMatrix> M_tmp(
        ParMult(&M, const_cast<mfem::HypreParMatrix*>(&edge_d_td)));
    hM_.reset(ParMult(const_cast<mfem::HypreParMatrix*>(&edge_td_d), M_tmp.get()));
    hypre_ParCSRMatrixSetNumNonzeros(*hM_);

    hM_->CopyRowStarts();
    hM_->CopyColStarts();

    if (use_W_)
    {
        hD_.reset(edge_d_td.LeftDiagMult(D_, D_row_start));
        hDt_.reset(hD_->Transpose());

        mfem::SparseMatrix* W = mgL.getW();
        assert(W);
        mfem::SparseMatrix W_copy(*W);

        W_.Swap(W_copy);
        W_.Finalize();

        hW_ = make_unique<mfem::HypreParMatrix>(comm_, hD_->GetGlobalNumRows(),
                                                D_row_start, &W_);

        hW_->CopyRowStarts();
        hW_->CopyColStarts();
    }
    else
    {
        if (remove_one_dof_ && myid_ == 0)
        {
            D_.EliminateRow(0);
        }

        hD_.reset(edge_d_td.LeftDiagMult(D_, D_row_start));
        hDt_.reset(hD_->Transpose());
    }

    hD_->CopyRowStarts();
    hDt_->CopyRowStarts();

    hD_->CopyColStarts();
    hDt_->CopyColStarts();


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

    *rhs_ = rhs;

    if (!use_W_ && remove_one_dof_ && myid_ == 0)
    {
        rhs_->GetBlock(1)(0) = 0.0;
    }

    minres_.Mult(*rhs_, sol);

    chrono.Stop();
    timing_ += chrono.RealTime();

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

    num_iterations_ += minres_.GetNumIterations();
}

/**
   MinresBlockSolver acts on "true" dofs, this one does not.
*/
MinresBlockSolverFalse::MinresBlockSolverFalse(MPI_Comm comm, const MixedMatrix& mgL, bool remove_one_dof)
    :
    MinresBlockSolver(comm, mgL, remove_one_dof),
    mixed_matrix_(mgL),
    true_rhs_(mgL.get_blockTrueOffsets()),
    true_sol_(mgL.get_blockTrueOffsets())
{
}

MinresBlockSolverFalse::~MinresBlockSolverFalse()
{
}

void MinresBlockSolverFalse::Mult(const mfem::BlockVector& rhs,
                                  mfem::BlockVector& sol) const
{
    mfem::StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    const mfem::HypreParMatrix& edgedof_d_td = mixed_matrix_.get_edge_d_td();

    edgedof_d_td.MultTranspose(rhs.GetBlock(0), true_rhs_.GetBlock(0));
    true_rhs_.GetBlock(1) = rhs.GetBlock(1);

    if (!use_W_ && remove_one_dof_ && myid_ == 0)
    {
        true_rhs_.GetBlock(1)(0) = 0.0;
    }

    minres_.Mult(true_rhs_, true_sol_);

    edgedof_d_td.Mult(true_sol_.GetBlock(0), sol.GetBlock(0));
    sol.GetBlock(1) = true_sol_.GetBlock(1);

    chrono.Stop();

    timing_ += chrono.RealTime();

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

    num_iterations_ += minres_.GetNumIterations();
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
