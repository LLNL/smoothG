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
                                     mfem::SparseMatrix* W,
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
                             mfem::SparseMatrix* W)
{
    assert(M && D);

    if (!hDt_)
    {
        hDt_.reset(D->Transpose());
    }

    operator_.SetBlock(0, 0, M);
    operator_.SetBlock(0, 1, hDt_.get());
    operator_.SetBlock(1, 0, D);
    if (W)
    {
        operator_.SetBlock(1, 1, W, -1.0);
    }

    mfem::Vector Md;
    M->GetDiag(Md);
    hDt_->InvScaleRows(Md);
    schur_block_.reset(mfem::ParMult(D, hDt_.get()));
    hDt_->ScaleRows(Md);

    nnz_ = M->NNZ() + D->NNZ() + hDt_->NNZ();

    if (W_is_nonzero_)
    {
        mfem::HypreParMatrix pW(comm_, D->M(), D->RowPart(), W);
        nnz_ += pW.NNZ();
        schur_block_.reset(ParAdd(pW, *schur_block_));
    }
    else if (remove_one_dof_)
    {
        nnz_ += 1;
    }

    Mprec_.reset(new mfem::HypreDiagScale(*M));
    Sprec_.reset(new mfem::HypreBoomerAMG(*schur_block_));
    Sprec_->SetPrintLevel(0);

    prec_.SetDiagonalBlock(0, Mprec_.get());
    prec_.SetDiagonalBlock(1, Sprec_.get());

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
    MixedLaplacianSolver(mgL.GetComm(), mgL.BlockTrueOffsets(), mgL.CheckW()),
    minres_(comm_),
    operator_(mgL.BlockTrueOffsets()),
    prec_(mgL.BlockTrueOffsets())
{
    MixedLaplacianSolver::Init(mgL, ess_attr);

    mfem::SparseMatrix M_proc(mgL.GetM());
    for (int mm = 0; mm < ess_edofs_.Size(); ++mm)
    {
        if (ess_edofs_[mm])
            M_proc.EliminateRowCol(mm, true); // assume essential data = 0
    }

    hM_.reset(mgL.MakeParallelM(M_proc));

    mfem::SparseMatrix D_proc(mgL.GetD());
    if (ess_edofs_.Size())
    {
        D_proc.EliminateCols(ess_edofs_);
    }

    if (!W_is_nonzero_ && remove_one_dof_ && myid_ == 0)
    {
        D_proc.EliminateRow(0);
    }

    hD_.reset(mgL.MakeParallelD(D_proc));

    if (W_is_nonzero_)
    {
        W_.reset(new mfem::SparseMatrix(mgL.GetW()));
    }
    else if (remove_one_dof_ && myid_ == 0)
    {
        W_.reset(new mfem::SparseMatrix(mgL.NumVDofs()));
        W_->Add(0, 0, -1.0);
    }

    Init(hM_.get(), hD_.get(), W_.get());
}

void MinresBlockSolver::Mult(const mfem::BlockVector& rhs,
                             mfem::BlockVector& sol) const
{
    mfem::StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    double rhs0 = rhs.GetBlock(1)[0];

    if (!W_is_nonzero_ && remove_one_dof_ && myid_ == 0)
    {
        const_cast<mfem::Vector&>(rhs.GetBlock(1))[0] = 0.0;
    }

    minres_.Mult(rhs, sol);

    const_cast<mfem::Vector&>(rhs.GetBlock(1))[0] = rhs0;

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

void MinresBlockSolverFalse::UpdateElemScaling(const mfem::Vector& elem_scaling_inverse)
{
    auto M_proc = mixed_matrix_.GetMBuilder().BuildAssembledM(elem_scaling_inverse);

    for (int mm = 0; mm < ess_edofs_.Size(); ++mm)
    {
        if (ess_edofs_[mm])
            M_proc.EliminateRowCol(mm, true); // assume essential data = 0
    }

    hM_.reset(mixed_matrix_.MakeParallelM(M_proc));

    Init(hM_.get(), hD_.get(), W_.get());
}


MinresBlockSolverFalse::MinresBlockSolverFalse(const MixedMatrix& mgL,
                                               const mfem::Array<int>* ess_attr)
    :
    MinresBlockSolver(mgL, ess_attr),
    mixed_matrix_(mgL)
{
}

void MinresBlockSolverFalse::Mult(const mfem::BlockVector& rhs,
                                  mfem::BlockVector& sol) const
{
    const auto& edof_trueedof = mixed_matrix_.GetGraphSpace().EDofToTrueEDof();
    edof_trueedof.MultTranspose(rhs.GetBlock(0), rhs_.GetBlock(0));
    rhs_.GetBlock(1) = rhs.GetBlock(1);

    MinresBlockSolver::Mult(rhs_, sol_);

    edof_trueedof.Mult(sol_.GetBlock(0), sol.GetBlock(0));
    sol.GetBlock(1) = sol_.GetBlock(1);
}

void MinresBlockSolverFalse::Mult(const mfem::Vector& rhs, mfem::Vector& sol) const
{
    rhs_.GetBlock(0) = 0.0;
    rhs_.GetBlock(1) = rhs;
    rhs_.GetBlock(1) *= -1.0;

    MinresBlockSolver::Mult(rhs_, sol_);

    sol = sol_.GetBlock(1);
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
