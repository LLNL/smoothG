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

    Mprec_.reset(new mfem::HypreSmoother(*M));
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
    //    minres_.iterative_mode = false;
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
    mfem::SparseMatrix M_elim(M_proc.NumRows());

    for (int mm = 0; mm < ess_edofs_.Size(); ++mm)
    {
        if (ess_edofs_[mm])
            M_proc.EliminateRowCol(mm, M_elim);
    }

    M_elim_.Swap(M_elim);

    hM_.reset(mgL.MakeParallelM(M_proc));

    mfem::SparseMatrix D_proc(mgL.GetD());
    mfem::SparseMatrix D_elim(D_proc);
    if (ess_edofs_.Size())
    {
        D_proc.EliminateCols(ess_edofs_);
    }

    if (!W_is_nonzero_ && remove_one_dof_ && myid_ == 0)
    {
        D_proc.EliminateRow(0);
    }

    D_elim.Add(-1.0, D_proc);
    D_elim_.Swap(D_elim);

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

    auto solver = is_symmetric_ ? dynamic_cast<const mfem::IterativeSolver*>(&minres_)
                                : dynamic_cast<const mfem::IterativeSolver*>(&gmres_);

    solver->Mult(rhs, sol);

    const_cast<mfem::Vector&>(rhs.GetBlock(1))[0] = rhs0;

    if (!W_is_nonzero_ && remove_one_dof_)
    {
        Orthogonalize(sol.GetBlock(1));
    }

    chrono.Stop();
    timing_ = chrono.RealTime();

    num_iterations_ = solver->GetNumIterations();

    if (myid_ == 0 && print_level_ > 0)
    {
        std::string solver_name = is_symmetric_ ? "Minres" : "GMRES";

        std::cout << "  Timing " + solver_name + ": Solver done in "
                  << timing_ << "s. \n";

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
}

void MinresBlockSolverFalse::UpdateElemScaling(const mfem::Vector& elem_scaling_inverse)
{
    mfem::StopWatch chrono;
    chrono.Start();

    auto M_proc = mixed_matrix_.GetMBuilder().BuildAssembledM(elem_scaling_inverse);

    for (int mm = 0; mm < ess_edofs_.Size(); ++mm)
    {
        if (ess_edofs_[mm])
            M_proc.EliminateRowCol(mm, true); // assume essential data = 0
    }

    hM_.reset(mixed_matrix_.MakeParallelM(M_proc));

    Init(hM_.get(), hD_.get(), W_.get());

    if (myid_ == 0 && print_level_ > 0)
    {
        std::cout << "  BlockSolver: rescaled system assembled in "
                  << chrono.RealTime() << "s. \n";
    }
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
    mfem::Vector rhs_sigma(rhs.GetBlock(0));
    M_elim_.AddMult(sol.GetBlock(0), rhs_sigma, -1);

    for (int mm = 0; mm < ess_edofs_.Size(); ++mm)
    {
        if (ess_edofs_[mm])
            rhs_sigma[mm] = sol[mm];
    }

    const auto& edof_trueedof = mixed_matrix_.GetGraphSpace().EDofToTrueEDof();
    edof_trueedof.MultTranspose(rhs_sigma, rhs_.GetBlock(0));

    rhs_.GetBlock(1) = rhs.GetBlock(1);
    D_elim_.AddMult(sol.GetBlock(0), rhs_.GetBlock(1), -1);

    mfem::SparseMatrix edof_trueedof_diag = GetDiag(edof_trueedof);
    edof_trueedof_diag.MultTranspose(sol.GetBlock(0), sol_.GetBlock(0));
    sol_.GetBlock(1) = sol.GetBlock(1);

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

void MinresBlockSolverFalse::UpdateJacobian(const mfem::Vector& elem_scaling_inverse,
                                            const std::vector<mfem::DenseMatrix>& N_el)
{
    mfem::StopWatch chrono;
    chrono.Start();

    // Update M and Mprec
    auto M_proc = mixed_matrix_.GetMBuilder().BuildAssembledM(elem_scaling_inverse);
    for (int mm = 0; mm < ess_edofs_.Size(); ++mm)
    {
        if (ess_edofs_[mm])
            M_proc.EliminateRowCol(mm, true); // assume essential data = 0
    }
    hM_.reset(mixed_matrix_.MakeParallelM(M_proc));
    operator_.SetBlock(0, 0, hM_.get());

    Mprec_.reset(new mfem::HypreSmoother(*hM_));
    prec_.SetDiagonalBlock(0, Mprec_.get());

    // Update N and Sprec
    auto& space = mixed_matrix_.GetGraphSpace();

    mfem::Array<int> local_edofs, local_vdofs;
    mfem::SparseMatrix dMdp(mixed_matrix_.NumEDofs(), mixed_matrix_.NumVDofs());
    for (unsigned int i = 0; i < N_el.size(); ++i)
    {
        GetTableRow(space.VertexToEDof(), i, local_edofs);
        GetTableRow(space.VertexToVDof(), i, local_vdofs);
        dMdp.AddSubMatrix(local_edofs, local_vdofs, N_el[i]);
    }
    dMdp.Finalize();

    if (NNZ(dMdp) > 0)
    {
        mfem::SparseMatrix dMdp_copy(dMdp);
        for (int i = 0; i < ess_edofs_.Size(); ++i)
        {
            if (ess_edofs_[i])
            {
                dMdp_copy.EliminateRow(i);
            }
        }

        block_01_ = ParMult(space.TrueEDofToEDof(), dMdp_copy, space.VDofStarts());
        block_01_.reset(ParAdd(*block_01_, *hDt_));
    }
    else
    {
        block_01_.reset(hD_->Transpose());
    }

    operator_.SetBlock(0, 1, block_01_.get());

    mfem::Vector Md;
    hM_->GetDiag(Md);
    block_01_->InvScaleRows(Md);
    schur_block_.reset(mfem::ParMult(hD_.get(), block_01_.get()));

    if (W_is_nonzero_)
    {
        mfem::HypreParMatrix pW(comm_, hD_->M(), hD_->RowPart(), W_.get());
        nnz_ += pW.NNZ();
        schur_block_.reset(ParAdd(pW, *schur_block_));
    }

    block_01_->ScaleRows(Md);
    Sprec_.reset(new mfem::HypreBoomerAMG(*schur_block_));
    Sprec_->SetPrintLevel(0);
    prec_.SetDiagonalBlock(1, Sprec_.get());

    gmres_.SetPrintLevel(print_level_);
    gmres_.SetMaxIter(max_num_iter_);
    gmres_.SetRelTol(rtol_);
    gmres_.SetAbsTol(atol_);
    gmres_.SetOperator(operator_);
    gmres_.SetPreconditioner(prec_);
    gmres_.iterative_mode = false;

    is_symmetric_ = false;

    if (myid_ == 0 && print_level_ > 0)
    {
        std::cout << "  BlockSolver: rescaled system assembled in "
                  << chrono.RealTime() << "s. \n";
    }
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
