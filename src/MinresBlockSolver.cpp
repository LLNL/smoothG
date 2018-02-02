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
MinresBlockSolver::MinresBlockSolver(
    MPI_Comm comm, mfem::HypreParMatrix* M, mfem::HypreParMatrix* D,
    const mfem::Array<int>& block_true_offsets)
    :
    MixedLaplacianSolver(),
    minres_(comm),
    comm_(comm),
    operator_(block_true_offsets),
    prec_(block_true_offsets)
{
    init(comm, M, D);
}

MinresBlockSolver::MinresBlockSolver(const MixedMatrix& mgL, MPI_Comm comm)
    : MixedLaplacianSolver(), minres_(comm), comm_(comm),
      operator_(mgL.get_blockTrueOffsets()),
      prec_(mgL.get_blockTrueOffsets())
{
    init(comm, &mgL.get_pM(), &mgL.get_pD());
}

void MinresBlockSolver::init(
    MPI_Comm comm, mfem::HypreParMatrix* M, mfem::HypreParMatrix* D)
{
    int myid, num_procs;
    MPI_Comm_rank(comm_, &myid);
    MPI_Comm_size(comm_, &num_procs);

    Dt_ = D->Transpose();
    s_elim_null_ = new mfem::SparseMatrix(D->Height());
    if (myid == 0)
        s_elim_null_->Add(0, 0, 1.0);
    s_elim_null_->Finalize();
    elim_null_ = new mfem::HypreParMatrix(comm_, D->GetGlobalNumRows(),
                                          D->RowPart(), s_elim_null_);
    nnz_ = M->NNZ() + D->NNZ() + Dt_->NNZ() + 1;

    operator_.SetBlock(0, 0, M);
    operator_.SetBlock(0, 1, Dt_);
    operator_.SetBlock(1, 0, D);
    operator_.SetBlock(1, 1, elim_null_);

    mfem::HypreParMatrix* MinvDt = D->Transpose();
    mfem::HypreParVector* Md = new mfem::HypreParVector(
        comm, M->GetGlobalNumRows(), M->GetRowStarts());
    M->GetDiag(*Md);
    MinvDt->InvScaleRows(*Md);
    schur_block_ = ParMult(D, MinvDt);
    // D retains ownership of rows, but MinvDt will be deleted, so we need to
    // take ownership of col_starts here.
    hypre_ParCSRMatrixSetColStartsOwner(*schur_block_, 1);
    hypre_ParCSRMatrixSetColStartsOwner(*MinvDt, 0);

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

    delete MinvDt;
    delete Md;
}

MinresBlockSolver::~MinresBlockSolver()
{
    delete Dt_;
    delete schur_block_;
    delete s_elim_null_;
    delete elim_null_;
}

void MinresBlockSolver::Mult(const mfem::BlockVector& rhs,
                             mfem::BlockVector& sol) const
{
    int myid;
    MPI_Comm_rank(comm_, &myid);
    sol = 0.0;

    mfem::StopWatch chrono;
    chrono.Clear();
    chrono.Start();
    minres_.Mult(rhs, sol);
    chrono.Stop();
    timing_ = chrono.RealTime();
    if (myid == 0)
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
MinresBlockSolverFalse::MinresBlockSolverFalse(
    const MixedMatrix& mgL, MPI_Comm comm)
    :
    MinresBlockSolver(mgL, comm),
    mixed_matrix_(mgL)
{
}

MinresBlockSolverFalse::~MinresBlockSolverFalse()
{
}

void MinresBlockSolverFalse::Mult(const mfem::BlockVector& rhs,
                                  mfem::BlockVector& sol) const
{
    int myid;
    MPI_Comm_rank(comm_, &myid);
    sol = 0.0;

    mfem::StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    const mfem::HypreParMatrix& edgedof_d_td = mixed_matrix_.get_edge_d_td();
    mfem::BlockVector true_rhs(mixed_matrix_.get_blockTrueOffsets());
    edgedof_d_td.MultTranspose(rhs.GetBlock(0), true_rhs.GetBlock(0));
    true_rhs.GetBlock(1) = rhs.GetBlock(1);

    mfem::BlockVector true_sol(mixed_matrix_.get_blockTrueOffsets());
    true_sol = 0.0;

    minres_.Mult(true_rhs, true_sol);

    edgedof_d_td.Mult(true_sol.GetBlock(0), sol.GetBlock(0));
    sol.GetBlock(1) = true_sol.GetBlock(1);

    chrono.Stop();
    timing_ = chrono.RealTime();
    if (myid == 0)
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

} // namespace smoothg
