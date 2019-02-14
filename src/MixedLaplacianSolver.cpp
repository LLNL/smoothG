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

/** @file

    @brief Contains implementation of abstract base class MixedLaplacianSolver
*/

#include "MixedLaplacianSolver.hpp"
#include "MixedMatrix.hpp"
#include "utilities.hpp"

namespace smoothg
{

MixedLaplacianSolver::MixedLaplacianSolver(MPI_Comm comm,
                                           const mfem::Array<int>& block_offsets,
                                           bool W_is_nonzero)
    : comm_(comm), rhs_(block_offsets), sol_(block_offsets), nnz_(0),
      num_iterations_(0), timing_(0), remove_one_dof_(true), W_is_nonzero_(W_is_nonzero)
{ }

void MixedLaplacianSolver::Solve(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const
{
    Mult(rhs, sol);
}

void MixedLaplacianSolver::Solve(const mfem::Vector& rhs, mfem::Vector& sol) const
{
    Mult(rhs, sol);
}

void MixedLaplacianSolver::Mult(const mfem::Vector& rhs, mfem::Vector& sol) const
{
    rhs_.GetBlock(0) = 0.0;
    rhs_.GetBlock(1) = rhs;
    rhs_.GetBlock(1) *= -1.0;

    Mult(rhs_, sol_);

    sol = sol_.GetBlock(1);
}

void MixedLaplacianSolver::Init(const MixedMatrix& mgL, const mfem::Array<int>* ess_attr)
{
    MPI_Comm_rank(comm_, &myid_);
    const_rep_ = &(mgL.GetConstantRep());
    if (ess_attr)
    {
        assert(mgL.GetGraph().HasBoundary());
        ess_edofs_.SetSize(mgL.NumEDofs(), 0);
        BooleanMult(mgL.GetGraphSpace().EDofToBdrAtt(), *ess_attr, ess_edofs_);
        ess_edofs_.SetSize(mgL.NumEDofs());

        remove_one_dof_ = (ess_attr->Find(0) == -1); // all attributes are essential
    }
}

void MixedLaplacianSolver::Orthogonalize(mfem::Vector& vec) const
{
    assert(const_rep_);

    double local_dot = (vec * (*const_rep_));
    double global_dot;
    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm_);

    double local_scale = ((*const_rep_) * (*const_rep_));
    double global_scale;
    MPI_Allreduce(&local_scale, &global_scale, 1, MPI_DOUBLE, MPI_SUM, comm_);

    vec.Add(-global_dot / global_scale, *const_rep_);
}

PrimalSolver::PrimalSolver(const MixedMatrix& mgL, const mfem::Array<int>* ess_attr)
    : MixedLaplacianSolver(mgL.GetComm(), mgL.BlockTrueOffsets(), mgL.CheckW()),
      cg_(comm_), mixed_matrix_(mgL)
{
    MixedLaplacianSolver::Init(mgL, ess_attr);

    mfem::SparseMatrix D_proc(mgL.GetD());
    if (ess_edofs_.Size())
    {
        D_proc.EliminateCols(ess_edofs_);
    }
    if (!W_is_nonzero_ && remove_one_dof_ && myid_ == 0)
    {
        D_proc.EliminateRow(0);
    }
    D_.reset(mgL.MakeParallelD(D_proc));
    Dt_.reset(D_->Transpose());

    if (W_is_nonzero_)
    {
        W_.reset(new mfem::SparseMatrix(mgL.GetW()));
    }
    else if (remove_one_dof_ && myid_ == 0)
    {
        W_.reset(new mfem::SparseMatrix(mgL.NumVDofs()));
        W_->Add(0, 0, -1.0);
        W_->Finalize();
    }

    cg_.SetPrintLevel(print_level_);
    cg_.SetMaxIter(max_num_iter_);
    cg_.SetRelTol(rtol_);
    cg_.SetAbsTol(atol_);
    cg_.iterative_mode = true;

    Init(mgL.GetM());
}

void PrimalSolver::Init(mfem::SparseMatrix M_proc)
{
    for (int mm = 0; mm < ess_edofs_.Size(); ++mm)
    {
        if (ess_edofs_[mm])
            M_proc.EliminateRowCol(mm, true); // assume essential data = 0
    }
    std::unique_ptr<mfem::HypreParMatrix> M(mixed_matrix_.MakeParallelM(M_proc));

    M->GetDiag(M_diag_);
    Dt_->InvScaleRows(M_diag_);
    operator_.reset(mfem::ParMult(D_.get(), Dt_.get()));
    Dt_->ScaleRows(M_diag_);

    if (W_)
    {
        auto diag = GetDiag(*operator_);
        diag += *W_;
    }

    nnz_ = 2 * D_->NNZ() + M->NNZ() + (W_ ? W_->NumNonZeroElems() : 0);
    cg_.SetOperator(*operator_);

    prec_.reset(new mfem::HypreBoomerAMG(*operator_));
    prec_->SetPrintLevel(0);
    cg_.SetPreconditioner(*prec_);
}

void PrimalSolver::UpdateElemScaling(const mfem::Vector& elem_scaling_inverse)
{
    Init(mixed_matrix_.GetMBuilder().BuildAssembledM(elem_scaling_inverse));
}

void PrimalSolver::Mult(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const
{
    assert(rhs.GetBlock(0).Norml2() == 0.0);
    rhs_.GetBlock(1) = rhs.GetBlock(1);

    if (!W_is_nonzero_ && remove_one_dof_ && myid_ == 0)
    {
        rhs_.GetBlock(1)[0] = -sol.GetBlock(1)[0];
    }

    mfem::StopWatch chrono;
    chrono.Start();

    cg_.Mult(rhs_.GetBlock(1), sol.GetBlock(1));

    timing_ = chrono.RealTime();

    Dt_->Mult(sol.GetBlock(1), sol_.GetBlock(0));
    InvRescaleVector(M_diag_, sol_.GetBlock(0));
    mixed_matrix_.GetGraphSpace().EDofToTrueEDof().Mult(sol_.GetBlock(0), sol.GetBlock(0));

    sol.GetBlock(1) *= -1.0;
    if (!W_is_nonzero_ && remove_one_dof_)
    {
        Orthogonalize(sol.GetBlock(1));
    }

    num_iterations_ = cg_.GetNumIterations();
}

void PrimalSolver::SetPrintLevel(int print_level)
{
    MixedLaplacianSolver::SetPrintLevel(print_level);
    cg_.SetPrintLevel(print_level_);
}

void PrimalSolver::SetMaxIter(int max_num_iter)
{
    MixedLaplacianSolver::SetMaxIter(max_num_iter);
    cg_.SetMaxIter(max_num_iter_);
}

void PrimalSolver::SetRelTol(double rtol)
{
    MixedLaplacianSolver::SetRelTol(rtol);
    cg_.SetRelTol(rtol_);
}

void PrimalSolver::SetAbsTol(double atol)
{
    MixedLaplacianSolver::SetAbsTol(atol);
    cg_.SetAbsTol(atol_);
}

} // namespace smoothg
