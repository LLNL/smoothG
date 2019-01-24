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

} // namespace smoothg
