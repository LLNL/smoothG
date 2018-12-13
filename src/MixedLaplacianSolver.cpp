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

MixedLaplacianSolver::MixedLaplacianSolver(const MixedMatrix& mgL,
                                           const mfem::Array<int>* ess_attr)
    : comm_(mgL.GetGraph().GetComm()),
      rhs_(mgL.GetBlockOffsets()), sol_(mgL.GetBlockOffsets()),
      nnz_(0), num_iterations_(0), timing_(0), remove_one_dof_(true),
      W_is_nonzero_(mgL.CheckW()), const_rep_(mgL.GetConstantRep())
{
    if (ess_attr)
    {
        assert(mgL.GetGraph().HasBoundary());
        for (int i = 0; i < ess_attr->Size(); ++i)
        {
            if ((*ess_attr)[i] == 0) // if Dirichlet pressure boundary is not empty
            {
                remove_one_dof_ = false;
                break;
            }
        }

        ess_edofs_.SetSize(sol_.BlockSize(0), 0);
        BooleanMult(mgL.EDofToBdrAtt(), *ess_attr, ess_edofs_);
        ess_edofs_.SetSize(sol_.BlockSize(0));
    }
}

void MixedLaplacianSolver::Solve(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const
{
    Mult(rhs, sol);
}

void MixedLaplacianSolver::Solve(const mfem::Vector& rhs, mfem::Vector& sol) const
{
    assert(rhs_);
    assert(sol_);

    rhs_.GetBlock(0) = 0.0;
    rhs_.GetBlock(1) = rhs;

    Solve(rhs_, sol_);

    sol = sol_.GetBlock(1);
}

void MixedLaplacianSolver::Mult(const mfem::Vector& rhs, mfem::Vector& sol) const
{
    Solve(rhs, sol);
}

void MixedLaplacianSolver::Orthogonalize(mfem::Vector& vec) const
{
    double local_dot = (vec * const_rep_);
    double global_dot;
    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm_);

    double local_scale = (const_rep_ * const_rep_);
    double global_scale;
    MPI_Allreduce(&local_scale, &global_scale, 1, MPI_DOUBLE, MPI_SUM, comm_);

    vec.Add(-global_dot / global_scale, const_rep_);
}

} // namespace smoothg
