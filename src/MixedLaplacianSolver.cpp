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

namespace smoothg
{

MixedLaplacianSolver::MixedLaplacianSolver(const mfem::Array<int>& block_offsets)
    : rhs_(make_unique<mfem::BlockVector>(block_offsets)),
      sol_(make_unique<mfem::BlockVector>(block_offsets)),
      nnz_(0), num_iterations_(0), timing_(0)
{

}

void MixedLaplacianSolver::Solve(const mfem::Vector& rhs, mfem::Vector& sol) const
{
    assert(rhs_);
    assert(sol_);

    rhs_->GetBlock(0) = 0.0;
    rhs_->GetBlock(1) = rhs;

    Solve(*rhs_, *sol_);

    sol = sol_->GetBlock(1);
}

void MixedLaplacianSolver::Mult(const mfem::Vector& rhs, mfem::Vector& sol) const
{
    Solve(rhs, sol);
}

} // namespace smoothg
