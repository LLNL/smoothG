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

    @brief Contains implementation of abstract base class MGLSolver
*/

#include "MGLSolver.hpp"

namespace smoothg
{

MGLSolver::MGLSolver(const std::vector<int>& offsets)
    : rhs_(offsets),
      sol_(offsets),
      nnz_(0), num_iterations_(0), timing_(0)
{

}

MGLSolver::MGLSolver(const MGLSolver& other) noexcept
    : rhs_(other.rhs_), sol_(other.sol_),
      print_level_(other.print_level_),
      max_num_iter_(other.max_num_iter_),
      rtol_(other.rtol_), atol_(other.atol_),
      nnz_(other.nnz_), num_iterations_(other.num_iterations_),
      timing_(other.timing_)
{
}

void swap(MGLSolver& lhs, MGLSolver& rhs) noexcept
{
    swap(static_cast<linalgcpp::Operator&>(lhs),
         static_cast<linalgcpp::Operator&>(rhs));

    std::swap(lhs.rhs_, rhs.rhs_);
    std::swap(lhs.sol_, rhs.sol_);
    std::swap(lhs.print_level_, rhs.print_level_);
    std::swap(lhs.max_num_iter_, rhs.max_num_iter_);
    std::swap(lhs.rtol_, rhs.rtol_);
    std::swap(lhs.atol_, rhs.atol_);
    std::swap(lhs.nnz_, rhs.nnz_);
    std::swap(lhs.num_iterations_, rhs.num_iterations_);
    std::swap(lhs.timing_, rhs.timing_);
}

void MGLSolver::Solve(const VectorView& rhs, VectorView sol) const
{
    rhs_.GetBlock(0) = 0.0;
    rhs_.GetBlock(1) = rhs;

    Solve(rhs_, sol_);

    sol = sol_.GetBlock(1);
}

void MGLSolver::Mult(const VectorView& rhs, VectorView sol) const
{
    Solve(rhs, sol);
}

} // namespace smoothg
