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

    @brief GraphCoarsen class
*/

#include "GraphEdgeSolver.hpp"

namespace smoothg
{

GraphEdgeSolver::GraphEdgeSolver(const SparseMatrix& M, const SparseMatrix& D)
    : GraphEdgeSolver(M.GetData(), D)
{
}

GraphEdgeSolver::GraphEdgeSolver(const std::vector<double>& M_data,
                         const SparseMatrix& D)
    : MinvDT_(D.Transpose()), vect_sol_(D.Rows(), 0.0)
{
    MinvDT_.InverseScaleRows(M_data);

    SparseMatrix A = D.Mult(MinvDT_);
    A.EliminateRowCol(0);

    Ainv_ = SparseSolver(std::move(A));
}

GraphEdgeSolver::GraphEdgeSolver(const GraphEdgeSolver& other) noexcept
    : MinvDT_(other.MinvDT_),
      Ainv_(other.Ainv_),
      vect_sol_(other.vect_sol_)
{

}

GraphEdgeSolver::GraphEdgeSolver(GraphEdgeSolver&& other) noexcept
{
    swap(*this, other);
}

GraphEdgeSolver& GraphEdgeSolver::operator=(GraphEdgeSolver other) noexcept
{
    swap(*this, other);

    return *this;
}
    
void swap(GraphEdgeSolver& lhs, GraphEdgeSolver& rhs) noexcept
{
    swap(lhs.MinvDT_, rhs.MinvDT_);
    swap(lhs.Ainv_, rhs.Ainv_);
    swap(lhs.vect_sol_, rhs.vect_sol_);
}

Vector GraphEdgeSolver::Mult(const VectorView& input) const
{
    Vector vect(MinvDT_.Rows());

    Mult(input, vect);

    return vect;
}

void GraphEdgeSolver::Mult(const VectorView& input, VectorView& output) const
{
    Vector elim_input(input);
    elim_input[0] = 0.0;

    Ainv_.Mult(elim_input, vect_sol_);
    MinvDT_.Mult(vect_sol_, output);
}





} // namespace smoothg
