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

    @brief ParMixedMatrix class
*/

#include "ParMixedMatrix.hpp"

namespace smoothg
{

ParMixedMatrix::ParMixedMatrix(MPI_Comm comm, const Graph& graph, const MixedMatrix& local_mats)
{
    const SparseMatrix& M_local = local_mats.M_local_;
    const SparseMatrix& D_local = local_mats.D_local_;
    const SparseMatrix& W_local = local_mats.W_local_;
    const ParMatrix& edge_true_edge = graph.edge_true_edge_;

    auto starts = parlinalgcpp::GenerateOffsets(comm, {D_local.Rows(), D_local.Cols()});
    std::vector<HYPRE_Int>& vertex_starts = starts[0];
    std::vector<HYPRE_Int>& edge_starts = starts[1];

    ParMatrix M_d(comm, edge_starts, M_local);
    ParMatrix D_d(comm, vertex_starts, edge_starts, D_local);

    M_global_ = parlinalgcpp::RAP(M_d, edge_true_edge);
    D_global_ = D_d.Mult(edge_true_edge);
    W_global_ = ParMatrix(comm, vertex_starts, W_local);
    offsets_ = {0, M_global_.Rows(), M_global_.Rows() + D_global_.Rows()};
}

ParMixedMatrix::ParMixedMatrix(const ParMixedMatrix& other) noexcept
    : M_global_(other.M_global_),
      D_global_(other.D_global_),
      W_global_(other.W_global_),
      offsets_(other.offsets_)
{
}

ParMixedMatrix::ParMixedMatrix(ParMixedMatrix&& other) noexcept
{
    swap(*this, other);
}

ParMixedMatrix& ParMixedMatrix::operator=(ParMixedMatrix other) noexcept
{
    swap(*this, other);

    return *this;
}
    
void swap(ParMixedMatrix& lhs, ParMixedMatrix& rhs) noexcept
{
    swap(lhs.M_global_, rhs.M_global_);
    swap(lhs.D_global_, rhs.D_global_);
    swap(lhs.W_global_, rhs.W_global_);

    std::swap(lhs.offsets_, rhs.offsets_);
}



} // namespace smoothg
