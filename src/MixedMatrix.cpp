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

    @brief MixedMatrix class
*/

#include "MixedMatrix.hpp"

namespace smoothg
{

MixedMatrix::MixedMatrix(const Graph& graph, const std::vector<double>& global_weight)
{
    M_local_ = MakeLocalM(graph.edge_true_edge_, graph.edge_edge_, graph.edge_map_, global_weight);
    D_local_ = MakeLocalD(graph.edge_true_edge_, graph.vertex_edge_local_);
    W_local_ = SparseMatrix(std::vector<double>(D_local_.Rows(), 0.0));
    offsets_ = {0, M_local_.Rows(), M_local_.Rows() + D_local_.Rows()};
}

MixedMatrix::MixedMatrix(const MixedMatrix& other) noexcept
    : M_local_(other.M_local_),
      D_local_(other.D_local_),
      W_local_(other.W_local_),
      offsets_(other.offsets_)
{

}

MixedMatrix::MixedMatrix(MixedMatrix&& other) noexcept
{
    swap(*this, other);
}

MixedMatrix& MixedMatrix::operator=(MixedMatrix other) noexcept
{
    swap(*this, other);

    return *this;
}
    
void swap(MixedMatrix& lhs, MixedMatrix& rhs) noexcept
{
    swap(lhs.M_local_, rhs.M_local_);
    swap(lhs.D_local_, rhs.D_local_);
    swap(lhs.W_local_, rhs.W_local_);

    std::swap(lhs.offsets_, rhs.offsets_);
}



} // namespace smoothg
