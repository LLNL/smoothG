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

#ifndef __MIXEDMATRIX_HPP__
#define __MIXEDMATRIX_HPP__

#include "Utilities.hpp"
#include "Graph.hpp"

namespace smoothg
{

/**
   @brief Container for local mixed matrix information
          On false dofs.
*/

class MixedMatrix
{
    public:
        MixedMatrix() = default;
        MixedMatrix(const Graph& graph, const std::vector<double>& global_weight);
        MixedMatrix(SparseMatrix M_local, SparseMatrix D_local, SparseMatrix W_local, ParMatrix edge_true_edge);

        ~MixedMatrix() noexcept = default;

        MixedMatrix(const MixedMatrix& other) noexcept;
        MixedMatrix(MixedMatrix&& other) noexcept;
        MixedMatrix& operator=(MixedMatrix other) noexcept;

        friend void swap(MixedMatrix& lhs, MixedMatrix& rhs) noexcept;

        SparseMatrix M_local_;
        SparseMatrix D_local_;
        SparseMatrix W_local_;

        ParMatrix M_global_;
        ParMatrix D_global_;
        ParMatrix W_global_;

        std::vector<int> offsets_;
        std::vector<int> true_offsets_;

    private:
        void Init(const ParMatrix& edge_true_edge);

};

} // namespace smoothg

#endif /* __MIXEDMATRIX_HPP__ */
