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

#ifndef __PARMIXEDMATRIX_HPP__
#define __PARMIXEDMATRIX_HPP__

#include "Utilities.hpp"
#include "MixedMatrix.hpp"

namespace smoothg
{

/**
   @brief Container for local mixed matrix information
          On false dofs.
*/

struct ParMixedMatrix
{
    ParMixedMatrix() = default;
    ParMixedMatrix(MPI_Comm comm, const Graph& graph, const MixedMatrix& local_mats);

    ~ParMixedMatrix() noexcept = default;

    ParMixedMatrix(const ParMixedMatrix& other) noexcept;
    ParMixedMatrix(ParMixedMatrix&& other) noexcept;
    ParMixedMatrix& operator=(ParMixedMatrix other) noexcept;

    friend void swap(ParMixedMatrix& lhs, ParMixedMatrix& rhs) noexcept;

    ParMatrix M_global_;
    ParMatrix D_global_;
    ParMatrix W_global_;
    std::vector<int> offsets_;
};

} // namespace smoothg

#endif /* __PARMIXEDMATRIX_HPP__ */
