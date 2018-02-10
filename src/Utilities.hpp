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

    @brief Implementations of some utility routines for linear algebra.

    These are implemented with and operate on linalgcpp data structures.
*/

#ifndef __UTILITIES_HPP__
#define __UTILITIES_HPP__

#include "linalgcpp.hpp"
#include "parlinalgcpp.hpp"

namespace smoothg
{

template <typename T = int>
linalgcpp::SparseMatrix<T> MakeAggVertex(const std::vector<int>& part)
{
    int nparts = *std::max(std::begin(part), std::end(part)) + 1;
    int nvertices = part.size();
    std::vector<int> indptr(nparts + 1);
    std::vector<int> indices(nvertices);
    std::vector<T> data(nvertices, 1.0);

    for (int i = 0; i < nvertices; ++i)
    {
        indptr[part[i] + 1]++;
    }

    for (int i = 1; i < nparts; ++i)
    {
        indptr[i + 1] += indptr[i];
    }

    for (int i = 0; i < nvertices; ++i)
    {
        indices[indptr[part[i]]++] = i;
    }

    assert(indptr[nparts - 1] == indptr[nparts]);

    for (int i = nparts - 1; i > 0; --i)
    {
        indptr[i] = indptr[i - 1];
    }

    indptr[0] = 0;

    return linalgcpp::SparseMatrix<T>(std::move(indptr), std::move(indices), std::move(data),
                                      nparts, nvertices);
}

template <typename T = int>
linalgcpp::SparseMatrix<T> MakeProcAgg(int num_procs, int num_aggs_global)
{
    int num_aggs_local = num_aggs_global / num_procs;
    int num_left = num_aggs_global % num_procs;

    std::vector<int> indptr(num_procs + 1);
    std::vector<int> indices(num_aggs_global);
    std::vector<T> data(num_aggs_global, 1.0);

    std::iota(std::begin(indices), std::end(indices), 0);

    for (int i = 0; i <= num_left; ++i)
    {
        indptr[i] = i * (num_aggs_local + 1);
    }

    for (int i = num_left + 1; i <= num_procs; ++i)
    {
        indptr[i] = indptr[i - 1] + num_aggs_local;
    }

    return linalgcpp::SparseMatrix<T>(std::move(indptr), std::move(indices), std::move(data),
                                      num_procs, num_aggs_global);
}

parlinalgcpp::ParMatrix MakeEdgeTrueEdge(MPI_Comm comm, const linalgcpp::SparseMatrix<int>& proc_edge, 
                                         const std::vector<int>& edge_map);

} //namespace smoothg

#endif // __UTILITIES_HPP__
