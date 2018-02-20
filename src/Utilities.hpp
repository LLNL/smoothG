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

template <typename T = double>
linalgcpp::SparseMatrix<T> MakeAggVertex(const std::vector<int>& partition)
{
    const int num_parts = *std::max_element(std::begin(partition), std::end(partition)) + 1;
    const int num_vert = partition.size();

    std::vector<int> indptr(num_vert + 1);
    std::vector<T> data(num_vert, 1);

    std::iota(std::begin(indptr), std::end(indptr), 0);

    linalgcpp::SparseMatrix<T> vertex_agg(std::move(indptr), partition, std::move(data), num_vert, num_parts);

    return vertex_agg.Transpose();
}

template <typename T = double>
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

parlinalgcpp::ParMatrix MakeEdgeTrueEdge(MPI_Comm comm, const linalgcpp::SparseMatrix<double>& proc_edge, 
                                         const std::vector<int>& edge_map);

linalgcpp::SparseMatrix<double> RestrictInterior(const linalgcpp::SparseMatrix<double>& mat);
parlinalgcpp::ParMatrix RestrictInterior(const parlinalgcpp::ParMatrix& mat);

linalgcpp::SparseMatrix<double> MakeFaceAggInt(const parlinalgcpp::ParMatrix& agg_agg);

linalgcpp::SparseMatrix<double> MakeFaceEdge(const parlinalgcpp::ParMatrix& agg_agg,
                                          const parlinalgcpp::ParMatrix& edge_edge,
                                          const linalgcpp::SparseMatrix<double>& agg_edge_ext,
                                          const linalgcpp::SparseMatrix<double>& face_edge_ext);

linalgcpp::SparseMatrix<double> ExtendFaceAgg(const parlinalgcpp::ParMatrix& agg_agg,
                                           const linalgcpp::SparseMatrix<double>& face_agg_int);

parlinalgcpp::ParMatrix MakeFaceTrueEdge(const parlinalgcpp::ParMatrix& face_face);
parlinalgcpp::ParMatrix MakeExtPermutation(MPI_Comm comm, const parlinalgcpp::ParMatrix& parmat);

linalgcpp::SparseMatrix<double> SparseIdentity(int size);
linalgcpp::SparseMatrix<double> SparseIdentity(int rows, int cols, int row_offset = 0, int col_offset = 0);

std::vector<int> GetExtDofs(const parlinalgcpp::ParMatrix& mat_ext, int row);

} //namespace smoothg

#endif // __UTILITIES_HPP__
