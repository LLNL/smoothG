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

using Vector = linalgcpp::Vector<double>;
using VectorView = linalgcpp::VectorView<double>;
using BlockVector = linalgcpp::BlockVector<double>;
using SparseMatrix = linalgcpp::SparseMatrix<double>;
using DenseMatrix = linalgcpp::DenseMatrix;
using BlockMatrix = linalgcpp::BlockMatrix<double>;
using ParMatrix = parlinalgcpp::ParMatrix;

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

ParMatrix MakeEdgeTrueEdge(MPI_Comm comm, const SparseMatrix& proc_edge, 
                                         const std::vector<int>& edge_map);

SparseMatrix RestrictInterior(const SparseMatrix& mat);
ParMatrix RestrictInterior(const ParMatrix& mat);

SparseMatrix MakeFaceAggInt(const ParMatrix& agg_agg);

SparseMatrix MakeFaceEdge(const ParMatrix& agg_agg,
                                  const ParMatrix& edge_edge,
                                  const SparseMatrix& agg_edge_ext,
                                  const SparseMatrix& face_edge_ext);

SparseMatrix ExtendFaceAgg(const ParMatrix& agg_agg,
                           const SparseMatrix& face_agg_int);

ParMatrix MakeFaceTrueEdge(const ParMatrix& face_face);
ParMatrix MakeExtPermutation(MPI_Comm comm, const ParMatrix& parmat);

SparseMatrix SparseIdentity(int size);
SparseMatrix SparseIdentity(int rows, int cols, int row_offset = 0, int col_offset = 0);

std::vector<int> GetExtDofs(const ParMatrix& mat_ext, int row);

void SetMarker(std::vector<int>& marker, const std::vector<int>& indices);
void ClearMarker(std::vector<int>& marker, const std::vector<int>& indices);

DenseMatrix Orthogonalize(DenseMatrix& mat);
DenseMatrix Orthogonalize(DenseMatrix& mat, Vector& vect);

void Deflate(DenseMatrix& A, const VectorView& vect);

DenseMatrix RestrictLocal(const DenseMatrix& ext_mat,
                          std::vector<int>& global_marker,
                          const std::vector<int>& ext_indices,
                          const std::vector<int>& local_indices);

} //namespace smoothg

#endif // __UTILITIES_HPP__
