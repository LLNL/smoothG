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

#include <map>

#include "linalgcpp.hpp"
#include "parlinalgcpp.hpp"

/// Call output only on processor 0
#define ParPrint(myid, output) if (myid == 0) output

#if __cplusplus > 201103L
using std::make_unique;
#else
template<typename T, typename... Ts>
std::unique_ptr<T> make_unique(Ts&& ... params)
{
    return std::unique_ptr<T>(new T(std::forward<Ts>(params)...));
}

#endif

namespace smoothg
{

using Vector = linalgcpp::Vector<double>;
using VectorView = linalgcpp::VectorView<double>;
using BlockVector = linalgcpp::BlockVector<double>;
using SparseMatrix = linalgcpp::SparseMatrix<double>;
using DenseMatrix = linalgcpp::DenseMatrix;
using CooMatrix = linalgcpp::CooMatrix<double>;
using BlockMatrix = linalgcpp::BlockMatrix<double>;
using ParMatrix = parlinalgcpp::ParMatrix;
using Timer = linalgcpp::Timer;

int MyId(MPI_Comm comm = MPI_COMM_WORLD);

ParMatrix MakeEdgeTrueEdge(MPI_Comm comm, const SparseMatrix& proc_edge,
                                         const std::vector<int>& edge_map);

SparseMatrix RestrictInterior(const SparseMatrix& mat);
ParMatrix RestrictInterior(const ParMatrix& mat);

ParMatrix MakeEntityTrueEntity(const ParMatrix& face_face);
ParMatrix MakeExtPermutation(const ParMatrix& parmat);

SparseMatrix SparseIdentity(int size);
SparseMatrix SparseIdentity(int rows, int cols, int row_offset = 0, int col_offset = 0);

std::vector<int> GetExtDofs(const ParMatrix& mat_ext, int row);

void SetMarker(std::vector<int>& marker, const std::vector<int>& indices);
void ClearMarker(std::vector<int>& marker, const std::vector<int>& indices);

DenseMatrix Orthogonalize(DenseMatrix& mat, int max_keep = -1);
DenseMatrix Orthogonalize(DenseMatrix& mat, const VectorView& vect, int max_keep = -1);

void OrthoConstant(DenseMatrix& mat);
void OrthoConstant(VectorView vect);
void OrthoConstant(MPI_Comm comm, VectorView vect, int global_size);

void Deflate(DenseMatrix& A, const VectorView& vect);

DenseMatrix RestrictLocal(const DenseMatrix& ext_mat,
                          std::vector<int>& global_marker,
                          const std::vector<int>& ext_indices,
                          const std::vector<int>& local_indices);
double DivError(MPI_Comm comm, const SparseMatrix& D, const VectorView& numer,
                const VectorView& denom);
double CompareError(MPI_Comm comm, const VectorView& numer, const VectorView& denom);
void ShowErrors(const std::vector<double>& error_info, std::ostream& out = std::cout, bool pretty = true);
std::vector<double> ComputeErrors(MPI_Comm comm, const SparseMatrix& M,
                                  const SparseMatrix& D,
                                  const BlockVector& upscaled_sol,
                                  const BlockVector& fine_sol);

void PrintJSON(const std::map<std::string, double>& values, std::ostream& out = std::cout,
               bool pretty = true);

SparseMatrix MakeAggVertex(const std::vector<int>& partition);
SparseMatrix MakeProcAgg(int num_procs, int num_aggs_global);

double PowerIterate(MPI_Comm comm, const linalgcpp::Operator& A, VectorView result,
                    int max_iter = 1000, double tol = 1e-8, bool verbose = false);

void BroadCast(MPI_Comm comm, SparseMatrix& mat);

} //namespace smoothg

#endif // __UTILITIES_HPP__
