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

// Aliases to avoid unnecessary type verbosity
using Vector = linalgcpp::Vector<double>;
using VectorView = linalgcpp::VectorView<double>;
using BlockVector = linalgcpp::BlockVector<double>;
using SparseMatrix = linalgcpp::SparseMatrix<double>;
using DenseMatrix = linalgcpp::DenseMatrix;
using CooMatrix = linalgcpp::CooMatrix<double>;
using BlockMatrix = linalgcpp::BlockMatrix<double>;
using ParMatrix = parlinalgcpp::ParMatrix;
using Timer = linalgcpp::Timer;

/** @brief Find processor id on given communicator
    @param comm MPI Communicator
    @returns processor id
*/
int MyId(MPI_Comm comm = MPI_COMM_WORLD);

/** @brief Create an edge to true edge relationship
    @param comm MPI Communicator
    @param proc_edge processor edge relationship
    @param edge_map local to global edge map
    @returns global edge to true edge
*/
ParMatrix MakeEdgeTrueEdge(MPI_Comm comm, const SparseMatrix& proc_edge,
                           const std::vector<int>& edge_map);

/** @brief Restricts aggregate relationship to interior only
    @param mat extended aggregate relationship
    @param mat interior aggregate relationship
*/
SparseMatrix RestrictInterior(const SparseMatrix& mat);

/** @brief Restricts aggregate relationship to interior only
    @param mat extended aggregate relationship
    @returns mat interior aggregate relationship
*/
ParMatrix RestrictInterior(const ParMatrix& mat);

/** @brief Create entity to true entity relationship
    @param entity_entity entity to entity relationship on false dofs
    @return entity_true_entity entity to true entity
*/
ParMatrix MakeEntityTrueEntity(const ParMatrix& entity_entity);

/** @brief Sparse identity of given size
    @param size square size of identity
    @return identity matrix
*/
SparseMatrix SparseIdentity(int size);

/** @brief Construct an rectangular identity matrix (as a SparseMatrix)
    @param rows number of row
    @param cols number of columns
    @param row_offset offset row where diagonal identity starts
    @param col_offset offset column where diagonal identity starts
*/
SparseMatrix SparseIdentity(int rows, int cols, int row_offset = 0, int col_offset = 0);

/** @brief Set a global marker with given local indices
    Such that marker[global_index] = local_index and
    all other entries are -1

    @param marker global marker to set
    @param indices local indices
*/
void SetMarker(std::vector<int>& marker, const std::vector<int>& indices);

/** @brief Clear a global marker with given local indices
    Such that marker[global_index] = -1

    @param marker global marker to clear
    @param indices local indices
*/
void ClearMarker(std::vector<int>& marker, const std::vector<int>& indices);

DenseMatrix Orthogonalize(DenseMatrix& mat, int max_keep = -1);
DenseMatrix Orthogonalize(DenseMatrix& mat, const VectorView& vect, int max_keep = -1);

/** @brief Orthogonalize all column vectors in the matrix from the constant vector.
    This is equivalent to shifting the vector so it has zero mean.

    @param mat Dense matrix of vectors to orthogonalize
*/
void OrthoConstant(DenseMatrix& mat);

/** @brief Orthogonalize this vector from the constant vector.
    This is equivalent to shifting the vector so it has zero mean.

    @param vect vector to orthogonalize
*/
void OrthoConstant(VectorView vect);

/** @brief Orthogonalize this vector from the constant vector in parallel
    This is equivalent to shifting the vector so it has zero mean.

    @param comm MPI Communicator
    @param vect vector to orthogonalize
    @param global_size global size of the vector
*/
void OrthoConstant(MPI_Comm comm, VectorView vect, int global_size);

/** @brief Make all column vectors of a dense matrix orthogonal to v

    @param A matrix to deflate
    @param vect vector to deflate by, assumed to be a unit vector
*/
void Deflate(DenseMatrix& A, const VectorView& vect);

/** @brief Compute D(numer - denom) / D(denom)
    @param comm MPI Communicator
    @param D D matrix
    @param numer numerator vector
    @param denom denominator vector
*/
double DivError(MPI_Comm comm, const SparseMatrix& D, const VectorView& numer,
                const VectorView& denom);

/** @brief Compute l2 error (numer - denom) / denom
    @param comm MPI Communicator
    @param numer numerator vector
    @param denom denominator vector
*/
double CompareError(MPI_Comm comm, const VectorView& numer, const VectorView& denom);

/** @brief Show error information.
    @param error_info array of size 3 or 4 that has vertex, edge, div errors, and optionally operator complexity.
*/
void ShowErrors(const std::vector<double>& error_info, std::ostream& out = std::cout,
                bool pretty = true);

/** @brief Compare errors between upscaled and fine solution.
    @param M matrix to scale edge values
    @param D matrix to compute div error
    @param upscaled_sol coarse approximation solution
    @param fine_sol fine level solution
    @returns array of {vertex_error, edge_error, div_error}
*/
std::vector<double> ComputeErrors(MPI_Comm comm, const SparseMatrix& M,
                                  const SparseMatrix& D,
                                  const BlockVector& upscaled_sol,
                                  const BlockVector& fine_sol);

/** @brief Print (string, double) pairs in JSON
    @param values values to print
    @param out output stream
    @param pretty print each pair on its own line if true
*/
void PrintJSON(const std::map<std::string, double>& values, std::ostream& out = std::cout,
               bool pretty = true);

/** @brief Create aggregate to vertex relationship
    @param partition partition of vertices
    @returns agg_vertex aggregate to vertex relationship
*/
SparseMatrix MakeAggVertex(const std::vector<int>& partition);

/** @brief Create processor to aggregate relationship
    @param num_procs number of processors
    @param num_aggs_global number of global aggregates
    @returns proc_agg processor to aggregate relationship
*/
SparseMatrix MakeProcAgg(int num_procs, int num_aggs_global);

/** @brief Use power iterations to find the maximum eigenpair of A
    @param comm MPI Communicator
    @param A operator to apply action of A
    @param result result on output, initial guess on input
    @param max_iter maxiumum number of iterations to perform
    @param tol tolerance to iterate to
    @param verbose print additional information if true
*/
double PowerIterate(MPI_Comm comm, const linalgcpp::Operator& A, VectorView result,
                    int max_iter = 1000, double tol = 1e-8, bool verbose = false);

/** @brief Broadcast a matrix from rank 0 to all other ranks
    @param comm MPI Communicator
    @param mat matrix to broadcast
*/
void BroadCast(MPI_Comm comm, SparseMatrix& mat);

/** @brief Extract a dense submatrix from a sparse matrix
    @param A matrix from which to extract
    @param row row indices to extract
    @param col column indices to extract
    @param colMapper map of global to local indices
    @param A_sub holds the extracted submatrix
*/
void ExtractSubMatrix(const SparseMatrix& A, const std::vector<int>& rows,
                      const std::vector<int>& cols, const std::vector<int>& colMapper,
                      DenseMatrix& A_sub);

/** @brief Compute the (scaled) outer product \f$ a v v^T \f$.

    @param a scalar multiple
    @param v the vector to outer product.
    @param aVVt the returned dense matrix.
*/
void MultScalarVVt(double a, const VectorView& v, DenseMatrix& aVVt);

/** @brief Assemble element matrices

    @param elem_dof element to dof relationship
    @param elems set of elements to assemble
    @returns assembled matrix
*/
SparseMatrix AssembleElemMat(const SparseMatrix& elem_dof, const std::vector<DenseMatrix>& elems);

/** @brief Adds two sparse matrices C = alpha * A + beta * B

    @param alpha scale for A
    @param A A matrix
    @param beta scale for B
    @param B B matrix
    @returns C such that C = alpha * A + beta * B
*/
SparseMatrix Add(double alpha, const SparseMatrix& A, double beta, const SparseMatrix& B);

} //namespace smoothg

#endif // __UTILITIES_HPP__
