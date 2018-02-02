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

    @brief A collection of utility linear algebra routines.
*/

#ifndef __MATRIXUTILITIES_HPP__
#define __MATRIXUTILITIES_HPP__

#include "mfem.hpp"
#include <memory>

namespace smoothg
{

/**
    @brief Compute \f$ C = AB \f$, where \f$ A \f$ is sparse and
           \f$ B \f$ is dense.
*/
void MultSparseDense(const mfem::SparseMatrix& A, mfem::DenseMatrix& B,
                     mfem::DenseMatrix& C);

/**
    @brief Compute the (scaled) outer product \f$ a v v^T \f$.

    @param a scalar multiple
    @param v the vector to outer product.
    @param aVVt the returned dense matrix.
*/
void Mult_a_VVt(const double a, const mfem::Vector& v, mfem::DenseMatrix& aVVt);

/**
    @brief Set values of the non zero entries of a HypreParMatrix to 'c'.
*/
void SetConstantValue(mfem::HypreParMatrix* pmat, double c);

/**
    @brief Construct the relation table aggregate to vertex from partition
*/
std::unique_ptr<mfem::SparseMatrix> PartitionToMatrix(
    const mfem::Array<int>& partition, int nparts);

/**
   @brief Construct an identity matrix (as a SparseMatrix) of size 'size'
*/
std::unique_ptr<mfem::SparseMatrix> SparseIdentity(int size);

/**
   @brief Extract a submatrix from a matrix

   @param A the matrix to extract from
   @param rows the rows to extract
   @param cols the columns to extract
   @param colMapper basically a data workspace

   @returns the extracted submatrix
*/
std::unique_ptr<mfem::SparseMatrix> ExtractRowAndColumns(
    const mfem::SparseMatrix& A, const mfem::Array<int>& rows,
    const mfem::Array<int>& cols, mfem::Array<int>& colMapper,
    bool colMapper_not_filled = true);

/**
   @brief Extract a submatrix from a sparse matrix, return it dense

   @param A the matrix to extract from
   @param rows the rows to extract
   @param cols the columns to extract
   @param colMapper basically a data workspace
   @param A_sub the returned (dense) submatrix)
*/
void ExtractSubMatrix(
    const mfem::SparseMatrix& A, const mfem::Array<int>& rows,
    const mfem::Array<int>& cols, const mfem::Array<int>& colMapper,
    mfem::DenseMatrix& A_sub);

/**
   @brief Fill a DenseMatrix with the entries of a SparseMatrix

   The size of the matrix Adense is set to be same as the size of Asparse
*/
void Full(const mfem::SparseMatrix& Asparse, mfem::DenseMatrix& Adense);

/**
   @brief Fill a DenseMatrix with the entries of transpose of a SparseMatrix

   The size of the matrix AdenseT is set to be same as the size of Asparse^T
*/
void FullTranspose(const mfem::SparseMatrix& Asparse, mfem::DenseMatrix& AdenseT);

/**
   @brief Prepend the (column) vector a to the matrix b.
*/
void Concatenate(const mfem::Vector& a, const mfem::DenseMatrix& b,
                 mfem::DenseMatrix& C);

/**
   @brief Make all column vectors of a orthogonal to v

   The input vector v is assumed to be a unit vector
*/
void Deflate(mfem::DenseMatrix& a, const mfem::Vector& v);

/**
   @brief Orthogonalize this vector from the constant vector.

   This is equivalent to shifting the vector so it has zero mean.

   The correct way to do this is with respect to a finite element space,
   take an FiniteElementSpace argument or a list of volumes or something.
   For now we assume equal size volumes, or a graph, and just take
   vec.Sum() / vec.Size()

   @todo improve this for the finite volume case
*/
void orthogonalize_from_constant(mfem::Vector& vec);

/**
   @brief Orthogonalize this vector from the constant vector.

   This is equivalent to shifting the vector so it has zero mean.

   The correct way to do this is with respect to a finite element space,
   take an FiniteElementSpace argument or a list of volumes or something.
   For now we assume equal size volumes, or a graph, and just take
   vec.Sum() / vec.Size()

   @todo improve this for the finite volume case
*/
void par_orthogonalize_from_constant(mfem::Vector& vec, int globalsize);
void par_orthogonalize_from_constant(mfem::HypreParVector& vec);

/** Create a std::vector of mfem::Vectors from a std::vector of mfem::BlockVectors
 *
 *  Given a block number, pulls out the mfem::Vectors from each
 *  BlockVector associated with that block number and puts it into a
 *  std::vector. Each mfem::Vector is only a view, so this does not do
 *  a deep copy.
 */
std::vector<mfem::Vector> get_blocks(
    const std::vector<std::unique_ptr<mfem::BlockVector>>& blockvecs, int block_num);

/** Get a matrix of squared errors.
 *
 *  Produces a strictly lower triangular matrix of squared differences
 *  between vectors under the given inner-product norm. If
 *  diag_sq_norms, the diagonal contains the squared norms of the
 *  vectors in the inner-product norm.
 *
 * \param vecs The std::vector of mfem::Vectors to compare.
 * \param inner_prod_mats The inner product matrix defining the norm to use.
 * \param diag_sq_norms If true, put the squared norms of the vectors on the diagonal.
 */
mfem::DenseMatrix get_sq_differences_matrix(const std::vector<mfem::Vector>& vecs,
                                            const mfem::SparseMatrix* inner_prod_mats,
                                            bool diag_sq_norms = false);

/**
   @brief Generate the "start" array for HypreParMatrix based on the number of
   local true dofs
*/
void GenerateOffsets(MPI_Comm comm, int N, HYPRE_Int loc_sizes[],
                     mfem::Array<HYPRE_Int>* offsets[]);

/**
   @brief Solver for local saddle point problems, see the formula below.

   This routine solves local saddle point problems of the form
   \f[
     \left( \begin{array}{cc}
       M&  D^T \\
       D&
     \end{array} \right)
     \left( \begin{array}{c}
       \sigma \\ u
     \end{array} \right)
     =
     \left( \begin{array}{c}
       0 \\ -g
     \end{array} \right)
   \f]

   This local solver is called when computing PV vectors, bubbles, and trace
   extensions.
*/
class LocalGraphEdgeSolver
{
public:
    /**
       @brief Constructor of the local saddle point solver.

       @param M matrix \f$ M \f$ in the formula in the class description
       @param D matrix \f$ D \f$ in the formula in the class description

       M is assumed to be diagonal (TODO: should assert?)
       We construct the matrix \f$ A = D M^{-1} D^T \f$, eliminate the zeroth
       degree of freedom to ensure it is solvable. LU factorization of \f$ A \f$
       is computed and stored (until the object is deleted) for potential
       multiple solves.
    */
    LocalGraphEdgeSolver(const mfem::SparseMatrix& M,
                         const mfem::SparseMatrix& D);

    /// M is the diagonal of the matrix \f$ M \f$ in the formula above
    LocalGraphEdgeSolver(const mfem::Vector& M, const mfem::SparseMatrix& D);

    /**
       @brief Solves \f$ (D M^{-1} D^T) u = g\f$, \f$ \sigma = M^{-1} D^T u \f$.

       @param rhs \f$ g \f$ in the formula above
       @param sol_sigma \f$ \sigma \f$ in the formula above
    */
    void Mult(const mfem::Vector& rhs, mfem::Vector& sol_sigma);

    /**
       @brief Solves \f$ (D M^{-1} D^T) u = g\f$, \f$ \sigma = M^{-1} D^T u \f$.

       @param rhs \f$ g \f$ in the formula above
       @param sol_sigma \f$ \sigma \f$ in the formula above
       @param sol_u \f$ u \f$ in the formula above
    */
    void Mult(const mfem::Vector& rhs, mfem::Vector& sol_sigma, mfem::Vector& sol_u);
private:
    void Init(double* M_data, const mfem::SparseMatrix& D);

    std::unique_ptr<mfem::UMFPackSolver> solver_;
    std::unique_ptr<mfem::SparseMatrix> A_;
    std::unique_ptr<mfem::SparseMatrix> MinvDT_;
};

/**
   @brief Compute the weighted l2 inner product between u and v
*/
double InnerProduct(const mfem::Vector& weight, const mfem::Vector& u,
                    const mfem::Vector& v);

/**
   @brief Compute the usual l2 inner product between u and v
*/
double InnerProduct(const mfem::Vector& u, const mfem::Vector& v);

} // namespace smoothg

#endif /* __MATRIXUTILITIES_HPP__ */
