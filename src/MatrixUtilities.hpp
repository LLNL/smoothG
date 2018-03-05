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
#include <iomanip>

namespace smoothg
{

/**
    @brief Prints dense matrices
*/
void Print(const mfem::DenseMatrix& mat, const std::string& label = "",
           std::ostream& out = std::cout);
void Print(const mfem::SparseMatrix& mat, const std::string& label = "",
           std::ostream& out = std::cout);

/**
    @brief Remove small entries from a matrix
*/
mfem::SparseMatrix Threshold(const mfem::SparseMatrix& mat, double tol = 1e-8);

/**
    @brief Creates a sparse matrix from a table
*/
mfem::SparseMatrix TableToMatrix(const mfem::Table& table);

/**
    @brief Creates a table from a sparse matrix's graph
*/
mfem::Table MatrixToTable(const mfem::SparseMatrix& mat);

// Rap by hand that seems to be faster than the mfem rap but uses more memory
// Use mfem::RAP if memory is more important than cycles
mfem::HypreParMatrix* RAP(const mfem::HypreParMatrix& R, const mfem::HypreParMatrix& A,
                          const mfem::HypreParMatrix& P);
mfem::HypreParMatrix* RAP(const mfem::HypreParMatrix& A, const mfem::HypreParMatrix& P);

/**
    @brief Broadcast a SparseMatrix on processor 0 to all other processors
*/
void BroadCast(MPI_Comm comm, mfem::SparseMatrix& mat);

/**
    @brief Compute transpose of a matrix
*/
mfem::SparseMatrix Transpose(const mfem::SparseMatrix& A);

/**
    @brief Multiply two sparse matrices C = A * B
*/
mfem::SparseMatrix Mult(const mfem::SparseMatrix& A, const mfem::SparseMatrix& B);

/**
    @brief Compute \f$ C = AB \f$, where \f$ A \f$ is sparse and
           \f$ B \f$ is dense.
*/
void MultSparseDense(const mfem::SparseMatrix& A, mfem::DenseMatrix& B,
                     mfem::DenseMatrix& C);

/**
    @brief Compute \f$ C = AB \f$, where \f$ A \f$ is sparse and
           \f$ B \f$ is dense, but C is kept transposed.
*/
void MultSparseDenseTranspose(const mfem::SparseMatrix& A, mfem::DenseMatrix& B,
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
void SetConstantValue(mfem::HypreParMatrix& pmat, double c);

/**
    @brief Construct the relation table aggregate to vertex from partition
*/
mfem::SparseMatrix PartitionToMatrix(const mfem::Array<int>& partition, int nparts);

/**
   @brief Construct an identity matrix (as a SparseMatrix) of size 'size'
*/
mfem::SparseMatrix SparseIdentity(int size);

/**
   @brief Construct an rectangular identity matrix (as a SparseMatrix)
   @param rows number of row
   @param cols number of columns
   @param row_offset offset row where diagonal identity starts
   @param col_offset offset column where diagonal identity starts
*/
mfem::SparseMatrix SparseIdentity(int rows, int cols, int row_offset = 0, int col_offset = 0);

/**
   @brief mat = a * mat + b * diag(vec) or diag(vec^{-1}) if invert_vec = true

   mat must have nonzeros on the diagonal
*/
void Add(const double a, mfem::SparseMatrix& mat, const double b,
         const mfem::Vector& vec, const bool invert_vec = false);

/**
   @brief mat = mat + diag(vec) or diag(vec^{-1}) if invert_vec = true

   mat must have nonzeros on the diagonal
*/
void Add(mfem::SparseMatrix& mat, const mfem::Vector& vec,
         const bool invert_vec = false);

/**
   @brief Compute A^t * diag(D) * A
*/
mfem::SparseMatrix Mult_AtDA(const mfem::SparseMatrix& A, const mfem::Vector& D);

/**
   @brief Construct a diagonal matrix with the entries specified by a vector
   @param vect diagonal entries
*/
mfem::SparseMatrix VectorToMatrix(const mfem::Vector& vect);

/**
   @brief Add two parallel matrices C = A + B
   @param A left hand side matrix
   @param B right hand side matrix
   @note can be removed with MFEM version > 3.3.2
*/
mfem::HypreParMatrix* ParAdd(const mfem::HypreParMatrix& A, const mfem::HypreParMatrix& B);

/**
   @brief Compute max norm of parallel matrix
   @param A Parallel Matrix
*/
double MaxNorm(const mfem::HypreParMatrix& A);

/**
   @brief Extract a submatrix from a matrix

   @param A the matrix to extract from
   @param rows the rows to extract
   @param cols the columns to extract
   @param colMapper basically a data workspace

   @returns the extracted submatrix
*/
mfem::SparseMatrix ExtractRowAndColumns(
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
   @brief Extract columns from a dense matrix (A) to another dense matrix (A_sub)

   @param A the matrix to extract from
   @param ref_to_col mapping from reference index to column index of A
   @param subcol_to_ref mapping from column index of A_sub to reference index
   @param A_sub the returned matrix where the extracted columns are collected
   @param row_offset which row of A_sub to start putting the extracted columns
*/
void ExtractColumns(
    const mfem::DenseMatrix& A, const mfem::Array<int>& ref_to_col,
    const mfem::Array<int>& subcol_to_ref, mfem::DenseMatrix& A_sub,
    const int row_offset = 0);

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
   @brief Generate the "start" array for HypreParMatrix based on the number of
   local true dofs
   Single case
*/
void GenerateOffsets(MPI_Comm comm, int local_size, mfem::Array<HYPRE_Int>& offsets);

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
    mfem::SparseMatrix A_;
    mfem::SparseMatrix MinvDT_;
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

/**
   @brief Construct entity to true entity table from entity_trueentity_entity

   Pick one of the processors sharing a true entity to own the true entity
   (pick the processor with a smaller id)

   @param entity_trueentity_entity = entity_trueentity * trueentity_entity
*/
std::unique_ptr<mfem::HypreParMatrix> BuildEntityToTrueEntity(
    const mfem::HypreParMatrix& entity_trueentity_entity);

} // namespace smoothg

#endif /* __MATRIXUTILITIES_HPP__ */
