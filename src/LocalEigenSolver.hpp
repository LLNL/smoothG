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

/** @file LocalEigenSolver.hpp

    @brief Wrapper for LAPACK and ARPACK (via arpackpp), solving (generalized)
    eigenproblem of symmetric matrices.

   ARPACK implementation is based on
   https://perso.univ-rennes1.fr/yvon.lafranche/mintera/index.html
*/

#ifndef __LocalEigenSolver_HPP
#define __LocalEigenSolver_HPP

#include "smoothG_config.h"

#include "mfem.hpp"

namespace smoothg
{
/**
   @brief Wrapper class for eigen solvers in LAPACK and ARPACK
*/
class LocalEigenSolver
{
public:
    /**
       Eigen solver for serial symmetric eigen systems.

       @param max_num_evects (in) maximum number of eigenpairs to be computed
       @param rel_tol (in) relative tolerance for eigenvalues, only eigenvectors
              corresponding to eigenvalues < rel_tol * eig_max will be computed,
              up to max_num_evects total.
       @param size_offset size offset for decision on dense or sparse solver
    */
    LocalEigenSolver(int max_num_evects, double rel_tol, int size_offset = 20);

    /**
       Given a symmetric matrix \f$ A \f$, find the eigenpairs
       corresponding to the smallest few eigenvalues.

       @param A (in) the matrix
       @param evals (out) eigenvalues
       @param evects (out) eigenvectors
    */
    template<typename T>
    void Compute(T& A, mfem::Vector& evals, mfem::DenseMatrix& evects);

    /**
       Given symmetric matrices \f$ A, B \f$, find the eigenpairs
       corresponding to the smallest few eigenvalues of the generalized eigen
       problem \f$ Ax = \lambda Bx \f$. B needs to be positive-definite.

       @param A (in) the matrix on the left
       @param B (in) the matrix on the right
       @param evals (out) eigenvalues
       @param evects (out) eigenvectors
    */
    template<typename T>
    void Compute(T& A, T& B, mfem::Vector& evals, mfem::DenseMatrix& evects);


    /**
       Given matrices \f$ M, D \f$, find the eigenvectors corresponding to the
       smallest few eigenvalues of the eigen problem
       \f$ (DM^{-1}D^T)x = \lambda x \f$.

       If size of A > size_offset_ and SMOOTHG_USE_ARPACK is on, ARPACK-based
       solver will be used, otherwise LAPACK-based solver is used.

       @param M (in) the matrix M
       @param D (in) the matrix D
       @param evals (out) eigenvalues
       @param evects (out) eigenvectors
    */
    template<typename T>
    void BlockCompute(T& M, T& D, mfem::Vector& evals, mfem::DenseMatrix& evects);

    /**
       Given a symmetric matrix \f$ A \f$, find the eigenvectors
       corresponding to the smallest few eigenvalues.

       If size of A > size_offset_ and SMOOTHG_USE_ARPACK is on, ARPACK-based
       solver will be used, otherwise LAPACK-based solver is used.

       @param A (in) the matrix
       @param evects (out) eigenvectors
       @return smallest eigenvalue
    */
    double Compute(mfem::SparseMatrix& A, mfem::DenseMatrix& evects);

    /**
       Given symmetric matrices \f$ A, B \f$, find the eigenvectors
       corresponding to the smallest few eigenvalues of the generalized eigen
       problem \f$ Ax = \lambda Bx \f$. B needs to be positive-definite.

       If size of A > size_offset_ and SMOOTHG_USE_ARPACK is on, ARPACK-based
       solver will be used, otherwise LAPACK-based solver is used.

       @param A (in) the matrix on the left
       @param B (in) the matrix on the right
       @param evects (out) eigenvectors
       @return smallest eigenvalue
    */
    double Compute(
        mfem::SparseMatrix& A, mfem::SparseMatrix& B, mfem::DenseMatrix& evects);

    /**
       If mat has 1 element, solve \f$ mat[0] x = \lambda x \f$
       If mat has 2 elements, solve \f$ mat[0] x = \lambda mat[1] x \f$

       If matrix size > size_offset_ and SMOOTHG_USE_ARPACK is on, ARPACK-based
       solver will be used, otherwise LAPACK-based solver is used.

       @param mat (in) contain either 1 or 2 matrices
       @param evects (out) eigenvectors
       @return smallest eigenvalue
    */
    double Compute(
        std::vector<mfem::SparseMatrix>& mat, mfem::DenseMatrix& evects);

    /**
       Given matrices \f$ M, D \f$, find the eigenvectors corresponding to the
       smallest few eigenvalues of the eigen problem
       \f$ (DM^{-1}D^T)x = \lambda x \f$.

       If size of A > size_offset_ and SMOOTHG_USE_ARPACK is on, ARPACK-based
       solver will be used, otherwise LAPACK-based solver is used.

       @param M (in) the matrix M
       @param D (in) the matrix D
       @param evects (out) eigenvectors
       @return smallest eigenvalue
    */
    double BlockCompute(
        mfem::SparseMatrix& M, mfem::SparseMatrix& D, mfem::DenseMatrix& evects);

    ~LocalEigenSolver() = default;
private:
    // Allocate workspace for LAPACK
    void AllocateWorkspace(int n, bool is_gev = false);

    std::vector<double*> EigenPairsSetSizeAndData(
        int size, int num_evects, mfem::Vector& evals, mfem::DenseMatrix& evects);

    int FindNumberOfEigenPairs(
        const mfem::Vector& evals, int max_num_evects, double eig_max);

    /**
       Calculate smallest eigenvalues and the associated eigenvectors for
       a dense symmetric matrix A (size and data given by n and a).

       Returns eigenvalues < rel_tol_ * eig_max_, up to max_num_evects_ total.
    */
    int Compute(int n, double* a, mfem::Vector& evals, mfem::DenseMatrix& evects);

    int max_num_evects_;
    double rel_tol_;
    int size_offset_;

    mfem::Vector evals_;
    mfem::DenseMatrix dense_A_;
    mfem::DenseMatrix dense_B_;
    double eig_max_;
    double* eig_max_ptr_;

    ///@name LAPACK parameters and workspace
    ///@{
    char uplo_;
    char side_;
    char trans_;
    double abstol_;

    int info_;
    int n_max_;
    int lwork_;

    int itype_;
    char diag_;

    std::vector<double> A_;
    std::vector<double> B_;
    std::vector<double> work_;
    std::vector<int> iwork_;

    // Triangularization info
    std::vector<double> d_;
    std::vector<double> e_;
    std::vector<double> tau_;

    // Block info for dstein_ / dormtr_
    std::vector<int> iblock_;
    std::vector<int> isplit_;
    std::vector<int> ifail_;
    ///@}

    ///@name ARPACK parameters
    ///@{
    const int num_arnoldi_vectors_;
    const double tolerance_;
    const int max_iterations_;
    const double shift_;
    ///@}
};

} // namespace smoothg

#endif // __LocalEigenSolver_HPP
