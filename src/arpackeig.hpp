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

/**
   @file arpackeig.hpp

   @brief Interface with ARPACK via arpackpp, to find eigenvalues of a
   symmetric hypre_CSRMatrix object.
*/

#ifndef __ARPACKEIG_HPP
#define __ARPACKEIG_HPP

#include "smoothG_config.h"
#if SMOOTHG_USE_ARPACK

// arpackpp include
#define ARPACK_SILENT_MODE
#include "arssym.h"

#include "mfem.hpp"

namespace smoothg
{

/// Interface to be similar to Eigensolver in utilities.hpp
class SparseEigensolver
{
public:
    SparseEigensolver();
    /**
       Given a (sparse) matrix \f$ A \f$, find the eigenvectors
       corresponding to the smallest few eigenvalues.

       Uses ARPACK for the implementation.

       @param A (in) the matrix
       @param evals (out) eigenvalues
       @param evects (out) eigenvectors
       @param numEvals (in) number of eigenvalues/vectors to compute
    */
    int Compute(
        mfem::SparseMatrix& A, mfem::Vector& evals,
        mfem::DenseMatrix& evects, int numEvals);
    ~SparseEigensolver() = default;
private:
    int num_arnoldi_vectors_;
    double tolerance_;
    int max_iterations_;
    int num_converged_;
};

} // namespace smoothg

#endif // SMOOTHG_USE_ARPACK

#endif // __ARPACKEIG_HPP
