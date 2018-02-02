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
   @file

   @brief Implements SparseEigensolver object
*/

#include "arpackeig.hpp"
#if SMOOTHG_USE_ARPACK
#include <cassert>

namespace smoothg
{

class ARPACK_mfem_adapter
{
public:
    ARPACK_mfem_adapter(const mfem::SparseMatrix& mat);
    ~ARPACK_mfem_adapter();

    void MultOP(double* in, double* out);

private:
    const mfem::SparseMatrix& mat_;
    mfem::UMFPackSolver mat_inv;
    int size_;
};

ARPACK_mfem_adapter::ARPACK_mfem_adapter(const mfem::SparseMatrix& mat) :
    mat_(mat),
    size_(mat.Size())
{
    mat_inv.SetOperator(mat_);
}

ARPACK_mfem_adapter::~ARPACK_mfem_adapter()
{
}

void ARPACK_mfem_adapter::MultOP(double* in, double* out)
{
    mfem::Vector m_in(in, size_);
    mfem::Vector m_out(out, size_);

    mat_inv.Mult(m_in, m_out);
}

SparseEigensolver::SparseEigensolver() :
    num_arnoldi_vectors_(-1),
    tolerance_(1.e-10),
    max_iterations_(1000)
{
}

int SparseEigensolver::Compute(mfem::SparseMatrix& A, mfem::Vector& evals,
                               mfem::DenseMatrix& evects, int num_evects)
{
    int size = A.Size();
    num_evects = std::min(size, num_evects);
    int ncv;
    if (num_arnoldi_vectors_ < 0)
        ncv = 2 * num_evects + 10;
    else
        ncv = num_arnoldi_vectors_;
    ncv = std::min(size, ncv);

    ARPACK_mfem_adapter adapter(A);
    evals.SetSize(num_evects);
    evects.SetSize(size, num_evects);
    ARSymStdEig<double, ARPACK_mfem_adapter>
    eigprob(A.Height(), num_evects, &adapter, &ARPACK_mfem_adapter::MultOP,
            "LM", ncv, tolerance_, max_iterations_);

    double* vec_data = evects.Data();
    double* val_data = evals.GetData();
    num_converged_ = eigprob.EigenValVectors(vec_data, val_data);

    return num_evects - num_converged_; // ???
}

} // namespace smoothg

#endif
