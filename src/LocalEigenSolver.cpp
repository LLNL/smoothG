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

   @brief Implements LocalEigensolver object
*/

#include "LocalEigenSolver.hpp"
#include "MatrixUtilities.hpp"
#include "utilities.hpp"
#include <cassert>

#if SMOOTHG_USE_ARPACK
// arpackpp include
#define ARPACK_SILENT_MODE
#include "arssym.h"
#include "argsym.h"
#endif

extern "C"
{
    void dsytrd_(char* uplo, const int* n, double* a, const int* lda,
                 double* d, double* e, double* tau, double* work,
                 int* lwork, int* info );

    void dsterf_(const int* n, double* d, double* e, int* info);

    void dstein_(const int* n, const double* d, const double* e,
                 int* m, const double* w, const int* iblock,
                 const int* isplit, double* z, const int* ldz,
                 double* work, int* iwork, int* ifailv,
                 int* info);

    void dormtr_(char* side, char* uplo, char* trans, const int* m,
                 const int* n, const double* a, const int* lda,
                 const double* tau, double* c, const int* ldc, double* work,
                 int* lwork, int* info);

    void dpotrf_(char* uplo, const int* n, double* a, const int* lda, int* info);

    void dsygst_(const int* itype, char* uplo, const int* n, double* a,
                 const int* lda, const double* b, const int* ldb, int* info);

    void dtrtrs_(char* uplo, char* trans, char* diag, const int* n,
                 const int* nrhs, double* a, const int* lda, double* b,
                 const int* ldb, int* info);
}

namespace smoothg
{

LocalEigenSolver::LocalEigenSolver(
    int max_num_evects, double rel_tol, int size_offset)
    :
    max_num_evects_(max_num_evects),
    rel_tol_(rel_tol),
    size_offset_(size_offset),
    eig_max_ptr_(&eig_max_),
    uplo_('U'),
    side_('L'),
    trans_('N'),
    abstol_(2 * std::numeric_limits<double>::min()),
    info_(0),
    n_max_(-1),
    lwork_(-1),
    itype_(1),
    diag_('N'),
    num_arnoldi_vectors_(-1),
    tolerance_(1e-16),
    max_iterations_(10000),
    shift_(-1e-1)  // shift_ may need to be adjusted
{
}

void LocalEigenSolver::AllocateWorkspace(int n, bool is_gev)
{
    A_.resize(n * n, 0.0);
    if (is_gev)
    {
        B_.resize(n * n, 0.0);
    }
    n_max_ = n;

    int lwork = -1;
    double wkopt;

    // find max workspace between dsytrd_, dstein_ and dormtr_
    dsytrd_(&uplo_, &n, nullptr, &n,
            d_.data(), e_.data(), tau_.data(), &wkopt,
            &lwork, &info_ );

    // 5n is for dstein_
    lwork_ = std::max(5 * n, (int)wkopt);

    dormtr_( &side_, &uplo_, &trans_, &n,
             &n, nullptr, &n,
             tau_.data(), nullptr, &n, &wkopt,
             &lwork, &info_ );

    lwork_ = std::max(lwork_, (int)wkopt);

    work_.resize(lwork_);
    iwork_.resize(n);

    d_.resize(n);
    e_.resize(n); // could be n - 1?
    tau_.resize(n);

    iblock_.resize(n, 1);
    isplit_.resize(n, 0);
    iwork_.resize(n);
}

std::vector<double*> LocalEigenSolver::EigenPairsSetSizeAndData(
    int size, int num_evects, mfem::Vector& evals, mfem::DenseMatrix& evects)
{
    evals.SetSize(num_evects);
    evects.SetSize(size, num_evects);
    std::vector<double*> out(2);
    out[0] = evects.Data();
    out[1] = evals.GetData();
    return out;
}

int LocalEigenSolver::FindNumberOfEigenPairs(
    const mfem::Vector& evals, int max_num_evects, double eig_max)
{
    if (rel_tol_ >= 1.0)
    {
        return max_num_evects;
    }

    const double tol = eig_max * rel_tol_;
    int m = 1;
    while (m < max_num_evects && evals(m) < tol)
    {
        ++m;
    }
    return m;
}

int LocalEigenSolver::Compute(
    int n, double* a, mfem::Vector& evals, mfem::DenseMatrix& evects)
{
    evals.SetSize(n);
    assert(n <= (int) d_.size());

    // Triangularize A = Q * T * Q^T
    dsytrd_(&uplo_, &n, a, &n, d_.data(), e_.data(), tau_.data(), work_.data(),
            &lwork_, &info_ );

    // d_ and e_ changed by dsterf_
    // copy since they are needed for dstein_
    // std::copy(std::begin(d_), std::end(d_), evals.GetData());
    std::copy(std::begin(d_), std::begin(d_) + n, evals.GetData());
    auto e_copy = e_;

    // Compute all eigenvalues
    dsterf_(&n, evals.GetData(), e_copy.data(), &info_);

    // Determine how many eigenvectors to be computed
    int max_num_evects = max_num_evects_ == -1 ? n : std::min(n, max_num_evects_);
    int m = FindNumberOfEigenPairs(evals, max_num_evects, evals(n - 1));

    evects.SetSize(n, m);

    ifail_.resize(m);
    isplit_[0] = n;

    // Calculate Eigenvectors of T
    dstein_(&n, d_.data(), e_.data(),
            &m, evals.GetData(), iblock_.data(),
            isplit_.data(), evects.Data(), &n,
            work_.data(), iwork_.data(), ifail_.data(),
            &info_);

    // Compute Q * (eigenvectors of T)
    dormtr_(&side_, &uplo_, &trans_, &n,
            &m, a, &n,
            tau_.data(), evects.Data(), &n, work_.data(),
            &lwork_, &info_);

    return m;
}

template<>
void LocalEigenSolver::Compute(
   const  mfem::DenseMatrix& A, mfem::Vector& evals, mfem::DenseMatrix& evects)
{
    const int n = A.Size();

    if (n < 1)
    {
        EigenPairsSetSizeAndData(n, 0, evals, evects);
        return;
    }

    if (n_max_ < n) { AllocateWorkspace(n, false); }

    std::copy(A.Data(), A.Data() + n * n, begin(A_));
    Compute(n, A_.data(), evals, evects);
}

template<>
void LocalEigenSolver::Compute(
    const mfem::DenseMatrix& A, const mfem::DenseMatrix& B,
    mfem::Vector& evals, mfem::DenseMatrix& evects)
{
    const int n = A.Size();
    if (n < 1)
    {
        EigenPairsSetSizeAndData(n, 0, evals, evects);
        return;
    }

    if (n_max_ < n) { AllocateWorkspace(n, true); }

    // Compute Cholesky factorization of B = U^T * U
    std::copy(B.Data(), B.Data() + n * n, begin(B_));
    double* b = B_.data();
    dpotrf_(&uplo_, &n, b, &n, &info_);

    std::copy(A.Data(), A.Data() + n * n, begin(A_));
    double* a = A_.data();

    // Reduce Ax = \lambda Bx to Cy = \lambda y, C = U^{-T}AU^{-1}, y = Ux
    dsygst_(&itype_, &uplo_, &n, a, &n, b, &n, &info_);

    // Solve Cy = \lambda y
    int m = Compute(n, a, evals, evects);

    // Compute U^{-1} * (eigenvectors of C)
    dtrtrs_(&uplo_, &trans_, &diag_, &n, &m, b, &n, evects.Data(), &n, &info_);
}

template<>
void LocalEigenSolver::BlockCompute(
    const mfem::DenseMatrix& M, const mfem::DenseMatrix& D,
    mfem::Vector& evals, mfem::DenseMatrix& evects)
{
    mfem::DenseMatrixInverse Minv(M);
    mfem::DenseMatrix Dt(D, 't');
    mfem::DenseMatrix MinvDt = smoothg::Mult(Minv, Dt);
    mfem::DenseMatrix DMinvDt = smoothg::Mult(D, MinvDt);

    Compute(DMinvDt, evals, evects);
}

#if SMOOTHG_USE_ARPACK
/// Adapter for applying the action of a certain operator in ARPACK
class ARPACK_operator_adapter
{
public:
    ARPACK_operator_adapter(const int size)
        :
        size_(size)
    {
    }

    virtual void MultOP(double* in, double* out) = 0;
protected:
    const int size_;
};

// Evaluate y = A x
class A_adapter : public ARPACK_operator_adapter
{
public:
    A_adapter(const mfem::SparseMatrix& A)
        :
        ARPACK_operator_adapter(A.Height()),
        A_(A)
    { }

    void MultOP(double* in, double* out) override
    {
        mfem::Vector v_in(in, size_);
        mfem::Vector v_out(out, size_);
        A_.Mult(v_in, v_out);
    }

private:
    const mfem::SparseMatrix& A_;
};

// Evaluate y = (A-shift*B)^{-1} x
class A_B_shift_adapter : public ARPACK_operator_adapter
{
public:
    A_B_shift_adapter(const mfem::SparseMatrix& A,
                      const mfem::SparseMatrix& B,
                      const double shift)
        :
        ARPACK_operator_adapter(A.Size())
    {
        A_minus_sigma_B_.reset(mfem::Add(1.0, A, -shift, B));
        A_minus_sigma_B_inv_.SetOperator(*A_minus_sigma_B_);
    }

    void MultOP(double* in, double* out) override
    {
        mfem::Vector v_in(in, size_);
        mfem::Vector v_out(out, size_);
        A_minus_sigma_B_inv_.Mult(v_in, v_out);
    }

private:
    std::unique_ptr<mfem::SparseMatrix> A_minus_sigma_B_;
    mfem::UMFPackSolver A_minus_sigma_B_inv_;
};

// Evaluate y = ((DM^{-1}D^T)-shift*B)^{-1} x
// TODO: this may use some modified version of LocalEdgeSolver
class DMD_shift_adapter : public ARPACK_operator_adapter
{
public:
    DMD_shift_adapter(const mfem::SparseMatrix& M,
                      const mfem::SparseMatrix& D,
                      const double shift)
        :
        ARPACK_operator_adapter(D.Height())
    {
        offsets_.SetSize(3);
        offsets_[0] = 0;
        offsets_[1] = M.Height();
        offsets_[2] = M.Height() + D.Height();

        mfem::SparseMatrix DT = smoothg::Transpose(D);
        mfem::SparseMatrix W = SparseIdentity(D.Height());
        W *= shift;

        mfem::BlockMatrix block_A(offsets_);
        block_A.SetBlock(0, 0, const_cast<mfem::SparseMatrix*>(&M));
        block_A.SetBlock(1, 0, const_cast<mfem::SparseMatrix*>(&D));
        block_A.SetBlock(0, 1, &DT);
        block_A.SetBlock(1, 1, &W);

        block_system_.reset(block_A.CreateMonolithic());
        block_system_inv_.SetOperator(*block_system_);

        rhs_ = make_unique<mfem::BlockVector>(offsets_);
        sol_ = make_unique<mfem::BlockVector>(offsets_);
        rhs_->GetBlock(0) = 0.0;
    }

    void MultOP(double* in, double* out) override
    {
        std::copy_n(in, size_, rhs_->GetBlock(1).GetData());
        rhs_->GetBlock(1) *= -1.0;

        block_system_inv_.Mult(*rhs_, *sol_);

        std::copy_n(sol_->GetBlock(1).GetData(), size_, out);
    }

private:
    mfem::UMFPackSolver block_system_inv_;
    std::unique_ptr<mfem::SparseMatrix> block_system_;
    mfem::Array<int> offsets_;
    mutable std::unique_ptr<mfem::BlockVector> rhs_;
    mutable std::unique_ptr<mfem::BlockVector> sol_;
};

// Evaluate y = (B^{-1}A) x, x = A x
class B_inv_A_adapter : public ARPACK_operator_adapter
{
public:
    B_inv_A_adapter(const mfem::SparseMatrix& A,
                    const mfem::SparseMatrix& B)
        :
        ARPACK_operator_adapter(A.Height()),
        A_(A),
        B_inv_(const_cast<mfem::SparseMatrix&>(B))
    { }

    void MultOP(double* in, double* out) override
    {
        mfem::Vector v_in(in, size_);
        mfem::Vector v_help(size_);
        mfem::Vector v_out(out, size_);

        A_.Mult(v_in, v_help);
        B_inv_.Mult(v_help, v_out);

        v_in = v_help;
    }

private:
    const mfem::SparseMatrix& A_;
    mfem::UMFPackSolver B_inv_;
};

/*! @brief Derived class to avoid uninitialized value Valgrind error */
template<class ARFLOAT, class ARFOP>
class ARSymStdEig_: public virtual ARSymStdEig<ARFLOAT, ARFOP>
{
public:
    using ARSymStdEig<ARFLOAT, ARFOP>::HowMny;

    ARSymStdEig_(int np, int nevp, ARFOP* objOPp,
                 void (ARFOP::* MultOPxp)(ARFLOAT[], ARFLOAT[]),
                 const std::string& whichp = "LM", int ncvp = 0, ARFLOAT tolp = 0.0,
                 int maxitp = 0, ARFLOAT* residp = NULL, bool ishiftp = true)
        : ARSymStdEig<ARFLOAT, ARFOP>(np, nevp, objOPp, MultOPxp,
                                      whichp, ncvp, tolp, maxitp, residp, ishiftp)
    {
        HowMny = 'A';
    }

    virtual ~ARSymStdEig_() { }

}; // class ARSymStdEig_.

/*! @brief Derived class to avoid uninitialized value Valgrind error */
template<class ARFLOAT, class ARFOP, class ARFB>
class ARSymGenEig_: public virtual ARSymGenEig<ARFLOAT, ARFOP, ARFB>
{
public:
    using ARSymGenEig<ARFLOAT, ARFOP, ARFB>::HowMny;

    ARSymGenEig_(int np, int nevp, ARFOP* objOPp,
                 void (ARFOP::* MultOPxp)(ARFLOAT[], ARFLOAT[]), ARFB* objBp,
                 void (ARFB::* MultBxp)(ARFLOAT[], ARFLOAT[]),
                 const std::string& whichp = "LM", int ncvp = 0, ARFLOAT tolp = 0.0,
                 int maxitp = 0, ARFLOAT* residp = NULL, bool ishiftp = true)
        : ARSymGenEig<ARFLOAT, ARFOP, ARFB>(np, nevp, objOPp, MultOPxp, objBp, MultBxp,
                                            whichp, ncvp, tolp, maxitp, residp, ishiftp)
    {
        HowMny = 'A';
    }

    virtual ~ARSymGenEig_() { }

}; // class ARSymGenEig_.

// ncv is the number of Arnoldi vectors generated at each iteration of ARPACK
int ComputeNCV(int size, int num_evects, int num_arnoldi_vectors)
{
    int ncv;
    if (num_arnoldi_vectors < 0)
        ncv = 2 * num_evects + 20;
    else
        ncv = num_arnoldi_vectors;
    return std::min(size, ncv);
}

void CheckNotConverged(int num_evects, int num_converged)
{
    int num_not_converged = num_evects - num_converged;
    if (num_not_converged)
    {
        std::cout << "Sparse eigen solver warning: " << num_not_converged
                  << " eigenvectors did not converge!\n";
    }
}

template<>
void LocalEigenSolver::Compute(
    const mfem::SparseMatrix& A, mfem::Vector& evals, mfem::DenseMatrix& evects)
{
    int n = A.Size();
    int max_num_evects = max_num_evects_ == -1 ? n : std::min(n, max_num_evects_);
    int ncv = ComputeNCV(n, max_num_evects, num_arnoldi_vectors_);

    // Find the eigenvectors associated with num_evects smallest eigenvalues
    auto identity = SparseIdentity(A.Height());
    A_B_shift_adapter adapter_A_I_shift(A, identity, shift_);

    ARSymStdEig<double, A_B_shift_adapter>
    eigprob(n, max_num_evects, &adapter_A_I_shift, &A_B_shift_adapter::MultOP,
            shift_, "LM", ncv, tolerance_, max_iterations_);
    auto data_ptr = EigenPairsSetSizeAndData(n, max_num_evects, evals, evects);
    int num_converged = eigprob.EigenValVectors(data_ptr[0], data_ptr[1]);
    CheckNotConverged(max_num_evects, num_converged);

    if (rel_tol_ < 1.0)
    {
        // Find the largest eigenvalue
        A_adapter adapter_A(A);
        ARSymStdEig_<double, A_adapter>
        eigvalueprob(n, 1, &adapter_A, &A_adapter::MultOP,
                     "LM", ncv, tolerance_, max_iterations_);
        eigvalueprob.Eigenvalues(eig_max_ptr_);

        int num_evects = FindNumberOfEigenPairs(evals, max_num_evects, eig_max_);
        EigenPairsSetSizeAndData(n, num_evects, evals, evects);
    }
}

template<>
void LocalEigenSolver::Compute(
    const mfem::SparseMatrix& A, const mfem::SparseMatrix& B,
    mfem::Vector& evals, mfem::DenseMatrix& evects)
{
    int n = A.Height();
    int max_num_evects = max_num_evects_ == -1 ? n : std::min(n, max_num_evects_);
    int ncv = ComputeNCV(n, max_num_evects, num_arnoldi_vectors_);

    A_B_shift_adapter adapter_A_B_shift(A, B, shift_);
    A_adapter adapter_B(B);

    ARSymGenEig<double, A_B_shift_adapter, A_adapter>
    eigprob('S', n, max_num_evects, &adapter_A_B_shift,
            &A_B_shift_adapter::MultOP, &adapter_B, &A_adapter::MultOP,
            shift_, "LM", ncv, tolerance_, max_iterations_);
    auto data_ptr = EigenPairsSetSizeAndData(n, max_num_evects, evals, evects);
    int num_converged = eigprob.EigenValVectors(data_ptr[0], data_ptr[1]);
    CheckNotConverged(max_num_evects, num_converged);

    if (rel_tol_ < 1.0)
    {
        // TODO: more efficient way to find eig_max of generalized eigen problem
        // Find the largest eigenvalue
        B_inv_A_adapter adapter_B_inv_A(A, B);
        ARSymGenEig_<double, B_inv_A_adapter, A_adapter>
        eigvalueprob(A.Height(), 1,
                     &adapter_B_inv_A, &B_inv_A_adapter::MultOP,
                     &adapter_B, &A_adapter::MultOP,
                     "LM", ncv, tolerance_, max_iterations_);
        eigvalueprob.Eigenvalues(eig_max_ptr_);

        int num_evects = FindNumberOfEigenPairs(evals, max_num_evects, eig_max_);
        EigenPairsSetSizeAndData(n, num_evects, evals, evects);
    }
}

template<>
void LocalEigenSolver::BlockCompute(
    const mfem::SparseMatrix& M, const mfem::SparseMatrix& D,
    mfem::Vector& evals, mfem::DenseMatrix& evects)
{
    int n = D.Height();
    int max_num_evects = max_num_evects_ == -1 ? n : std::min(n, max_num_evects_);
    int ncv = ComputeNCV(n, max_num_evects, num_arnoldi_vectors_);

    // Find the eigenvectors associated with num_evects smallest eigenvalues
    DMD_shift_adapter DMD_I_shift(M, D, shift_);

    ARSymStdEig<double, DMD_shift_adapter>
    eigprob(n, max_num_evects, &DMD_I_shift, &DMD_shift_adapter::MultOP,
            shift_, "LM", ncv, tolerance_, max_iterations_);
    auto data_ptr = EigenPairsSetSizeAndData(n, max_num_evects, evals, evects);
    int num_converged = eigprob.EigenValVectors(data_ptr[0], data_ptr[1]);
    CheckNotConverged(max_num_evects, num_converged);

    // TODO: a fast way of computing largest eigenvalue?
    //    if (rel_tol_ < 1.0)
    //    {
    //        // Find the largest eigenvalue
    //        A_adapter adapter_A(A);
    //        ARSymStdEig_<double, A_adapter>
    //        eigvalueprob(n, 1, &adapter_A, &A_adapter::MultOP,
    //                     "LM", ncv, tolerance_, max_iterations_);
    //        eigvalueprob.Eigenvalues(eig_max_ptr_);

    //        int num_evects = FindNumberOfEigenPairs(evals, max_num_evects, eig_max_);
    //        EigenPairsSetSizeAndData(n, num_evects, evals, evects);
    //    }
}
#endif // SMOOTHG_USE_ARPACK

double LocalEigenSolver::Compute(const mfem::SparseMatrix& A, mfem::DenseMatrix& evects)
{
#if SMOOTHG_USE_ARPACK
    if (A.Size() > size_offset_)
    {
        Compute(A, evals_, evects);
    }
    else
#endif
    {
        Full(A, dense_A_);
        Compute(dense_A_, evals_, evects);
    }
    return evals_(0);
}

double LocalEigenSolver::Compute(
   const mfem::SparseMatrix& A, const mfem::SparseMatrix& B, mfem::DenseMatrix& evects)
{
#if SMOOTHG_USE_ARPACK
    if (A.Size() > size_offset_)
    {
        Compute(A, B, evals_, evects);
    }
    else
#endif
    {
        Full(A, dense_A_);
        Full(B, dense_B_);
        Compute(dense_A_, dense_B_, evals_, evects);
    }
    return evals_(0);
}

double LocalEigenSolver::Compute(
    const std::vector<mfem::SparseMatrix>& mat, mfem::DenseMatrix& evects)
{
    assert(mat.size() == 1 || mat.size() == 2);
    if (mat.size() == 1)
    {
        return Compute(mat[0], evects);
    }
    else
    {
        return Compute(mat[0], mat[1], evects);
    }
}

double LocalEigenSolver::BlockCompute(
    const mfem::SparseMatrix& M, const mfem::SparseMatrix& D, mfem::DenseMatrix& evects)
{
#if SMOOTHG_USE_ARPACK
    if (D.NumRows() > size_offset_)
    {
        BlockCompute(M, D, evals_, evects);
    }
    else
#endif
    {
        Full(M, dense_A_);
        Full(D, dense_B_);
        BlockCompute(dense_A_, dense_B_, evals_, evects);
    }
    return evals_(0);
}

} // namespace smoothg

