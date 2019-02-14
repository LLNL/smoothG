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

    @brief Contains abstract base class MixedLaplacianSolver
*/

#ifndef __MIXEDLAPLACIANSOLVER_HPP__
#define __MIXEDLAPLACIANSOLVER_HPP__

#include "mfem.hpp"
#include "utilities.hpp"

namespace smoothg
{

/**
   @brief Abstract base class for solvers of graph Laplacian problems
*/
class MixedLaplacianSolver : public mfem::Operator
{
public:
    MixedLaplacianSolver(MPI_Comm comm, const mfem::Array<int>& block_offsets,
                         bool W_is_nonzero);
    MixedLaplacianSolver() = delete;

    virtual ~MixedLaplacianSolver() = default;

    /**
       Solve the mixed form of the graph Laplacian problem

       The BlockVectors here are in "dof" numbering, rather than "truedof" numbering.
       That is, dofs on processor boundaries are *repeated* in the vectors that
       come into and go out of this method.
    */
    void Solve(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const;
    virtual void Mult(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const = 0;

    /// Solve the primal form of the graph Laplacian problem (DM^{-1}D^T) sol = rhs
    void Solve(const mfem::Vector& rhs, mfem::Vector& sol) const;
    virtual void Mult(const mfem::Vector& rhs, mfem::Vector& sol) const;

    /// Update solver based on new "element" scaling for M matrix
    virtual void UpdateElemScaling(const mfem::Vector& elem_scaling_inverse) = 0;

    ///@name Set solver parameters
    ///@{
    virtual void SetPrintLevel(int print_level) { print_level_ = print_level; }
    virtual void SetMaxIter(int max_num_iter) { max_num_iter_ = max_num_iter; }
    virtual void SetRelTol(double rtol) { rtol_ = rtol; }
    virtual void SetAbsTol(double atol) { atol_ = atol; }
    ///@}

    ///@name Get results of iterative solve
    ///@{
    virtual int GetNumIterations() const { return num_iterations_; }
    virtual int GetNNZ() const { return nnz_; }
    virtual double GetTiming() const { return timing_; }
    ///@}

protected:
    void Init(const MixedMatrix& mgL, const mfem::Array<int>* ess_attr);
    void Orthogonalize(mfem::Vector& vec) const;

    MPI_Comm comm_;
    int myid_;

    mutable mfem::BlockVector rhs_;
    mutable mfem::BlockVector sol_;

    mfem::Vector elem_scaling_;

    // default linear solver options
    int print_level_ = 0;
    int max_num_iter_ = 5000;
    double rtol_ = 1e-9;
    double atol_ = 1e-12;

    int nnz_;
    mutable int num_iterations_;
    mutable double timing_;

    bool remove_one_dof_;
    bool W_is_nonzero_;

    mfem::Array<int> ess_edofs_;
    const mfem::Vector* const_rep_;
};

class PrimalSolver : public MixedLaplacianSolver
{
public:
    /**
       @brief Constructor from individual M and D matrices.

       @param mgL containing information of mixed system
       @param ess_attr marker for essential boundary attributes
    */
    PrimalSolver(const MixedMatrix& mgL,
                 const mfem::Array<int>* ess_attr = nullptr);

    /**
       @brief Use CG preconditioned by BoomerAMG to solve the primal problem.
    */
    virtual void Mult(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const;

    virtual void UpdateElemScaling(const mfem::Vector& elem_scaling_inverse);

    ///@name Set solver parameters
    ///@{
    virtual void SetPrintLevel(int print_level) override;
    virtual void SetMaxIter(int max_num_iter) override;
    virtual void SetRelTol(double rtol) override;
    virtual void SetAbsTol(double atol) override;
    ///@}

protected:
    void Init(mfem::SparseMatrix M_proc);

    mfem::CGSolver cg_;

    mfem::Vector M_diag_;
    std::unique_ptr<mfem::HypreParMatrix> D_;
    std::unique_ptr<mfem::HypreParMatrix> Dt_;
    std::unique_ptr<mfem::SparseMatrix> W_;

    std::unique_ptr<mfem::HypreParMatrix> operator_;
    std::unique_ptr<mfem::HypreBoomerAMG> prec_;

    const MixedMatrix& mixed_matrix_;
};

} // namespace smoothg

#endif /* __MIXEDLAPLACIANSOLVER_HPP__ */
