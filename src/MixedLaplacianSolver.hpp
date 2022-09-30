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

#include "utilities.hpp"

namespace smoothg
{

enum class KrylovMethod { CG, MINRES, GMRES };

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
    mfem::BlockVector Solve(const mfem::BlockVector& rhs) const;
    virtual void Mult(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const = 0;

    /// Solve the primal form of the graph Laplacian problem (DM^{-1}D^T) sol = rhs
    void Solve(const mfem::Vector& rhs, mfem::Vector& sol) const;
    virtual void Mult(const mfem::Vector& rhs, mfem::Vector& sol) const;

    /// Update solver based on new "element" scaling for M matrix
    virtual void UpdateElemScaling(const mfem::Vector& elem_scaling_inverse) = 0;

    /// Update the Jacobian associated with the nonlinear graph Laplacian problem
    virtual void UpdateJacobian(const mfem::Vector& elem_scaling_inverse,
                                const std::vector<mfem::DenseMatrix>& N_el) = 0;

    ///@name Set solver parameters
    ///@{
    void SetPrintLevel(int l) { print_level_ = l; solver_->SetPrintLevel(l); }
    void SetMaxIter(int it) { max_num_iter_ = it; solver_->SetMaxIter(it); }
    void SetRelTol(double rtol) { rtol_ = rtol; solver_->SetRelTol(rtol); }
    void SetAbsTol(double atol) { atol_ = atol; solver_->SetAbsTol(atol); }
    ///@}

    ///@name Get results of iterative solve
    ///@{
    int GetNumIterations() const { return num_iterations_; }
    int GetNNZ() const { return nnz_; }
    double GetTiming() const { return timing_; }
    ///@}

protected:
    void Init(const MixedMatrix& mgL, const mfem::Array<int>* ess_attr);
    void Orthogonalize(mfem::Vector& vec) const;
    std::unique_ptr<mfem::IterativeSolver> InitKrylovSolver(KrylovMethod method);
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

    std::unique_ptr<mfem::IterativeSolver> solver_;
    bool is_symmetric_;
};



/// Constructs a solver which is a combination of a given pair of solvers
/// TwoStageSolver * x = solver2 * (I - A * solver1 ) * x + solver1 * x
class TwoStageSolver : public mfem::Solver
{
protected:
    const mfem::Operator& solver1_;
    const mfem::Operator& solver2_;
    const mfem::Operator& op_;
    // additional memory for storing intermediate results
    mutable mfem::Vector tmp1;
    mutable mfem::Vector tmp2;

public:
    virtual void SetOperator(const Operator &op) override { }
    TwoStageSolver(const mfem::Operator& solver1, const mfem::Operator& solver2, const mfem::Operator& op) :
        solver1_(solver1), solver2_(solver2), op_(op),  tmp1(op.NumRows()), tmp2(op.NumRows()) { }

    void Mult(const mfem::Vector & x, mfem::Vector & y) const override
    {
        solver1_.Mult(x, y);
        op_.Mult(y, tmp1);
        tmp1 -= x;
        solver2_.Mult(tmp1, tmp2);
        y -= tmp2;
    }
};

/// Hypre ILU Preconditioner
class HypreILU : public mfem::HypreSolver
{
   HYPRE_Solver ilu_precond;
public:
   HypreILU(mfem::HypreParMatrix &A, int type = 0, int fill_level = 1)
       : HypreSolver(&A)
   {
      HYPRE_ILUCreate(&ilu_precond);
      HYPRE_ILUSetMaxIter( ilu_precond, 1 );
      HYPRE_ILUSetTol( ilu_precond, 0.0 );
      HYPRE_ILUSetType( ilu_precond, type );
      HYPRE_ILUSetLevelOfFill( ilu_precond, fill_level );
      HYPRE_ILUSetDropThreshold( ilu_precond, 1e-2 );
      HYPRE_ILUSetMaxNnzPerRow( ilu_precond, 100 );
      HYPRE_ILUSetLocalReordering(ilu_precond, type == 0 ? false : true);
   }

   virtual void SetOperator(const mfem::Operator &op) { }

   virtual operator HYPRE_Solver() const { return ilu_precond; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ILUSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ILUSolve; }

   virtual ~HypreILU() { HYPRE_ILUDestroy(ilu_precond); }
};

} // namespace smoothg

#endif /* __MIXEDLAPLACIANSOLVER_HPP__ */
