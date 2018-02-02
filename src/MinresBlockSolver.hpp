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
   @file MinresBlockSolver.hpp

   @brief Given a graph in mixed form, solve the resulting system with
   preconditioned MINRES
*/

#ifndef __MINRESBLOCKSOLVER_HPP
#define __MINRESBLOCKSOLVER_HPP

#include <memory>
#include <assert.h>

#include "mfem.hpp"
#include "MixedMatrix.hpp"
#include "MixedLaplacianSolver.hpp"

namespace smoothg
{

/**
   @brief Block diagonal preconditioned MINRES solver for saddle point
   problem.

   Given matrix M and D, setup and solve the graph Laplacian problem
   \f[
     \left( \begin{array}{cc}
       M&  D^T \\
       D&
     \end{array} \right)
     \left( \begin{array}{c}
       u \\ p
     \end{array} \right)
     =
     \left( \begin{array}{c}
       f \\ g
     \end{array} \right)
   \f]
   using MinRes with a block-diagonal preconditioner.

   @todo should this inherit from mfem::Solver or IterativeSolver?

   This class and its implementation owes a lot to MFEM example ex5p
*/
class MinresBlockSolver : public MixedLaplacianSolver
{
public:
    /**
       @brief Constructor from individual M and D matrices.

       @param comm communicator on which to construct parallel MINRES solver
       @param M weighting matrix for graph edges
       @param D describes vertex-edge relation
       @param block_true_offsets describes parallel partitioning (@todo can this be inferred from the matrices?)
    */
    MinresBlockSolver(
        MPI_Comm comm, mfem::HypreParMatrix* M, mfem::HypreParMatrix* D,
        const mfem::Array<int>& block_true_offsets);

    /**
       @brief Constructor from a single MixedMatrix
    */
    MinresBlockSolver(const MixedMatrix& mgL, MPI_Comm comm);

    ~MinresBlockSolver();

    /**
       @brief Use block-preconditioned MINRES to solve the problem.
    */
    void solve(const mfem::BlockVector& rhs, mfem::BlockVector& sol)
    {
        Mult(rhs, sol);
    }

    /// Same as solve()
    virtual void Mult(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const;

protected:
    mfem::MINRESSolver minres_;
    MPI_Comm comm_;

private:
    void init(MPI_Comm comm, mfem::HypreParMatrix* M, mfem::HypreParMatrix* D);

    mfem::BlockOperator operator_;
    mfem::BlockDiagonalPreconditioner prec_;

    mfem::HypreParMatrix* Dt_;
    mfem::HypreParMatrix* schur_block_;
    /// change to lower-right block to eliminate nullspace, ensure solvability
    mfem::SparseMatrix* s_elim_null_;
    mfem::HypreParMatrix* elim_null_;

    // Are these needed? I think the answer is yes, they are
    // needed. We might be able to find efficiency gains by refrencing
    // already-existing matrices, though D gets altered so we'll have
    // to be careful about that. Maybe we should give
    // MixedMatrix a function to release D.
    std::unique_ptr<mfem::SparseMatrix> M_;
    std::unique_ptr<mfem::SparseMatrix> D_;
    HYPRE_Int row_starts_M_[2];
    HYPRE_Int row_starts_D_[2];
    HYPRE_Int col_starts_D_[2];
    std::unique_ptr<mfem::HypreParMatrix> hM_;
    std::unique_ptr<mfem::HypreParMatrix> hD_;
};

/**
   @brief MinresBlockSolver acts on "true" dofs, this one does not.
*/
class MinresBlockSolverFalse : public MinresBlockSolver
{
public:
    MinresBlockSolverFalse(const MixedMatrix& mgL, MPI_Comm comm);
    ~MinresBlockSolverFalse();

    virtual void Mult(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const;

private:
    const MixedMatrix& mixed_matrix_;
};

} // namespace smoothg

#endif
