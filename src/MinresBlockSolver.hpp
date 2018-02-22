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
#include "utilities.hpp"
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
       D&  -W
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
       @param use_W use the W block
    */
    MinresBlockSolver(MPI_Comm comm, mfem::HypreParMatrix* M, mfem::HypreParMatrix* D, mfem::HypreParMatrix* W,
        const mfem::Array<int>& block_true_offsets, bool remove_one_dof = true, bool use_W = false);

    MinresBlockSolver(
        MPI_Comm comm, mfem::HypreParMatrix* M, mfem::HypreParMatrix* D,
        const mfem::Array<int>& block_true_offsets, bool remove_one_dof = true);

    /**
       @brief Constructor from a single MixedMatrix
    */
    MinresBlockSolver(MPI_Comm comm, const MixedMatrix& mgL, bool remove_one_dof = true);

    ~MinresBlockSolver();

    /**
       @brief Use block-preconditioned MINRES to solve the problem.
    */
    void Solve(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const
    {
        Mult(rhs, sol);
    }

    /// Same as Solve()
    virtual void Mult(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const;

    ///@name Set solver parameters
    ///@{
    virtual void SetPrintLevel(int print_level) override;
    virtual void SetMaxIter(int max_num_iter) override;
    virtual void SetRelTol(double rtol) override;
    virtual void SetAbsTol(double atol) override;
    ///@}

protected:
    mfem::MINRESSolver minres_;
    MPI_Comm comm_;
    int myid_;

    bool remove_one_dof_;
    bool use_W_;

private:
    void Init(mfem::HypreParMatrix* M, mfem::HypreParMatrix* D,
              mfem::HypreParMatrix* W);

    mfem::BlockOperator operator_;
    mfem::BlockDiagonalPreconditioner prec_;

    std::unique_ptr<mfem::HypreParMatrix> schur_block_;

    // Solvers' copy of potentially modified data
    mfem::SparseMatrix M_;
    mfem::SparseMatrix D_;
    mfem::SparseMatrix W_;

    std::unique_ptr<mfem::HypreParMatrix> hM_;
    std::unique_ptr<mfem::HypreParMatrix> hD_;
    std::unique_ptr<mfem::HypreParMatrix> hDt_;
    std::unique_ptr<mfem::HypreParMatrix> hW_;
};

/**
   @brief MinresBlockSolver acts on "true" dofs, this one does not.
*/
class MinresBlockSolverFalse : public MinresBlockSolver
{
public:
    MinresBlockSolverFalse(MPI_Comm comm, const MixedMatrix& mgL, bool remove_one_dof = true);
    ~MinresBlockSolverFalse();

    virtual void Mult(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const;

private:
    const MixedMatrix& mixed_matrix_;

    mutable mfem::BlockVector true_rhs_;
    mutable mfem::BlockVector true_sol_;
};

} // namespace smoothg

#endif
