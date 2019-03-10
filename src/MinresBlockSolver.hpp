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
    MinresBlockSolver(mfem::HypreParMatrix* M, mfem::HypreParMatrix* D, mfem::SparseMatrix* W,
                      const mfem::Array<int>& block_true_offsets);

    /**
       @brief Constructor from a single MixedMatrix
    */
    MinresBlockSolver(const MixedMatrix& mgL,
                      const mfem::Array<int>* ess_attr = nullptr);

    /**
       @brief Use block-preconditioned MINRES to solve the problem.
    */
    virtual void Mult(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const;

    virtual void UpdateJacobian(const mfem::Vector& elem_scaling_inverse,
                                const std::vector<mfem::DenseMatrix>& N_el)
    {
        mfem::mfem_error("not implemented!\n");
    }

    virtual void UpdateElemScaling(const mfem::Vector& elem_scaling_inverse)
    {
        mfem::mfem_error("This is currently not supported!\n");
    }

    ///@name Set solver parameters
    ///@{
    virtual void SetPrintLevel(int print_level) override;
    virtual void SetMaxIter(int max_num_iter) override;
    virtual void SetRelTol(double rtol) override;
    virtual void SetAbsTol(double atol) override;
    ///@}

protected:
    mfem::MINRESSolver minres_;

    void Init(mfem::HypreParMatrix* M, mfem::HypreParMatrix* D,
              mfem::SparseMatrix* W);

    mfem::BlockOperator operator_;
    mfem::BlockDiagonalPreconditioner prec_;

    std::unique_ptr<mfem::HypreParMatrix> schur_block_;

    std::unique_ptr<mfem::SparseMatrix> W_;

    std::unique_ptr<mfem::HypreParMatrix> hM_;
    std::unique_ptr<mfem::HypreParMatrix> hD_;
    std::unique_ptr<mfem::HypreParMatrix> hDt_;

    std::unique_ptr<mfem::HypreSmoother> Mprec_;
    std::unique_ptr<mfem::HypreBoomerAMG> Sprec_;
};

/**
   @brief MinresBlockSolver acts on "true" dofs, this one does not.
*/
class MinresBlockSolverFalse : public MinresBlockSolver
{
public:
    MinresBlockSolverFalse(const MixedMatrix& mgL,
                           const mfem::Array<int>* ess_attr = nullptr);

    virtual void Mult(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const;

    virtual void Mult(const mfem::Vector& rhs, mfem::Vector& sol) const;

    virtual void UpdateElemScaling(const mfem::Vector& elem_scaling_inverse);

    virtual void UpdateJacobian(const mfem::Vector& elem_scaling_inverse,
                                const std::vector<mfem::DenseMatrix>& N_el) override;

private:
    const MixedMatrix& mixed_matrix_;
    std::unique_ptr<mfem::HypreParMatrix> block_01_;
};

} // namespace smoothg

#endif
