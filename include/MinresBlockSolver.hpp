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

#include "Utilities.hpp"
#include "MixedMatrix.hpp"
#include "MGLSolver.hpp"

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
class MinresBlockSolver : public MGLSolver
{
public:
    /** @brief Default Constructor */
    MinresBlockSolver() = default;

    /** @brief Constructor from a mixed matrix
        @param mgl mixed matrix information
    */
    MinresBlockSolver(const MixedMatrix& mgl);

    /** @brief Copy Constructor */
    MinresBlockSolver(const MinresBlockSolver& other) noexcept;

    /** @brief Move Constructor */
    MinresBlockSolver(MinresBlockSolver&& other) noexcept;

    /** @brief Assignment Operator */
    MinresBlockSolver& operator=(MinresBlockSolver other) noexcept;

    /** @brief Swap two solvers */
    friend void swap(MinresBlockSolver& lhs, MinresBlockSolver& rhs) noexcept;

    /** @brief Default Destructor */
    ~MinresBlockSolver() noexcept = default;

    /** @brief Use block-preconditioned MINRES to solve the problem.
        @param rhs Right hand side
        @param sol Solution
    */
    void Solve(const BlockVector& rhs, BlockVector& sol) const override;

    ///@name Set solver parameters
    ///@{
    virtual void SetPrintLevel(int print_level) override;
    virtual void SetMaxIter(int max_num_iter) override;
    virtual void SetRelTol(double rtol) override;
    virtual void SetAbsTol(double atol) override;
    ///@}

protected:

    ParMatrix M_;
    ParMatrix D_;
    ParMatrix DT_;
    ParMatrix W_;

    ParMatrix edge_true_edge_;

    MPI_Comm comm_;
    int myid_;

private:
    linalgcpp::BlockOperator op_;
    linalgcpp::BlockOperator prec_;

    parlinalgcpp::ParDiagScale M_prec_;
    parlinalgcpp::BoomerAMG schur_prec_;

    linalgcpp::PMINRESSolver pminres_;

    mutable BlockVector true_rhs_;
    mutable BlockVector true_sol_;

    bool use_w_;
};

/**
   @brief MinresBlockSolver acts on "true" dofs, this one does not.
*/
class MinresBlockSolverFalse : public MinresBlockSolver
{
public:
    MinresBlockSolverFalse() = default;
    MinresBlockSolverFalse(const MixedMatrix& mgl);

    ~MinresBlockSolverFalse() = default;

    virtual void Mult(const BlockVector& rhs, BlockVector& sol) const;

private:
    //const MixedMatrix& mixed_matrix_;

    mutable BlockVector true_rhs_;
    mutable BlockVector true_sol_;
};

} // namespace smoothg

#endif
