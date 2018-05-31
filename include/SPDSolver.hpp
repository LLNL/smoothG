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
   @file SPDSolver.hpp

   @brief Given a graph in mixed form, solve the resulting system with
   preconditioned CG
*/

#ifndef SPDSOLVER_HPP
#define SPDSOLVER_HPP

#include <memory>
#include <assert.h>

#include "Utilities.hpp"
#include "MixedMatrix.hpp"
#include "MGLSolver.hpp"

namespace smoothg
{

/**
   @brief BoomerAMG Preconditioned CG solver for saddle point
   problem in primal form.

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
*/
class SPDSolver : public MGLSolver
{
public:
    /** @brief Default Constructor */
    SPDSolver() = default;

    /** @brief Constructor from a mixed matrix
        @param mgl mixed matrix information
    */
    SPDSolver(const MixedMatrix& mgl);

    /** @brief Constructor from a mixed matrix, with eliminated edge dofs
        @param mgl mixed matrix information
        @param elim_dofs dofs to eliminate
    */
    SPDSolver(const MixedMatrix& mgl, const std::vector<int>& elim_dofs);

    /** @brief Copy Constructor */
    SPDSolver(const SPDSolver& other) noexcept;

    /** @brief Move Constructor */
    SPDSolver(SPDSolver&& other) noexcept;

    /** @brief Assignment Operator */
    SPDSolver& operator=(SPDSolver other) noexcept;

    /** @brief Swap two solvers */
    friend void swap(SPDSolver& lhs, SPDSolver& rhs) noexcept;

    /** @brief Default Destructor */
    ~SPDSolver() noexcept = default;

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
    ParMatrix A_;
    ParMatrix MinvDT_;

private:
    parlinalgcpp::BoomerAMG prec_;
    linalgcpp::PCGSolver pcg_;
};


} // namespace smoothg

#endif // SPDSOLVER_HPP

