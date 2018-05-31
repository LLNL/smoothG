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

    @brief Routines for setup and implementing the hybridization solver.

           The setup involves forming the hybridized system and constructing a
           preconditioner for the hybridized system.

           In the solving phase (Mult), a given right hand side is transformed
           and the hybridized system is solved by calling an iterative method
           preconditioned by the preconditioner constructed in the setup.
           Lastly, the solution of the original system is computed from the
           solution (Lagrange multiplier) of the hybridized system through
           back substition.
*/

#ifndef __HYBRIDSOLVER_HPP
#define __HYBRIDSOLVER_HPP

#include "Utilities.hpp"
#include "MGLSolver.hpp"
#include "GraphCoarsen.hpp"
#include "MixedMatrix.hpp"

namespace smoothg
{

/**
   @brief Hybridization solver for saddle point problems

   This solver is intended to solve saddle point problems of the form
   \f[
     \left( \begin{array}{cc}
       M&  D^T \\
       D&  0
     \end{array} \right)
     \left( \begin{array}{cc}
       \sigma \\
       u
     \end{array} \right)
     =\left( \begin{array}{cc}
       0 \\
       f
     \end{array} \right)
   \f]

   Given \f$ \widehat{M}, \widehat{D} \f$, the "element" matrices of
   \f$M\f$ and \f$D\f$, the following hybridized system is formed

   \f[
     H = C (\widehat{M}^{-1}-\widehat{M}^{-1}\widehat{D}^T
           (\widehat{D}\widehat{M}^{-1}\widehat{D}^T)^{-1}
           \widehat{D}\widehat{M}^{-1}) C^T
   \f]

   The \f$C\f$ matrix is the constraint matrix for enforcing the continuity of
   the "broken" edge space as well as the boundary conditions. This is
   created inside the class.

   Each constraint in turn creates a dual variable (Lagrange multiplier).
   The construction is done locally in each element.
*/
class HybridSolver : public MGLSolver
{
public:
    /**
       @brief Constructor for hybridiziation solver.

       @param mgL Mixed matrices for the graph Laplacian
    */
    HybridSolver(const MixedMatrix& mgL);

    virtual ~HybridSolver() = default;

    /// Wrapper for solving the saddle point system through hybridization
    void Solve(const BlockVector& Rhs, BlockVector& Sol) const override;

    /// Transform original RHS to the RHS of the hybridized system
    void RHSTransform(const BlockVector& OriginalRHS, VectorView HybridRHS) const;

    /**
       @brief Recover the solution of the original system from multiplier \f$ \mu \f$.

       \f[
         \left( \begin{array}{c} u \\ p \end{array} \right)
         =
         \left( \begin{array}{c} f \\ g \end{array} \right) -
         \left( \begin{array}{cc} M&  B^T \\ B& \end{array} \right)^{-1}
         \left( \begin{array}{c} C \\ 0 \end{array} \right)
         \mu
       \f]

       This procedure is done locally in each element

       This function assumes the offsets of RecoveredSol have been defined
    */
    void RecoverOriginalSolution(const VectorView& HybridSol,
                                 BlockVector& RecoveredSol) const;

    /**
       @brief Update weights of local M matrices on aggregates
       @param agg_weights weights per aggregate

       @todo when W is non-zero, Aloc and Hybrid_el need to be recomputed
    */
    void UpdateAggScaling(const std::vector<double>& agg_weight);

    ///@name Set solver parameters
    ///@{
    virtual void SetPrintLevel(int print_level) override;
    virtual void SetMaxIter(int max_num_iter) override;
    virtual void SetRelTol(double rtol) override;
    virtual void SetAbsTol(double atol) override;
    ///@}

private:

    SparseMatrix AssembleHybridSystem(const MixedMatrix& mgl,
                                      const std::vector<int>& j_multiplier_edgedof);

    SparseMatrix MakeEdgeDofMultiplier() const;

    SparseMatrix MakeLocalC(int agg, const ParMatrix& edge_true_edge,
                            const std::vector<int>& j_multiplier_edgedof,
                            std::vector<int>& edge_map,
                            std::vector<bool>& edge_marker) const;

    void InitSolver(SparseMatrix local_hybrid);

    SparseMatrix agg_vertexdof_;
    SparseMatrix agg_edgedof_;
    SparseMatrix agg_multiplier_;

    int num_aggs_;
    int num_edge_dofs_;
    int num_multiplier_dofs_;

    ParMatrix multiplier_d_td_;

    ParMatrix pHybridSystem_;

    linalgcpp::PCGSolver cg_;
    parlinalgcpp::BoomerAMG prec_;

    std::vector<DenseMatrix> MinvDT_;
    std::vector<DenseMatrix> MinvCT_;
    std::vector<DenseMatrix> AinvDMinvCT_;
    std::vector<DenseMatrix> Ainv_;
    std::vector<DenseMatrix> hybrid_elem_;

    mutable std::vector<Vector> Ainv_f_;

    std::vector<double> agg_weights_;

    mutable Vector trueHrhs_;
    mutable Vector trueMu_;
    mutable Vector Hrhs_;
    mutable Vector Mu_;
};


} // namespace smoothg

#endif /* HYBRIDSOLVER_HPP_ */
