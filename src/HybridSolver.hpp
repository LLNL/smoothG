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

#ifndef __HYBRIDIZATION_HPP
#define __HYBRIDIZATION_HPP

#include "LocalMixedGraphSpectralTargets.hpp"
#include "utilities.hpp"
#include "MixedLaplacianSolver.hpp"
#include "MixedMatrix.hpp"

#if SMOOTHG_USE_SAAMGE
#include "saamge.hpp"
#endif

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
class HybridSolver : public MixedLaplacianSolver
{
public:
    /**
       @brief Constructor for fine-level hybridiziation solver.

       @param comm MPI communicator
       @param mgL Mixed matrices for the graph Laplacian in the fine level
       @param face_bdrattr Boundary edge to boundary attribute table
       @param ess_edge_dofs An array indicating essential edge dofs
       @param rescale_iter number of iterations to compute diagonal scaling
              vector for hybridized system. No rescaling if set to 0.
       @param saamge_param SAAMGe parameters. Use SAAMGe as preconditioner for
              hybridized system if saamge_param is not nullptr, otherwise
              BoomerAMG is used instead.
    */
    HybridSolver(const MixedMatrix& mgL,
                 const mfem::Array<int>* ess_attr = nullptr,
                 const int rescale_iter = 0,
                 const SAAMGeParam* saamge_param = nullptr);

    virtual ~HybridSolver();

    /// Wrapper for solving the saddle point system through hybridization
    void Mult(const mfem::BlockVector& Rhs, mfem::BlockVector& Sol) const;

    /**
       @brief Update weights of local M matrices on "elements"

       elem_scaling_inverse in the input is like the coefficient in
       a finite volume problem, elem_scaling is the weights on the mass matrix
       in the mixed form, which is the reciprocal of that.

       @todo when W is non-zero, Aloc and Hybrid_el need to be recomputed
    */
    virtual void UpdateElemScaling(const mfem::Vector& elem_scaling_inverse);

    virtual void UpdateJacobian(const mfem::Vector& elem_scaling_inverse,
                                const std::vector<mfem::DenseMatrix>& N_el);

    ///@name Set solver parameters
    ///@{
    virtual void SetPrintLevel(int print_level) override;
    virtual void SetMaxIter(int max_num_iter) override;
    virtual void SetRelTol(double rtol) override;
    virtual void SetAbsTol(double atol) override;
    ///@}

private:
    void Init(const mfem::SparseMatrix& face_edgedof,
              const std::vector<mfem::DenseMatrix>& M_el,
              const mfem::HypreParMatrix& edgedof_d_td,
              const mfem::SparseMatrix& face_bdrattr);

    void CreateMultiplierRelations(const mfem::SparseMatrix& face_edgedof,
                                   const mfem::HypreParMatrix& edgedof_d_td);

    mfem::SparseMatrix AssembleHybridSystem(
        const std::vector<mfem::DenseMatrix>& M_el);

    mfem::SparseMatrix AssembleHybridSystem(
        const mfem::Vector& elem_scaling_inverse,
        const std::vector<mfem::DenseMatrix>& N_el);

    // Compute scaling vector and the scaled hybridized system
    void ComputeScaledHybridSystem(const mfem::HypreParMatrix& H_d);

    // Construct spectral AMGe preconditioner
    void BuildSpectralAMGePreconditioner();

    // Assemble parallel hybridized system and build a solver for it
    void BuildParallelSystemAndSolver(mfem::SparseMatrix& H_proc);

    void CollectEssentialDofs(const mfem::SparseMatrix& edof_bdrattr);

    void CheckSharing();

    /// Transform original RHS to the RHS of the hybridized system
    void RHSTransform(const mfem::BlockVector& OriginalRHS,
                      mfem::Vector& HybridRHS) const;

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
    void RecoverOriginalSolution(const mfem::Vector& HybridSol,
                                 mfem::BlockVector& RecoveredSol) const;

    mfem::Vector MakeInitialGuess(const mfem::BlockVector& sol,
                                  const mfem::BlockVector& rhs) const;

    const MixedMatrix& mgL_;

    mfem::SparseMatrix Agg_multiplier_;

    mfem::Array<int> edgedof_is_shared_;

    std::unique_ptr<mfem::HypreParMatrix> H_;
    std::unique_ptr<mfem::Solver> prec_;
    mfem::CGSolver cg_;

    // eliminated part of H_ (for applying elimination in repeated solves)
    std::unique_ptr<mfem::HypreParMatrix> H_elim_;

    std::vector<mfem::DenseMatrix> Hybrid_el_;

    std::vector<mfem::DenseMatrix> MinvN_;
    std::vector<mfem::DenseMatrix> DMinv_;
    std::vector<mfem::DenseMatrix> MinvCT_;
    std::vector<mfem::DenseMatrix> AinvDMinvCT_;
    std::vector<mfem::DenseMatrix> CMinvNAinv_;
    std::vector<mfem::DenseMatrix> Ainv_;
    std::vector<mfem::DenseMatrix> Minv_;
    std::vector<mfem::DenseMatrix> Minv_ref_;
    std::vector<mfem::SparseMatrix> C_;
    std::vector<mfem::DenseMatrix> CM_;
    std::vector<mfem::SparseMatrix> CDT_;

    mutable std::vector<mfem::Vector> Minv_g_;
    mutable std::vector<mfem::Vector> local_rhs_;

    mfem::Array<int> ess_true_multipliers_;
    mfem::Array<int> multiplier_to_edof_;
    mfem::Array<int> ess_true_mult_to_edof_;
    mfem::Array<HYPRE_Int> multiplier_start_;
    mfem::Array<bool> mult_on_bdr_;

    std::unique_ptr<mfem::HypreParMatrix> multiplier_d_td_;
    std::unique_ptr<mfem::HypreParMatrix> multiplier_td_d_;

    mutable mfem::Vector trueHrhs_;
    mutable mfem::Vector trueMu_;
    mutable mfem::Vector Hrhs_;
    mutable mfem::Vector Mu_;

    int nAggs_;
    int num_multiplier_dofs_;

    int rescale_iter_;
    mfem::Vector diagonal_scaling_;

    const SAAMGeParam* saamge_param_;
#if SMOOTHG_USE_SAAMGE
    std::vector<int> sa_nparts_;
    saamge::agg_partitioning_relations_t* sa_apr_;
    saamge::ml_data_t* sa_ml_data_;
#endif
};


/// assuming symmetric problems
class AuxSpacePrec : public mfem::Solver
{
public:
    /// dofs are in true dofs numbering, coarse_map: coarse to fine
    AuxSpacePrec(mfem::HypreParMatrix& op, mfem::SparseMatrix aux_map,
                 const std::vector<mfem::Array<int>>& loc_dofs);

    virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const;
    virtual void SetOperator(const mfem::Operator& op) {}

private:

    void Smoothing(const mfem::Vector& x, mfem::Vector& y) const;
    std::vector<mfem::Array<int>> local_dofs_;
    std::vector<mfem::DenseMatrix> local_ops_;
    std::vector<mfem::DenseMatrix> local_solvers_;

    mfem::HypreParMatrix& op_;
    mfem::SparseMatrix op_diag_;
    mfem::SparseMatrix aux_map_;
    std::unique_ptr<mfem::HypreParMatrix> aux_op_;
    std::unique_ptr<mfem::HypreBoomerAMG> aux_solver_;
};

} // namespace smoothg

#endif /* __HYBRIDIZATION_HPP */
