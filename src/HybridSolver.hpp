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

#include "mfem.hpp"

#include "LocalMixedGraphSpectralTargets.hpp"
#include "utilities.hpp"
#include "MixedLaplacianSolver.hpp"
#include "Mixed_GL_Coarsener.hpp"
#include "MixedMatrix.hpp"

#if SMOOTHG_USE_SAAMGE
#include "saamge.hpp"
#endif

namespace smoothg
{

/// Container for SAAMGe parameters
struct SAAMGeParam
{
    int num_levels = 2;

    /// Parameters for all levels
    int nu_relax = 2;
    bool use_arpack = false;
    bool correct_nulspace = false;
    bool do_aggregates = true;

    /// Parameters for the first coarsening
    int first_coarsen_factor = 64;
    int first_nu_pro = 1;
    double first_theta = 1e-3;

    /// Parameters for all later coarsenings (irrelevant if num_levels = 2)
    int coarsen_factor = 8;
    int nu_pro = 1;
    double theta = 1e-3;
};

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
    /// Construct local mass matrix for the fine level edge space
    static void BuildFineLevelLocalMassMatrix(
        const mfem::SparseMatrix& vertex_edge,
        const mfem::SparseMatrix& M,
        std::vector<mfem::Vector>& M_el);

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
    HybridSolver(MPI_Comm comm,
                 const MixedMatrix& mgL,
                 const mfem::SparseMatrix* face_bdrattr = nullptr,
                 const mfem::Array<int>* ess_edge_dofs = nullptr,
                 const int rescale_iter = 0,
                 const SAAMGeParam* saamge_param = nullptr);

    /**
       @brief Constructor for coarse-level hybridiziation solver.

       @param comm MPI communicator
       @param mgL Mixed matrices for the graph Laplacian in the coarse level
       @param mgLc Mixed graph Laplacian Coarsener from fine to coarse level
       @param face_bdrattr Boundary edge to boundary attribute table
       @param ess_edge_dofs An array indicating essential edge dofs
       @param rescale_iter number of iterations to compute diagonal scaling
              vector for hybridized system. No rescaling if set to 0.
       @param saamge_param SAAMGe parameters. Use SAAMGe as preconditioner for
              hybridized system if saamge_param is not nullptr, otherwise
              BoomerAMG is used instead.
    */
    HybridSolver(MPI_Comm comm,
                 const MixedMatrix& mgL,
                 const Mixed_GL_Coarsener& mgLc,
                 const mfem::SparseMatrix* face_bdrattr = nullptr,
                 const mfem::Array<int>* ess_edge_dofs = nullptr,
                 const int rescale_iter = 0,
                 const SAAMGeParam* saamge_param = nullptr);

    virtual ~HybridSolver();

    /// Wrapper for solving the saddle point system through hybridization
    void Mult(const mfem::BlockVector& Rhs,
              mfem::BlockVector& Sol) const;

    /// Same as Mult()
    void Solve(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const
    {
        Mult(rhs, sol);
    }

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

    ///@name Set solver parameters
    ///@{
    virtual void SetPrintLevel(int print_level) override;
    virtual void SetMaxIter(int max_num_iter) override;
    virtual void SetRelTol(double rtol) override;
    virtual void SetAbsTol(double atol) override;
    ///@}

protected:
    template<typename T>
    void Init(const mfem::SparseMatrix& face_edgedof,
              const std::vector<T>& M_el,
              const mfem::HypreParMatrix& edgedof_d_td,
              const mfem::SparseMatrix* face_bdrattr,
              const mfem::Array<int>* ess_edge_dofs);

    /**
       @todo this method and its cousin share a lot of duplicated code
    */
    void AssembleHybridSystem(
        const std::vector<mfem::DenseMatrix>& M_el,
        const mfem::Array<int>& j_multiplier_edgedof);

    void AssembleHybridSystem(
        const std::vector<mfem::Vector>& M_el,
        const mfem::Array<int>& j_multiplier_edgedof);

    // Compute scaling vector and the scaled hybridized system
    void ComputeScaledHybridSystem(const mfem::HypreParMatrix& H_d);

    // Construct spectral AMGe preconditioner
    void BuildSpectralAMGePreconditioner();

private:
    MPI_Comm comm_;
    int myid_;

    mfem::SparseMatrix Agg_multiplier_;
    mfem::SparseMatrix Agg_vertexdof_;
    mfem::SparseMatrix Agg_edgedof_;
    mfem::SparseMatrix edgedof_IsOwned_;

    const mfem::SparseMatrix& D_;
    const mfem::SparseMatrix* W_;

    std::unique_ptr<mfem::SparseMatrix> HybridSystem_;
    std::unique_ptr<mfem::SparseMatrix> HybridSystemElim_;
    std::unique_ptr<mfem::HypreParMatrix> pHybridSystem_;
    std::unique_ptr<mfem::Solver> prec_;
    std::unique_ptr<mfem::CGSolver> cg_;


    std::vector<mfem::DenseMatrix> Hybrid_el_;

    std::vector<mfem::DenseMatrix> MinvDT_;
    std::vector<mfem::DenseMatrix> MinvCT_;
    std::vector<mfem::DenseMatrix> AinvDMinvCT_;
    std::vector<mfem::DenseMatrix> Ainv_;

    mutable std::vector<mfem::Vector> Ainv_f_;

    bool ess_multiplier_bc_;
    mfem::Array<int> ess_multiplier_dofs_;
    mfem::Array<HYPRE_Int> multiplier_start_;

    std::unique_ptr<mfem::HypreParMatrix> multiplier_d_td_;

    mutable mfem::Vector trueHrhs_;
    mutable mfem::Vector trueMu_;
    mutable mfem::Vector Hrhs_;
    mutable mfem::Vector Mu_;

    int nAggs_;
    int num_edge_dofs_;
    int num_multiplier_dofs_;

    bool use_spectralAMGe_;
    bool use_w_;

    int rescale_iter_;
    mfem::Vector diagonal_scaling_;

    const SAAMGeParam* saamge_param_;
#if SMOOTHG_USE_SAAMGE
    std::vector<int> sa_nparts_;
    saamge::agg_partitioning_relations_t* sa_apr_;
    saamge::ml_data_t* sa_ml_data_;
#endif
};

} // namespace smoothg

#endif /* __HYBRIDIZATION_HPP */
