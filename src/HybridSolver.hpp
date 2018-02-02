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
    /// Construct local mass matrix for the fine level edge space
    static void BuildFineLevelLocalMassMatrix(
        const mfem::SparseMatrix& vertex_edge,
        const mfem::SparseMatrix& M,
        std::vector<std::unique_ptr<mfem::Vector> >& M_el);

public:
    /**
       @brief Constructor for fine-level hybridiziation solver.

       @param comm MPI communicator
       @param mgL Mixed matrices for the graph Laplacian in the fine level
       @param face_bdrattr Boundary edge to boundary attribute table
       @param ess_edge_dofs An array indicating essential edge dofs
       @param spectralAMGe Whether to use spectral AMGe as the preconditioner
              for the CG iteration (not implemented yet).
    */
    HybridSolver(MPI_Comm comm,
                 const MixedMatrix& mgL,
                 std::shared_ptr<const mfem::SparseMatrix> face_bdrattr = nullptr,
                 std::shared_ptr<const mfem::Array<int> > ess_edge_dofs = nullptr,
                 bool spectralAMGe = false);

    /**
       @brief Constructor for fine-level hybridiziation solver.

       @param comm MPI communicator
       @param mgL Mixed matrices for the graph Laplacian in the coarse level
       @param mgLc Mixed graph Laplacian Coarsener from fine to coarse level
       @param face_bdrattr Boundary edge to boundary attribute table
       @param ess_edge_dofs An array indicating essential edge dofs
       @param spectralAMGe Whether to use spectral AMGe as the preconditioner
              for the CG iteration (not implemented yet).
    */
    HybridSolver(MPI_Comm comm,
                 const MixedMatrix& mgL,
                 const Mixed_GL_Coarsener& mgLc,
                 std::shared_ptr<const mfem::SparseMatrix> face_bdrattr = nullptr,
                 std::shared_ptr<const mfem::Array<int> > ess_edge_dofs = nullptr,
                 bool spectralAMGe = false);

    virtual ~HybridSolver() {}

    /// Wrapper for solving the saddle point system through hybridization
    void Mult(const mfem::BlockVector& Rhs,
              mfem::BlockVector& Sol);

    /// Same as Mult()
    /// @todo should be const
    void solve(const mfem::BlockVector& rhs, mfem::BlockVector& sol)
    {
        Mult(rhs, sol);
    }

    /// Transform original RHS to the RHS of the hybridized system
    void RHSTransform(const mfem::BlockVector& OriginalRHS,
                      mfem::Vector& HybridRHS);

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
                                 mfem::BlockVector& RecoveredSol);

protected:
    template<typename T>
    void Init(const mfem::SparseMatrix& face_edgedof,
              const std::vector<std::unique_ptr<T> >& M_el,
              const mfem::HypreParMatrix& edgedof_d_td,
              std::shared_ptr<const mfem::SparseMatrix> face_bdrattr,
              std::shared_ptr<const mfem::Array<int> > ess_edge_dofs,
              bool spectralAMGe);

    /**
       @todo this method and its cousin share a lot of duplicated code
    */
    void AssembleHybridSystem(
        const std::vector<std::unique_ptr<mfem::DenseMatrix> >& M_el,
        mfem::Array<int>& edgedof_global_to_local_map,
        mfem::Array<bool>& edge_marker,
        int* j_multiplier_edgedof,
        bool spectralAMGe);

    void AssembleHybridSystem(
        const std::vector<std::unique_ptr<mfem::Vector> >& M_el,
        mfem::Array<int>& edgedof_global_to_local_map,
        mfem::Array<bool>& edge_marker,
        int* j_multiplier_edgedof,
        bool spectralAMGe);
private:
    MPI_Comm comm_;
    int myid_;
    mfem::SparseMatrix Agg_vertexdof_;
    mfem::SparseMatrix Agg_edgedof_;
    mfem::SparseMatrix edgedof_IsOwned_;

    const mfem::SparseMatrix& D_;

    std::unique_ptr<mfem::SparseMatrix> HybridSystem_;
    std::unique_ptr<mfem::SparseMatrix> HybridSystemElim_;
    std::unique_ptr<mfem::HypreParMatrix> pHybridSystem_;
    std::unique_ptr<mfem::HypreBoomerAMG> prec_;
    std::unique_ptr<mfem::CGSolver> cg_;

    std::unique_ptr<mfem::SparseMatrix> Agg_multiplier_;

    std::vector<std::unique_ptr<mfem::DenseMatrix>> Hybrid_el_;
    std::vector<std::unique_ptr<mfem::DenseMatrix>> MinvDT_;
    std::vector<std::unique_ptr<mfem::DenseMatrix>> MinvCT_;
    std::vector<std::unique_ptr<mfem::DenseMatrix>> AinvDMinvCT_;
    std::vector<std::unique_ptr<mfem::Vector>> Ainv_f_;
    std::vector<std::unique_ptr<mfem::Operator>> Ainv_;

    bool ess_multiplier_bc_;
    mfem::Array<int> ess_multiplier_dofs_;
    mfem::Array<HYPRE_Int> multiplier_start_;
    mfem::Array<HYPRE_Int> truemultiplier_start_;

    std::unique_ptr<mfem::HypreParMatrix> multiplier_d_td_;

    int nAggs_;
    int num_edge_dofs_;
    int num_multiplier_dofs_;
};

} // namespace smoothg

#endif /* HYBRIDIZATION_HPP_ */
