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
   @file

   @brief Use spectral methods to find edge-based and vertex-based target
   functions that should be represented on the coarse grid.

   Contains LocalMixedGraphSpectralTargets class.
*/

#ifndef __LOCALMIXEDGRAPHSPECTRALTARGETS_HPP
#define __LOCALMIXEDGRAPHSPECTRALTARGETS_HPP

#include <memory>
#include <assert.h>

#include "mfem.hpp"
#include "GraphTopology.hpp"

namespace smoothg
{

/**
   @brief Take a mixed form graph Laplacian, do local eigenvalue problems, and
   generate targets in parallel.
*/
class LocalMixedGraphSpectralTargets
{
public:
    /**
       @brief Construct based on mixed form graph Laplacian.

       @param rel_tol tolerance for including small eigenvectors
       @param max_evects max eigenvectors to include per aggregate
       @param dual_target get traces from eigenvectors of dual graph Laplacian
       @param scaled_dual scale dual graph Laplacian by inverse edge weight.
              Typically coarse problem gets better accuracy but becomes harder
              to solve when this option is turned on.
       @param energy_dual use energy matrix in (RHS of) dual graph eigen problem
              (guarantees approximation property in edge energy norm)
       @param M_local is mass matrix on edge-based (velocity) space
       @param D_local is a divergence-like operator
       @param graph_topology the partitioning relations for coarsening

       M_local should have the coefficients (edge weights) in it.
       D_local should be all 1 and -1.

       And the graph Laplacian in mixed form is
       \f[
          \left( \begin{array}{cc}
            M&  D^T \\
            D&
          \end{array} \right)
       \f]
    */
    LocalMixedGraphSpectralTargets(
        double rel_tol, int max_evects,
        bool dual_target, bool scaled_dual, bool energy_dual,
        const mfem::SparseMatrix& M_local,
        const mfem::SparseMatrix& D_local,
        const GraphTopology& graph_topology);

    LocalMixedGraphSpectralTargets(
        double rel_tol, int max_evects,
        bool dual_target, bool scaled_dual, bool energy_dual,
        const mfem::SparseMatrix& M_local,
        const mfem::SparseMatrix& D_local,
        const mfem::SparseMatrix* W_local,
        const GraphTopology& graph_topology);

    ~LocalMixedGraphSpectralTargets() {}

    /**
       @brief Return targets as result of eigenvalue computations.

       @param local_edge_trace_targets traces of the vertex targets
       @param local_vertex_targets vectors corresponding to smallest eigenvalues
                                   on the vertex space.
    */
    void Compute(std::vector<mfem::DenseMatrix>& local_edge_trace_targets,
                 std::vector<mfem::DenseMatrix>& local_vertex_targets);
private:
    /**
       @brief Solve an eigenvalue problem on each agglomerate, put the result in
       local_vertex_targets.

       Put the normal trace of these into AggExt_sigma

       @param local_vertex targets is a std::vector of size nAEs when it comes in.
           When it comes out, each entry is a DenseMatrix with one column for each
           eigenvector selected.
    */
    void ComputeVertexTargets(std::vector<mfem::DenseMatrix>& AggExt_sigmaT,
                              std::vector<mfem::DenseMatrix>& local_vertex_targets);

    /**
       @brief Given normal traces of eigenvectors in AggExt_sigma, put those as well as
       some kind of PV vector into local_edge_trace_targets.

       @param AggExt_sigma (IN)
       @param local_edge_trace_targets (OUT)
    */
    void ComputeEdgeTargets(const std::vector<mfem::DenseMatrix>& AggExt_sigmaT,
                            std::vector<mfem::DenseMatrix>& local_edge_trace_targets);

    void BuildExtendedAggregates();

    // TODO: better naming - this is not really a permutation because it is not one to one
    // the returned matrix makes a copy of extended part (offd) and add it to local
    std::unique_ptr<mfem::HypreParMatrix> DofPermutation(mfem::HypreParMatrix& ExtAgg_dof);

    std::vector<mfem::SparseMatrix> BuildEdgeEigenSystem(
        const mfem::SparseMatrix& Lloc,
        const mfem::SparseMatrix& Dloc,
        const mfem::Vector& Mloc_diag_inv);

    void Orthogonalize(mfem::DenseMatrix& vectors, mfem::Vector& single_vec,
                       int offset, mfem::DenseMatrix& out);

    void CheckMinimalEigenvalue(double eval_min, int aggregate_id, std::string entity);

    MPI_Comm comm_;

    const double rel_tol_;
    const int max_evects_;
    const bool dual_target_;
    const bool scaled_dual_;
    const bool energy_dual_;

    const mfem::SparseMatrix& M_local_;
    const mfem::SparseMatrix& D_local_;
    const mfem::SparseMatrix* W_local_;

    std::unique_ptr<mfem::HypreParMatrix> M_global_;
    std::unique_ptr<mfem::HypreParMatrix> D_global_;
    std::unique_ptr<mfem::HypreParMatrix> W_global_;

    const GraphTopology& graph_topology_;
    const double zero_eigenvalue_threshold_;

    /// Extended aggregate to vertex dof relation table
    std::unique_ptr<mfem::HypreParMatrix> ExtAgg_vdof_;

    /// Extended aggregate to edge dof relation table
    std::unique_ptr<mfem::HypreParMatrix> ExtAgg_edof_;

    /// face to permuted edge relation table
    std::unique_ptr<mfem::HypreParMatrix> face_permedge_;

    mfem::Array<HYPRE_Int> edgedof_starts;
    mfem::Array<HYPRE_Int> vertdof_starts;
    mfem::Array<HYPRE_Int> edgedof_ext_starts;
    mfem::Array<int> Agg_start_;

    mfem::Array<int> col_mapper_;
};

} // namespace smoothg

#endif
