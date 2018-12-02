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
#include "LocalEigenSolver.hpp"

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
   Collection of parameters for upscaling methods

   @param spect_tol spectral tolerance determines how many eigenvectors to
          keep per aggregate
   @param max_evects maximum number of eigenvectors to keep per aggregate
   @param trace_method methods for getting edge trace samples
   @param hybridization use hybridization as solver
   @param coefficient use coarse coefficient rescaling construction
   @param saamge_param SAAMGe paramters, use SAAMGe as preconditioner for
          coarse hybridized system if saamge_param is not nullptr
*/
class UpscaleParameters
{
public:
    int max_levels;
    double spect_tol;
    int max_evects;
    bool dual_target;
    bool scaled_dual;
    bool energy_dual;
    bool hybridization;
    bool coarse_components;
    int coarse_factor;
    SAAMGeParam* saamge_param;
    // possibly also boundary condition information?

    UpscaleParameters() : max_levels(2),
        spect_tol(0.001),
        max_evects(4),
        dual_target(false),
        scaled_dual(false),
        energy_dual(false),
        hybridization(false),
        coarse_components(false),
        coarse_factor(64),
        saamge_param(NULL)
    {}

    void RegisterInOptionsParser(mfem::OptionsParser& args)
    {
        args.AddOption(&max_levels, "--max-levels", "--max-levels",
                       "Number of levels in multilevel hierarchy");
        args.AddOption(&max_evects, "-m", "--max-evects",
                       "Maximum eigenvectors per aggregate.");
        args.AddOption(&spect_tol, "-t", "--spect-tol",
                       "Spectral tolerance for eigenvalue problems.");
        args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                       "--no-hybridization", "Enable hybridization.");
        args.AddOption(&dual_target, "-dt", "--dual-target", "-no-dt",
                       "--no-dual-target", "Use dual graph Laplacian in trace generation.");
        args.AddOption(&scaled_dual, "-sd", "--scaled-dual", "-no-sd",
                       "--no-scaled-dual", "Scale dual graph Laplacian by (inverse) edge weight.");
        args.AddOption(&energy_dual, "-ed", "--energy-dual", "-no-ed",
                       "--no-energy-dual", "Use energy matrix in trace generation.");
        args.AddOption(&coarse_components, "-coarse-comp", "--coarse-components", "-no-coarse-comp",
                       "--no-coarse-components", "Store trace, bubble components of coarse M.");
        args.AddOption(&coarse_factor, "--coarse-factor", "--coarse-factor",
                       "Coarsening factor for metis agglomeration.");
    }
};

/**
   @brief Take a mixed form graph Laplacian, do local eigenvalue problems, and
   generate targets in parallel.
*/
class LocalMixedGraphSpectralTargets
{
public:
    /**
       @brief Construct based on mixed form graph Laplacian.

       @param mixed_graph_laplacian container for M, D, W
       @param graph_topology the partitioning relations for coarsening
       @param param container for rel_tol, max_evects, various dual target flags

       And the graph Laplacian in mixed form is
       \f[
          \left( \begin{array}{cc}
            M&  D^T \\
            D&  -W
          \end{array} \right)
       \f]
    */
    LocalMixedGraphSpectralTargets(
        const MixedMatrix& mixed_graph_laplacian,
        const GraphTopology& graph_topology,
        const UpscaleParameters& param);

    ~LocalMixedGraphSpectralTargets() {}

    /**
       @brief Return targets as result of eigenvalue computations.

       @param local_edge_trace_targets traces of the vertex targets
       @param local_vertex_targets vectors corresponding to smallest eigenvalues
                                   on the vertex space.
       @param constant_rep representation of constant vertex vector on finer
                           space
    */
    void Compute(std::vector<mfem::DenseMatrix>& local_edge_trace_targets,
                 std::vector<mfem::DenseMatrix>& local_vertex_targets);
private:
    enum DofType { VDOF, EDOF }; // vertex-based and edge-based dofs

    /**
       @brief Compute spectral vectex targets for each aggregate

       Solve an eigenvalue problem on each extended aggregate, restrict eigenvectors
       to original aggregate and use SVD to remove linear dependence, put resulting
       vectors in local_vertex_targets as column vectors.

       Put edge traces into ExtAgg_sigmaT as row vectors
       Trace generation depends on dual_target_, scaled_dual_, and energy_dual_

       @param ExtAgg_sigmaT (OUT)
       @param local_vertex targets (OUT)
    */
    void ComputeVertexTargets(std::vector<mfem::DenseMatrix>& ExtAgg_sigmaT,
                              std::vector<mfem::DenseMatrix>& local_vertex_targets);

    /**
       @brief Compute edge trace targets for each coarse face

       Given edge traces in aggregates (from ExtAgg_sigmaT), restrict them to coarse
       face, put those restricted vectors as well as some kind of PV vector into
       local_edge_trace_targets after SVD.

       @param ExtAgg_sigmaT (IN)
       @param local_edge_trace_targets (OUT)
    */
    void ComputeEdgeTargets(const std::vector<mfem::DenseMatrix>& ExtAgg_sigmaT,
                            std::vector<mfem::DenseMatrix>& local_edge_trace_targets);

    void BuildExtendedAggregates();

    // TODO: better naming - this is not really a permutation because it is not one to one
    // the returned matrix makes a copy of extended part (offd) and add it to local
    std::unique_ptr<mfem::HypreParMatrix> DofPermutation(DofType dof_type);

    void GetExtAggDofs(DofType dof_type, int iAgg, mfem::Array<int>& dofs);

    void Orthogonalize(mfem::DenseMatrix& vectors, mfem::Vector& single_vec,
                       int offset, mfem::DenseMatrix& out);

    /**
       @brief Fill onenegone partly with a constant positive value, partly with
       a constant negative value, so that it has zero average

       (on coarser levels this does something different)
    */
    mfem::Vector MakeOneNegOne(const mfem::Vector& constant, int split);

    /// given an assembled vector on vertices, return extracted value on (possibly shared) faces
    mfem::Vector** CollectConstant(const mfem::Vector& constant_vect);

    /// shared_constant expected to be an array of legth 2, just returns them
    /// stacked on top of each other
    mfem::Vector ConstantLocal(mfem::Vector* shared_constant);

    MPI_Comm comm_;

    const double rel_tol_;
    const int max_evects_;
    const bool dual_target_;
    const bool scaled_dual_;
    const bool energy_dual_;

    const mfem::SparseMatrix& M_local_;
    const mfem::SparseMatrix& D_local_;
    const mfem::SparseMatrix* W_local_;
    const mfem::Vector& constant_rep_;

    std::unique_ptr<mfem::HypreParMatrix> M_global_;
    std::unique_ptr<mfem::HypreParMatrix> D_global_;
    std::unique_ptr<mfem::HypreParMatrix> W_global_;

    const GraphTopology& graph_topology_;
    const double zero_eigenvalue_threshold_;

    /// Extended aggregate to vertex dof relation table
    std::unique_ptr<mfem::HypreParMatrix> ExtAgg_vdof_;
    mfem::SparseMatrix ExtAgg_vdof_diag_;
    mfem::SparseMatrix ExtAgg_vdof_offd_;

    /// Extended aggregate to edge dof relation table
    std::unique_ptr<mfem::HypreParMatrix> ExtAgg_edof_;
    mfem::SparseMatrix ExtAgg_edof_diag_;
    mfem::SparseMatrix ExtAgg_edof_offd_;

    /// face to permuted edge dof relation table
    std::unique_ptr<mfem::HypreParMatrix> face_perm_edof_;

    mfem::Array<HYPRE_Int> edgedof_starts;
    mfem::Array<HYPRE_Int> vertdof_starts;
    mfem::Array<HYPRE_Int> edgedof_ext_starts;
    mfem::Array<int> Agg_start_;

    mfem::Array<int> col_map_;
};

} // namespace smoothg

#endif
