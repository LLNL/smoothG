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

#include "GraphTopology.hpp"
#include "GraphSpace.hpp"
#include "LocalEigenSolver.hpp"
#include "MatrixUtilities.hpp"

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
   @param max_traces maximum number of edge traces to keep per coarse face
   @param trace_method methods for getting edge trace samples
   @param hybridization use hybridization as solver
   @param coefficient use coarse coefficient rescaling construction
   @param rescale_iter number of iteration to compute scaling in hybridization
   @param saamge_param SAAMGe paramters, use SAAMGe as preconditioner for
          coarse hybridized system if saamge_param is not nullptr
*/
class UpscaleParameters
{
public:
    int max_levels;
    double spect_tol;
    int max_evects;
    int max_traces;
    bool dual_target;
    bool scaled_dual;
    bool energy_dual;
    bool hybridization;
    bool coarse_components;
    int coarse_factor;
    int num_iso_verts;
    int rescale_iter;
    SAAMGeParam* saamge_param;
    // possibly also boundary condition information?

    UpscaleParameters() : max_levels(2),
        spect_tol(0.001),
        max_evects(4),
        max_traces(4),
        dual_target(false),
        scaled_dual(false),
        energy_dual(false),
        hybridization(false),
        coarse_components(false),
        coarse_factor(64),
        num_iso_verts(0),
        rescale_iter(-1),
        saamge_param(NULL)
    {}

    void RegisterInOptionsParser(mfem::OptionsParser& args)
    {
        args.AddOption(&max_levels, "--max-levels", "--max-levels",
                       "Number of levels in multilevel hierarchy");
        args.AddOption(&max_evects, "-m", "--max-evects",
                       "Maximum number of eigenvectors per aggregate.");
        args.AddOption(&max_traces, "-mt", "--max-traces",
                       "Maximum number of edge traces per coarse face.");
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
        args.AddOption(&num_iso_verts, "--num-iso-verts", "--num-iso-verts",
                       "Number of isolated vertices.");
        args.AddOption(&rescale_iter, "--rescale-iter", "--rescale-iter",
                       "Number of iteration to compute rescale vector in hybridization.");
    }
};

/**
   Collection of relation tables concerning degrees of freedom aggregations

   @param agg_vdof_ aggregate to vertex-based dof relation table
   @param agg_edof_ aggregate to edge-based dof relation table
   @param face_edof_ face to edge-based dof relation table
   @param topology_ associated topology
*/
struct DofAggregate
{
    mfem::SparseMatrix agg_vdof_;
    mfem::SparseMatrix face_edof_;
    mfem::SparseMatrix agg_edof_;  // the edofs here belong to one and only one agg
    const GraphTopology* topology_;

    DofAggregate(const GraphTopology& topology, const GraphSpace& space)
        : agg_vdof_(smoothg::Mult(topology.Agg_vertex_, space.VertexToVDof())),
          face_edof_(smoothg::Mult(topology.face_edge_, space.EdgeToEDof())),
          agg_edof_(DropSmall(smoothg::Mult(topology.Agg_vertex_, space.VertexToEDof()), 1.5)),
          topology_(&topology)
    {
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
       @param dof_agg container for various dofs aggregation
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
        const Graph& coarse_graph,
        const DofAggregate& dof_agg,
        const UpscaleParameters& param);

    ~LocalMixedGraphSpectralTargets() {}

    /**
       @brief Compute spectral vectex targets for each aggregate

       Solve an eigenvalue problem on each extended aggregate, restrict eigenvectors
       to original aggregate and use SVD to remove linear dependence, put resulting
       vectors in local_vertex_targets as column vectors.

       Put edge traces into ExtAgg_sigmaT as row vectors
       Trace generation depends on dual_target_, scaled_dual_, and energy_dual_

       @param local vertex targets in each aggregate
    */
    std::vector<mfem::DenseMatrix> ComputeVertexTargets();

    /**
       @brief Compute edge trace targets for each coarse face

       Given edge traces in aggregates (from ExtAgg_sigmaT), restrict them to coarse
       face, put those restricted vectors as well as some kind of PV vector into
       local_edge_trace_targets after SVD.

       @return local edge trace targets on each face
    */
    std::vector<mfem::DenseMatrix> ComputeEdgeTargets(
            const std::vector<mfem::DenseMatrix>& local_vertex_targets);
private:
    enum DofType { VDOF, EDOF }; // vertex-based and edge-based dofs

    /// Build extended aggregates to vertex-based and edge-based dofs relation
    void BuildExtendedAggregates(const GraphSpace& space);

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
    mfem::Vector** CollectConstant(const mfem::Vector& constant_vect,
                                   const mfem::SparseMatrix& agg_vdof);

    /// shared_constant expected to be an array of legth 2, just returns them
    /// stacked on top of each other
    mfem::Vector ConstantLocal(mfem::Vector* shared_constant);

    MPI_Comm comm_;

    const double rel_tol_;
    const int max_loc_vdofs_;
    const int max_loc_edofs_;
    const bool dual_target_;
    const bool scaled_dual_;
    const bool energy_dual_;

    const MixedMatrix& mgL_;
    const mfem::Vector& constant_rep_;

    const Graph& coarse_graph_;
    const DofAggregate& dof_agg_;
    const double zero_eigenvalue_threshold_;

    /// Extended aggregate to vertex dof relation table
    std::unique_ptr<mfem::HypreParMatrix> ExtAgg_vdof_;
    mfem::SparseMatrix ExtAgg_vdof_diag_;
    mfem::SparseMatrix ExtAgg_vdof_offd_;

    /// Extended aggregate to edge dof relation table
    std::unique_ptr<mfem::HypreParMatrix> ExtAgg_edof_;
    mfem::SparseMatrix ExtAgg_edof_diag_;
    mfem::SparseMatrix ExtAgg_edof_offd_;

    /// face to extended edge dof relation table
    std::unique_ptr<mfem::HypreParMatrix> face_ext_edof_;

    /// Transpose of M^{-1}D^T to vertex targets (candidates for traces)
    std::vector<mfem::DenseMatrix> ExtAgg_sigmaT_;

    mfem::Array<int> col_map_;
};

} // namespace smoothg

#endif
