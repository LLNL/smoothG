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

    @brief Implements SpectralAMG_MGL_Coarsener object.
*/

#ifndef __SPECTRALAMG_MGL_COARSENER_IMPL_HPP__
#define __SPECTRALAMG_MGL_COARSENER_IMPL_HPP__

#include "SpectralAMG_MGL_Coarsener.hpp"
#include "LocalMixedGraphSpectralTargets.hpp"
#include "GraphCoarsen.hpp"

using std::unique_ptr;

namespace smoothg
{

SpectralAMG_MGL_Coarsener::SpectralAMG_MGL_Coarsener(const UpscaleParameters& param)
    : Mixed_GL_Coarsener(), param_(param)
{
}

MixedMatrix SpectralAMG_MGL_Coarsener::do_construct_coarse_subspace(
    const MixedMatrix& mgL, const mfem::Array<int>* partitioning)
{
    GraphTopology topology(mgL.GetGraph());
    std::shared_ptr<Graph> coarse_graph = partitioning ? topology.Coarsen(*partitioning)
                                          : topology.Coarsen(param_.coarse_factor);

    DofAggregate dof_agg(topology, mgL.GetGraphSpace());

    std::vector<mfem::DenseMatrix> local_edge_traces;
    std::vector<mfem::DenseMatrix> local_spectral_vertex_targets;

    LocalMixedGraphSpectralTargets localtargets(mgL, dof_agg, param_);
    localtargets.Compute(local_edge_traces, local_spectral_vertex_targets);

    GraphCoarsen graph_coarsen(mgL, dof_agg);
    GraphSpace coarse_space = graph_coarsen.BuildCoarseSpace(local_edge_traces,
                                                             local_spectral_vertex_targets,
                                                             std::move(*coarse_graph));

    graph_coarsen.BuildInterpolation(local_edge_traces,
                                     local_spectral_vertex_targets,
                                     coarse_space, param_.coarse_components,
                                     Pu_, Psigma_);

    auto tmp = graph_coarsen.BuildEdgeProjection(local_edge_traces,
                                                 local_spectral_vertex_targets,
                                                 coarse_space);
    Proj_sigma_.Swap(tmp);

#ifdef SMOOTHG_DEBUG
    Debug_tests(mgL.GetD());
#endif

    return graph_coarsen.BuildCoarseMatrix(std::move(coarse_space), mgL, Pu_);
}

} // namespace smoothg

#endif /* __SPECTRALAMG_MGL_COARSENER_IMPL_HPP__ */
