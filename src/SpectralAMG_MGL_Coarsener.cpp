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

SpectralAMG_MGL_Coarsener::SpectralAMG_MGL_Coarsener(const MixedMatrix& mgL,
                                                     const GraphTopology& gt,
                                                     double spectral_tol,
                                                     unsigned int max_evecs_per_agg,
                                                     bool dual_target,
                                                     bool scaled_dual,
                                                     bool energy_dual,
                                                     bool is_hybridization_used)
    : Mixed_GL_Coarsener(mgL, gt)
{
    coarsen_param_.hybridization = is_hybridization_used;
    coarsen_param_.spec_tol = spectral_tol;
    coarsen_param_.max_evects = max_evecs_per_agg;
    coarsen_param_.dual_target = dual_target;
    coarsen_param_.scaled_dual = scaled_dual;
    coarsen_param_.energy_dual = energy_dual;
}

SpectralAMG_MGL_Coarsener::SpectralAMG_MGL_Coarsener(
        const MixedMatrix& mgL, const GraphTopology& gt, const SpectralCoarsenParam& param)
    : Mixed_GL_Coarsener(mgL, gt), coarsen_param_(param)
{
}

void SpectralAMG_MGL_Coarsener::do_construct_coarse_subspace()
{
    std::vector<mfem::DenseMatrix> local_edge_traces;
    std::vector<mfem::DenseMatrix> local_spectral_vertex_targets;

    LocalMixedGraphSpectralTargets localtargets(mgL_, graph_topology_, coarsen_param_);
    localtargets.Compute(local_edge_traces, local_spectral_vertex_targets);

    graph_coarsen_->BuildInterpolation(local_edge_traces,
                                       local_spectral_vertex_targets,
                                       Pu_, Psigma_, face_facedof_table_,
                                       CM_el_, coarsen_param_.hybridization);

    CoarseD_ = graph_coarsen_->GetCoarseD();
    CoarseM_ = graph_coarsen_->GetCoarseM();
    CoarseW_ = graph_coarsen_->GetCoarseW();
}

} // namespace smoothg

#endif /* __SPECTRALAMG_MGL_COARSENER_IMPL_HPP__ */
