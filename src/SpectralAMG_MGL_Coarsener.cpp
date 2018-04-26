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
                                                     std::unique_ptr<GraphTopology> gt,
                                                     double spectral_tol,
                                                     unsigned int max_evecs_per_agg,
                                                     bool dual_target,
                                                     bool scaled_dual,
                                                     bool energy_dual,
                                                     bool coarse_coefficient)
    : Mixed_GL_Coarsener(mgL, std::move(gt)),
      is_hybridization_used_(false),
      spectral_tol_(spectral_tol),
      max_evecs_per_agg_(max_evecs_per_agg),
      dual_target_(dual_target),
      scaled_dual_(scaled_dual),
      energy_dual_(energy_dual),
      coarse_coefficient_(coarse_coefficient)
{
}

void SpectralAMG_MGL_Coarsener::do_construct_coarse_subspace()
{
    using LMGST = LocalMixedGraphSpectralTargets;

    std::vector<mfem::DenseMatrix> local_edge_traces;
    std::vector<mfem::DenseMatrix> local_spectral_vertex_targets;

    LMGST localtargets(spectral_tol_, max_evecs_per_agg_, dual_target_,
                       scaled_dual_, energy_dual_, mgL_.GetM(),
                       mgL_.GetD(), mgL_.GetW(), *graph_topology_);
    localtargets.Compute(local_edge_traces, local_spectral_vertex_targets);

    if (coarse_coefficient_)
    {
        coarse_m_builder_ = make_unique<CoefficientMBuilder>(*graph_topology_);
    }
    else
    {
        coarse_m_builder_ = make_unique<ElementMBuilder>();
    }

    graph_coarsen_->BuildInterpolation(local_edge_traces,
                                       local_spectral_vertex_targets,
                                       Pu_, Psigma_, face_facedof_table_,
                                       *coarse_m_builder_);

    CoarseD_ = graph_coarsen_->GetCoarseD();
    CoarseW_ = graph_coarsen_->GetCoarseW();
}

} // namespace smoothg

#endif /* __SPECTRALAMG_MGL_COARSENER_IMPL_HPP__ */
