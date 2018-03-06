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

   @brief  SpectralAMG_MGL_Coarsener is a concrete realization of Mixed_GL_Coarsener
*/

#ifndef __SPECTRALAMG_MGL_COARSENER_HPP__
#define __SPECTRALAMG_MGL_COARSENER_HPP__

#include "Mixed_GL_Coarsener.hpp"
#include "GraphTopology.hpp"
#include "GraphCoarsen.hpp"
#include "mfem.hpp"
#include <memory>

namespace smoothg
{

/**
   @brief Use spectral AMG to coarsen the mixed graph Laplacian.
*/
class SpectralAMG_MGL_Coarsener : public Mixed_GL_Coarsener
{
public:
    /**
       @brief Build a coarsener based on spectral AMG.

       @param mgL the actual mixed graph Laplacian
       @param gt the topology describing how vertices and edges are agglomerated
       @param spectral_tol number specifying how small eigenvalues must be before they
              are included in the coarse space. A larger number here leads to a more
              accurate, more expensive coarse space.
       @param max_evecs_per_agg cap the number of eigenvectors per aggregate that will
              be used in the coarse space.
       @param dual_target get traces from eigenvectors of dual graph Laplacian
       @param scaled_dual scale dual graph Laplacian by inverse edge weight.
              Typically coarse problem gets better accuracy but becomes harder
              to solve when this option is turned on.
       @param energy_dual use energy matrix in (RHS of) dual graph eigen problem
              (guarantees approximation property in edge energy norm)
       @param is_hybridization_used whether to prepare the coarse space to use the
              HybridSolver
    */
    SpectralAMG_MGL_Coarsener(const MixedMatrix& mgL,
                              const GraphTopology& gt,
                              double spectral_tol,
                              unsigned int max_evecs_per_agg,
                              bool dual_target,
                              bool scaled_dual,
                              bool energy_dual,
                              bool is_hybridization_used);

    SpectralAMG_MGL_Coarsener(const MixedMatrix& mgL, const GraphTopology& gt,
                              const SpectralCoarsenParam& param);

private:
    /**
       @brief Coarsen the graph, constructing projectors, coarse operators, etc.
    */
    void do_construct_coarse_subspace();

private:
    // TODO: make it a const reference if the first constructor is not kept
    SpectralCoarsenParam coarsen_param_;
}; // SpectralAMG_MGL_Coarsener

} // namespace smoothg

#endif /* __SPECTRALAMG_MGL_COARSENER_HPP__ */
