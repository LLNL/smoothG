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
    */
    SpectralAMG_MGL_Coarsener(const MixedMatrix& mgL,
                              GraphTopology gt,
                              const UpscaleParameters& param = UpscaleParameters());

private:
    /**
       @brief Coarsen the graph, constructing projectors, coarse operators, etc.

       @param constant_rep representation of constant on finer level
    */
    void do_construct_coarse_subspace(const mfem::Vector& constant_rep);

private:
    const UpscaleParameters& param_;
}; // SpectralAMG_MGL_Coarsener

} // namespace smoothg

#endif /* __SPECTRALAMG_MGL_COARSENER_HPP__ */
