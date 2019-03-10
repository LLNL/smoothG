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

/** @file smoothG.hpp
    @brief Contains all header files for easy include
*/

#include "LocalMixedGraphSpectralTargets.hpp"
#include "GraphCoarsen.hpp"
#include "utilities.hpp"
#include "HybridSolver.hpp"
#include "MixedMatrix.hpp"
#include "GraphTopology.hpp"
#include "MinresBlockSolver.hpp"
#include "MetisGraphPartitioner.hpp"
#include "MatrixUtilities.hpp"
#include "GraphGenerator.hpp"
#include "Upscale.hpp"
#include "UpscaleOperators.hpp"
#include "Graph.hpp"
#include "Sampler.hpp"
#include "GraphSpace.hpp"
#include "MLMCManager.hpp"
#include "Hierarchy.hpp"
#include "NonlinearSolver.hpp"
