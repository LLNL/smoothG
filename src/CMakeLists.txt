#!/bin/sh
# BHEADER ####################################################################
#
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# LLNL-CODE-745247. All Rights reserved. See file COPYRIGHT for details.
#
# This file is part of smoothG. For more information and source code
# availability, see https://www.github.com/llnl/smoothG.
#
# smoothG is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
#################################################################### EHEADER #

#####
# generate lib file
#####

add_library(smoothg LocalMixedGraphSpectralTargets.cpp GraphCoarsenBuilder.cpp
  GraphCoarsen.cpp utilities.cpp MinresBlockSolver.cpp HybridSolver.cpp
  sharedentitycommunication.cpp GraphTopology.cpp Mixed_GL_Coarsener.cpp
  SpectralAMG_MGL_Coarsener.cpp MetisGraphPartitioner.cpp MatrixUtilities.cpp
  MixedMatrix.cpp LocalEigenSolver.cpp GraphGenerator.cpp GraphUpscale.cpp FiniteVolumeUpscale.cpp
  FiniteVolumeMLMC.cpp Upscale.cpp MixedLaplacianSolver.cpp)

