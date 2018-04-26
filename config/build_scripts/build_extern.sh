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

# Builds all external libraries, Metis, SuiteSparse, MPICH, and Hypre
# into the extern directory

if [ -z "$INSTALL_DIR" ]; then INSTALL_DIR=${PWD}/extern; fi
mkdir -p ${INSTALL_DIR}

export PATH=$INSTALL_DIR/mpich/bin:$PATH
INSTALL_DIR=$INSTALL_DIR sh config/build_scripts/build_mpich.sh
INSTALL_DIR=$INSTALL_DIR sh config/build_scripts/build_metis.sh
INSTALL_DIR=$INSTALL_DIR sh config/build_scripts/build_suitesparse.sh
INSTALL_DIR=$INSTALL_DIR sh config/build_scripts/build_hypre.sh
INSTALL_DIR=$INSTALL_DIR sh config/build_scripts/build_linalg.sh
