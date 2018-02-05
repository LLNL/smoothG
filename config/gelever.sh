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


# This is the path to the root of the git repo
# the BASE_DIR should contain smoothG_config.h.in
BASE_DIR=${PWD}

# this is where we actually build binaries and so forth
BUILD_DIR=${BASE_DIR}/build

EXTRA_ARGS=$@

mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Force a reconfigure
rm CMakeCache.txt
rm -rf CMakeFiles

cmake \
    -DSPARSESOLVE_DIR=${HOME}/Code/umfpack \
    -DPARTITION_DIR=${HOME}/Code/partition \
    -DLINALGCPP_DIR=${HOME}/Code/linalgcpp \
    -DPARLINALGCPP_DIR=${HOME}/Code/parlinalgcpp \
    -DHYPRE_DIR=${HOME}/hypre \
    -DSuiteSparse_DIR=${HOME}/SuiteSparse \
    -DSPE10_DIR=${HOME}/spe10 \
    -DCMAKE_BUILD_TYPE=Debug \
    -DUSE_ARPACK=OFF \
    ${EXTRA_ARGS} \
    ${BASE_DIR}
