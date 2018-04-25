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

# Downloads and builds Linear algebra libraries
if [ -z "$INSTALL_DIR" ]; then INSTALL_DIR=${PWD}/extern; fi
if [ -z "$METIS_DIR" ]; then METIS_DIR=$INSTALL_DIR/metis; fi
if [ -z "$SUITESPARSE_DIR" ]; then SUITESPARSE_DIR=$INSTALL_DIR/SuiteSparse; fi
if [ -z "$HYPRE_DIR" ]; then HYPRE_DIR=$INSTALL_DIR/hypre; fi

#############
# Linalgcpp #
#############

cd $INSTALL_DIR
git clone -b develop https://github.com/gelever/linalgcpp.git linalgcpp
cd linalgcpp
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/linalgcpp
make -j3 install

################
# ParLinalgcpp #
################

cd $INSTALL_DIR
git clone https://github.com/gelever/parlinalgcpp.git parlinalgcpp
cd parlinalgcpp
mkdir -p build && cd build
CC=mpicc CXX=mpic++ cmake .. \
    -DHypre_INC_DIR=$HYPRE_DIR/include \
    -DHypre_LIB_DIR=$HYPRE_DIR/lib \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/parlinalgcpp
make -j3 install

#############
# Partition #
#############

cd $INSTALL_DIR
git clone https://github.com/gelever/partition.git partition
cd partition
mkdir -p build && cd build
cmake .. \
    -DMETIS_DIR=$METIS_DIR \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/partition
make -j3 install

###############
# SparseSolve #
###############

cd $INSTALL_DIR
git clone https://github.com/gelever/sparsesolver.git sparsesolve
cd sparsesolve
mkdir -p build && cd build
cmake .. \
    -DSUITESPARSE_INCLUDE_DIR_HINTS=$SUITESPARSE_DIR/include \
    -DSUITESPARSE_LIBRARY_DIR_HINTS=$SUITESPARSE_DIR/lib \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/sparsesolve
make -j3 install
