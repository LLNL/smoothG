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

# Downloads and builds SuiteSparse
if [ -z "$INSTALL_DIR" ]; then INSTALL_DIR=${PWD}/extern; fi

TMP_DIR=/tmp/hypre

mkdir -p $TMP_DIR
cd $TMP_DIR

wget https://computation.llnl.gov/project/linear_solvers/download/hypre-2.10.0b.tar.gz --no-check-certificate
tar xzf hypre-2.10.0b.tar.gz
cd hypre-2.10.0b/src

./configure --disable-fortran --without-fei CC=mpicc CXX=mpic++ prefix=$INSTALL_DIR/hypre
make -j3
make install

rm -r $TMP_DIR
