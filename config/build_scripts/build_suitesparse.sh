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
if [ -z "$BLAS_LIB" ]; then BLAS_LIB=-lblas; fi
if [ -z "$METIS_DIR" ]; then METIS_DIR=$INSTALL_DIR/metis; fi


TMP_DIR=/tmp/suitesparse

mkdir -p $TMP_DIR
cd $TMP_DIR

wget http://faculty.cse.tamu.edu/davis/SuiteSparse/SuiteSparse-4.5.4.tar.gz
tar xzf SuiteSparse-4.5.4.tar.gz
cd SuiteSparse

make -j3 install \
INSTALL=$INSTALL_DIR/SuiteSparse \
BLAS=$BLAS_LIB \
CFOPENMP="" \
MY_METIS_LIB="-L$METIS_DIR/lib -lmetis" \
MY_METIS_INC=$METIS_DIR/include/

rm -r $TMP_DIR
