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

TMP_DIR=/tmp/mpich

mkdir -p $TMP_DIR
cd $TMP_DIR

wget http://www.mpich.org/static/downloads/3.2.1/mpich-3.2.1.tar.gz
tar xzf mpich-3.2.1.tar.gz
cd mpich-3.2.1

./configure --disable-fortran --prefix=$INSTALL_DIR/mpich
make -j3
make install

rm -r $TMP_DIR
