<!-- BHEADER ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 +
 + Copyright (c) 2018, Lawrence Livermore National Security, LLC.
 + Produced at the Lawrence Livermore National Laboratory.
 + LLNL-CODE-745247. All Rights reserved. See file COPYRIGHT for details.
 +
 + This file is part of smoothG. For more information and source code
 + availability, see https://www.github.com/llnl/smoothG.
 +
 + smoothG is free software; you can redistribute it and/or modify it under the
 + terms of the GNU Lesser General Public License (as published by the Free
 + Software Foundation) version 2.1 dated February 1999.
 +
 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ EHEADER -->

Installing smoothG            {#INSTALL}
==========

The following instructions will install smoothG and all of its
dependencies.

# Dependencies:

* blas
* lapack
* [metis-5.1.0](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview)
* [hypre-2.10.0b](https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods/software)
* [SuiteSparse-4.5.4](http://faculty.cse.tamu.edu/davis/suitesparse.html)
* [mfem-3.3](http://mfem.org/)

# Build Dependencies:

These instructions will build dependencies in the your home folder: `${HOME}`

If not installing in standard system locations (`/usr/`, `/usr/local/`, etc),
you will need to export the appropriate `LIBRARY_PATH` and `LD_LIBRARY_PATH`
so that the linker/loader can find them.

For example the final `LIBRARY_PATH` will look like:

    export LD_LIBRARY_PATH=${HOME}/local/lib:$LD_LIBRARY_PATH


## blas

    check if exists or install from package manager

## lapack

    check if exists or install from package manager

## metis-5.1.0

    tar -xvzf metis-5.1.0.tar.gz
    cd metis-5.1.0

    make config prefix=${HOME}/metis
    make install

## hypre-2.10.0b

    tar -xvfz hypre-2.10.0b.gz
    cd hypre-2.10.0b/src

    ./configure --disable-fortran --prefix=${HOME}/hypre
    make install

## SuiteSparse-4.5.4

    tar -xvfz SuiteSparse-4.5.4.tar.gz
    cd SuiteSparse-4.5.4

    make install BLAS=/usr/lib64/libblas.so.3 LAPACK=/usr/lib64/liblapack.so.3 \
        INSTALL=${HOME}/SuiteSparse

    #(Replace blas and lapack library locations appropriately)

## mfem-3.3

    tar -xvzf mfem-3.3.tar.gz
    cd mfem-3.3

    make config

Choose one of the following:

    edit config/config.mk with the correct parameters

    make parallel
    make install

--or--

    make parallel \
        PREFIX=${HOME}/mfem \
        MFEM_USE_METIS_5=YES \
        MFEM_USE_LAPACK=YES \
        MFEM_USE_SUITESPARSE=YES \
        HYPRE_DIR=${HOME}/hypre \
        SUITESPARSE_DIR=${HOME}/SuiteSparse  \
        METIS_DIR=${HOME}/metis \
        LAPACKLIB="/usr/lib64/liblapack.so.3 /usr/lib64/libblas.so.3"
    make install


# Optional Dependencies:

* [SPE10 dataset](http://www.spe.org/web/csp/datasets/set02.htm)
* [Valgrind](http://valgrind.org/)

## spe10 dataset

    unzip por_perm_case2a.zip -d ${HOME}/spe10

## Valgrind

    tar -xvf valgrind-3.12.0
    cd valgrind-3.12.0

    ./configure --prefix=${HOME}/valgrind
    make
    make install

# Build smoothG

Clone the smoothG repo and cd into smoothG directory.

To build smoothG, either copy, modify, and run a config file from config/
or pass the parameters directly to cmake:

    mkdir -p build
    cd build

    cmake \
        -DMFEM_DIR=${HOME}/mfem \
        -DMETIS_DIR=${HOME}/metis \
        -DHYPRE_DIR=${HOME}/hypre \
        -DSuiteSparse_DIR=${HOME}/SuiteSparse \
        -DCMAKE_BUILD_TYPE=DEBUG \
        -DBLAS_LIBRARIES=/usr/lib64/libblas.so.3 \
        -DLAPACK_LIBRARIES=/usr/lib64/liblapack.so.3 \
        -DSPE10_DIR=${HOME}/spe10 \
        ..

    make
    make test
    make doc

# Notes:

Metis gives you the option of choosing between float and double
as your real type by altering the REALTYPEWIDTH constant in
metis.h. To pass our tests, you need to have REALTYPEWIDTH set to 32
(resulting in float for your real type).

