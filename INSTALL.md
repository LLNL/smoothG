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
* [hypre-2.15.1](https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods/software)
* [SuiteSparse-4.5.4](http://faculty.cse.tamu.edu/davis/suitesparse.html)
* [mfem-3.4](http://mfem.org/)


# Build Dependencies:

These instructions will build dependencies in the your home folder: `${HOME}`

## blas

Check if you already have this, and if not, install from a package manager

## lapack

Check if you already have this, and if not, install from a package manager

## metis-5.1.0

Download with a browser from [here](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) or type:

    wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
    
Then:
    
    tar xvzf metis-5.1.0.tar.gz
    cd metis-5.1.0
    make config prefix=${HOME}/metis-install
    make install
    cd ..

## hypre-2.15.1

    git clone https://github.com/hypre-space/hypre.git hypre
    cd hypre
    git checkout v2.15.1
    cd src
    ./configure --disable-fortran CC=mpicc CXX=mpicxx prefix=${HOME}/hypre-install
    make
    make install
    cd ../..

## SuiteSparse-4.5.4

Download with a browser from [here](http://faculty.cse.tamu.edu/davis/suitesparse.html) or type:

    wget http://faculty.cse.tamu.edu/davis/SuiteSparse/SuiteSparse-4.5.4.tar.gz

Then:

    tar xvzf SuiteSparse-4.5.4.tar.gz
    cd SuiteSparse-4.5.4
    make install INSTALL=${HOME}/suitesparse-install BLAS=-lblas MY_METIS_LIB=${HOME}/metis-install/lib/libmetis.a \
        MY_METIS_INC=${HOME}/metis-install/include/
    cd ..

## mfem-3.4

    git clone https://github.com/mfem/mfem.git mfem
    cd mfem
    git checkout v3.4
    make config MFEM_USE_METIS_5=YES MFEM_USE_LAPACK=YES MFEM_USE_SUITESPARSE=YES MFEM_USE_MPI=YES \
        HYPRE_DIR=${HOME}/hypre-install SUITESPARSE_DIR=${HOME}/suitesparse-install METIS_DIR=${HOME}/metis-install \
        PREFIX=${HOME}/mfem-install
    CC=mpicc CXX=mpicxx make install


# Optional Dependencies:

* [SPE10 dataset](http://www.spe.org/web/csp/datasets/set02.htm)
* [Valgrind](http://valgrind.org/)

## spe10 dataset

Available at [this page](https://www.spe.org/web/csp/datasets/set02.htm)

    unzip por_perm_case2a.zip -d ${HOME}/spe10

## Valgrind

Some of our tests require valgrind, you can probably get it from your package manager or [here](https://valgrind.org/). Then:

    tar xvf valgrind-3.12.0
    cd valgrind-3.12.0
    ./configure --prefix=${HOME}/valgrind-install
    make
    make install

# Build smoothG

Clone the smoothG repo and `cd` into smoothG directory.

To build smoothG, either copy, modify, and run a config file from config/
or pass the parameters directly to cmake:

    mkdir -p build
    cd build

    cmake \
        -DMFEM_DIR=${HOME}/mfem-install \
        -DMETIS_DIR=${HOME}/metis-install \
        -DHYPRE_DIR=${HOME}/hypre-install \
        -DSuiteSparse_DIR=${HOME}/suitesparse-install \
        -DCMAKE_BUILD_TYPE=DEBUG \
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

