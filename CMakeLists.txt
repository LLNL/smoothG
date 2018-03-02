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

cmake_minimum_required(VERSION 2.6)

set(PROJECT_NAME smoothG)
project(${PROJECT_NAME})
enable_testing()

set(HOME_DIR $ENV{HOME})
set(${PROJECT_NAME}_CMAKE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH ${${PROJECT_NAME}_CMAKE_PATH}/modules)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DSMOOTHG_DEBUG")

set(${PROJECT_NAME}_GRAPHDATA ${PROJECT_SOURCE_DIR}/graphdata)

#####
# set up external dependencies
# (this is a mess, seems to work, but I don't know what
#  I'm doing and I'm sure it can be greatly simplified)
# (also should take some directory hints from config.sh)
#####

# This is a list of TPLs that are used by all targets
set(TPL_LIBRARIES "")

# This is a list of linker flags to be used with TPLs for all targets
set(TPL_LINKER_FLAGS "")

# SAAMGE_PARALLEL
option(USE_SAAMGE "Should SAAMGE be enabled?" OFF)
if(USE_SAAMGE)
  find_path(SAAMGE_INCLUDE_PATH saamge.hpp
    HINTS ${SAAMGE_DIR}/include $ENV{SAAMGE_DIR}/include)
  set(SAAMGE_LIB_NAME libsaamge.a)
  find_path(SAAMGE_LIBRARY_PATH ${SAAMGE_LIB_NAME}
             HINTS ${SAAMGE_DIR}
                   ${SAAMGE_DIR}/lib
                   $ENV{SAAMGE_DIR}/lib)
  include_directories(${SAAMGE_INCLUDE_PATH})
  add_library(SAAMGE_LIB STATIC IMPORTED)
  set_property(TARGET SAAMGE_LIB PROPERTY IMPORTED_LOCATION
    ${SAAMGE_LIBRARY_PATH}/${SAAMGE_LIB_NAME})
  set(${PROJECT_NAME}_USE_SAAMGE 1)
  list(APPEND TPL_LIBRARIES ${SAAMGE_LIBRARY_PATH}/${SAAMGE_LIB_NAME})
else()
  set(${PROJECT_NAME}_USE_SAAMGE 0)
endif()

# MFEM
find_path(MFEM_INCLUDE_PATH mfem.hpp
          HINTS ${MFEM_DIR}/include)
set(MFEM_LIB_NAME libmfem.a)
find_path(MFEM_LIBRARY_PATH ${MFEM_LIB_NAME}
  ${MFEM_DIR}/lib)
include_directories(${MFEM_INCLUDE_PATH})
add_library(MFEM_LIB STATIC IMPORTED)
set_property(TARGET MFEM_LIB PROPERTY IMPORTED_LOCATION ${MFEM_LIBRARY_PATH}/${MFEM_LIB_NAME})
list(APPEND TPL_LIBRARIES ${MFEM_LIBRARY_PATH}/${MFEM_LIB_NAME})

# Hypre
find_path(HYPRE_INCLUDE_PATH HYPRE.h
          HINTS ${HYPRE_DIR}/include)
include_directories(${HYPRE_INCLUDE_PATH})
set(HYPRE_LIB_NAME libHYPRE.a)
find_library(HYPRE_LIB HYPRE
  ${HYPRE_DIR}/lib)
list(APPEND TPL_LIBRARIES ${HYPRE_LIB})

# MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
list(APPEND TPL_LIBRARIES ${MPI_LIBRARIES})
if(MPI_CXX_COMPILE_FLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
endif()

# Metis
find_path(METIS_INCLUDE_PATH metis.h
  HINTS ${METIS_DIR}/include)
set(METIS_LIB_NAME libmetis.a)
find_path(METIS_LIBRARY_PATH ${METIS_LIB_NAME}
  ${METIS_DIR}/lib)
include_directories(${METIS_INCLUDE_PATH})
add_library(METIS_LIB STATIC IMPORTED)
set_property(TARGET METIS_LIB PROPERTY IMPORTED_LOCATION ${METIS_LIBRARY_PATH}/${METIS_LIB_NAME})
list(APPEND TPL_LIBRARIES ${METIS_LIBRARY_PATH}/${METIS_LIB_NAME})

# SuiteSparse
find_package(SuiteSparse REQUIRED UMFPACK KLU AMD BTF CHOLMOD COLAMD CAMD CCOLAMD config)
include_directories(${SuiteSparse_INCLUDE_DIRS})
list(APPEND TPL_LIBRARIES ${SuiteSparse_LIBRARIES})

# ARPACK
# there are several sources for ARPACK, I recommend
#    https://github.com/opencollab/arpack-ng.git
# and
#    https://github.com/m-reuter/arpackpp.git
option(USE_ARPACK "Should ARPACK be enabled?" ON)
if(USE_ARPACK)
  find_path(ARPACK_INCLUDE_PATH argsym.h
    HINTS ${ARPACKPP_DIR}/include $ENV{ARPACK_DIR}/include)
  set(ARPACK_LIB_NAME libarpack.a)
  if(APPLE)
    set(ARPACK_LIB_NAME libarpack.dylib)
  endif(APPLE)
  find_path(ARPACK_LIBRARY_PATH ${ARPACK_LIB_NAME}
             HINTS ${ARPACK_DIR}
                   ${ARPACK_DIR}/lib64
                   $ENV{ARPACK_DIR}/lib)
  include_directories(${ARPACK_INCLUDE_PATH})
  add_library(ARPACK_LIB STATIC IMPORTED)
  set_property(TARGET ARPACK_LIB PROPERTY IMPORTED_LOCATION
    ${ARPACK_LIBRARY_PATH}/${ARPACK_LIB_NAME})
  set(${PROJECT_NAME}_USE_ARPACK 1)
  list(APPEND TPL_LIBRARIES ${ARPACK_LIBRARY_PATH}/${ARPACK_LIB_NAME})
else()
  set(${PROJECT_NAME}_USE_ARPACK 0)
endif()

# BLAS/LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
list(APPEND TPL_LIBRARIES ${LAPACK_LIBRARIES})
list(APPEND TPL_LIBRARIES ${BLAS_LIBRARIES})
list(APPEND TPL_LIBRARIES "gfortran")

list(REMOVE_DUPLICATES TPL_LIBRARIES)

# SPE10 data (not a package, but we want it for testing)
# (this file is available in http://www.spe.org/web/csp/datasets/por_perm_case2a.zip)
find_file(SPE10_PERM spe_perm.dat
  HINTS ${SPE10_DIR})

# for valgrind memory testing
find_program(MEMORYCHECK_COMMAND valgrind)
set(MEMORYCHECK_COMMAND_OPTIONS "")
list(APPEND MEMORYCHECK_COMMAND_OPTIONS "--leak-check=full")
list(APPEND MEMORYCHECK_COMMAND_OPTIONS "--error-exitcode=1")
function(add_valgrind_test name binary)
  add_test(${name} ${MEMORYCHECK_COMMAND} ${MEMORYCHECK_COMMAND_OPTIONS} ./${binary})
endfunction(add_valgrind_test)

#####
# documentation
#####

# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${PROJECT_SOURCE_DIR}/Doxyfile.in ${PROJECT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif(DOXYGEN_FOUND)

#####
# Astyle
#####

find_program(ASTYLE_COMMAND astyle HINTS ${ASTYLE_DIR})

# add a target to format code
add_custom_target(style
    ${ASTYLE_COMMAND} --options=smoothg.astylerc src/*.?pp examples/*.?pp testcode/*.?pp
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Formating source code" VERBATIM
    )

#####
# export configuration of optional packages
#####
configure_file(
  "${PROJECT_SOURCE_DIR}/${PROJECT_NAME}_config.h.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}_config.h"
  )
include_directories(${PROJECT_BINARY_DIR}) # for _config.h

#####
# generate lib file
#####
add_subdirectory(src)

#####
# actual executables and tests
#####
add_subdirectory(examples)
add_subdirectory(testcode)
