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

set(EI_INCLUDES "")
set(EI_LIBRARIES "")

# Hypre
find_path(HYPRE_INCLUDE_PATH HYPRE.h
          HINTS ${HYPRE_DIR}/include)
list(APPEND EI_INCLUDES ${HYPRE_INCLUDE_PATH})
set(HYPRE_LIB_NAME libHYPRE.a)
find_library(HYPRE_LIB HYPRE
  ${HYPRE_DIR}/lib)
list(APPEND EI_LIBRARIES ${HYPRE_LIB})

# Metis
find_path(METIS_INCLUDE_PATH metis.h
  HINTS ${METIS_DIR}/include)
set(METIS_LIB_NAME libmetis.a)
find_path(METIS_LIBRARY_PATH ${METIS_LIB_NAME}
  ${METIS_DIR}/lib)
list(APPEND EI_INCLUDES ${METIS_INCLUDE_PATH})
add_library(METIS_LIB STATIC IMPORTED)
set_property(TARGET METIS_LIB PROPERTY IMPORTED_LOCATION ${METIS_LIBRARY_PATH}/${METIS_LIB_NAME})
list(APPEND EI_LIBRARIES ${METIS_LIBRARY_PATH}/${METIS_LIB_NAME})

# SuiteSparse
find_package(SuiteSparse REQUIRED UMFPACK KLU AMD BTF CHOLMOD COLAMD CAMD CCOLAMD config)
list(APPEND EI_INCLUDES ${SuiteSparse_INCLUDE_DIRS})
list(APPEND EI_LIBRARIES ${SuiteSparse_LIBRARIES})

# BLAS/LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
list(APPEND EI_LIBRARIES ${LAPACK_LIBRARIES})
list(APPEND EI_LIBRARIES ${BLAS_LIBRARIES})
list(APPEND EI_LIBRARIES "gfortran")

list(REMOVE_DUPLICATES EI_LIBRARIES)
