# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

# Sets the following variables
#   - SuiteSparse_FOUND
#   - SuiteSparse_INCLUDE_DIRS
#   - SuiteSparse_LIBRARIES
#
# We need the following libraries:
#   umfpack, cholmod, amd, camd, colamd, ccolamd, suitesparseconfig, klu, btf
#

include(MfemCmakeUtilities)
mfem_find_package(SuiteSparse SuiteSparse SuiteSparse_DIR "" "" "" ""
  "Paths to headers required by SuiteSparse."
  "Libraries required by SuiteSparse."
  ADD_COMPONENT "UMFPACK" "include;suitesparse" umfpack.h "lib" umfpack
  ADD_COMPONENT "KLU" "include;suitesparse" klu.h "lib" klu
  ADD_COMPONENT "AMD" "include;suitesparse" amd.h "lib" amd
  ADD_COMPONENT "BTF" "include;suitesparse" btf.h "lib" btf
  ADD_COMPONENT "CHOLMOD" "include;suitesparse" cholmod.h "lib" cholmod
  ADD_COMPONENT "COLAMD" "include;suitesparse" colamd.h "lib" colamd
  ADD_COMPONENT "CAMD" "include;suitesparse" camd.h "lib" camd
  ADD_COMPONENT "CCOLAMD" "include;suitesparse" ccolamd.h "lib" ccolamd
  ADD_COMPONENT "config" "include;suitesparse" SuiteSparse_config.h "lib"
    suitesparseconfig)

if (SuiteSparse_FOUND AND METIS_VERSION_5)
  message(STATUS " *** Warning: using SuiteSparse with METIS v5!")
endif()


# OLD CMAKE
# # BHEADER ####################################################################
# #
# # Copyright (c) 2018, Lawrence Livermore National Security, LLC.
# # Produced at the Lawrence Livermore National Laboratory.
# # LLNL-CODE-745247. All Rights reserved. See file COPYRIGHT for details.
# #
# # This file is part of smoothG. For more information and source code
# # availability, see https://www.github.com/llnl/smoothG.
# #
# # smoothG is free software; you can redistribute it and/or modify it under the
# # terms of the GNU Lesser General Public License (as published by the Free
# # Software Foundation) version 2.1 dated February 1999.
# #
# #################################################################### EHEADER #
# 
# # Sets the following variables
# #   - SuiteSparse_FOUND
# #   - SuiteSparse_INCLUDE_DIRS
# #   - SuiteSparse_LIBRARIES
# #
# # This is the one FindXXX.cmake module that ParELAG has that is
# # component-aware. That is, the following import targets are created,
# # each aware of its own dependencies:
# #
# # SuiteSparse::amd
# # SuiteSparse::btf
# # SuiteSparse::camd
# # SuiteSparse::ccolamd
# # SuiteSparse::cholmod
# # SuiteSparse::colamd
# # SuiteSparse::cxsparse
# # SuiteSparse::klu
# # SuiteSparse::ldl
# # SuiteSparse::rbio
# # SuiteSparse::spqr
# # SuiteSparse::suitesparseconfig
# # SuiteSparse::umfpack
# #
# # TODO: Actually use a "COMPONENTS" framework so that this module
# # becomes useful outside of ParELAG. See FindBoost or FindQt* for details
# 
# # My SuiteSparse looks like:
# #
# # libamd.a
# # libbtf.a
# # libcamd.a
# # libccolamd.a
# # libcholmod.a
# # libcolamd.a
# # libcxsparse.a
# # libklu.a
# # libldl.a
# # librbio.a
# # libspqr.a
# # libsuitesparseconfig.a
# # libumfpack.a
# #
# # RBio.h
# # SuiteSparseQR.hpp
# # SuiteSparse_config.h
# # amd.h
# # btf.h
# # camd.h
# # ccolamd.h
# # cholmod.h
# # colamd.h
# # cs.h
# # klu.h
# # ldl.h
# # spqr.hpp
# # umfpack.h
# 
# # The components that we need
# set(${PROJECT_NAME}_SUITESPARSE_COMPONENTS
#   umfpack klu btf amd colamd cholmod config)
# 
# # These are used only if they are found on the system
# set(${PROJECT_NAME}_SUITESPARSE_OPTIONAL_COMPONENTS
#   camd ccolamd)
# 
# # These are the full list
# set(${PROJECT_NAME}_SUITESPARSE_ALL_COMPONENTS
#   ${${PROJECT_NAME}_SUITESPARSE_COMPONENTS}
#   ${${PROJECT_NAME}_SUITESPARSE_OPTIONAL_COMPONENTS})
# 
# # SuiteSparse_Config is a little weird...
# set(config_HEADER_NAME "SuiteSparse_config.h")
# set(config_LIBRARY_NAME "suitesparseconfig")
# 
# 
# # Find and add the components
# foreach (component ${${PROJECT_NAME}_SUITESPARSE_ALL_COMPONENTS})
# 
#   # Set the header for the component
#   if (NOT ${component}_HEADER_NAME)
#     set(${component}_HEADER_NAME ${component}.h)
#   endif (NOT ${component}_HEADER_NAME)
# 
#   # Set the library for the component
#   if (NOT ${component}_LIBRARY_NAME)
#     set(${component}_LIBRARY_NAME ${component})
#   endif (NOT ${component}_LIBRARY_NAME)
# 
#   # Go searching for the header
#   message("Looking for " ${${component}_HEADER_NAME} " in " ${SuiteSparse_DIR})
#   find_path(${component}_INCLUDE_DIR ${${component}_HEADER_NAME}
#     HINTS ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR} ${SuiteSparse_INCLUDE_DIRS}
#     PATH_SUFFIXES suitesparse include
#     NO_DEFAULT_PATH
#     DOC "Directory where SuiteSparse component ${component} headers live.")
#   find_path(${component}_INCLUDE_DIR ${${component}_HEADER_NAME}
#     PATH_SUFFIXES suitesparse)
# 
#   # Add to the list of SuiteSparse headers
#   if (${component}_INCLUDE_DIR)
#     list(APPEND SuiteSparse_INCLUDE_DIRS ${${component}_INCLUDE_DIR})
#     list(REMOVE_DUPLICATES SuiteSparse_INCLUDE_DIRS)
#   endif (${component}_INCLUDE_DIR)
# 
#   # Go searching for the library
#   find_library(${component}_LIBRARY ${${component}_LIBRARY_NAME}
#     HINTS ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
#     PATH_SUFFIXES lib
#     NO_DEFAULT_PATH
#     DOC "The SuiteSparse component ${component} library.")
#   find_library(${component}_LIBRARY ${${component}_LIBRARY_NAME})
# 
#   if (${component}_LIBRARY AND ${component}_INCLUDE_DIR)
# 
#     # Setup the imported target
#     if (NOT TARGET SuiteSparse::${component})
#       # Check if we have shared or static libraries
#       include(CMakeUtilities)
#       parelag_determine_library_type(${${component}_LIBRARY} ${component}_LIB_TYPE)
# 
#       add_library(SuiteSparse::${component} ${${component}_LIB_TYPE} IMPORTED)
#     endif (NOT TARGET SuiteSparse::${component})
# 
#     # Set library
#     set_property(TARGET SuiteSparse::${component}
#       PROPERTY IMPORTED_LOCATION ${${component}_LIBRARY})
# 
#     # Add include path
#     set_property(TARGET SuiteSparse::${component} APPEND
#       PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${${component}_INCLUDE_DIR})
# 
#     # Add BLAS/LAPACK dependencies
#     set_property(TARGET SuiteSparse::${component} APPEND
#       PROPERTY INTERFACE_LINK_LIBRARIES
#       ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
# 
#     # Set the libraries
#     mark_as_advanced(FORCE ${component}_INCLUDE_DIR)
#     mark_as_advanced(FORCE ${component}_LIBRARY)
# 
#   else (${component}_LIBRARY AND ${component}_INCLUDE_DIR)
# 
#     list(FIND ${PROJECT_NAME}_SUITESPARSE_OPTIONAL_COMPONENTS
#       ${component} COMPONENT_IS_OPTIONAL)
# 
#     if (COMPONENT_IS_OPTIONAL EQUAL -1)
#       message(FATAL_ERROR "Required SuiteSparse component ${component} has not been found.")
#     else ()
#       message("WARNING: Optional SuiteSparse component ${component} has not been found.")
#     endif (COMPONENT_IS_OPTIONAL EQUAL -1)
#   endif (${component}_LIBRARY AND ${component}_INCLUDE_DIR)
# 
# endforeach (component ${${PROJECT_NAME}_SUITESPARSE_ALL_COMPONENTS})
# 
# #
# # Add dependencies
# #
# 
# # TODO: Finish figuring this out
# set(umfpack_DEPENDENCIES amd cholmod config)
# set(cholmod_DEPENDENCIES colamd camd ccolamd)
# set(klu_DEPENDENCIES btf amd)
# 
# # Set the dependencies
# foreach (component ${${PROJECT_NAME}_SUITESPARSE_COMPONENTS})
#   if (${component}_DEPENDENCIES)
#     foreach (dep ${${component}_DEPENDENCIES})
#       if (TARGET SuiteSparse::${dep})
#         set_property(TARGET SuiteSparse::${component} APPEND
#           PROPERTY INTERFACE_LINK_LIBRARIES SuiteSparse::${dep})
#       endif (TARGET SuiteSparse::${dep})
#     endforeach (dep ${${component}_DEPENDENCIES})
#   endif (${component}_DEPENDENCIES)
# endforeach (component ${${PROJECT_NAME}_SUITESPARSE_COMPONENTS})
# 
# #
# # Set the output LIBRARIES variable and cache INCLUDE_DIRS
# #
# set(SuiteSparse_LIBRARIES SuiteSparse::umfpack SuiteSparse::klu)
# 
# # Set the include directories
# set(SuiteSparse_INCLUDE_DIRS ${SuiteSparse_INCLUDE_DIRS}
#   CACHE PATH
#   "Directories in which to find headers for SuiteSparse.")
# mark_as_advanced(FORCE SuiteSparse_INCLUDE_DIRS)
# 
# # This handles "REQUIRED" etc keywords
# include(FindPackageHandleStandardArgs)
# find_package_handle_standard_args(SuiteSparse
#   "SuiteSparse could not be found. Be sure to set SuiteSparse_DIR."
#   SuiteSparse_LIBRARIES SuiteSparse_INCLUDE_DIRS)
