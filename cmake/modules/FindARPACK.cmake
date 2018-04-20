# - Try to find ARPACK
# Once done this will define
#
#  ARPACK_FOUND        - system has ARPACK
#  ARPACK_INCLUDE_DIRS - include directories for ARPACK
#  ARPACK_LIBRARIES    - libraries for ARPACK
#
# Variables used by this module. They can change the default behaviour and
# need to be set before calling find_package:
#
#  ARPACK_DIR          - Prefix directory of the ARPACK installation
#  ARPACK_INCLUDE_DIR  - Include directory of the ARPACK installation
#  ARPACK                (set only if different from ${ARPACK_DIR}/include)
#  ARPACK_LIB_DIR      - Library directory of the ARPACK installation
#  ARPACK                (set only if different from ${ARPACK_DIR}/lib)
#  ARPACK_TEST_RUNS    - Skip tests building and running a test
#  ARPACK                executable linked against ARPACK libraries
#  ARPACK_LIB_SUFFIX   - Also search for non-standard library names with the
#                       given suffix appended
#
# NOTE: This file was modified from a METIS detection script 

#=============================================================================
# Copyright (C) 2015 Jack Poulson. All rights reserved.
#
# Copyright (C) 2010-2012 Garth N. Wells, Anders Logg, Johannes Ring
# and Florian Rathgeber. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

if(NOT ARPACK_INCLUDE_DIR)
    find_path(ARPACK_INCLUDE_DIR argsym.h
        HINTS ${ARPACK_INCLUDE_DIR} ENV ARPACK_INCLUDE_DIR ${ARPACK_DIR} ENV ARPACK_DIR
    PATH_SUFFIXES include
    DOC "Directory where the ARPACK header files are located"
  )
endif()

if(ARPACK_LIBRARIES)
    set(ARPACK_LIBRARY ${ARPACK_LIBRARIES})
endif()
if(NOT ARPACK_LIBRARY)
    find_library(ARPACK_LIBRARY
        NAMES arpack arpack${ARPACK_LIB_SUFFIX}
        HINTS ${ARPACK_LIB_DIR} ENV ARPACK_LIB_DIR ${ARPACK_DIR} ENV ARPACK_DIR
    PATH_SUFFIXES lib
    DOC "Directory where the ARPACK library is located"
  )
endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
if(CMAKE_VERSION VERSION_GREATER 2.8.2)
    find_package_handle_standard_args(ARPACK
        REQUIRED_VARS ARPACK_LIBRARY ARPACK_INCLUDE_DIR
        VERSION_VAR ARPACK_VERSION_STRING)
else()
    find_package_handle_standard_args(ARPACK
        REQUIRED_VARS ARPACK_LIBRARY ARPACK_INCLUDE_DIR
    )
endif()

if(ARPACK_FOUND)
    set(ARPACK_LIBRARIES ${ARPACK_LIBRARY})
    set(ARPACK_INCLUDE_DIRS ${ARPACK_INCLUDE_DIR})
endif()

mark_as_advanced(ARPACK_INCLUDE_DIR ARPACK_LIBRARY)
if (ARPACK_FOUND AND NOT TARGET Arpack::Arpack)
    add_library(Arpack::Arpack INTERFACE IMPORTED)
    set_target_properties(Arpack::Arpack PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ARPACK_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${ARPACK_LIBRARIES}"
        )
endif()
