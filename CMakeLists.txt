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

###############################################################
# MIT License
# 
# Copyright (c) 2018 Pablo Arias
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
# OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 
###############################################################

###############################################################
# Using https://github.com/pabloariasal/modern-cmake-sample
# Modified by gelever on April 1st, 2018 #
###############################################################


cmake_minimum_required(VERSION 3.5)
project(smoothG VERSION 1.0 LANGUAGES CXX)

find_package(linalgcpp REQUIRED)
find_package(parlinalgcpp REQUIRED)
find_package(partition REQUIRED)
find_package(sparsesolve REQUIRED)

add_library(smoothG
    src/GraphCoarsen.cpp
    src/Graph.cpp
    src/GraphEdgeSolver.cpp
    src/GraphTopology.cpp
    src/GraphUpscale.cpp
    src/MGLSolver.cpp
    src/MinresBlockSolver.cpp
    src/MixedMatrix.cpp
    src/SharedEntityComm.cpp
    src/Upscale.cpp
    src/Utilities.cpp
)

set_target_properties(smoothG PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_compile_options(smoothG PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wall>)

target_link_libraries(smoothG
    PUBLIC
        parlinalgcpp::parlinalgcpp
        partition::partition
        sparsesolve::sparsesolve
        linalgcpp::linalgcpp
)

target_include_directories(smoothG
    PUBLIC
		$<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

add_subdirectory(examples)

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/libsmoothg)

install(TARGETS smoothG
    EXPORT smoothg-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT smoothg-targets
    FILE smoothgTargets.cmake
    NAMESPACE smoothg::
    DESTINATION ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/smoothgConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/smoothgConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/smoothgConfig.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/smoothgConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/smoothgConfigVersion.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)

export(EXPORT smoothg-targets
       FILE ${CMAKE_CURRENT_BINARY_DIR}/smoothgTargets.cmake
       NAMESPACE smoothg::
       )

export(PACKAGE smoothG)
