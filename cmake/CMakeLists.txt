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

# Build the CMake configuration files
include(CMakePackageConfigHelpers)

set(MODULES_TO_INSTALL
  modules/FindMETIS.cmake
  modules/FindSuiteSparse.cmake
  modules/FindHYPRE.cmake
  modules/FindMFEM.cmake
  modules/CMakeUtilities.cmake
  )

# Create the version for the build tree
write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  VERSION "${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}"
  COMPATIBILITY SameMajorVersion)

# Create the build-tree configuration file
set(INCLUDE_INSTALL_DIRS "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/src")
set(LIB_INSTALL_DIR "${PROJECT_BINARY_DIR}/src")
set(MODULE_INSTALL_DIR "${PROJECT_SOURCE_DIR}/cmake/modules")

configure_package_config_file(smoothgConfig.cmake.in
  ${CMAKE_BINARY_DIR}/smoothgConfig.cmake
  INSTALL_DESTINATION ${CMAKE_BINARY_DIR}
  PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR MODULE_INSTALL_DIR)

#
# Install tree stuff
#

# Create the install-tree configuration file
set(INCLUDE_INSTALL_DIRS include)
set(LIB_INSTALL_DIR lib)
set(CMAKE_INSTALL_DIR lib/cmake/smoothg)
set(MODULE_INSTALL_DIR ${CMAKE_INSTALL_DIR}/modules)

configure_package_config_file(
  smoothgConfig.cmake.in
  smoothgConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_DIR}
  PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR MODULE_INSTALL_DIR)

# Install the version file
install(
  FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${CMAKE_INSTALL_DIR})

# Install the modules
install(FILES ${MODULES_TO_INSTALL} DESTINATION ${MODULE_INSTALL_DIR})
