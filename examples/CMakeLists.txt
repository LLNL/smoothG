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

# Apply flags to all test targets
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

add_executable(generalgraph generalgraph.cpp)
target_link_libraries(generalgraph smoothG)

add_executable(poweriter poweriter.cpp)
target_link_libraries(poweriter smoothG)

add_executable(graphupscale graphupscale.cpp)
target_link_libraries(graphupscale smoothG)

if (NOT DEFINED SMOOTHG_TEST_TOL)
    set(SMOOTHG_TEST_TOL 1e-4)
endif()
if (NOT DEFINED SMOOTHG_TEST_PROCS)
    set(SMOOTHG_TEST_PROCS 4)
endif()

configure_file(
  "${PROJECT_SOURCE_DIR}/examples/stest.py"
  "${PROJECT_BINARY_DIR}/examples/stest.py" @ONLY)

add_test(samplegraph1 python stest.py samplegraph1)
add_test(samplegraph4 python stest.py samplegraph4)
add_test(graph-hybridization python stest.py graph-hybridization)
add_test(poweriter python stest.py poweriter)
add_test(graph-weight python stest.py graph-weight)

add_test(parsamplegraph1 python stest.py parsamplegraph1)
add_test(parsamplegraph4 python stest.py parsamplegraph4)
add_test(pargraph-hybridization python stest.py pargraph-hybridization)
add_test(parpoweriter python stest.py parpoweriter)

if (APPLE)
  add_test(graph-metis python stest.py graph-metis-mac)
  add_test(graph-usegenerator python stest.py graph-usegenerator-mac)
  add_test(pargraph-usegenerator python stest.py pargraph-usegenerator-mac)
  add_test(pargraph-metis python stest.py pargraph-metis-mac)
else()
  add_test(graph-metis python stest.py graph-metis)
  add_test(graph-usegenerator python stest.py graph-usegenerator)
  add_test(pargraph-usegenerator python stest.py pargraph-usegenerator)
  add_test(pargraph-metis python stest.py pargraph-metis)
endif()

#add_test(vgraph-small-usegenerator python stest.py vgraph-small-usegenerator)
#add_test(isolate_coarsen python stest.py isolate-coarsen)
