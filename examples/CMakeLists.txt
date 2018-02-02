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


# add executables
add_executable(finitevolume finitevolume.cpp)
target_link_libraries(finitevolume smoothg ${TPL_LIBRARIES})

add_executable(fv_weakscaling fv_weakscaling.cpp)
target_link_libraries(fv_weakscaling smoothg ${TPL_LIBRARIES})

add_executable(generalgraph generalgraph.cpp)
target_link_libraries(generalgraph smoothg ${TPL_LIBRARIES})

# add tests
configure_file(
  "${PROJECT_SOURCE_DIR}/examples/stest.py"
  "${PROJECT_BINARY_DIR}/examples/stest.py" @ONLY)
configure_file(
  "${PROJECT_SOURCE_DIR}/examples/readjson.py"
  "${PROJECT_BINARY_DIR}/examples/readjson.py"
  COPYONLY)

add_test(eigenvector1 python stest.py eigenvector1)
add_test(eigenvector4 python stest.py eigenvector4)
add_test(fv-hybridization python stest.py fv-hybridization)
add_test(slice19 python stest.py slice19)
add_test(samplegraph1 python stest.py samplegraph1)
add_test(samplegraph4 python stest.py samplegraph4)
add_test(graph-hybridization python stest.py graph-hybridization)
add_test(pareigenvector1 python stest.py pareigenvector1)
add_test(pareigenvector4 python stest.py pareigenvector4)
add_test(parfv-hybridization python stest.py parfv-hybridization)
add_test(parslice19 python stest.py parslice19)
add_test(parsamplegraph1 python stest.py parsamplegraph1)
add_test(parsamplegraph4 python stest.py parsamplegraph4)
add_test(pargraph-hybridization python stest.py pargraph-hybridization)

if (APPLE)
  add_test(fv-metis python stest.py fv-metis-mac)
  add_test(graph-metis python stest.py graph-metis-mac)
  add_test(graph-usegenerator python stest.py graph-usegenerator-mac)
  add_test(pargraph-usegenerator python stest.py pargraph-usegenerator-mac)
  add_test(pargraph-metis python stest.py pargraph-metis-mac)
else()
  add_test(fv-metis python stest.py fv-metis)
  add_test(graph-metis python stest.py graph-metis)
  add_test(graph-usegenerator python stest.py graph-usegenerator)
  add_test(pargraph-usegenerator python stest.py pargraph-usegenerator)
  add_test(pargraph-metis python stest.py pargraph-metis)
endif()
add_test(isolate_coarsen python stest.py isolate-coarsen)

add_test(NAME style
  COMMAND ${ASTYLE_COMMAND} --options=smoothg.astylerc --dry-run src/*.?pp examples/*.?pp testcode/*.?pp
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(style
  PROPERTIES FAIL_REGULAR_EXPRESSION
  "Formatted")

# this test takes too long so we are deprecating it, but you should run it once in a while
# add_test(veigenvector python stest.py veigenvector)
