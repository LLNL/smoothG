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
add_executable(lineargraph lineargraph.cpp)
target_link_libraries(lineargraph smoothg ${TPL_LIBRARIES})

add_executable(tinygraphsolver tinygraphsolver.cpp)
target_link_libraries(tinygraphsolver smoothg ${TPL_LIBRARIES})

add_executable(partinygraph partinygraph.cpp)
target_link_libraries(partinygraph smoothg ${TPL_LIBRARIES})

add_executable(eigen eigen.cpp)
target_link_libraries(eigen smoothg ${TPL_LIBRARIES})

add_executable(test_MetisGraphPartitioner test_MetisGraphPartitioner.cpp)
target_link_libraries(test_MetisGraphPartitioner smoothg ${TPL_LIBRARIES})

add_executable(test_IsolatePartitioner test_IsolatePartitioner.cpp)
target_link_libraries(test_IsolatePartitioner smoothg ${TPL_LIBRARIES})

add_executable(wattsstrogatz wattsstrogatz.cpp)
target_link_libraries(wattsstrogatz smoothg ${TPL_LIBRARIES})

add_executable(rescaling rescaling.cpp)
target_link_libraries(rescaling smoothg ${TPL_LIBRARIES})

# add tests
add_test(lineargraph lineargraph)
add_test(lineargraph64 lineargraph --size 64)
add_valgrind_test(vlineargraph lineargraph)

add_test(tinygraphsolver tinygraphsolver)
add_test(tinygraphsolver_w tinygraphsolver -w)
add_test(tinygraphsolver_m tinygraphsolver -m)
add_test(tinygraphsolver_wm tinygraphsolver -w -m)

add_valgrind_test(vtinygraph tinygraphsolver)
add_valgrind_test(vtinygraph_w tinygraphsolver -w)
add_valgrind_test(vtinygraph_m tinygraphsolver -m)
add_valgrind_test(vtinygraph_wm tinygraphsolver -w -m)

add_test(partinygraph mpirun -np 2 ./partinygraph)
add_test(partinygraph_w mpirun -np 2 ./partinygraph -w)
add_test(partinygraph_m mpirun -np 2 ./partinygraph -m)
add_test(partinygraph_wm mpirun -np 2 ./partinygraph -w -m)

add_test(eigen eigen)
add_valgrind_test(veigen eigen)

add_test(test_MetisGraphPartitioner test_MetisGraphPartitioner)
add_test(test_IsolatePartitioner test_IsolatePartitioner)
add_valgrind_test(vtest_IsolatePartitioner test_IsolatePartitioner)

add_test(wattsstrogatz wattsstrogatz)
add_test(parwattsstrogatz mpirun -np 2 ./wattsstrogatz)
add_valgrind_test(vwattsstrogatz wattsstrogatz)

add_test(rescaling rescaling)
add_test(parrescaling mpirun -np 2 ./rescaling)
add_valgrind_test(vrescaling rescaling)

add_test(NAME style
  COMMAND ${ASTYLE_COMMAND} --options=smoothg.astylerc --dry-run src/*.?pp examples/*.?pp testcode/*.?pp
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(style
  PROPERTIES FAIL_REGULAR_EXPRESSION
  "Formatted")
