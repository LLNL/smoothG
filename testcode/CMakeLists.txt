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

# add executables
#add_executable(lineargraph lineargraph.cpp)
#target_link_libraries(lineargraph smoothg ${TPL_LIBRARIES})

add_executable(tinygraphsolver tinygraphsolver.cpp)
target_link_libraries(tinygraphsolver smoothG)

add_executable(eigen eigen.cpp)
target_link_libraries(eigen smoothG)

add_executable(test_MetisGraphPartitioner test_MetisGraphPartitioner.cpp)
target_link_libraries(test_MetisGraphPartitioner smoothG)

#add_executable(test_IsolatePartitioner test_IsolatePartitioner.cpp)
#target_link_libraries(test_IsolatePartitioner smoothg ${TPL_LIBRARIES})

add_executable(wattsstrogatz wattsstrogatz.cpp)
target_link_libraries(wattsstrogatz smoothG)

add_executable(test_Graph test_Graph.cpp)
target_link_libraries(test_Graph smoothG)

# add tests
# add_test(lineargraph lineargraph)
# add_test(lineargraph64 lineargraph --size 64)
# add_valgrind_test(vlineargraph lineargraph)


add_test(tinygraphsolver tinygraphsolver)
add_test(tinygraphsolver_wg tinygraphsolver -wg)
add_test(tinygraphsolver_wb tinygraphsolver -wb)
add_test(tinygraphsolver_wg_wb tinygraphsolver -wg -wb)

add_test(partinygraphsolver mpirun -np 2 ./tinygraphsolver)
add_test(partinygraphsolver_wg mpirun -np 2 ./tinygraphsolver -wg)
add_test(partinygraphsolver_wb mpirun -np 2 ./tinygraphsolver -wb)
add_test(partinygraphsolver_wg_wb mpirun -np 2 ./tinygraphsolver -wg -wb)

add_test(eigen eigen)

add_test(test_MetisGraphPartitioner test_MetisGraphPartitioner)

add_test(test_Graph test_Graph)
add_test(parttest_Graph mpirun -np 2 ./test_Graph)

# add_test(test_IsolatePartitioner test_IsolatePartitioner)
# add_valgrind_test(vtest_IsolatePartitioner test_IsolatePartitioner)

add_test(wattsstrogatz wattsstrogatz)
add_test(parwattsstrogatz mpirun -np 2 ./wattsstrogatz)

if (NOT APPLE)
    add_valgrind_test(vtinygraph tinygraphsolver)
    add_valgrind_test(vtinygraph_wg tinygraphsolver -wg)
    add_valgrind_test(vtinygraph_wb tinygraphsolver -wb)
    add_valgrind_test(vtinygraph_wg_wb tinygraphsolver -wg -wb)
    add_valgrind_test(veigen eigen)
    add_valgrind_test(vtest_MetisGraphPartitioner test_MetisGraphPartitioner)
    add_valgrind_test(vwattsstrogatz wattsstrogatz)
endif()

# add_test(NAME style
#   COMMAND ${ASTYLE_COMMAND} --options=smoothg.astylerc --dry-run src/*.?pp examples/*.?pp testcode/*.?pp
#   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
# set_tests_properties(style
#   PROPERTIES FAIL_REGULAR_EXPRESSION
#   "Formatted")
