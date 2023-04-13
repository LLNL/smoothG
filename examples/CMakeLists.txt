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

add_executable(sampler sampler.cpp)
target_link_libraries(sampler smoothg ${TPL_LIBRARIES})

add_executable(generalgraph generalgraph.cpp)
target_link_libraries(generalgraph smoothg ${TPL_LIBRARIES})

add_executable(timestep timestep.cpp)
target_link_libraries(timestep smoothg ${TPL_LIBRARIES})

add_executable(graphupscale graphupscale.cpp)
target_link_libraries(graphupscale smoothg ${TPL_LIBRARIES})

add_executable(poweriter poweriter.cpp)
target_link_libraries(poweriter smoothg ${TPL_LIBRARIES})

add_executable(mltopo mltopo.cpp)
target_link_libraries(mltopo smoothg ${TPL_LIBRARIES})

add_executable(samplesolve samplesolve.cpp)
target_link_libraries(samplesolve smoothg ${TPL_LIBRARIES})

add_executable(qoi qoi.cpp)
target_link_libraries(qoi smoothg ${TPL_LIBRARIES})

add_executable(nldarcy nldarcy.cpp)
target_link_libraries(nldarcy smoothg ${TPL_LIBRARIES})

# add tests
if (NOT DEFINED SMOOTHG_TEST_PROCS)
    set(SMOOTHG_TEST_PROCS 2)
endif()
if (NOT DEFINED SMOOTHG_TEST_TOL)
    set(SMOOTHG_TEST_TOL 1e-4)
endif()

configure_file(
  "${PROJECT_SOURCE_DIR}/examples/stest.py"
  "${PROJECT_BINARY_DIR}/examples/stest.py" @ONLY)
configure_file(
  "${PROJECT_SOURCE_DIR}/examples/readjson.py"
  "${PROJECT_BINARY_DIR}/examples/readjson.py"
  COPYONLY)

if (APPLE)
  add_test(ml-sampler python stest.py ml-sampler-mac)
  add_test(ml-sampler4 python stest.py ml-sampler4-mac)
else()
  add_test(ml-sampler python stest.py ml-sampler)
  add_test(ml-sampler4 python stest.py ml-sampler4)
endif()
add_test(sampler python stest.py sampler)
add_test(fas-picard-one-level python stest.py fas-picard-one-level)
add_test(fas-picard python stest.py fas-picard)
add_test(fas-newton-one-level python stest.py fas-newton-one-level)
add_test(fas-newton python stest.py fas-newton)

# these tests take too long so we are deprecating them, but you should run them once in a while
# add_test(veigenvector python stest.py veigenvector)
# add_valgrind_test(vmltopo mltopo)
