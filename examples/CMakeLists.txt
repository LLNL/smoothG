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

add_executable(twophase twophase.cpp)
target_link_libraries(twophase smoothg ${TPL_LIBRARIES})

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

add_test(eigenvector1 python stest.py eigenvector1)
add_test(eigenvector4 python stest.py eigenvector4)
add_test(fv-hybridization python stest.py fv-hybridization)
add_test(slice19 python stest.py slice19)
add_test(dual-trace python stest.py dual-trace)
add_test(scaled-dual-trace python stest.py scaled-dual-trace)
add_test(energy-dual-trace python stest.py energy-dual-trace)
add_test(scaled-energy-dual-trace python stest.py scaled-energy-dual-trace)
add_test(samplegraph1 python stest.py samplegraph1)
add_test(samplegraph1-coeff python stest.py samplegraph1-coeff)
add_test(samplegraph1-coeff-hb python stest.py samplegraph1-coeff-hb)
add_test(samplegraph4 python stest.py samplegraph4)
add_test(samplegraph4-coeff python stest.py samplegraph4-coeff)
add_test(samplegraph4-coeff-hb python stest.py samplegraph4-coeff-hb)
add_test(graph-hybridization python stest.py graph-hybridization)
add_test(graph-weight python stest.py graph-weight)
add_test(samplesolve-sanity python stest.py samplesolve-sanity)
add_test(samplesolve-pde-sampler python stest.py samplesolve-pde-sampler)
add_test(samplesolve-pde-sampler-hb python stest.py samplesolve-pde-sampler-hb)
add_test(par-samplesolve python stest.py par-samplesolve)
add_test(timestep python stest.py timestep)
add_test(graphupscale graphupscale)
add_test(mltopo mltopo --no-visualization)
add_test(qoi-one-level python stest.py qoi-one-level)

add_test(pareigenvector1 python stest.py pareigenvector1)
add_test(pareigenvector4 python stest.py pareigenvector4)
add_test(parfv-hybridization python stest.py parfv-hybridization)
add_test(parslice19 python stest.py parslice19)
add_test(pardual-trace python stest.py pardual-trace)
add_test(parscaled-dual-trace python stest.py parscaled-dual-trace)
add_test(parenergy-dual-trace python stest.py parenergy-dual-trace)
add_test(parscaled-energy-dual-trace python stest.py parscaled-energy-dual-trace)
add_test(partimestep python stest.py partimestep)
add_test(pargraphupscale mpirun -np 2 ./graphupscale)
add_test(parmltopo mpirun -np 2 ./mltopo --no-visualization)

if (APPLE)
  add_test(poweriter python stest.py poweriter-mac)
  add_test(fv-metis python stest.py fv-metis-mac)
  add_test(graph-metis python stest.py graph-metis-mac)
  add_test(graph-usegenerator python stest.py graph-usegenerator-mac)
  add_test(pargraph-usegenerator python stest.py pargraph-usegenerator-mac)
  add_test(pargraph-metis python stest.py pargraph-metis-mac)
  add_test(parsamplegraph1 python stest.py parsamplegraph1-mac)
  add_test(parsamplegraph1-coeff python stest.py parsamplegraph1-coeff-mac)
  add_test(parsamplegraph1-coeff-hb python stest.py parsamplegraph1-coeff-hb-mac)
  add_test(parsamplegraph4 python stest.py parsamplegraph4-mac)
  add_test(pargraph-hybridization python stest.py pargraph-hybridization-mac)
  add_test(parpoweriter python stest.py parpoweriter-mac)
  add_test(fv-ml-4 python stest.py fv-ml-4-mac)
  add_test(qoi python stest.py qoi-mac)
  add_test(qoi-hb python stest.py qoi-hb-mac)
  add_test(pardirichlet python stest.py pardirichlet-mac)
  add_test(pardirichlet-hb python stest.py pardirichlet-hb-mac)
  add_test(ml-sampler python stest.py ml-sampler-mac)
  add_test(ml-sampler4 python stest.py ml-sampler4-mac)
else()
  add_test(poweriter python stest.py poweriter)
  add_test(fv-metis python stest.py fv-metis)
  add_test(graph-metis python stest.py graph-metis)
  add_test(graph-usegenerator python stest.py graph-usegenerator)
  add_test(pargraph-usegenerator python stest.py pargraph-usegenerator)
  add_test(pargraph-metis python stest.py pargraph-metis)
  add_test(parsamplegraph1 python stest.py parsamplegraph1)
  add_test(parsamplegraph1-coeff python stest.py parsamplegraph1-coeff)
  add_test(parsamplegraph1-coeff-hb python stest.py parsamplegraph1-coeff-hb)
  add_test(parsamplegraph4 python stest.py parsamplegraph4)
  add_test(pargraph-hybridization python stest.py pargraph-hybridization)
  add_test(parpoweriter python stest.py parpoweriter)
  add_test(fv-ml-4 python stest.py fv-ml-4)
  add_test(qoi python stest.py qoi)
  add_test(qoi-hb python stest.py qoi-hb)
  add_test(pardirichlet python stest.py pardirichlet)
  add_test(pardirichlet-hb python stest.py pardirichlet-hb)
  add_test(ml-sampler python stest.py ml-sampler)
  add_test(ml-sampler4 python stest.py ml-sampler4)
endif()
add_test(isolate_coarsen python stest.py isolate-coarsen)
add_test(sampler python stest.py sampler)
add_test(fas-picard-one-level python stest.py fas-picard-one-level)
add_test(fas-picard python stest.py fas-picard)
add_test(fas-newton-one-level python stest.py fas-newton-one-level)
add_test(fas-newton python stest.py fas-newton)

add_test(vgraph-small-usegenerator python stest.py vgraph-small-usegenerator)
add_test(vgraph-small-usegenerator-hb python stest.py vgraph-small-usegenerator-hb)

# these tests take too long so we are deprecating them, but you should run them once in a while
# add_test(veigenvector python stest.py veigenvector)
# add_valgrind_test(vmltopo mltopo)
