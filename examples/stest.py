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

"""
A way to interface some basic JSON-parsing tests in python
with the cmake/ctest testing framework.

Stephan Gelever
gelever1@llnl.gov
17 July 2017
"""

from __future__ import print_function

import subprocess

import readjson
import sys
import platform

spe10_perm_file = "@SPE10_PERM@"
graph_data = "@smoothG_GRAPHDATA@"
memorycheck_command = "@MEMORYCHECK_COMMAND@"

# Test paramaters
num_procs = "@SMOOTHG_TEST_PROCS@"
default_test_tol = float("@SMOOTHG_TEST_TOL@")

def run_test(command, expected={}, verbose=False):
    """ Executes test

    Args:
        command:    command to run test
        expected:   expected result of test
        verbose:    display additional info

    Returns:
        bool:       true if test passes

    """
    if verbose:
        print(command)

    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = p.communicate()

    if verbose:
        print(stdout)
        print(stderr)

    if p.returncode != 0:
        return False

    output = readjson.json_parse_lines(stdout.splitlines())

    for key, expected_valpair in expected.items():
        test_val = output[key]

        try:
            expected_val = expected_valpair[0]
            test_tol = expected_valpair[1]
        except TypeError:
            expected_val = expected_valpair
            test_tol = default_test_tol
        fexpected = float(expected_val)
        ftest = float(test_val)
        if abs(fexpected - ftest) > test_tol:
            print("test {0:s} failed, expected {1:f}, got {2:f}\n".format(key, fexpected, ftest))
            return False

    return True


def make_tests():
    """ Generates test dictionary

    Tests are the following format:
        - dictionary key is test name
        - dictionary value is an array containing:
            - command to execute
            - expected result [optional]

    Returns:
        dict:     collection of tests

    """
    tests = dict()

    tests["eigenvector1"] = \
        [["./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "1",
          "--max-traces", "1",
          "--perm", spe10_perm_file],
         {"relative-D-edge-error": (0.0, 1.e-7),
          "relative-vertex-error": 0.14743131732550618,
          "relative-edge-error": 0.22621045683612057,
          "operator-complexity": 1.0221724964280585}]

    tests["eigenvector4"] = \
        [["./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--perm", spe10_perm_file],
         {"relative-D-edge-error": (0.0, 1.e-7),
          "relative-vertex-error": 0.05516198497834629,
          "relative-edge-error": 0.052317636963252999,
          "operator-complexity": 1.3017591339648173}]

    tests["fv-hybridization"] = \
        [["./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--hybridization",
          "--perm", spe10_perm_file],
         {"relative-D-edge-error": (0.0, 1.e-7),
          "relative-vertex-error": 0.055161984984368362,
          "relative-edge-error": 0.052317636981330032,
          "operator-complexity": 1.1362437864707153}]

    tests["slice19"] = \
        [["./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "19",
          "--max-evects", "1",
          "--max-traces", "1",
          "--perm", spe10_perm_file],
         {"relative-D-edge-error": (0.0, 1.e-7),
          "relative-vertex-error": 0.23763409361749516,
          "relative-edge-error": 0.16419932734829923,
          "operator-complexity": 1.0221724964280585}]

    tests["fv-metis"] = \
        [["./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "1",
          "--max-traces", "1",
          "--metis-agglomeration",
          "--coarse-factor", "100",
          "--perm", spe10_perm_file],
         {"relative-D-edge-error": 0.63485274406150471,
          "relative-vertex-error": 0.15480723435585672,
          "relative-edge-error": 0.25645221876785801,
          "operator-complexity": 1.04042908656572}]

    tests["fv-metis-mac"] = \
        [["./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "1",
          "--max-traces", "1",
          "--metis-agglomeration",
          "--coarse-factor", "100",
          "--perm", spe10_perm_file],
         {"relative-D-edge-error": 0.5505999771706922,
          "relative-vertex-error": 0.16109495317744718,
          "relative-edge-error": 0.27438846301628711,
          "operator-complexity": 1.0398091940641514}]

    tests["dual-trace"] = \
        [["./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--dual-target",
          "--perm", spe10_perm_file],
         {"relative-D-edge-error": (0.0, 1.e-7),
          "relative-vertex-error": 0.055207481027916193,
          "relative-edge-error": 0.06430185063505546,
          "operator-complexity": 1.3017591339648173}]

    tests["scaled-dual-trace"] = \
        [["./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--dual-target",
          "--scaled-dual",
          "--perm", spe10_perm_file],
         {"relative-D-edge-error": (0.0, 1.e-7),
          "relative-vertex-error": 0.055054636384856817,
          "relative-edge-error": 0.034260930604399109,
          "operator-complexity": 1.3017591339648173}]

    tests["energy-dual-trace"] = \
        [["./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--dual-target",
          "--energy-dual",
          "--perm", spe10_perm_file],
         {"relative-D-edge-error": (0.0, 1.e-7),
          "relative-vertex-error": 0.05531441553069754,
          "relative-edge-error": 0.068870924043507808,
          "operator-complexity": 1.3017591339648173}]

    tests["scaled-energy-dual-trace"] = \
        [["./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--dual-target",
          "--scaled-dual",
          "--energy-dual",
          "--perm", spe10_perm_file],
         {"relative-D-edge-error": (0.0, 1.e-7),
          "relative-vertex-error": 0.055053504716695102,
          "relative-edge-error": 0.034903738719505632,
          "operator-complexity": 1.3017591339648173}]

    tests["fv-ml-4"] = \
        [["./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--max-levels", "3",
          "--coarse-factor", "8",
          "--perm", spe10_perm_file],
         {"relative-D-edge-error": 0.82980244867656872,
          "relative-vertex-error": 0.069590757228569236,
          "relative-edge-error": 0.14631573149454258,
          "operator-complexity": 1.3403134236965248}]

    tests["fv-ml-4-mac"] = \
        [["./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--max-levels", "3",
          "--coarse-factor", "8",
          "--perm", spe10_perm_file],
         {"relative-D-edge-error": 0.80539906744851109,
          "relative-vertex-error": 0.096644730987530583,
          "relative-edge-error": 0.14688172254757564,
          "operator-complexity": 1.3381135612824215}]

    tests["samplegraph1"] = \
        [["./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "1",
          "--max-traces", "1"],
         {"relative-D-edge-error": 0.37918423747873353,
          "relative-vertex-error": 0.38013398274257243,
          "relative-edge-error": 0.38079825403520218,
          "operator-complexity": 1.016509834901651}]

    tests["graph-metis"] = \
        [["./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "1",
          "--max-traces", "1",
          "--metis-agglomeration"],
         {"relative-D-edge-error": 0.44710819907744104,
          "relative-vertex-error": 0.44939226988126274,
          "relative-edge-error": 0.42773807524771068,
          "operator-complexity": 1.016509834901651}]

    tests["graph-metis-mac"] = \
        [["./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "1",
          "--max-traces", "1",
          "--metis-agglomeration"],
         {"relative-D-edge-error": 0.22228470008233389,
          "relative-vertex-error": 0.22265174467689006,
          "relative-edge-error": 0.22168973853676807,
          "operator-complexity": 1.016509834901651}]

    tests["samplegraph4"] = \
        [["./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4"],
         {"relative-D-edge-error": 0.12043046187567592,
          "relative-vertex-error": 0.13514675917148347,
          "relative-edge-error": 0.19926779054787247,
          "operator-complexity": 1.2578874211257887}]

    tests["graph-hybridization"] = \
        [["./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4",
          "--hybridization"],
         {"relative-D-edge-error": 0.12051328492652449,
          "relative-vertex-error": 0.13514675917148347,
          "relative-edge-error": 0.19926779054787247,
          "operator-complexity": 1.013984620448976}]

    tests["graph-usegenerator"] = \
        [["./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4",
          "--generate-graph"],
         {"relative-D-edge-error": 0.11283262603381641,
          "relative-vertex-error": 0.1203852548326301,
          "relative-edge-error": 0.16674213482507089,
          "operator-complexity": 1.2578874211257887}]

    tests["graph-usegenerator-mac"] = \
        [["./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4",
          "--generate-graph"],
         {"relative-D-edge-error": 0.1070681247529167,
          "relative-vertex-error": 0.10863131137013603,
          "relative-edge-error": 0.12848813745253315,
          "operator-complexity": 1.2578874211257887}]

    tests["poweriter"] = \
        [["./poweriter"],
         {"coarse-error": 0.2050307003818391,
          "coarse-eval": 0.17663653196285808,
          "fine-error": 2.9382710663490486e-05,
          "fine-eval": 0.17545528997990226}]

    tests["poweriter-mac"] = \
        [["./poweriter"],
         {"coarse-error": 0.26120392031493855,
          "coarse-eval": 0.17978705998830949,
          "fine-error": 2.7998030058085634e-05,
          "fine-eval": 0.17545528997985335}]

    tests["graph-weight"] = \
        [["./generalgraph",
          "--graph", graph_data + "/vertex_edge_tiny.txt",
          "--weight", graph_data + "/tiny_weights.txt",
          "--generate-fiedler",
          "--metis-agglomeration",
          "--num-part", "2"],
         {"relative-D-edge-error": 0.3033520464019937,
          "relative-vertex-error": 0.31217311873637132,
          "relative-edge-error": 0.14767829457535478,
          "operator-complexity": 1.1666666666666667}]

    tests["mlmc-sanity"] = \
        [["./mlmc",
          "--perm", spe10_perm_file],
         {"relative-vertex-error": 0.10754186878360708}]

    tests["mlmc-pde-sampler"] = \
        [["./mlmc",
          "--sampler-type", "pde",
          "--kappa", "0.01"],
         {"relative-vertex-error": 0.1487505869352104}]

    tests["mlmc-pde-sampler-hb"] = \
        [["./mlmc",
          "--sampler-type", "pde",
          "--kappa", "0.01",
          "--hybridization"],
         {"relative-vertex-error": 0.14875751525009742}]

    tests["par-mlmc"] = \
        [["mpirun", "-n", "2", "./mlmc",
          "--sampler-type", "pde",
          "--kappa", "0.01"],
         {"relative-vertex-error": 0.38805214759478951}]

    tests["timestep"] = \
        [["./timestep",
          "--total-time", "100.0",
          "--perm", spe10_perm_file]]

    tests["pareigenvector1"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "1",
          "--max-traces", "1",
          "--perm", spe10_perm_file],
         {"relative-D-edge-error": (0.0, 1.e-7),
          "relative-vertex-error": 0.14743131732550618,
          "relative-edge-error": 0.22621045683612057,
          "operator-complexity": 1.0221724964280585}]

    tests["pareigenvector4"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--perm", spe10_perm_file],
         {"relative-D-edge-error": (0.0, 1.e-7),
          "relative-vertex-error": 0.05516198497834629,
          "relative-edge-error": 0.052317636963252999,
          "operator-complexity": 1.3017591339648173}]

    tests["parfv-hybridization"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--hybridization",
          "--perm", spe10_perm_file],
         {"relative-D-edge-error": (0.0, 1.e-7),
          "relative-vertex-error": 0.055161984984368362,
          "relative-edge-error": 0.052317636981330032,
          "operator-complexity": 1.1362437864707153}]

    tests["parslice19"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "19",
          "--max-evects", "1",
          "--max-traces", "1",
          "--perm", spe10_perm_file],
         {"relative-D-edge-error": (0.0, 1.e-7),
          "relative-vertex-error": 0.23763409361749516,
          "relative-edge-error": 0.16419932734829923,
          "operator-complexity": 1.0221724964280585}]

    tests["pardual-trace"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--dual-target",
          "--perm", spe10_perm_file],
         {"relative-D-edge-error": (0.0, 1.e-7),
          "relative-vertex-error": 0.055207481027916193,
          "relative-edge-error": 0.06430185063505546,
          "operator-complexity": 1.3017591339648173}]

    tests["parscaled-dual-trace"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--dual-target",
          "--scaled-dual",
          "--perm", spe10_perm_file],
         {"relative-D-edge-error": (0.0, 1.e-7),
          "relative-vertex-error": 0.055054636384856817,
          "relative-edge-error": 0.034260930604399109,
          "operator-complexity": 1.3017591339648173}]

    tests["parenergy-dual-trace"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--dual-target",
          "--energy-dual",
          "--perm", spe10_perm_file],
         {"relative-D-edge-error": (0.0, 1.e-7),
          "relative-vertex-error": 0.05531441553069754,
          "relative-edge-error": 0.068870924043507808,
          "operator-complexity": 1.3017591339648173}]

    tests["parscaled-energy-dual-trace"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--dual-target",
          "--scaled-dual",
          "--energy-dual",
          "--perm", spe10_perm_file],
         {"relative-D-edge-error": (0.0, 1.e-7),
          "relative-vertex-error": 0.055053504716695102,
          "relative-edge-error": 0.034903738719505632,
          "operator-complexity": 1.3017591339648173}]

    tests["pardirichlet"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--lateral-pressure",
          "--spect-tol", "1.0",
          "--max-levels", "3",
          "--coarse-factor", "8",
          "--perm", spe10_perm_file],
         {"quantity-error-level-1": 0.0022603778822020588,
          "quantity-error-level-2": 0.0034320940221200902}]

    tests["pardirichlet-mac"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--lateral-pressure",
          "--spect-tol", "1.0",
          "--max-levels", "3",
          "--coarse-factor", "8",
          "--perm", spe10_perm_file],
         {"quantity-error-level-1": 0.002291327159532812,
          "quantity-error-level-2": 0.0039259944921827836}]

    tests["pardirichlet-hb"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--lateral-pressure",
          "--spect-tol", "1.0",
          "--max-levels", "3",
          "--coarse-factor", "8",
          "--hybridization",
          "--perm", spe10_perm_file],
         {"quantity-error-level-1": 0.0022603778822020588,
          "quantity-error-level-2": 0.0034320940221200902}]

    tests["pardirichlet-hb-mac"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--lateral-pressure",
          "--spect-tol", "1.0",
          "--max-levels", "3",
          "--coarse-factor", "8",
          "--hybridization",
          "--perm", spe10_perm_file],
         {"quantity-error-level-1": 0.002291327159532812,
          "quantity-error-level-2": 0.0039259944921827836}]

    tests["parsamplegraph1"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "1",
          "--max-traces", "1"],
         {"relative-D-edge-error": 0.79929158208615803,
          "relative-vertex-error": 0.80196594708999625,
          "relative-edge-error": 0.78297677149740286,
          "operator-complexity": 1.016509834901651}]

    tests["parsamplegraph1-mac"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "1",
          "--max-traces", "1"],
         {"relative-D-edge-error": 0.63689693723000385,
          "relative-vertex-error": 0.64187135487003799,
          "relative-edge-error": 0.59087512350995663,
          "operator-complexity": 1.008739912600874}]

    tests["pargraph-metis"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "1",
          "--max-traces", "1",
          "--metis-agglomeration"],
         {"relative-D-edge-error": 0.79929158208615803,
          "relative-vertex-error": 0.80196594708999625,
          "relative-edge-error": 0.78297677149740286,
          "operator-complexity": 1.016509834901651}]

    tests["pargraph-metis-mac"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "1",
          "--max-traces", "1",
          "--metis-agglomeration"],
         {"relative-D-edge-error": 0.63689693723000385,
          "relative-vertex-error": 0.64187135487003799,
          "relative-edge-error": 0.59087512350995663,
          "operator-complexity": 1.008739912600874}]

    tests["parsamplegraph4"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4"],
         {"relative-D-edge-error": 0.1438138616203242,
          "relative-vertex-error": 0.1874559440644907,
          "relative-edge-error": 0.25470044682041143,
          "operator-complexity": (1.230, 1.e-3)}]

    tests["parsamplegraph4-mac"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4"],
         {"relative-D-edge-error": 0.094122554052011503,
          "relative-vertex-error": 0.11511310888163191,
          "relative-edge-error": 0.18119613898323036,
          "operator-complexity": 1.1303686963130368}]

    tests["pargraph-hybridization"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4",
          "--hybridization"],
         {"relative-D-edge-error": 0.1438138616203242,
          "relative-vertex-error": 0.1874559440644907,
          "relative-edge-error": 0.25470044682041143,
          "operator-complexity": 1.0124793314423153}]

    tests["pargraph-hybridization-mac"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4",
          "--hybridization"],
         {"relative-D-edge-error": 0.094122554052011503,
          "relative-vertex-error": 0.11511310888163191,
          "relative-edge-error": 0.18119613898323036,
          "operator-complexity": 1.0067715933613413}]

    tests["pargraph-usegenerator"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4",
          "--generate-graph"],
         {"relative-D-edge-error": 0.19793206480307649,
          "relative-vertex-error": 0.28945297307624301,
          "relative-edge-error": 0.29617987947507896,
          "operator-complexity": (1.189778, 3.e-4)}]

    tests["pargraph-usegenerator-mac"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4",
          "--generate-graph"],
         {"relative-D-edge-error": 0.10696386867442335,
          "relative-vertex-error": 0.19594367090858048,
          "relative-edge-error": 0.2468153494604973,
          "operator-complexity": 1.1347886521134789}]

    tests["parpoweriter"] = \
        [["mpirun", "-n", num_procs, "./poweriter"],
         {"coarse-error": 1.4795742061962853,
          "coarse-eval": 0.33533401525270057,
          "fine-error": 2.950192232309819e-05,
          "fine-eval": 0.17545528997987314}]

    tests["parpoweriter-mac"] = \
        [["mpirun", "-n", num_procs, "./poweriter"],
         {"coarse-error": 1.3769646029608467,
          "coarse-eval": 0.24465778499404064,
          "fine-error": 2.9877711869866266e-05,
          "fine-eval": 0.1754552899797806}]

    tests["partimestep"] = \
        [["mpirun", "-n", num_procs, "./timestep",
          "--total-time", "100.0",
          "--perm", spe10_perm_file]]

    tests["isolate-coarsen"] = \
        [["./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4",
          "--metis-agglomeration",
          "--isolate", "0"],
         {"operator-complexity": 1.2736672633273667}]

    tests["sampler"] = \
        [["./sampler",
          "--kappa", "0.01",
          "--num-samples", "2"],
         {"fine-mean-l1": 0.54235348335037126,
          "p-error-level-1": 0.42784788897003051}]

    tests["ml-sampler"] = \
        [["./sampler",
          "--num-samples", "1",
          "--max-levels", "3",
          "--spect-tol", "1.0",
          "--max-evects", "1",
          "--max-traces", "1"],
         {"p-error-level-1": 0.197584266889065,
          "p-error-level-2": 0.38747838996654665}]

    tests["ml-sampler-mac"] = \
        [["./sampler",
          "--num-samples", "1",
          "--max-levels", "3",
          "--spect-tol", "1.0",
          "--max-evects", "1",
          "--max-traces", "1"],
         {"p-error-level-1": 0.20833920377622106,
          "p-error-level-2": 0.39357489172376314}]

    tests["ml-sampler4"] = \
        [["./sampler",
          "--num-samples", "1",
          "--max-levels", "3",
          "--spect-tol", "1.0",
          "--max-evects", "4"],
         {"p-error-level-1": 0.15325177356696879,
          "p-error-level-2": 0.28438100342155059}]

    tests["ml-sampler4-mac"] = \
        [["./sampler",
          "--num-samples", "1",
          "--max-levels", "3",
          "--spect-tol", "1.0",
          "--max-evects", "4"],
         {"p-error-level-1": 0.15902531435127545,
          "p-error-level-2": 0.29696652801349377}]

    # this is supposed to mimic using --choose-samples, but the choice
    # depends on cpu time, so for reproducibility we fix everything
    tests["qoi"] = \
        [["./qoi",
          "--coarse-factor", "16",
          "--max-levels", "2",
          "--fine-samples", "0",
          "--coarse-samples", "82",
          "--shared-samples", "18",
          "--choose-samples", "0",
          "--seed", "1"],
         {"coarse-variance":0.00019821590592782775,
          "correction-variance":5.532948306092317e-05,
          "mlmc-estimate":-0.0059401689449150533}]

    tests["qoi-mac"] = \
        [["./qoi",
          "--coarse-factor", "16",
          "--max-levels", "2",
          "--fine-samples", "0",
          "--coarse-samples", "82",
          "--shared-samples", "18",
          "--choose-samples", "0",
          "--seed", "1"],
         {"coarse-variance":0.00010774720329605776,
          "correction-variance":1.765201628796675e-05,
          "mlmc-estimate":-0.010527920345162218}]

    tests["qoi-hb"] = \
        [["./qoi",
          "--coarse-factor", "16",
          "--max-levels", "2",
          "--fine-samples", "0",
          "--coarse-samples", "82",
          "--shared-samples", "18",
          "--choose-samples", "0",
          "--hybridization",
          "--seed", "1"],
         {"coarse-variance":0.00019821590592782775,
          "correction-variance":5.532948306092317e-05,
          "mlmc-estimate":-0.0059401689449150533}]

    tests["qoi-hb-mac"] = \
        [["./qoi",
          "--coarse-factor", "16",
          "--max-levels", "2",
          "--fine-samples", "0",
          "--coarse-samples", "82",
          "--shared-samples", "18",
          "--choose-samples", "0",
          "--hybridization",
          "--seed", "1"],
         {"coarse-variance":0.00010774720329605776,
          "correction-variance":1.765201628796675e-05,
          "mlmc-estimate":-0.010527920345162218}]

    tests["qoi-one-level"] = \
        [["./qoi",
          "--coarse-factor", "16",
          "--max-levels", "1",
          "--fine-samples", "50",
          "--coarse-samples", "0",
          "--shared-samples", "0",
          "--choose-samples", "0",
          "--seed", "1"],
         {"correction-variance":0.00017825875122906521,
          "mlmc-estimate":-0.013182028965699699}]

    tests["fas-picard-one-level"] = \
        [["./nldarcy",
          "--alpha", "1.0",
          "--max-levels", "1",
          "--diff-tol", "5",
          "--num-relax-fine", "1",
          "--use-picard",
          "--perm", spe10_perm_file],
         {"nonlinear-iterations":59}]

    tests["fas-picard"] = \
        [["./nldarcy",
          "--alpha", "1.0",
          "--max-levels", "3",
          "--coarse-factor", "32",
          "--diff-tol", "5",
          "--num-relax-fine", "1",
          "--num-relax-mid", "1",
          "--num-relax-coarse", "30",
          "--max-traces", "1",
          "--max-evects", "1",
          "--use-picard",
          "--perm", spe10_perm_file],
         {"nonlinear-iterations":16}]

    tests["fas-newton-one-level"] = \
        [["./nldarcy",
          "--alpha", "1.0",
          "--max-levels", "1",
          "--diff-tol", "5",
          "--num-relax-fine", "1",
          "--use-newton",
          "--perm", spe10_perm_file],
         {"nonlinear-iterations":22}]

    tests["fas-newton"] = \
        [["./nldarcy",
          "--alpha", "1.0",
          "--max-levels", "3",
          "--coarse-factor", "32",
          "--diff-tol", "5",
          "--num-relax-fine", "1",
          "--num-relax-mid", "1",
          "--num-relax-coarse", "30",
          "--max-traces", "1",
          "--max-evects", "1",
          "--use-newton",
          "--perm", spe10_perm_file],
         {"nonlinear-iterations":3}]

    if "tux" in platform.node():
        tests["veigenvector"] = \
            [[memorycheck_command, "--leak-check=full",
              "mpirun", "-n", num_procs, "./finitevolume",
              "--max-evects", "1",
              "--spe10-scale", "1",
              "--perm", spe10_perm_file]]

        tests["vgraph-small-usegenerator"] = \
            [[memorycheck_command, "--leak-check=full",
              "./generalgraph",
              "--num-vert", "20",
              "--mean-degree", "4",
              "--spect-tol", "1.0",
              "--max-evects", "1",
              "--generate-graph"]]

        tests["vgraph-small-usegenerator-hb"] = \
            [[memorycheck_command, "--leak-check=full",
              "./generalgraph",
              "--hybridization",
              "--num-vert", "20",
              "--mean-degree", "4",
              "--spect-tol", "1.0",
              "--max-evects", "1",
              "--generate-graph"]]

    return tests


def run_all_tests(tests, verbose=False):
    """ Execute all tests and display results

    Any exception raised during a test counts as a
    failure

    Args:
        tests (dict):    tests to perform,
                         see make_tests for format

    Returns:
        int:     number of failed tests

    """
    totaltests = len(tests)
    success = 0

    for i, (name, test) in enumerate(tests.items()):
        try:
            result = run_test(*test, verbose=verbose)
        except BaseException as err:
            print("{0} Failed: {1}".format(name, err))
            result = False

        success += result
        status = "passed" if result else "FAILED"

        print("  ({0}/{1}) [{2}] {3}.".format(i + 1, totaltests, name, status))

    failures = totaltests - success

    print("Ran {0} tests with {1} successes and {2} failures.".format(
        totaltests, success, failures))

    return failures


def main(argv):
    """ Parses command line options and runs tests

    Empty commandline runs all tests
    Otherwise individual tests can be specified by name
    Pass in '-nv' with args to remove additional information

    Args:
        argv (list):     command line parameters

    Returns:
        int:     number of failed tests

    """
    verbose = True

    if "-nv" in argv:
        verbose = False
        argv.remove("-nv")

    tests = make_tests()

    if argv:
        tests = dict((name, tests[name]) for name in argv if name in tests)

    return run_all_tests(tests, verbose)


if __name__ == "__main__":
    exit(main(sys.argv[1:]))
