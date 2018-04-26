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
test_tol = float("@SMOOTHG_TEST_TOL@")

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

    for key, expected_val in expected.items():
        test_val = output[key]

        if abs(float(expected_val) - float(test_val)) > test_tol:
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
          "--perm", spe10_perm_file],
         {"finest-div-error": 2.0312444586906591e-08,
          "finest-p-error": 0.14743131732550618,
          "finest-u-error": 0.22621045683612057,
          "operator-complexity": 1.0221724964280585}]

    tests["eigenvector4"] = \
        [["./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--perm", spe10_perm_file],
         {"finest-div-error": 2.0336350399372878e-08,
          "finest-p-error": 0.05516198497834629,
          "finest-u-error": 0.052317636963252999,
          "operator-complexity": 1.3017591339648173}]

    tests["fv-hybridization"] = \
        [["./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--hybridization",
          "--perm", spe10_perm_file],
         {"finest-div-error": 1.3301680521537587e-08,
          "finest-p-error": 0.055161984984368362,
          "finest-u-error": 0.052317636981330032,
          "operator-complexity": 1.1362437864707153}]

    tests["slice19"] = \
        [["./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "19",
          "--max-evects", "1",
          "--perm", spe10_perm_file],
         {"finest-div-error": 1.2837519341678676e-08,
          "finest-p-error": 0.23763409361749516,
          "finest-u-error": 0.16419932734829923,
          "operator-complexity": 1.0221724964280585}]

    tests["fv-metis"] = \
        [["./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "1",
          "--metis-agglomeration",
          "--perm", spe10_perm_file],
         {"finest-div-error": 0.5640399150429396,
          "finest-p-error": 0.17385749780334459,
          "finest-u-error": 0.29785869880514693,
          "operator-complexity": 1.04042908656572}]

    tests["fv-metis-mac"] = \
        [["./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "1",
          "--metis-agglomeration",
          "--perm", spe10_perm_file],
         {"finest-div-error": 0.5420467322660617,
          "finest-p-error": 0.17088288700278217,
          "finest-u-error": 0.2031768008190909,
          "operator-complexity": 1.0398091940641514}]

    tests["dual-trace"] = \
        [["./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--dual-target",
          "--perm", spe10_perm_file],
         {"finest-div-error": 3.2049690562060094e-08,
          "finest-p-error": 0.055207481027916193,
          "finest-u-error": 0.06430185063505546,
          "operator-complexity": 1.3017591339648173}]

    tests["scaled-dual-trace"] = \
        [["./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--dual-target",
          "--scaled-dual",
          "--perm", spe10_perm_file],
         {"finest-div-error": 1.9821133537907875e-08,
          "finest-p-error": 0.055054636384856817,
          "finest-u-error": 0.034260930604399109,
          "operator-complexity": 1.3017591339648173}]

    tests["energy-dual-trace"] = \
        [["./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--dual-target",
          "--energy-dual",
          "--perm", spe10_perm_file],
         {"finest-div-error": 3.2032854597960414e-8,
          "finest-p-error": 0.055279347333696799,
          "finest-u-error": 0.068336534035533678,
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
         {"finest-div-error": 3.2079666982238907e-8,
          "finest-p-error": 0.055052992284074398,
          "finest-u-error": 0.035336370431801843,
          "operator-complexity": 1.3017591339648173}]

    tests["samplegraph1"] = \
        [["./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "1"],
         {"finest-div-error": 0.37918423747873353,
          "finest-p-error": 0.38013398274257243,
          "finest-u-error": 0.38079825403520218,
          "operator-complexity": 1.016509834901651}]

    tests["samplegraph1-coeff"] = \
        [["./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "1",
          "--coarse-coefficient"],
         {"finest-div-error": 0.37918423747873353,
          "finest-p-error": 0.38013398274257243,
          "finest-u-error": 0.38079825403520218,
          "operator-complexity": 1.016509834901651}]

    tests["samplegraph1-coeff-hb"] = \
        [["./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "1",
          "--coarse-coefficient",
          "--hybridization"],
         {"finest-div-error": 0.37918423747873353,
          "finest-p-error": 0.38013398274257243,
          "finest-u-error": 0.38079825403520218,
          "operator-complexity": 1.000874038778061}]

    tests["parsamplegraph1-coeff"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "1",
          "--coarse-coefficient"],
         {"finest-div-error": 0.37918423727222522,
          "finest-p-error": 0.38013398274257243,
          "finest-u-error": 0.38079825403520218,
          "operator-complexity": 1.016509834901651}]

    tests["parsamplegraph1-coeff-hb"] = \
        [["mpirun", "-n", "4", "./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "1",
          "--coarse-coefficient",
          "--hybridization"],
         {"finest-div-error": 0.37918423727222522,
          "finest-p-error": 0.38013398274257243,
          "finest-u-error": 0.38079825403520218,
          "operator-complexity": 1.000874038778061}]

    tests["graph-metis"] = \
        [["./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "1",
          "--metis-agglomeration"],
         {"finest-div-error": 0.44710819907744104,
          "finest-p-error": 0.44939226988126274,
          "finest-u-error": 0.42773807524771068,
          "operator-complexity": 1.016509834901651}]

    tests["graph-metis-mac"] = \
        [["./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "1",
          "--metis-agglomeration"],
         {"finest-div-error": 0.22228470008233389,
          "finest-p-error": 0.22265174467689006,
          "finest-u-error": 0.22168973853676807,
          "operator-complexity": 1.016509834901651}]

    tests["samplegraph4"] = \
        [["./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4"],
         {"finest-div-error": 0.12043046187567592,
          "finest-p-error": 0.13514675917148347,
          "finest-u-error": 0.19926779054787247,
          "operator-complexity": 1.2578874211257887}]

    tests["samplegraph4-coeff"] = \
        [["./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4",
          "--coarse-coefficient"],
         {"finest-div-error": 0.12043046187567592,
          "finest-p-error": 0.13514675917148347,
          "finest-u-error": 0.19926779054787247,
          "operator-complexity": 1.2578874211257887}]

    tests["samplegraph4-coeff-hb"] = \
        [["./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4",
          "--coarse-coefficient",
          "--hybridization"],
         {"finest-div-error": 0.12043046187567592,
          "finest-p-error": 0.13514675917148347,
          "finest-u-error": 0.19926779054787247,
          "operator-complexity": 1.013984620448976}]

    tests["graph-hybridization"] = \
        [["./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4",
          "--hybridization"],
         {"finest-div-error": 0.12051328492652449,
          "finest-p-error": 0.13514675917148347,
          "finest-u-error": 0.19926779054787247,
          "operator-complexity": 1.013984620448976}]

    tests["graph-usegenerator"] = \
        [["./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4",
          "--generate-graph"],
         {"finest-div-error": 0.11283262603381641,
          "finest-p-error": 0.1203852548326301,
          "finest-u-error": 0.16674213482507089,
          "operator-complexity": 1.2578874211257887}]

    tests["graph-usegenerator-mac"] = \
        [["./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4",
          "--generate-graph"],
         {"finest-div-error": 0.1070681247529167,
          "finest-p-error": 0.10863131137013603,
          "finest-u-error": 0.12848813745253315,
          "operator-complexity": 1.2578874211257887}]

    tests["poweriter"] = \
        [["./poweriter"],
         {"coarse-error": 0.2050307003818391,
          "coarse-eval": 0.17663653196285808,
          "fine-error": 2.9382710663490486e-05,
          "fine-eval": 0.17545528997990226}]

    tests["graph-weight"] = \
        [["./generalgraph",
          "--graph", graph_data + "/vertex_edge_tiny.txt",
          "--weight", graph_data + "/tiny_weights.txt",
          "--generate-fiedler",
          "--metis-agglomeration",
          "--num-part", "2"],
         {"finest-div-error": 0.3033520464019937,
          "finest-p-error": 0.31217311873637132,
          "finest-u-error": 0.14767829457535478,
          "operator-complexity": 1.1666666666666667}]

    tests["timestep"] = \
        [["./timestep",
          "--total-time", "100.0",
          "--perm", spe10_perm_file]]

    tests["pareigenvector1"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "1",
          "--perm", spe10_perm_file],
         {"finest-div-error": 2.0312444586906591e-08,
          "finest-p-error": 0.14743131732550618,
          "finest-u-error": 0.22621045683612057,
          "operator-complexity": 1.0221724964280585}]

    tests["pareigenvector4"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--perm", spe10_perm_file],
         {"finest-div-error": 2.0336350399372878e-08,
          "finest-p-error": 0.05516198497834629,
          "finest-u-error": 0.052317636963252999,
          "operator-complexity": 1.3017591339648173}]

    tests["parfv-hybridization"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--hybridization",
          "--perm", spe10_perm_file],
         {"finest-div-error": 1.3301680521537587e-08,
          "finest-p-error": 0.055161984984368362,
          "finest-u-error": 0.052317636981330032,
          "operator-complexity": 1.1362437864707153}]

    tests["parslice19"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "19",
          "--max-evects", "1",
          "--perm", spe10_perm_file],
         {"finest-div-error": 1.2837519341678676e-08,
          "finest-p-error": 0.23763409361749516,
          "finest-u-error": 0.16419932734829923,
          "operator-complexity": 1.0221724964280585}]

    tests["pardual-trace"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--dual-target",
          "--perm", spe10_perm_file],
         {"finest-div-error": 3.2049690562060094e-08,
          "finest-p-error": 0.055207481027916193,
          "finest-u-error": 0.06430185063505546,
          "operator-complexity": 1.3017591339648173}]

    tests["parscaled-dual-trace"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--dual-target",
          "--scaled-dual",
          "--perm", spe10_perm_file],
         {"finest-div-error": 1.9821133537907875e-08,
          "finest-p-error": 0.055054636384856817,
          "finest-u-error": 0.034260930604399109,
          "operator-complexity": 1.3017591339648173}]

    tests["parenergy-dual-trace"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--spect-tol", "1.0",
          "--slice", "0",
          "--max-evects", "4",
          "--dual-target",
          "--energy-dual",
          "--perm", spe10_perm_file],
         {"finest-div-error": 3.2032854597960414e-8,
          "finest-p-error": 0.055279347333696799,
          "finest-u-error": 0.068336534035533678,
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
         {"finest-div-error": 3.2079666982238907e-8,
          "finest-p-error": 0.055052992284074398,
          "finest-u-error": 0.035336370431801843,
          "operator-complexity": 1.3017591339648173}]

    tests["parsamplegraph1"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "1"],
         {"finest-div-error": 0.37918423727222522,
          "finest-p-error": 0.38013398274257243,
          "finest-u-error": 0.38079825403520218,
          "operator-complexity": 1.016509834901651}]

    tests["pargraph-metis"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "1",
          "--metis-agglomeration"],
         {"finest-div-error": 0.44710819906667049,
          "finest-p-error": 0.44939226988126274,
          "finest-u-error": 0.42773807524771068,
          "operator-complexity": 1.016509834901651}]

    tests["pargraph-metis-mac"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "1",
          "--metis-agglomeration"],
         {"finest-div-error": 0.22228470008233389,
          "finest-p-error": 0.22265174467689006,
          "finest-u-error": 0.22168973853676807,
          "operator-complexity": 1.016509834901651}]

    tests["parsamplegraph4"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4"],
         {"finest-div-error": 0.12043046187567592,
          "finest-p-error": 0.13514675917148347,
          "finest-u-error": 0.19926779054787247,
          "operator-complexity": 1.2578874211257887}]

    tests["pargraph-hybridization"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4",
          "--hybridization"],
         {"finest-div-error": 0.12051328492652449,
          "finest-p-error": 0.13514675917148347,
          "finest-u-error": 0.19926779054787247,
          "operator-complexity": 1.013984620448976}]

    tests["pargraph-usegenerator"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4",
          "--generate-graph"],
         {"finest-div-error": 0.11283262603381641,
          "finest-p-error": 0.1203852548326301,
          "finest-u-error": 0.16674213482507089,
          "operator-complexity": 1.2578874211257887}]

    tests["pargraph-usegenerator-mac"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--spect-tol", "1.0",
          "--max-evects", "4",
          "--generate-graph"],
         {"finest-div-error": 0.1070681247529167,
          "finest-p-error": 0.10863131137013603,
          "finest-u-error": 0.12848813745253315,
          "operator-complexity": 1.2578874211257887}]

    tests["parpoweriter"] = \
        [["mpirun", "-n", num_procs, "./poweriter"],
         {"coarse-error": 0.20499789652195419,
          "coarse-eval": 0.17663653207421526,
          "fine-error": 2.9887390635842169e-05,
          "fine-eval": 0.17545528997977797}]

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
