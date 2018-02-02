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
Try to automate tests for smoothG code within a cmake
framework.

Andrew T. Barker
atb@llnl.gov
22 June 2016
"""
from __future__ import print_function

import subprocess

import readjson
import sys
import platform

spe10_perm_file = "@SPE10_PERM@"

class BasicTest:
    def __init__(self, triple):
        self.triple = triple
    def get_name(self):
        return self.triple[0]
    def get_test_output(self):
        commandline = self.triple[1]
        p = subprocess.Popen(commandline,stdout=subprocess.PIPE)
        stdout,stderr = p.communicate()
        return stdout
    def get_json(self,verbose):
        try:
            stdout = self.get_test_output()
        except OSError:
            return False
        if verbose:
            print(stdout)
        s = readjson.json_parse_lines(stdout.splitlines())
        return s
    def run_test(self,verbose=False):
        tests = self.triple[2]
        try:
            s = self.get_json(verbose)
        except OSError:
            return False
        out = True
        for t in tests.keys():
            try:
                expected = tests[t]
                actual = s[t]
            except KeyError:
                return False
            try:
                ev = float(expected)
                av = float(actual)
                out = out and (abs(ev-av) < 1.e-4)
            except ValueError:
                out = out and (actual == expected)
        return out

class ValgrindTest:
    def __init__(self,name,commandline):
        self.name = name
        self.commandline = commandline
    def get_name(self):
        return self.name
    def get_test_pair(self):
        precl = ["valgrind","--leak-check=full"]
        p = subprocess.Popen(precl + self.commandline, 
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout,stderr = p.communicate()
        return (stdout,stderr)
    def get_test_output(self):
        stdout,stderr = self.get_test_pair()
        out = stdout + stderr
        return out
    def run_test(self,verbose=False):
        try:
            stdout,stderr = self.get_test_pair()
        except OSError:
            return False
        if verbose:
            print(stdout)
            print(stderr)
        if ("All heap blocks were freed" in stderr
            and "ERROR SUMMARY: 0 errors" in stderr):
            return True
        else:
            return False

class TestSuite:
    def __init__(self):
        self.tests = []
    def add_test(self, name, commandline, expectation):
        self.tests.append( BasicTest((name, commandline, expectation)) )
    def add_ptest(self, test):
        self.tests.append(test)
    def run_particular(self,testname):
        for t in self.tests:
            if testname == t.get_name():
                result = t.run_test(True)
                return result
        return False
    def run(self):
        totaltests = len(self.tests)
        successes = 0
        failures = 0
        i = 0
        for test in self.tests:
            result = test.run_test(False)
            if result:
                print("  ({0:d}/{1:d}) [{2}] passed.".format(i,totaltests,test.get_name()))
                successes = successes + 1
            else:
                print("  ({0:d}/{1:d}) [{2}] FAILED.".format(i,totaltests,test.get_name()))
                failures = failures + 1
            i = i + 1
        print("Ran {0:d} tests with {1:d} successes and {2:d} failures.".format(totaltests, successes, failures))
        return failures
    
def make_test_suite():
    ts = TestSuite()

    ts.add_test("eigenvector1",
                ["./finitevolume",
                 "--spect-tol","1.0",
                 "--slice","0",
                 "--max-evects","1",
                 "--perm",spe10_perm_file],
                {"finest-div-error":2.0312444586906591e-08,
                 "finest-p-error":0.14743131732550618,
                 "finest-u-error":0.22621045683612057,
                 "operator-complexity":1.0221724964280585})
    ts.add_test("eigenvector4",
                ["./finitevolume",
                 "--spect-tol","1.0",
                 "--slice","0",
                 "--max-evects","4",
                 "--perm",spe10_perm_file],
                {"finest-div-error":2.0336350399372878e-08,
                 "finest-p-error":0.05516198497834629,
                 "finest-u-error":0.052317636963252999,
                 "operator-complexity":1.3017591339648173})
    ts.add_test("fv-hybridization",
                ["./finitevolume",
                 "--spect-tol","1.0",
                 "--slice","0",
                 "--max-evects","4",
                 "--hybridization",
                 "--perm",spe10_perm_file],
                {"finest-div-error":1.3301680521537587e-08,
                 "finest-p-error":0.055161984984368362,
                 "finest-u-error":0.052317636981330032,
                 "operator-complexity":1.1362437864707153})
    ts.add_test("slice19",
                ["./finitevolume",
                 "--spect-tol","1.0",
                 "--slice","19",
                 "--max-evects","1",
                 "--perm",spe10_perm_file],
                {"finest-div-error":1.2837519341678676e-08,
                 "finest-p-error":0.23763409361749516,
                 "finest-u-error":0.16419932734829923,
                 "operator-complexity":1.0221724964280585})
    ts.add_test("fv-metis",
                ["./finitevolume",
                 "--spect-tol","1.0",
                 "--slice","0",
                 "--max-evects","1",
                 "--metis-agglomeration",
                 "--perm",spe10_perm_file],
                {"finest-div-error":0.5640399150429396,
                 "finest-p-error":0.17385749780334459,
                 "finest-u-error":0.29785869880514693,
                 "operator-complexity":1.04042908656572})
    ts.add_test("fv-metis-mac",
                ["./finitevolume",
                 "--spect-tol","1.0",
                 "--slice","0",
                 "--max-evects","1",
                 "--metis-agglomeration",
                 "--perm",spe10_perm_file],
                {"finest-div-error":0.5420467322660617,
                 "finest-p-error":0.17088288700278217,
                 "finest-u-error":0.2031768008190909,
                 "operator-complexity":1.0398091940641514})
    ts.add_test("samplegraph1",
                ["./generalgraph",
                 "--spect-tol","1.0",
                 "--max-evects","1"],
                {"finest-div-error":0.37903965884799579,
                 "finest-p-error":0.38013398274257243,
                 "finest-u-error":0.38079825403520218,
                 "operator-complexity":1.016509834901651})
    ts.add_test("graph-metis",
                ["./generalgraph",
                 "--spect-tol","1.0",
                 "--max-evects","1",
                 "--metis-agglomeration"],
                {"finest-div-error":0.44735096079313674,
                 "finest-p-error":0.44939226988126274,
                 "finest-u-error":0.42773807524771068,
                 "operator-complexity":1.016509834901651})
    ts.add_test("graph-metis-mac",
                ["./generalgraph",
                "--spect-tol","1.0",
                "--max-evects","1",
                "--metis-agglomeration"],
                {"finest-div-error":0.22240585195370702,
                 "finest-p-error":0.22265174467689006,
                 "finest-u-error":0.22168973853676807,
                 "operator-complexity":1.016509834901651})
    ts.add_test("samplegraph4",
                ["./generalgraph",
                 "--spect-tol","1.0",
                 "--max-evects","4"],
                {"finest-div-error":0.12043046187567592,
                 "finest-p-error":0.13514675917148347,
                 "finest-u-error":0.19926779054787247,
                 "operator-complexity":1.2578874211257887})
    ts.add_test("graph-hybridization",
                ["./generalgraph",
                 "--spect-tol","1.0",
                 "--max-evects","4",
                 "--hybridization"],
                {"finest-div-error":0.12051328492652449,
                 "finest-p-error":0.13514675917148347,
                 "finest-u-error":0.19926779054787247,
                 "operator-complexity":1.013984620448976})
    ts.add_test("graph-usegenerator",
                ["./generalgraph",
                 "--spect-tol","1.0",
                 "--max-evects","4",
                 "--generate-graph"],
                {"finest-div-error":0.11283262603381641,
                 "finest-p-error":0.1203852548326301,
                 "finest-u-error":0.16674213482507089,
                 "operator-complexity":1.2578874211257887})
    ts.add_test("graph-usegenerator-mac",
                ["./generalgraph",
                 "--spect-tol","1.0",
                 "--max-evects","4",
                 "--generate-graph"],
                {"finest-div-error":0.10665922015088503,
                 "finest-p-error":0.10863131137013603,
                 "finest-u-error":0.12848813745253315,
                 "operator-complexity":1.2578874211257887})
    ts.add_test("pareigenvector1",
                ["mpirun","-n","4","./finitevolume",
                 "--spect-tol","1.0",
                 "--slice","0",
                 "--max-evects","1",
                 "--perm",spe10_perm_file],
                {"finest-div-error":2.0312444586906591e-08,
                 "finest-p-error":0.14743131732550618,
                 "finest-u-error":0.22621045683612057,
                 "operator-complexity":1.0221724964280585})
    ts.add_test("pareigenvector4",
                ["mpirun","-n","4","./finitevolume",
                 "--spect-tol","1.0",
                 "--slice","0",
                 "--max-evects","4",
                 "--perm",spe10_perm_file],
                {"finest-div-error":2.0336350399372878e-08,
                 "finest-p-error":0.05516198497834629,
                 "finest-u-error":0.052317636963252999,
                 "operator-complexity":1.3017591339648173})
    ts.add_test("parfv-hybridization",
                ["mpirun","-n","4","./finitevolume",
                 "--spect-tol","1.0",
                 "--slice","0",
                 "--max-evects","4",
                 "--hybridization",
                 "--perm",spe10_perm_file],
                {"finest-div-error":1.3301680521537587e-08,
                 "finest-p-error":0.055161984984368362,
                 "finest-u-error":0.052317636981330032,
                 "operator-complexity":1.1362437864707153})
    ts.add_test("parslice19",
                ["mpirun","-n","4","./finitevolume",
                 "--spect-tol","1.0",
                 "--slice","19",
                 "--max-evects","1",
                 "--perm",spe10_perm_file],
                {"finest-div-error":1.2837519341678676e-08,
                 "finest-p-error":0.23763409361749516,
                 "finest-u-error":0.16419932734829923,
                 "operator-complexity":1.0221724964280585})
    ts.add_test("parsamplegraph1",
                ["mpirun","-n","4","./generalgraph",
                 "--spect-tol","1.0",
                 "--max-evects","1"],
                {"finest-div-error":0.37903965884799579,
                 "finest-p-error":0.38013398274257243,
                 "finest-u-error":0.38079825403520218,
                 "operator-complexity":1.016509834901651})
    ts.add_test("pargraph-metis",
                ["mpirun","-n","4","./generalgraph",
                 "--spect-tol","1.0",
                 "--max-evects","1",
                 "--metis-agglomeration"],
                {"finest-div-error":0.44735096079313674,
                 "finest-p-error":0.44939226988126274,
                 "finest-u-error":0.42773807524771068,
                 "operator-complexity":1.016509834901651})
    ts.add_test("pargraph-metis-mac",
                ["mpirun","-n","4","./generalgraph",
                 "--spect-tol","1.0",
                 "--max-evects","1",
                 "--metis-agglomeration"],
                {"finest-div-error":0.22240585195370702,
                 "finest-p-error":0.22265174467689006,
                 "finest-u-error":0.22168973853676807,
                 "operator-complexity":1.016509834901651})
    ts.add_test("parsamplegraph4",
                ["mpirun","-n","4","./generalgraph",
                 "--spect-tol","1.0",
                 "--max-evects","4"],
                {"finest-div-error":0.12043046187567592,
                 "finest-p-error":0.13514675917148347,
                 "finest-u-error":0.19926779054787247,
                 "operator-complexity":1.2578874211257887})
    ts.add_test("pargraph-hybridization",
                ["mpirun","-n","4","./generalgraph",
                 "--spect-tol","1.0",
                 "--max-evects","4",
                 "--hybridization"],
                {"finest-div-error":0.12051328492652449,
                 "finest-p-error":0.13514675917148347,
                 "finest-u-error":0.19926779054787247,
                 "operator-complexity":1.013984620448976})
    ts.add_test("pargraph-usegenerator",
                ["mpirun","-n","4","./generalgraph",
                 "--spect-tol","1.0",
                 "--max-evects","4",
                 "--generate-graph"],
                {"finest-div-error":0.11283262603381641,
                 "finest-p-error":0.1203852548326301,
                 "finest-u-error":0.16674213482507089,
                 "operator-complexity":1.2578874211257887})
    ts.add_test("pargraph-usegenerator-mac",
                ["mpirun","-n","4","./generalgraph",
                 "--spect-tol","1.0",
                 "--max-evects","4",
                 "--generate-graph"],
                {"finest-div-error":0.10665922015088503,
                 "finest-p-error":0.10863131137013603,
                 "finest-u-error":0.12848813745253315,
                 "operator-complexity":1.2578874211257887})
    ts.add_test("isolate-coarsen",
                ["./generalgraph",
                 "--metis-agglomeration",
                 "--spect-tol","1.0",
                 "--max-evects","4",
                 "--isolate","0"],
                {"operator-complexity":1.2736672633273667})

    if "tux" in platform.node():
        ts.add_ptest(ValgrindTest("veigenvector",
                                  ["mpirun","-n","4","./finitevolume",
                                   "--max-evects","1",
                                   "--spe10-scale","1",
                                   "--perm",spe10_perm_file]))

    return ts

def main():
    ts = make_test_suite()
    if len(sys.argv) == 1:
        return ts.run()
    else:
        result = ts.run_particular(sys.argv[1])
        if result:
            return 0
        return 1

if __name__ == "__main__":
    exit(main())
