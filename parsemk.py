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
This essentially translates the config.mk file from MFEM
into a out.cmake file to import the dependencies that MFEM
uses into our configuration process.
"""
from __future__ import print_function
from __future__ import division

import sys


def matchopencheck(item, op):
    if len(item) > len(op) and item[:len(op)] == op:
        return True
    else:
        return False


def matchopen(item, op, l):
    b = matchopencheck(item, op)
    if b:
        l.append(item[len(op):])
    return b


def name_package(p, expected_names):
    """
    loop order here together with order of
    expected_names expresses the naming preference I want
    (ie, I want ["HYPRE", "lapack", "blas"] to be named "HYPRE", not "lapack")
    """
    for en in expected_names:
        for l in p["libraries"]:
            if en == l:
                p["name"] = en
                return
    p["name"] = "unexpected:" + p["libraries"][0]


def parse_packages(filename="config.mk", verbose=False):
    # print("Opening", filename)
    rpathopen = "-Wl,-rpath,"
    linkpathopen = "-L"
    libraryopen = "-l"
    other = []
    packages = []
    includes = []
    status = "library"  # switch *from* library to rpath triggers new package
    with open(filename, "r") as fd:
        for line in fd:
            p = line.split()
            if len(p) > 0 and p[0] == "MFEM_TPLFLAGS":
                for item in p[2:]:
                    if len(item) > 2 and item[0:2] == "-I":
                        includes.append(item[2:])
            if len(p) > 0 and p[0] == "MFEM_EXT_LIBS":
                # print("Found MFEM_EXT_LIBS.")
                for item in p[2:]:
                    b = False
                    rp = matchopencheck(item, rpathopen)
                    if status == "library" and rp:
                        packages.append({"rpaths": [],
                                         "linkpaths": [],
                                         "libraries": []})
                        status = "rpath"
                    b = b or matchopen(item, rpathopen, packages[-1]["rpaths"])
                    lp = matchopen(item, linkpathopen, packages[-1]["linkpaths"])
                    if lp:
                        status = "library"
                    b = b or lp
                    b = b or matchopen(item, libraryopen, packages[-1]["libraries"])
                    if not b:
                        other.append(item)
    if len(other) > 0:
        print("WARNING: could not parse MFEM_EXT_LIBS: did not understand following tokens:")
        for o in other:
            print("  ", o)
    expected_names = ["HYPRE", "metis", "suitesparseconfig",
                      "unwind", "z", "lapack"]
    for p in packages:
        name_package(p, expected_names)
    if verbose:
        print("packages:")
        for p in packages:
            print("\n  ", p["name"])
            print("  ", "rpaths:")
            for r in p["rpaths"]:
                print("    ", r)
            print("  ", "linkpaths:")
            for l in p["linkpaths"]:
                print("    ", l)
            print("  ", "libraries:")
            for l in p["libraries"]:
                print("    ", l)
    return packages, includes


def save_cmake_packages(packages, includes, filename="out.cmake"):
    with open(filename, "w") as fd:
        fd.write('set(MK_INCLUDES "")\n\n')
        for i in includes:
            fd.write("list(APPEND MK_INCLUDES " + i + ")\n")
        fd.write("\n")
        fd.write('set(MK_LIBRARIES "")\n\n')
        for p in packages:
            fd.write("# " + p["name"] + "\n")
            for l in p["libraries"]:
                cmake_name = l + "_LIB"
                fd.write("find_library(" + cmake_name + " " + l + " PATHS ")
                for lp in p["linkpaths"]:
                    fd.write(lp + " ")
                fd.write(")\n")
                fd.write("list(APPEND MK_LIBRARIES ${" + cmake_name + "})\n")
            fd.write("\n")
        fd.write("list(REMOVE_DUPLICATES MK_LIBRARIES)\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        packages, includes = parse_packages(filename=sys.argv[1], verbose=False)
    else:
        packages, includes = parse_packages(verbose=False)
    save_cmake_packages(packages, includes)
