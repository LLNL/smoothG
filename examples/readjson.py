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
Look for a JSON object on last lines of a file.
Used to parse results from fv-mfem.cpp, eventually replacing fvparse.py

Copied from saamge-multilevel/test/startfromcoarse/readlog.py

Andrew T. Barker
atb@llnl.gov
1 December 2015
"""
from __future__ import print_function

import json
import sys


def json_parse_lines(lines, max_depth=10, max_height=6):
    for index in range(-1, -max_depth, -1):
        for i in range(max_height):
            try:
                name = "".join(lines[index - i:])
                return json.loads(name)
            except ValueError:
                pass
    return {}


def json_parse(fd, max_depth=10):
    """
    Look for a JSON object on the last few (max_depth) lines of a file.
    This is expecting the object to be on a single line.
    """
    lines = fd.readlines()
    return json_parse_lines(lines, max_depth)


def flatten_dict(d, name=""):
    """
    What we normally get out of json_parse is a dictionary, but maybe
    a too complicated one with dictionaries inside it. To make this
    play nice with standardized (old) table routines, this function "flattens"
    the dictionary, returning a dictionary without any dictionaries in it.

    Works fine in simple cases, but not expected to be totally general for all JSON objects.
    """
    out = {}
    for key, value in d.iteritems():
        # in python 3 we would use "abstract base class" here...
        if isinstance(value, dict):
            out.update(flatten_dict(value, key + "-"))
        else:
            out[name + key] = value
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "cosine.out"
    fd = open(filename, "r")
    print(json_parse(fd))
    fd.close()
