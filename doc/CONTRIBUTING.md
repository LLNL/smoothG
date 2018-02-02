<!-- BHEADER ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 +
 + Copyright (c) 2018, Lawrence Livermore National Security, LLC.
 + Produced at the Lawrence Livermore National Laboratory.
 + LLNL-CODE-745247. All Rights reserved. See file COPYRIGHT for details.
 +
 + This file is part of smoothG. For more information and source code
 + availability, see https://www.github.com/llnl/smoothG.
 +
 + smoothG is free software; you can redistribute it and/or modify it under the
 + terms of the GNU Lesser General Public License (as published by the Free
 + Software Foundation) version 2.1 dated February 1999.
 +
 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ EHEADER -->

# Contributing Guidelines

## Naming conventions

* Member variable names all have trailing underscores.
* Classes are capitalized.
* Words in classes are differentiated by capitalizing each word. E.g. "MyClass"
* Functions and variables are not capitalized.
* Words in functions and variables are separated by an underscore. E.g. "my_member_var_"

## Language feature conventions

* We require C++11, but not C++14.

## Programming style conventions

* Each class receives its own file, with an exception for multiple tiny highly-related classes
* Each file has the same name as the class.
* The body of a function should fit onto your screen. If it doesn't, split it up into subfunctions.
* We use 4 space indent (no tabs) and Allman or BSD style braces, as enforced by `make style`. See smoothg.astylerc for details.

## Documentation conventions

* We use Doxygen for inline documentation, see `make doc`.
* At a minimum each class's public interface should be documented.
