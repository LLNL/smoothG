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

# Contents of this file stolen from Parelag's ParELAGCMakeUtilities.cmake

# Function that uses "dumb" logic to try to figure out if a library
# file is a shared or static library. This won't work on Windows; it
# will just return "unknown" for everything.
function(parelag_determine_library_type lib_name output_var)

  # Test if ends in ".a"
  string(REGEX MATCH "\\.a$" _static_match ${lib_name})
  if (_static_match)
    set(${output_var} STATIC PARENT_SCOPE)
    return()
  endif (_static_match)

  # Test if ends in ".so(.version.id.whatever)"
  string(REGEX MATCH "\\.so($|..*$)" _shared_match ${lib_name})
  if (_shared_match)
    set(${output_var} SHARED PARENT_SCOPE)
    return()
  endif (_shared_match)

  # Test if ends in ".dylib(.version.id.whatever)"
  string(REGEX MATCH "\\.dylib($|\\..*$)" _mac_shared_match ${lib_name})
  if (_mac_shared_match)
    set(${output_var} SHARED PARENT_SCOPE)
    return()
  endif (_mac_shared_match)

  set(${output_var} "UNKNOWN" PARENT_SCOPE)
endfunction(parelag_determine_library_type lib_name output)
