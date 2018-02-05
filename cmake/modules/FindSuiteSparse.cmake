# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

# Sets the following variables
#   - SuiteSparse_FOUND
#   - SuiteSparse_INCLUDE_DIRS
#   - SuiteSparse_LIBRARIES
#
# We need the following libraries:
#   umfpack, cholmod, amd, camd, colamd, ccolamd, suitesparseconfig, klu, btf
#

# Auxiliary function, used in find_package().
function(mfem_find_component Prefix DirVar IncSuffixes Header LibSuffixes Lib
         IncDoc LibDoc)

  if (Lib)
    if (${DirVar} OR EnvDirVar)
      find_library(${Prefix}_LIBRARY ${Lib}
        HINTS ${${DirVar}} ENV ${DirVar}
        PATH_SUFFIXES ${LibSuffixes}
        NO_DEFAULT_PATH
        DOC "${LibDoc}")
    endif()
    find_library(${Prefix}_LIBRARY ${Lib}
      PATH_SUFFIXES ${LibSuffixes}
      DOC "${LibDoc}")
  endif()

  if (Header)
    if (${DirVar} OR EnvDirVar)
      find_path(${Prefix}_INCLUDE_DIR ${Header}
        HINTS ${${DirVar}} ENV ${DirVar}
        PATH_SUFFIXES ${IncSuffixes}
        NO_DEFAULT_PATH
        DOC "${IncDoc}")
    endif()
    find_path(${Prefix}_INCLUDE_DIR ${Header}
      PATH_SUFFIXES ${IncSuffixes}
      DOC "${IncDoc}")
  endif()

endfunction(mfem_find_component)

function(mfem_find_package Name Prefix DirVar IncSuffixes Header LibSuffixes
         Lib IncDoc LibDoc)

  # Quick return
  if (${Prefix}_FOUND)
    return()
  elseif (${Prefix}_INCLUDE_DIRS OR ${Prefix}_LIBRARIES)
    # If ${Prefix}_INCLUDE_DIRS or ${Prefix}_LIBRARIES are defined, accept them
    # silently.
    set(${Prefix}_FOUND TRUE CACHE BOOL "${Name} was found." FORCE)
    return()
  endif()

  set(EnvDirVar "$ENV{${DirVar}}")
  if (NOT ${Name}_FIND_QUIETLY)
    if (NOT ${Name}_SKIP_LOOKING_MSG)
      message(STATUS "Looking for ${Name} ...")
    endif()
    if (${DirVar})
      message(STATUS "   in ${DirVar} = ${${DirVar}}")
    endif()
    if (EnvDirVar)
      message(STATUS "   in ENV{${DirVar}} = ${EnvDirVar}")
    endif()
  endif()

  mfem_find_component("${Prefix}" "${DirVar}" "${IncSuffixes}" "${Header}"
    "${LibSuffixes}" "${Lib}" "${IncDoc}" "${LibDoc}")

  if (((NOT Lib) OR ${Prefix}_LIBRARY) AND
      ((NOT Header) OR ${Prefix}_INCLUDE_DIR))
    set(Found TRUE)
  else()
    set(Found FALSE)
  endif()
  set(${Prefix}_LIBRARIES ${${Prefix}_LIBRARY})
  set(${Prefix}_INCLUDE_DIRS ${${Prefix}_INCLUDE_DIR})

  set(ReqVars "")

  # Check for optional "ADD_COMPONENT" arguments.
  set(I 9) # 9 is the number of required arguments
  while(I LESS ARGC)
    if ("${ARGV${I}}" STREQUAL "CHECK_BUILD")
      # "CHECK_BUILD" has 3 arguments, handled below
      math(EXPR I "${I}+3")
    elseif ("${ARGV${I}}" STREQUAL "ADD_COMPONENT")
      # "ADD_COMPONENT" has 5 arguments:
      # CompPrefix CompIncSuffixes CompHeader CompLibSuffixes CompLib
      math(EXPR I "${I}+1")
      set(CompPrefix "${ARGV${I}}")
      math(EXPR I "${I}+1")
      set(CompIncSuffixes "${ARGV${I}}")
      math(EXPR I "${I}+1")
      set(CompHeader "${ARGV${I}}")
      math(EXPR I "${I}+1")
      set(CompLibSuffixes "${ARGV${I}}")
      math(EXPR I "${I}+1")
      set(CompLib "${ARGV${I}}")
      # Determine if the component is requested.
      list(FIND ${Name}_FIND_COMPONENTS ${CompPrefix} CompIdx)
      if (CompIdx GREATER -1)
        set(CompRequested TRUE)
      else()
        set(CompRequested FALSE)
      endif()
      # Determine if the component is optional or required.
      set(CompRequired ${${Name}_FIND_REQUIRED_${CompPrefix}})
      if (CompRequested)
        set(FullPrefix "${Prefix}_${CompPrefix}")
        mfem_find_component("${FullPrefix}" "${DirVar}"
          "${CompIncSuffixes}" "${CompHeader}"
          "${CompLibSuffixes}" "${CompLib}" "" "")
        if (CompRequired)
          if (CompLib)
            list(APPEND ReqVars ${FullPrefix}_LIBRARY)
          endif()
          if (CompHeader)
            list(APPEND ReqVars ${FullPrefix}_INCLUDE_DIR)
          endif()
        endif(CompRequired)
        if (((NOT CompLib) OR ${FullPrefix}_LIBRARY) AND
            ((NOT CompHeader) OR ${FullPrefix}_INCLUDE_DIR))
          # Component found
          list(APPEND ${Prefix}_LIBRARIES ${${FullPrefix}_LIBRARY})
          list(APPEND ${Prefix}_INCLUDE_DIRS ${${FullPrefix}_INCLUDE_DIR})
          if (NOT ${Name}_FIND_QUIETLY)
            message(STATUS
              "${Name}: ${CompPrefix}: ${${FullPrefix}_LIBRARY}")
            # message(STATUS
            #   "${Name}: ${CompPrefix}: ${${FullPrefix}_INCLUDE_DIR}")
          endif()
        else()
          # Let FindPackageHandleStandardArgs() handle errors
          if (NOT ${Name}_FIND_QUIETLY)
            message(STATUS "${Name}: ${CompPrefix}: *** NOT FOUND ***")
          endif()
        endif()
      endif(CompRequested)
    else()
      message(FATAL_ERROR "Unknown argument: ${ARGV${I}}")
    endif()
    math(EXPR I "${I}+1")
  endwhile()

  # Add required / optional / alternative packages.
  set(Required "REQUIRED")
  set(Quiet "")
  if (${Name}_FIND_QUIETLY)
    set(Quiet "QUIET")
  endif()
  set(Alternative FALSE)
  foreach(ReqPack IN LISTS ${Name}_REQUIRED_PACKAGES)
    # Parse the pattern: <PackName>[/<CompName>]...
    string(REPLACE "/" ";" PackComps "${ReqPack}")
    list(GET PackComps 0 PackName)
    list(REMOVE_AT PackComps 0)
    set(ReqPack "${PackName}")
    set(ReqPackM "${ReqPack}")
    if (NOT ("${PackComps}" STREQUAL ""))
       set(ReqPackM "${ReqPackM}, COMPONENTS: ${PackComps}")
    endif()
    if (Quiet)
      set(ReqPackM "${ReqPackM} (quiet)")
    endif()
    if ("${ReqPack}" STREQUAL "REQUIRED:")
      set(Required "REQUIRED")
    elseif ("${ReqPack}" STREQUAL "OPTIONAL:")
      set(Required "")
    elseif ("${ReqPack}" STREQUAL "QUIET:")
      set(Quiet "QUIET")
    elseif ("${ReqPack}" STREQUAL "VERBOSE:")
      set(Quiet "")
      if (${Name}_FIND_QUIETLY)
        set(Quiet "QUIET")
      endif()
    elseif ("${ReqPack}" STREQUAL "ALT:")
      set(Alternative TRUE)
    elseif ((NOT Found) AND Alternative)
      set(Alternative FALSE)
      if (NOT ${Name}_FIND_QUIETLY)
        message(STATUS "${Name}: trying alternative package: ${ReqPackM}")
      endif()
      # Do not add ${Required} here, since that will prevent other potential
      # alternative packages from being found.
      find_package(${ReqPack} ${Quiet} COMPONENTS ${PackComps})
      string(TOUPPER ${ReqPack} ReqPACK)
      if (${ReqPack}_FOUND)
        set(Found TRUE)
        set(${Prefix}_LIBRARIES ${${ReqPack}_LIBRARIES})
        set(${Prefix}_INCLUDE_DIRS ${${ReqPack}_INCLUDE_DIRS})
      elseif (${ReqPACK}_FOUND)
        set(Found TRUE)
        set(${Prefix}_LIBRARIES ${${ReqPACK}_LIBRARIES})
        set(${Prefix}_INCLUDE_DIRS ${${ReqPACK}_INCLUDE_DIRS})
      endif()
    elseif (Alternative)
      set(Alternative FALSE)
    elseif (Found)
      if (NOT ${Name}_FIND_QUIETLY)
        if (Required)
          message(STATUS "${Name}: looking for required package: ${ReqPackM}")
        else()
          message(STATUS "${Name}: looking for optional package: ${ReqPackM}")
        endif()
      endif()
      string(TOUPPER ${ReqPack} ReqPACK)
      if (NOT (${ReqPack}_FOUND OR ${ReqPACK}_FOUND))
        if (NOT ${ReqPack}_TARGET_NAMES)
          find_package(${ReqPack} ${Required} ${Quiet} COMPONENTS ${PackComps})
        else()
          foreach(_target ${ReqPack} ${${ReqPack}_TARGET_NAMES})
            # Do not use ${Required} here:
            find_package(${_target} NAMES ${_target} ${ReqPack} ${Quiet}
              COMPONENTS ${PackComps})
            string(TOUPPER ${_target} _TARGET)
            if (${_target}_FOUND OR ${_TARGET}_FOUND)
              set(${ReqPack}_FOUND TRUE)
              break()
            endif()
          endforeach()
          if (${Required} AND NOT ${ReqPack}_FOUND)
            message(FATAL_ERROR " *** Required package ${ReqPack} not found."
              "Checked target names: ${ReqPack} ${${ReqPack}_TARGET_NAMES}")
          endif()
        endif()
      endif()
      if (Required AND NOT (${ReqPack}_FOUND OR ${ReqPACK}_FOUND))
        message(FATAL_ERROR " --------- INTERNAL ERROR")
      endif()
      if ("${ReqPack}" STREQUAL "MPI" AND MPI_CXX_FOUND)
        list(APPEND ${Prefix}_LIBRARIES ${MPI_CXX_LIBRARIES})
        list(APPEND ${Prefix}_INCLUDE_DIRS ${MPI_CXX_INCLUDE_PATH})
      elseif (${ReqPack}_FOUND OR ${ReqPACK}_FOUND)
        if (${ReqPack}_FOUND)
          set(_Pack ${ReqPack})
        else()
          set(_Pack ${ReqPACK})
        endif()
        set(_Pack_LIBS)
        set(_Pack_INCS)
        # - ${_Pack}_CONFIG is defined by find_package() when a config file was
        #   loaded
        # - If ${ReqPack}_TARGET_NAMES is defined, use target mode
        if (NOT ((DEFINED ${_Pack}_CONFIG) OR
                 (DEFINED ${ReqPack}_TARGET_NAMES)))
          # Defined variables expected:
          # - ${ReqPack}_LIB_VARS, optional, default: ${_Pack}_LIBRARIES
          # - ${ReqPack}_INCLUDE_VARS, optional, default: ${_Pack}_INCLUDE_DIRS
          set(_lib_vars ${${ReqPack}_LIB_VARS})
          if (NOT _lib_vars)
            set(_lib_vars ${_Pack}_LIBRARIES)
          endif()
          foreach (_var ${_lib_vars})
            if (${_var})
              list(APPEND _Pack_LIBS ${${_var}})
            endif()
          endforeach()
          # Includes
          set(_inc_vars ${${ReqPack}_INCLUDE_VARS})
          if (NOT _inc_vars)
            set(_inc_vars ${_Pack}_INCLUDE_DIRS)
          endif()
          foreach (_include ${_inc_vars})
            # message(STATUS "${Name}: ${ReqPack}: ${_include}")
            if (${_include})
              list(APPEND _Pack_INCS ${${_include}})
            endif()
          endforeach()
        else()
          # Target mode: check for a valid target:
          # - an entry in the variable ${ReqPack}_TARGET_NAMES (optional)
          # - ${_Pack}
          # Other optional variables:
          # - ${ReqPack}_IMPORT_CONFIG, default value: "RELEASE"
          # - ${ReqPack}_TARGET_FORCE, default value: "FALSE"
          set(TargetName)
          foreach (_target ${${ReqPack}_TARGET_NAMES} ${_Pack})
            if (TARGET ${_target})
              set(TargetName ${_target})
              break()
            endif()
          endforeach()
          if ("${TargetName}" STREQUAL "")
            message(FATAL_ERROR " *** ${ReqPack}: unknown target. "
              "Please set ${ReqPack}_TARGET_NAMES.")
          endif()
          get_target_property(IsImported ${TargetName} IMPORTED)
          if (IsImported)
            set(ImportConfig ${${ReqPack}_IMPORT_CONFIG})
            if (NOT ImportConfig)
              set(ImportConfig RELEASE)
            endif()
            get_target_property(ImpConfigs ${TargetName} IMPORTED_CONFIGURATIONS)
            list(FIND ImpConfigs ${ImportConfig} _Index)
            if (_Index EQUAL -1)
              message(FATAL_ERROR " *** ${ReqPack}: configuration "
                "${ImportConfig} not found. Set ${ReqPack}_IMPORT_CONFIG "
                "from the list: ${ImpConfigs}.")
            endif()
          endif()
          # Set _Pack_LIBS
          if (NOT IsImported OR ${ReqPack}_TARGET_FORCE)
            # Set _Pack_LIBS to be the target itself
            set(_Pack_LIBS ${TargetName})
            if (NOT ${Name}_FIND_QUIETLY)
              message(STATUS "Found ${ReqPack}: ${_Pack_LIBS} (target)")
            endif()
          else()
            # Set _Pack_LIBS from the target properties for ImportConfig
            foreach (_prop IMPORTED_LOCATION_${ImportConfig}
                IMPORTED_LINK_INTERFACE_LIBRARIES_${ImportConfig})
              get_target_property(_value ${TargetName} ${_prop})
              if (_value)
                list(APPEND _Pack_LIBS ${_value})
              endif()
            endforeach()
            if (NOT ${Name}_FIND_QUIETLY)
              message(STATUS
                "Imported ${ReqPack}[${ImportConfig}]: ${_Pack_LIBS}")
            endif()
          endif()
          # Set _Pack_INCS
          foreach (_prop INCLUDE_DIRECTORIES)
            get_target_property(_value ${TargetName} ${_prop})
            if (_value)
              list(APPEND _Pack_INCS ${_value})
            endif()
          endforeach()
        endif()
        # _Pack_LIBS and _Pack_INCS should be fully defined here
        list(APPEND ${Prefix}_LIBRARIES ${_Pack_LIBS})
        list(APPEND ${Prefix}_INCLUDE_DIRS ${_Pack_INCS})
      endif()
    endif()
  endforeach()

  if (Found AND ${Name}_REQUIRED_LIBRARIES)
    list(APPEND ${Prefix}_LIBRARIES ${${Name}_REQUIRED_LIBRARIES})
  endif()

  if (NOT ("${${Prefix}_INCLUDE_DIRS}" STREQUAL ""))
    list(INSERT ReqVars 0 ${Prefix}_INCLUDE_DIRS)
    set(ReqHeaders 1)
  endif()
  if (NOT ("${${Prefix}_LIBRARIES}" STREQUAL ""))
    list(INSERT ReqVars 0 ${Prefix}_LIBRARIES)
    set(ReqLibs 1)
  endif()

  if (Found)
    if (ReqLibs)
      list(REMOVE_DUPLICATES ${Prefix}_LIBRARIES)
    endif()
    if (ReqHeaders)
      list(REMOVE_DUPLICATES ${Prefix}_INCLUDE_DIRS)
    endif()

    # Check for optional "CHECK_BUILD" arguments.
    set(I 9) # 9 is the number of required arguments
    while(I LESS ARGC)
      if ("${ARGV${I}}" STREQUAL "CHECK_BUILD")
        math(EXPR I "${I}+1")
        set(TestVar "${ARGV${I}}")
        math(EXPR I "${I}+1")
        set(TestReq "${ARGV${I}}")
        math(EXPR I "${I}+1")
        set(TestSrc "${ARGV${I}}")
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_INCLUDES ${${Prefix}_INCLUDE_DIRS})
        set(CMAKE_REQUIRED_LIBRARIES ${${Prefix}_LIBRARIES})
        set(CMAKE_REQUIRED_QUIET ${${Name}_FIND_QUIETLY})
        check_cxx_source_compiles("${TestSrc}" ${TestVar})
        if (TestReq)
          if (NOT ${TestVar})
            set(Found FALSE)
            unset(${TestVar} CACHE)
          endif()
          list(APPEND ReqVars ${TestVar})
        endif()
      elseif("${ARGV${I}}" STREQUAL "ADD_COMPONENT")
        # "ADD_COMPONENT" has 5 arguments, handled above
        math(EXPR I "${I}+5")
      else()
        message(FATAL_ERROR "Unknown argument: ${ARGV${I}}")
      endif()
      math(EXPR I "${I}+1")
    endwhile()
  endif()
  if ("_x_${ReqVars}" STREQUAL "_x_")
    set(${Prefix}_FOUND ${Found})
    set(ReqVars ${Prefix}_FOUND)
  endif()
  # foreach(ReqVar ${ReqVars})
  #   message(STATUS " *** ${ReqVar}=${${ReqVar}}")
  #   get_property(IsCached CACHE ${ReqVar} PROPERTY "VALUE" SET)
  #   if (IsCached)
  #     get_property(CachedVal CACHE ${ReqVar} PROPERTY "VALUE")
  #     message(STATUS " *** ${ReqVar}[cached]=${CachedVal}")
  #   endif()
  # endforeach()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(${Name}
    " *** ${Name} not found. Please set ${DirVar}." ${ReqVars})

  string(TOUPPER ${Name} UName)
  if (${UName}_FOUND)
    # Write the ${Prefix}_* variables to the cache.
    set(${Prefix}_LIBRARIES ${${Prefix}_LIBRARIES} CACHE STRING
        "${LibDoc}" FORCE)
    set(${Prefix}_INCLUDE_DIRS ${${Prefix}_INCLUDE_DIRS} CACHE STRING
        "${IncDoc}" FORCE)
    set(${Prefix}_FOUND TRUE CACHE BOOL "${Name} was found." FORCE)
    if (ReqHeaders AND (NOT ${Name}_FIND_QUIETLY))
      message(STATUS "${Prefix}_INCLUDE_DIRS=${${Prefix}_INCLUDE_DIRS}")
    endif()
  endif()

endfunction(mfem_find_package)


mfem_find_package(SuiteSparse SuiteSparse SuiteSparse_DIR "" "" "" ""
  "Paths to headers required by SuiteSparse."
  "Libraries required by SuiteSparse."
  ADD_COMPONENT "UMFPACK" "include;suitesparse" umfpack.h "lib" umfpack
  ADD_COMPONENT "KLU" "include;suitesparse" klu.h "lib" klu
  ADD_COMPONENT "AMD" "include;suitesparse" amd.h "lib" amd
  ADD_COMPONENT "BTF" "include;suitesparse" btf.h "lib" btf
  ADD_COMPONENT "CHOLMOD" "include;suitesparse" cholmod.h "lib" cholmod
  ADD_COMPONENT "COLAMD" "include;suitesparse" colamd.h "lib" colamd
  ADD_COMPONENT "CAMD" "include;suitesparse" camd.h "lib" camd
  ADD_COMPONENT "CCOLAMD" "include;suitesparse" ccolamd.h "lib" ccolamd
  ADD_COMPONENT "config" "include;suitesparse" SuiteSparse_config.h "lib"
    suitesparseconfig)

if (SuiteSparse_FOUND AND METIS_VERSION_5)
  message(STATUS " *** Warning: using SuiteSparse with METIS v5!")
endif()


# OLD CMAKE
# # BHEADER ####################################################################
# #
# # Copyright (c) 2018, Lawrence Livermore National Security, LLC.
# # Produced at the Lawrence Livermore National Laboratory.
# # LLNL-CODE-745247. All Rights reserved. See file COPYRIGHT for details.
# #
# # This file is part of smoothG. For more information and source code
# # availability, see https://www.github.com/llnl/smoothG.
# #
# # smoothG is free software; you can redistribute it and/or modify it under the
# # terms of the GNU Lesser General Public License (as published by the Free
# # Software Foundation) version 2.1 dated February 1999.
# #
# #################################################################### EHEADER #
# 
# # Sets the following variables
# #   - SuiteSparse_FOUND
# #   - SuiteSparse_INCLUDE_DIRS
# #   - SuiteSparse_LIBRARIES
# #
# # This is the one FindXXX.cmake module that ParELAG has that is
# # component-aware. That is, the following import targets are created,
# # each aware of its own dependencies:
# #
# # SuiteSparse::amd
# # SuiteSparse::btf
# # SuiteSparse::camd
# # SuiteSparse::ccolamd
# # SuiteSparse::cholmod
# # SuiteSparse::colamd
# # SuiteSparse::cxsparse
# # SuiteSparse::klu
# # SuiteSparse::ldl
# # SuiteSparse::rbio
# # SuiteSparse::spqr
# # SuiteSparse::suitesparseconfig
# # SuiteSparse::umfpack
# #
# # TODO: Actually use a "COMPONENTS" framework so that this module
# # becomes useful outside of ParELAG. See FindBoost or FindQt* for details
# 
# # My SuiteSparse looks like:
# #
# # libamd.a
# # libbtf.a
# # libcamd.a
# # libccolamd.a
# # libcholmod.a
# # libcolamd.a
# # libcxsparse.a
# # libklu.a
# # libldl.a
# # librbio.a
# # libspqr.a
# # libsuitesparseconfig.a
# # libumfpack.a
# #
# # RBio.h
# # SuiteSparseQR.hpp
# # SuiteSparse_config.h
# # amd.h
# # btf.h
# # camd.h
# # ccolamd.h
# # cholmod.h
# # colamd.h
# # cs.h
# # klu.h
# # ldl.h
# # spqr.hpp
# # umfpack.h
# 
# # The components that we need
# set(${PROJECT_NAME}_SUITESPARSE_COMPONENTS
#   umfpack klu btf amd colamd cholmod config)
# 
# # These are used only if they are found on the system
# set(${PROJECT_NAME}_SUITESPARSE_OPTIONAL_COMPONENTS
#   camd ccolamd)
# 
# # These are the full list
# set(${PROJECT_NAME}_SUITESPARSE_ALL_COMPONENTS
#   ${${PROJECT_NAME}_SUITESPARSE_COMPONENTS}
#   ${${PROJECT_NAME}_SUITESPARSE_OPTIONAL_COMPONENTS})
# 
# # SuiteSparse_Config is a little weird...
# set(config_HEADER_NAME "SuiteSparse_config.h")
# set(config_LIBRARY_NAME "suitesparseconfig")
# 
# 
# # Find and add the components
# foreach (component ${${PROJECT_NAME}_SUITESPARSE_ALL_COMPONENTS})
# 
#   # Set the header for the component
#   if (NOT ${component}_HEADER_NAME)
#     set(${component}_HEADER_NAME ${component}.h)
#   endif (NOT ${component}_HEADER_NAME)
# 
#   # Set the library for the component
#   if (NOT ${component}_LIBRARY_NAME)
#     set(${component}_LIBRARY_NAME ${component})
#   endif (NOT ${component}_LIBRARY_NAME)
# 
#   # Go searching for the header
#   message("Looking for " ${${component}_HEADER_NAME} " in " ${SuiteSparse_DIR})
#   find_path(${component}_INCLUDE_DIR ${${component}_HEADER_NAME}
#     HINTS ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR} ${SuiteSparse_INCLUDE_DIRS}
#     PATH_SUFFIXES suitesparse include
#     NO_DEFAULT_PATH
#     DOC "Directory where SuiteSparse component ${component} headers live.")
#   find_path(${component}_INCLUDE_DIR ${${component}_HEADER_NAME}
#     PATH_SUFFIXES suitesparse)
# 
#   # Add to the list of SuiteSparse headers
#   if (${component}_INCLUDE_DIR)
#     list(APPEND SuiteSparse_INCLUDE_DIRS ${${component}_INCLUDE_DIR})
#     list(REMOVE_DUPLICATES SuiteSparse_INCLUDE_DIRS)
#   endif (${component}_INCLUDE_DIR)
# 
#   # Go searching for the library
#   find_library(${component}_LIBRARY ${${component}_LIBRARY_NAME}
#     HINTS ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
#     PATH_SUFFIXES lib
#     NO_DEFAULT_PATH
#     DOC "The SuiteSparse component ${component} library.")
#   find_library(${component}_LIBRARY ${${component}_LIBRARY_NAME})
# 
#   if (${component}_LIBRARY AND ${component}_INCLUDE_DIR)
# 
#     # Setup the imported target
#     if (NOT TARGET SuiteSparse::${component})
#       # Check if we have shared or static libraries
#       include(CMakeUtilities)
#       parelag_determine_library_type(${${component}_LIBRARY} ${component}_LIB_TYPE)
# 
#       add_library(SuiteSparse::${component} ${${component}_LIB_TYPE} IMPORTED)
#     endif (NOT TARGET SuiteSparse::${component})
# 
#     # Set library
#     set_property(TARGET SuiteSparse::${component}
#       PROPERTY IMPORTED_LOCATION ${${component}_LIBRARY})
# 
#     # Add include path
#     set_property(TARGET SuiteSparse::${component} APPEND
#       PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${${component}_INCLUDE_DIR})
# 
#     # Add BLAS/LAPACK dependencies
#     set_property(TARGET SuiteSparse::${component} APPEND
#       PROPERTY INTERFACE_LINK_LIBRARIES
#       ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
# 
#     # Set the libraries
#     mark_as_advanced(FORCE ${component}_INCLUDE_DIR)
#     mark_as_advanced(FORCE ${component}_LIBRARY)
# 
#   else (${component}_LIBRARY AND ${component}_INCLUDE_DIR)
# 
#     list(FIND ${PROJECT_NAME}_SUITESPARSE_OPTIONAL_COMPONENTS
#       ${component} COMPONENT_IS_OPTIONAL)
# 
#     if (COMPONENT_IS_OPTIONAL EQUAL -1)
#       message(FATAL_ERROR "Required SuiteSparse component ${component} has not been found.")
#     else ()
#       message("WARNING: Optional SuiteSparse component ${component} has not been found.")
#     endif (COMPONENT_IS_OPTIONAL EQUAL -1)
#   endif (${component}_LIBRARY AND ${component}_INCLUDE_DIR)
# 
# endforeach (component ${${PROJECT_NAME}_SUITESPARSE_ALL_COMPONENTS})
# 
# #
# # Add dependencies
# #
# 
# # TODO: Finish figuring this out
# set(umfpack_DEPENDENCIES amd cholmod config)
# set(cholmod_DEPENDENCIES colamd camd ccolamd)
# set(klu_DEPENDENCIES btf amd)
# 
# # Set the dependencies
# foreach (component ${${PROJECT_NAME}_SUITESPARSE_COMPONENTS})
#   if (${component}_DEPENDENCIES)
#     foreach (dep ${${component}_DEPENDENCIES})
#       if (TARGET SuiteSparse::${dep})
#         set_property(TARGET SuiteSparse::${component} APPEND
#           PROPERTY INTERFACE_LINK_LIBRARIES SuiteSparse::${dep})
#       endif (TARGET SuiteSparse::${dep})
#     endforeach (dep ${${component}_DEPENDENCIES})
#   endif (${component}_DEPENDENCIES)
# endforeach (component ${${PROJECT_NAME}_SUITESPARSE_COMPONENTS})
# 
# #
# # Set the output LIBRARIES variable and cache INCLUDE_DIRS
# #
# set(SuiteSparse_LIBRARIES SuiteSparse::umfpack SuiteSparse::klu)
# 
# # Set the include directories
# set(SuiteSparse_INCLUDE_DIRS ${SuiteSparse_INCLUDE_DIRS}
#   CACHE PATH
#   "Directories in which to find headers for SuiteSparse.")
# mark_as_advanced(FORCE SuiteSparse_INCLUDE_DIRS)
# 
# # This handles "REQUIRED" etc keywords
# include(FindPackageHandleStandardArgs)
# find_package_handle_standard_args(SuiteSparse
#   "SuiteSparse could not be found. Be sure to set SuiteSparse_DIR."
#   SuiteSparse_LIBRARIES SuiteSparse_INCLUDE_DIRS)
