/*BHEADER**********************************************************************
 *
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * LLNL-CODE-745247. All Rights reserved. See file COPYRIGHT for details.
 *
 * This file is part of smoothG. For more information and source code
 * availability, see https://www.github.com/llnl/smoothG.
 *
 * smoothG is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 ***********************************************************************EHEADER*/

/** @file

   @brief A class to manage shared entity communication

   This implements a kind of general reduction algorithm, beyond what you
   can do with matrix-matrix multiplies or MPI_Reduce. In particular, for the
   spectral method we want to do reduction where the operation is some kind of
   SVD, which requires something more complicated.

   The complicated logic on the Reduce side is because we are leaving the actual
   reduce operation to the user, so you can think of it more as a "collect"
   operation onto the master, where the user is responsible to do what is
   necessary.

   This is "fairly" generic but not completely, if you want to use for a
   datatype other than mfem::DenseMatrix or mfem::Vector you need to implement:
   SetSizeSpecifer(), PackSizes(), SendData(), ReceiveData(), and CopyData()
   routines yourself.

   Significant improvements to handling the "tags" argument to honor the
   MPI_TAG_UB constraint are due to Alex Druinsky from Lawrence Berkeley
   (adruinksy@lbl.gov).
*/

#include "SharedEntityComm.hpp"

namespace smoothg
{

} // namespace smoothg
