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

/**
   @file

   @brief Implements Mixed_GL_Coarsener
*/

#include "Mixed_GL_Coarsener.hpp"
#include <assert.h>

namespace smoothg
{

const mfem::SparseMatrix& Mixed_GL_Coarsener::GetPu() const
{
    check_subspace_construction_("Pu");
    return Pu_;
}

const mfem::SparseMatrix& Mixed_GL_Coarsener::GetPsigma() const
{
    check_subspace_construction_("Psigma");
    return Psigma_;
}

void Mixed_GL_Coarsener::Restrict(const mfem::BlockVector& fine_vect,
                                  mfem::BlockVector& coarse_vect) const
{
    Psigma_.MultTranspose(fine_vect.GetBlock(0), coarse_vect.GetBlock(0));
    Pu_.MultTranspose(fine_vect.GetBlock(1), coarse_vect.GetBlock(1));
}

void Mixed_GL_Coarsener::Interpolate(const mfem::BlockVector& coarse_vect,
                                     mfem::BlockVector& fine_vect) const
{
    Psigma_.Mult(coarse_vect.GetBlock(0), fine_vect.GetBlock(0));
    Pu_.Mult(coarse_vect.GetBlock(1), fine_vect.GetBlock(1));
}

void Mixed_GL_Coarsener::Restrict(const mfem::Vector& fine_vect,
                                  mfem::Vector& coarse_vect) const
{
    Pu_.MultTranspose(fine_vect, coarse_vect);
}

void Mixed_GL_Coarsener::Interpolate(const mfem::Vector& coarse_vect,
                                     mfem::Vector& fine_vect) const
{
    Pu_.Mult(coarse_vect, fine_vect);
}

} // namespace smoothg

