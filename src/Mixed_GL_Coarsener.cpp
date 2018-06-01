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

#ifndef __MIXED_GL_COARSENER_IMPL_HPP__
#define __MIXED_GL_COARSENER_IMPL_HPP__

#include "Mixed_GL_Coarsener.hpp"
#include <assert.h>

namespace smoothg
{

const mfem::SparseMatrix& Mixed_GL_Coarsener::get_Pu() const
{
    check_subspace_construction_("Pu");
    return Pu_;
}

const mfem::SparseMatrix& Mixed_GL_Coarsener::get_Psigma() const
{
    check_subspace_construction_("Psigma");
    return Psigma_;
}

std::unique_ptr<mfem::BlockVector> Mixed_GL_Coarsener::restrict_rhs(
    const mfem::BlockVector& rhs) const
{
    auto coarse_rhs = make_unique<mfem::BlockVector>(get_blockoffsets());
    restrict(rhs, *coarse_rhs);

    return coarse_rhs;
}

void Mixed_GL_Coarsener::restrict(const mfem::BlockVector& fine_vect,
                                  mfem::BlockVector& coarse_vect) const
{
    Psigma_.MultTranspose(fine_vect.GetBlock(0), coarse_vect.GetBlock(0));
    Pu_.MultTranspose(fine_vect.GetBlock(1), coarse_vect.GetBlock(1));
}

void Mixed_GL_Coarsener::interpolate(const mfem::BlockVector& coarse_vect,
                                     mfem::BlockVector& fine_vect) const
{
    Psigma_.Mult(coarse_vect.GetBlock(0), fine_vect.GetBlock(0));
    Pu_.Mult(coarse_vect.GetBlock(1), fine_vect.GetBlock(1));
}

void Mixed_GL_Coarsener::restrict(const mfem::Vector& fine_vect,
                                  mfem::Vector& coarse_vect) const
{
    Pu_.MultTranspose(fine_vect, coarse_vect);
}

void Mixed_GL_Coarsener::interpolate(const mfem::Vector& coarse_vect,
                                     mfem::Vector& fine_vect) const
{
    Pu_.Mult(coarse_vect, fine_vect);
}


const mfem::SparseMatrix& Mixed_GL_Coarsener::construct_Agg_cvertexdof_table() const
{
    check_subspace_construction_("Agg_cvertexdof_table");
    return graph_coarsen_->GetAggToCoarseVertexDof();
}

const mfem::SparseMatrix& Mixed_GL_Coarsener::construct_Agg_cedgedof_table() const
{
    check_subspace_construction_("Agg_cedgedof_table");
    return graph_coarsen_->GetAggToCoarseEdgeDof();
}

const mfem::SparseMatrix& Mixed_GL_Coarsener::construct_face_facedof_table() const
{
    check_subspace_construction_("face_facedof_table");
    return face_facedof_table_;
}

MixedMatrix Mixed_GL_Coarsener::GetCoarse()
{
    return MixedMatrix(GetCoarseMBuilder(), GetCoarseD(), GetCoarseW(), get_face_dof_truedof_table());
}

const mfem::HypreParMatrix& Mixed_GL_Coarsener::get_face_dof_truedof_table() const
{
    check_subspace_construction_("face_dof_truedof_table");
    if (!face_dof_truedof_table_)
    {
        face_dof_truedof_table_ = graph_coarsen_->BuildEdgeCoarseDofTruedof(
                                      face_facedof_table_, get_Psigma());
        face_dof_truedof_table_->CopyRowStarts();
        face_dof_truedof_table_->CopyColStarts();
    }

    assert(face_dof_truedof_table_);
    return *face_dof_truedof_table_;
}

mfem::Array<int>& Mixed_GL_Coarsener::get_blockoffsets() const
{
    assert(face_dof_truedof_table_);

    if (!coarseBlockOffsets_)
    {
        coarseBlockOffsets_ = make_unique<mfem::Array<int>>(3);
        (*coarseBlockOffsets_)[0] = 0;
        (*coarseBlockOffsets_)[1] = face_dof_truedof_table_->GetNumRows();
        (*coarseBlockOffsets_)[2] = (*coarseBlockOffsets_)[1] + Pu_.Width();
    }
    return *coarseBlockOffsets_;
}

} // namespace smoothg

#endif /* __MIXED_GL_COARSENER_IMPL_HPP__ */
