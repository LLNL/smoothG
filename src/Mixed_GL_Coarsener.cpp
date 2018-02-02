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

const std::vector<std::unique_ptr<mfem::DenseMatrix>>&
                                                   Mixed_GL_Coarsener::get_CM_el() const
{
    check_subspace_construction_("CM_el");
    return CM_el_;
}

std::unique_ptr<mfem::BlockVector> Mixed_GL_Coarsener::coarsen_rhs(
    const mfem::BlockVector& rhs) const
{
    auto coarse_rhs = make_unique<mfem::BlockVector>(get_blockoffsets());
    Psigma_.MultTranspose(rhs.GetBlock(0), coarse_rhs->GetBlock(0));
    Pu_.MultTranspose(rhs.GetBlock(1), coarse_rhs->GetBlock(1));
    return coarse_rhs;
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

std::shared_ptr<mfem::HypreParMatrix>&
Mixed_GL_Coarsener::get_face_dof_truedof_table() const
{
    check_subspace_construction_("face_dof_truedof_table");
    if (!face_dof_truedof_table_)
    {
        face_dof_truedof_table_ = graph_coarsen_->BuildEdgeCoarseDofTruedof(
                                      face_facedof_table_, get_Psigma());
        face_dof_truedof_table_->CopyRowStarts();
        face_dof_truedof_table_->CopyColStarts();
    }
    return face_dof_truedof_table_;
}

mfem::Array<int>& Mixed_GL_Coarsener::get_blockoffsets() const
{
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
