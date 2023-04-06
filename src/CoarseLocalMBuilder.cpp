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

#include "CoarseLocalMBuilder.hpp"
#include "GraphTopology.hpp"
#include "MatrixUtilities.hpp"

namespace smoothg
{

CoarseLocalMBuilder::CoarseLocalMBuilder(const GraphSpace& coarse_space)
{
    auto& agg_cedof = coarse_space.VertexToEDof();
    int num_aggs = agg_cedof.NumRows();

    M_agg_.resize(num_aggs);
    for (int agg = 0; agg < num_aggs; agg++)
    {
        M_agg_[agg].SetSize(agg_cedof.RowSize(agg));
    }

    global_to_local_.resize(2);
    global_to_local_[0].resize(agg_cedof.NumCols(), -1);
    global_to_local_[1].resize(agg_cedof.NumCols(), -1);
}

void CoarseLocalMBuilder::RegisterTraceIndex(int agg_index, int dof_global, int dof_local)
{
    agg_index_ = agg_index;
    dof_loc_ = dof_local;
    global_to_local_[0][dof_global] = dof_local;
}

void CoarseLocalMBuilder::SetTraceBubbleBlock(int bubble_local, double value)
{
    mfem::DenseMatrix& M_agg_loc(M_agg_[agg_index_]);
    M_agg_loc(bubble_local, dof_loc_) = value;
    M_agg_loc(dof_loc_, bubble_local) = value;
}

void CoarseLocalMBuilder::AddTraceTraceBlockDiag(double value)
{
    M_agg_[agg_index_](dof_loc_, dof_loc_) += value;
}

void CoarseLocalMBuilder::AddTraceTraceBlock(int dof_global, double value)
{
    mfem::DenseMatrix& M_agg_loc(M_agg_[agg_index_]);
    M_agg_loc(global_to_local_[0][dof_global], dof_loc_) += value;
    M_agg_loc(dof_loc_, global_to_local_[0][dof_global]) += value;
}

void CoarseLocalMBuilder::SetBubbleBubbleBlock(int agg_index, int bubble_local1,
                                               int bubble_local2, double value)
{
    mfem::DenseMatrix& M_agg_loc(M_agg_[agg_index]);
    M_agg_loc(bubble_local1, bubble_local2) = value;
    M_agg_loc(bubble_local2, bubble_local1) = value;
}

void CoarseLocalMBuilder::SetDofGlobalToLocalMaps(int face, const mfem::SparseMatrix& face_agg,
                                                  const mfem::SparseMatrix& agg_cedof)
{
    mfem::Array<int> local_cedof;
    GetTableRow(face_agg, face, aggs_);
    for (int a = 0; a < aggs_.Size(); a++)
    {
        GetTableRow(agg_cedof, aggs_[a], local_cedof);
        for (int k = 0; k < local_cedof.Size(); k++)
        {
            global_to_local_[a][local_cedof[k]] = k;
        }
    }
}

void CoarseLocalMBuilder::AddTraceAcross(int dof_global1, int dof_global2, int agg, double value)
{
    mfem::DenseMatrix& M_agg_loc(M_agg_[aggs_[agg]]);
    int dof_local1 = global_to_local_[agg][dof_global1];
    int dof_local2 = global_to_local_[agg][dof_global2];
    M_agg_loc(dof_local1, dof_local2) += value;
}

}
