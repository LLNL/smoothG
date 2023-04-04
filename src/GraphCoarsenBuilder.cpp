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

#include "GraphCoarsenBuilder.hpp"
#include "GraphTopology.hpp"
#include "MatrixUtilities.hpp"

namespace smoothg
{

// mfem::SparseMatrix MBuilder::BuildAssembledM() const
// {
//     mfem::Vector agg_weights_inverse(num_aggs_);
//     agg_weights_inverse = 1.0;
//     return BuildAssembledM(agg_weights_inverse);
// }

MBuilder::MBuilder(const GraphSpace& coarse_space)
{
    elem_edgedof_.MakeRef(coarse_space.VertexToEDof());
    num_aggs_ = elem_edgedof_.NumRows();

    M_el_.resize(num_aggs_);
    for (unsigned int i = 0; i < num_aggs_; i++)
    {
        M_el_[i].SetSize(elem_edgedof_.RowSize(i));
    }

    edge_dof_markers_.resize(2);
    edge_dof_markers_[0].resize(elem_edgedof_.NumCols(), -1);
    edge_dof_markers_[1].resize(elem_edgedof_.NumCols(), -1);
}

void MBuilder::RegisterRow(int agg_index, int row, int dof_loc)
{
    agg_index_ = agg_index;
    dof_loc_ = dof_loc;
    edge_dof_markers_[0][row] = dof_loc;
}

void MBuilder::SetTraceBubbleBlock(int l, double value)
{
    mfem::DenseMatrix& M_el_loc(M_el_[agg_index_]);
    M_el_loc(l, dof_loc_) = value;
    M_el_loc(dof_loc_, l) = value;
}

void MBuilder::AddTraceTraceBlockDiag(double value)
{
    M_el_[agg_index_](dof_loc_, dof_loc_) += value;
}

void MBuilder::AddTraceTraceBlock(int l, double value)
{
    mfem::DenseMatrix& M_el_loc(M_el_[agg_index_]);
    M_el_loc(edge_dof_markers_[0][l], dof_loc_) += value;
    M_el_loc(dof_loc_, edge_dof_markers_[0][l]) += value;
}

void MBuilder::SetBubbleBubbleBlock(int agg_index, int l,
                                    int j, double value)
{
    mfem::DenseMatrix& M_el_loc(M_el_[agg_index]);
    M_el_loc(l, j) = value;
    M_el_loc(j, l) = value;
}

void MBuilder::FillEdgeCdofMarkers(int face_num, const mfem::SparseMatrix& face_Agg,
                                   const mfem::SparseMatrix& Agg_cdof_edge)
{
    mfem::Array<int> local_Agg_edge_cdof;
    GetTableRow(face_Agg, face_num, Aggs_);
    for (int a = 0; a < Aggs_.Size(); a++)
    {
        GetTableRow(Agg_cdof_edge, Aggs_[a], local_Agg_edge_cdof);
        for (int k = 0; k < local_Agg_edge_cdof.Size(); k++)
        {
            edge_dof_markers_[a][local_Agg_edge_cdof[k]] = k;
        }
    }
}

void MBuilder::AddTraceAcross(int row, int col, int agg, double value)
{
    mfem::DenseMatrix& M_el_loc(M_el_[Aggs_[agg]]);

    int id0_in_agg = edge_dof_markers_[agg][row];
    int id1_in_agg = edge_dof_markers_[agg][col];
    M_el_loc(id0_in_agg, id1_in_agg) += value;
}

// mfem::SparseMatrix MBuilder::BuildAssembledM(
//     const mfem::Vector& agg_weights_inverse) const
// {
//     mfem::Array<int> edofs;
//     mfem::SparseMatrix M(elem_edgedof_.Width());
//     for (int Agg = 0; Agg < elem_edgedof_.Height(); Agg++)
//     {
//         GetTableRow(elem_edgedof_, Agg, edofs);
//         const double agg_weight = 1. / agg_weights_inverse(Agg);
//         mfem::DenseMatrix agg_M = M_el_[Agg];
//         agg_M *= agg_weight;
//         M.AddSubMatrix(edofs, edofs, agg_M);
//     }
//     M.Finalize();
//     return M;
// }

// mfem::Vector MBuilder::Mult(
//     const mfem::Vector& elem_scaling_inv, const mfem::Vector& x) const
// {
//     mfem::Vector y(x.Size());
//     y = 0.0;

//     mfem::Array<int> local_edofs;
//     mfem::Vector x_loc;
//     mfem::Vector y_loc;
//     for (int elem = 0; elem < elem_edgedof_.NumRows(); ++elem)
//     {
//         GetTableRow(elem_edgedof_, elem, local_edofs);

//         x.GetSubVector(local_edofs, x_loc);

//         y_loc.SetSize(x_loc.Size());
//         M_el_[elem].Mult(x_loc, y_loc);
//         y_loc /= elem_scaling_inv[elem];

//         for (int j = 0; j < local_edofs.Size(); ++j)
//         {
//             y[local_edofs[j]] += y_loc[j];
//         }
//     }

//     return y;
// }

}
