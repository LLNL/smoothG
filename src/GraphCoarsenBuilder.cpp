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

namespace smoothg
{

CoarseMBuilder::CoarseMBuilder(std::vector<mfem::DenseMatrix>& edge_traces,
                               std::vector<mfem::DenseMatrix>& vertex_target,
                               std::vector<mfem::DenseMatrix>& CM_el,
                               const mfem::SparseMatrix& Agg_face,
                               int total_num_traces, int ncoarse_vertexdofs,
                               bool build_coarse_relation)
    :
    edge_traces_(edge_traces),
    vertex_target_(vertex_target),
    CM_el_(CM_el),
    total_num_traces_(total_num_traces),
    build_coarse_relation_(build_coarse_relation)
{
    const unsigned int nAggs = vertex_target.size();

    if (build_coarse_relation_)
    {
        CM_el.resize(nAggs);
        mfem::Array<int> faces;
        for (unsigned int i = 0; i < nAggs; i++)
        {
            int nlocal_coarse_dofs = vertex_target[i].Width() - 1;
            GetTableRow(Agg_face, i, faces);
            for (int j = 0; j < faces.Size(); ++j)
                nlocal_coarse_dofs += edge_traces[faces[j]].Width();
            CM_el[i].SetSize(nlocal_coarse_dofs);
        }
        edge_cdof_marker_.SetSize(total_num_traces + ncoarse_vertexdofs - nAggs);
        edge_cdof_marker_ = -1;
    }
    else
    {
        CoarseM_ = make_unique<mfem::SparseMatrix>(
                       total_num_traces + ncoarse_vertexdofs - nAggs,
                       total_num_traces + ncoarse_vertexdofs - nAggs);
    }
}

void CoarseMBuilder::RegisterRow(int agg_index, int row, int cdof_loc, int bubble_counter)
{
    agg_index_ = agg_index;
    row_ = row;
    cdof_loc_ = cdof_loc;
    bubble_counter_ = bubble_counter;
    if (build_coarse_relation_)
        edge_cdof_marker_[row] = cdof_loc;
}

void CoarseMBuilder::SetBubbleOffd(int l, double value)
{
    const int global_col = total_num_traces_ + bubble_counter_ + l;
    if (build_coarse_relation_)
    {
        mfem::DenseMatrix& CM_el_loc(CM_el_[agg_index_]);
        CM_el_loc(l, cdof_loc_) = value;
        CM_el_loc(cdof_loc_, l) = value;
    }
    else
    {
        CoarseM_->Set(row_, global_col, value);
        CoarseM_->Set(global_col, row_, value);
    }
}

void CoarseMBuilder::AddDiag(double value)
{
    if (build_coarse_relation_)
        CM_el_[agg_index_](cdof_loc_, cdof_loc_) = value;
    else
        CoarseM_->Add(row_, row_, value);
}

void CoarseMBuilder::AddTrace(int l, double value)
{
    if (build_coarse_relation_)
    {
        mfem::DenseMatrix& CM_el_loc(CM_el_[agg_index_]);
        CM_el_loc(edge_cdof_marker_[l], cdof_loc_) = value;
        CM_el_loc(cdof_loc_, edge_cdof_marker_[l]) = value;
    }
    else
    {
        CoarseM_->Add(row_, l, value);
        CoarseM_->Add(l, row_, value);
    }
}

void CoarseMBuilder::SetBubbleLocal(int l, int j, double value)
{
    if (build_coarse_relation_)
    {
        mfem::DenseMatrix& CM_el_loc(CM_el_[agg_index_]);
        CM_el_loc(l, j) = value;
        CM_el_loc(j, l) = value;
    }
    else
    {
        int global_row = total_num_traces_ + bubble_counter_ + l;
        int global_col = total_num_traces_ + bubble_counter_ + j;
        CoarseM_->Set(global_row, global_col, value);
        CoarseM_->Set(global_col, global_row, value);
    }
}

void CoarseMBuilder::ResetEdgeCdofMarkers(int size)
{
    if (build_coarse_relation_)
    {
        edge_cdof_marker_.SetSize(size);
        edge_cdof_marker_ = -1;
        edge_cdof_marker2_.SetSize(size);
        edge_cdof_marker2_ = -1;
    }
}

void CoarseMBuilder::RegisterTraceFace(int face_num, const mfem::SparseMatrix& face_Agg,
                                       const mfem::SparseMatrix& Agg_cdof_edge)
{
    mfem::Array<int> Aggs;
    mfem::Array<int> local_Agg_edge_cdof;
    if (build_coarse_relation_)
    {
        GetTableRow(face_Agg, face_num, Aggs);
        Agg0_ = Aggs[0];
        GetTableRow(Agg_cdof_edge, Agg0_, local_Agg_edge_cdof);
        for (int k = 0; k < local_Agg_edge_cdof.Size(); k++)
        {
            edge_cdof_marker_[local_Agg_edge_cdof[k]] = k;
        }
        if (Aggs.Size() == 2)
        {
            Agg1_ = Aggs[1];
            GetTableRow(Agg_cdof_edge, Agg1_, local_Agg_edge_cdof);
            for (int k = 0; k < local_Agg_edge_cdof.Size(); k++)
            {
                edge_cdof_marker2_[local_Agg_edge_cdof[k]] = k;
            }
        }
        else
        {
            Agg1_ = -1;
        }
    }
    else
    {
        Agg0_ = Agg1_ = 0;
    }
}

void CoarseMBuilder::AddTraceAcross(int row, int col, double value)
{
    if (build_coarse_relation_)
    {
        mfem::DenseMatrix& CM_el_loc1(CM_el_[Agg0_]);

        int id0_in_Agg0 = edge_cdof_marker_[row];
        int id1_in_Agg0 = edge_cdof_marker_[col];
        if (Agg1_ == -1)
        {
            CM_el_loc1(id0_in_Agg0, id1_in_Agg0) += value;
        }
        else
        {
            mfem::DenseMatrix& CM_el_loc2(CM_el_[Agg1_]);
            CM_el_loc1(id0_in_Agg0, id1_in_Agg0) += value / 2.;
            int id0_in_Agg1 = edge_cdof_marker2_[row];
            int id1_in_Agg1 = edge_cdof_marker2_[col];
            CM_el_loc2(id0_in_Agg1, id1_in_Agg1) += value / 2.;
        }
    }
    else
    {
        CoarseM_->Add(row, col, value);
    }
}

std::unique_ptr<mfem::SparseMatrix> CoarseMBuilder::GetCoarseM()
{
    if (!build_coarse_relation_)
        CoarseM_->Finalize(0);
    return std::move(CoarseM_);
}

Agg_cdof_edge_Builder::Agg_cdof_edge_Builder(std::vector<mfem::DenseMatrix>& edge_traces,
                                             std::vector<mfem::DenseMatrix>& vertex_target,
                                             const mfem::SparseMatrix& Agg_face,
                                             bool build_coarse_relation)
    :
    Agg_dof_nnz_(0),
    build_coarse_relation_(build_coarse_relation)
{
    const unsigned int nAggs = vertex_target.size();

    if (build_coarse_relation_)
    {
        Agg_dof_i_ = new int[nAggs + 1];
        Agg_dof_i_[0] = 0;

        mfem::Array<int> faces; // this is repetitive of InitializePEdgesNNZ
        for (unsigned int i = 0; i < nAggs; i++)
        {
            int nlocal_coarse_dofs = vertex_target[i].Width() - 1;
            GetTableRow(Agg_face, i, faces);
            for (int j = 0; j < faces.Size(); ++j)
                nlocal_coarse_dofs += edge_traces[faces[j]].Width();
            Agg_dof_i_[i + 1] = Agg_dof_i_[i] + nlocal_coarse_dofs;
        }
        Agg_dof_j_ = new int[Agg_dof_i_[nAggs]];
        Agg_dof_d_ = new double[Agg_dof_i_[nAggs]];
        std::fill(Agg_dof_d_, Agg_dof_d_ + Agg_dof_i_[nAggs], 1.);
    }
}

void Agg_cdof_edge_Builder::Register(int k)
{
    if (build_coarse_relation_)
        Agg_dof_j_[Agg_dof_nnz_++] = k;
}

std::unique_ptr<mfem::SparseMatrix> Agg_cdof_edge_Builder::GetAgg_cdof_edge(int rows, int cols)
{
    if (build_coarse_relation_)
    {
        return make_unique<mfem::SparseMatrix>(
            Agg_dof_i_, Agg_dof_j_, Agg_dof_d_, rows, cols);
    }
    return std::unique_ptr<mfem::SparseMatrix>(nullptr);
}

}
