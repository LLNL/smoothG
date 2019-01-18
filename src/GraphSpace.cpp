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

    @brief Implements GraphSpace object.
*/

#include "GraphSpace.hpp"
#include "MatrixUtilities.hpp"

using std::unique_ptr;

namespace smoothg
{

GraphSpace::GraphSpace(Graph graph)
    : graph_(std::move(graph)),
      vertex_vdof_(SparseIdentity(graph_.NumVertices())),
      edge_edof_(SparseIdentity(graph_.NumEdges()))
{
    vertex_edof_.MakeRef(graph_.VertexToEdge());
    edof_trueedof_ = make_unique<mfem::HypreParMatrix>();
    edof_trueedof_->MakeRef(graph_.EdgeToTrueEdge());
    if (graph_.HasBoundary())
    {
        edof_bdratt_.MakeRef(graph_.EdgeToBdrAtt());
    }
}

GraphSpace::GraphSpace(Graph graph, const std::vector<int> num_local_vdofs,
                       const std::vector<int> num_local_edofs)
    : graph_(std::move(graph)), vertex_vdof_(std::move(vertex_vdof)),
      vertex_edof_(std::move(vertex_edof)),
      edge_edof_(std::move(edge_edof)), edof_trueedof_(std::move(edof_trueedof)),

{
    if (graph_.HasBoundary())
    {
        mfem::SparseMatrix edof_edge = smoothg::Transpose(edge_edof_);
        mfem::SparseMatrix tmp = smoothg::Mult(edof_edge, graph_.EdgeToBdrAtt());
        edof_bdratt_.Swap(tmp);
    }
}

GraphSpace::GraphSpace(GraphSpace&& other) noexcept
{
    swap(*this, other);
}

GraphSpace& GraphSpace::operator=(GraphSpace other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(GraphSpace& lhs, GraphSpace& rhs) noexcept
{
    lhs.vertex_vdof_.Swap(rhs.vertex_vdof_);
    lhs.vertex_edof_.Swap(rhs.vertex_edof_);
    lhs.edge_edof_.Swap(rhs.edge_edof_);
    std::swap(lhs.edof_trueedof_, rhs.edof_trueedof_);
    lhs.edof_bdratt_.Swap(rhs.edof_bdratt_);

    swap(lhs.graph_, rhs.graph_);
}


mfem::SparseMatrix GraphCoarsen::BuildCoarseEntityToCoarseDof(
    const std::vector<mfem::DenseMatrix>& local_targets)
{
    const unsigned int num_entities = local_targets.size();
    int* I = new int[num_entities + 1]();
    for (unsigned int entity = 0; entity < num_entities; ++entity)
    {
        I[entity + 1] = I[entity] + local_targets[entity].Width();
    }

    int nnz = I[num_entities];
    int* J = new int[nnz];
    std::iota(J, J + nnz, 0);

    double* Data = new double[nnz];
    std::fill_n(Data, nnz, 1.);

    return mfem::SparseMatrix(I, J, Data, num_entities, nnz);
}

unique_ptr<mfem::HypreParMatrix> GraphCoarsen::BuildCoarseEdgeDofTruedof(
    const Graph& coarse_graph, const mfem::SparseMatrix& face_cdof, int num_coarse_edofs)
{
    const int ncdofs = num_coarse_edofs;
    const int nfaces = face_cdof.Height();

    // count edge coarse true dofs (if the dof is a bubble or on a true face)
    mfem::SparseMatrix face_d_td_diag;
    const mfem::HypreParMatrix& face_trueface_ = coarse_graph.EdgeToTrueEdge();
    mfem::HypreParMatrix& face_trueface_face_ =
        const_cast<mfem::HypreParMatrix&>(*topology_.face_trueface_face_);
    face_trueface_.GetDiag(face_d_td_diag);

    MPI_Comm comm = face_trueface_.GetComm();
    mfem::Array<HYPRE_Int> edge_cd_start;
    GenerateOffsets(comm, ncdofs, edge_cd_start);

    mfem::Array<HYPRE_Int>& face_start =
        const_cast<mfem::Array<HYPRE_Int>&>(topology_.GetFaceStarts());

    mfem::SparseMatrix face_cdof_tmp(face_cdof.GetI(), face_cdof.GetJ(),
                                     face_cdof.GetData(), nfaces, ncdofs,
                                     false, false, false);

    mfem::HypreParMatrix face_cdof_d(comm, face_start.Last(),
                                     edge_cd_start.Last(), face_start,
                                     edge_cd_start, &face_cdof_tmp);

    unique_ptr<mfem::HypreParMatrix> d_td_d_tmp(smoothg::RAP(face_trueface_face_, face_cdof_d));

    mfem::SparseMatrix d_td_d_tmp_offd;
    HYPRE_Int* d_td_d_map;
    d_td_d_tmp->GetOffd(d_td_d_tmp_offd, d_td_d_map);

    mfem::Array<int> d_td_d_diag_i(ncdofs + 1);
    std::iota(d_td_d_diag_i.begin(), d_td_d_diag_i.begin() + ncdofs + 1, 0);

    mfem::Array<double> d_td_d_diag_data(ncdofs);
    std::fill_n(d_td_d_diag_data.begin(), ncdofs, 1.0);
    mfem::SparseMatrix d_td_d_diag(d_td_d_diag_i.GetData(), d_td_d_diag_i.GetData(),
                                   d_td_d_diag_data.GetData(), ncdofs, ncdofs,
                                   false, false, false);

    int* d_td_d_offd_i = new int[ncdofs + 1];
    int d_td_d_offd_nnz = 0;
    for (int i = 0; i < ncdofs; i++)
    {
        d_td_d_offd_i[i] = d_td_d_offd_nnz;
        if (d_td_d_tmp_offd.RowSize(i))
            d_td_d_offd_nnz++;
    }
    d_td_d_offd_i[ncdofs] = d_td_d_offd_nnz;
    int* d_td_d_offd_j = new int[d_td_d_offd_nnz];
    d_td_d_offd_nnz = 0;

    mfem::SparseMatrix face_trueface_face_offd;
    HYPRE_Int* junk_map;
    face_trueface_face_.GetOffd(face_trueface_face_offd, junk_map);

    int face_1st_cdof, face_ncdofs;
    int* face_cdof_i = face_cdof.GetI();
    int* face_cdof_j = face_cdof.GetJ();
    mfem::Array<int> face_cdofs;
    for (int i = 0; i < nfaces; i++)
    {
        if (face_trueface_face_offd.RowSize(i))
        {
            face_ncdofs = face_cdof_i[i + 1] - face_cdof_i[i];
            face_1st_cdof = face_cdof_j[face_cdof_i[i]];
            GetTableRow(d_td_d_tmp_offd, face_1st_cdof, face_cdofs);
            assert(face_cdofs.Size() == face_ncdofs);
            for (int j = 0; j < face_ncdofs; j++)
                d_td_d_offd_j[d_td_d_offd_nnz++] = face_cdofs[j];
        }
    }
    assert(d_td_d_offd_i[ncdofs] == d_td_d_offd_nnz);
    mfem::SparseMatrix d_td_d_offd(d_td_d_offd_i, d_td_d_offd_j,
                                   d_td_d_diag_data.GetData(), ncdofs, d_td_d_offd_nnz,
                                   true, false, false);

    mfem::HypreParMatrix d_td_d(
        comm, edge_cd_start.Last(), edge_cd_start.Last(), edge_cd_start,
        edge_cd_start, &d_td_d_diag, &d_td_d_offd, d_td_d_map);

    return BuildEntityToTrueEntity(d_td_d);
}

mfem::SparseMatrix GraphCoarsen::BuildAggToCoarseEdgeDof(
    const Graph& coarse_graph,
    const mfem::SparseMatrix& agg_coarse_vdof,
    const mfem::SparseMatrix& face_coarse_edof)
{
    const unsigned int num_aggs = agg_coarse_vdof.NumRows();
    const mfem::SparseMatrix& agg_face = coarse_graph.VertexToEdge();

    int* I = new int[num_aggs + 1];
    I[0] = 0;

    mfem::Array<int> faces; // this is repetitive of InitializePEdgesNNZ
    for (unsigned int agg = 0; agg < num_aggs; agg++)
    {
        int nlocal_coarse_edofs = agg_coarse_vdof.RowSize(agg) - 1;
        GetTableRow(agg_face, agg, faces);
        for (int& face : faces)
        {
            nlocal_coarse_edofs += face_coarse_edof.RowSize(face);
        }
        I[agg + 1] = I[agg] + nlocal_coarse_edofs;
    }

    const int nnz = I[num_aggs];
    int* J = new int[nnz];
    double* data = new double[nnz];

    int edof_counter = face_coarse_edof.NumCols(); // start with num_traces

    int* J_begin = J;
    double* data_begin = data;

    // data values are chosen for the ease of extended aggregate construction
    for (unsigned int agg = 0; agg < num_aggs; agg++)
    {
        const int num_bubbles_agg = agg_coarse_vdof.RowSize(agg) - 1;

        int* J_end = J_begin + num_bubbles_agg;
        std::iota(J_begin, J_end, edof_counter);
        J_begin = J_end;

        double* data_end = data_begin + num_bubbles_agg;
        std::fill(data_begin, data_end, 2.0);
        data_begin = data_end;

        edof_counter += num_bubbles_agg;

        GetTableRow(agg_face, agg, faces);
        for (int& face : faces)
        {
            J_end += face_coarse_edof.RowSize(face);
            std::iota(J_begin, J_end, *face_coarse_edof.GetRowColumns(face));
            J_begin = J_end;

            data_end += face_coarse_edof.RowSize(face);
        }
        std::fill(data_begin, data_end, 1.0);
        data_begin = data_end;
    }

    return mfem::SparseMatrix(I, J, data, num_aggs, edof_counter);
}

} // namespace smoothg

