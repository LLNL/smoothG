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

    @brief GraphTopology class
*/

#include "GraphTopology.hpp"

namespace smoothg
{

GraphTopology::GraphTopology(MPI_Comm comm, const Graph& graph)
{
    agg_vertex_local_ = MakeAggVertex(graph.part_local_);

    SparseMatrix agg_edge_ext = agg_vertex_local_.Mult(graph.vertex_edge_local_);
    agg_edge_ext.SortIndices();

    agg_edge_local_ = RestrictInterior(agg_edge_ext);

    SparseMatrix edge_ext_agg = agg_edge_ext.Transpose();

    int num_vertices = graph.vertex_edge_local_.Rows();
    int num_edges = graph.vertex_edge_local_.Cols();
    int num_aggs = agg_edge_local_.Rows();

    auto starts = parlinalgcpp::GenerateOffsets(comm, {num_vertices, num_edges, num_aggs});
    const auto& vertex_starts = starts[0];
    const auto& edge_starts = starts[1];
    const auto& agg_starts = starts[2];

    ParMatrix edge_agg_d(comm, edge_starts, agg_starts, std::move(edge_ext_agg));
    ParMatrix agg_edge_d = edge_agg_d.Transpose();

    ParMatrix edge_agg_ext = graph.edge_edge_.Mult(edge_agg_d);
    ParMatrix agg_agg = agg_edge_d.Mult(edge_agg_ext);

    agg_edge_ext = 1.0;
    SparseMatrix face_agg_int = MakeFaceAggInt(agg_agg);
    SparseMatrix face_edge_ext = face_agg_int.Mult(agg_edge_ext);

    face_edge_local_ = MakeFaceEdge(agg_agg, edge_agg_ext,
                                    agg_edge_ext, face_edge_ext);

    face_agg_local_ = ExtendFaceAgg(agg_agg, face_agg_int);
    agg_face_local_ = face_agg_local_.Transpose();

    auto face_starts = parlinalgcpp::GenerateOffsets(comm, face_agg_local_.Rows());

    face_edge_ = ParMatrix(comm, face_starts, edge_starts, face_edge_local_);
    ParMatrix edge_face = face_edge_.Transpose();

    face_face_ = parlinalgcpp::RAP(graph.edge_edge_, edge_face);
    face_face_ = 1;

    face_true_face_ = MakeEntityTrueEntity(face_face_);

    ParMatrix vertex_edge_d(comm, vertex_starts, edge_starts, graph.vertex_edge_local_);
    ParMatrix vertex_edge = vertex_edge_d.Mult(graph.edge_true_edge_);
    ParMatrix edge_vertex = vertex_edge.Transpose();
    ParMatrix agg_edge = agg_edge_d.Mult(graph.edge_true_edge_);

    agg_ext_vertex_ = agg_edge.Mult(edge_vertex);
    agg_ext_vertex_ = 1.0;

    ParMatrix agg_ext_edge_ext = agg_ext_vertex_.Mult(vertex_edge);
    agg_ext_edge_ = RestrictInterior(agg_ext_edge_ext);
}

GraphTopology::GraphTopology(const GraphTopology& other) noexcept
    : agg_vertex_local_(other.agg_vertex_local_),
      agg_edge_local_(other.agg_edge_local_),
      face_edge_local_(other.face_edge_local_),
      face_agg_local_(other.face_agg_local_),
      agg_face_local_(other.agg_face_local_),
      face_face_(other.face_face_),
      face_true_face_(other.face_true_face_),
      face_edge_(other.face_edge_),
      agg_ext_vertex_(other.agg_ext_vertex_),
      agg_ext_edge_(other.agg_ext_edge_)
{

}

GraphTopology::GraphTopology(GraphTopology&& other) noexcept
{
    swap(*this, other);
}

GraphTopology& GraphTopology::operator=(GraphTopology other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(GraphTopology& lhs, GraphTopology& rhs) noexcept
{
    swap(lhs.agg_vertex_local_, rhs.agg_vertex_local_);
    swap(lhs.agg_edge_local_, rhs.agg_edge_local_);
    swap(lhs.face_edge_local_, rhs.face_edge_local_);
    swap(lhs.face_agg_local_, rhs.face_agg_local_);
    swap(lhs.agg_face_local_, rhs.agg_face_local_);

    swap(lhs.face_face_, rhs.face_face_);
    swap(lhs.face_true_face_, rhs.face_true_face_);
    swap(lhs.face_edge_, rhs.face_edge_);
    swap(lhs.agg_ext_vertex_, rhs.agg_ext_vertex_);
    swap(lhs.agg_ext_edge_, rhs.agg_ext_edge_);
}

SparseMatrix GraphTopology::MakeFaceAggInt(const ParMatrix& agg_agg)
{
    const auto& agg_agg_diag = agg_agg.GetDiag();

    int num_aggs = agg_agg_diag.Rows();
    int num_faces = agg_agg_diag.nnz() - agg_agg_diag.Rows();

    assert(num_faces % 2 == 0);
    num_faces /= 2;

    std::vector<int> indptr(num_faces + 1);
    std::vector<int> indices(num_faces * 2);
    std::vector<double> data(num_faces * 2, 1);

    indptr[0] = 0;

    const auto& agg_indptr = agg_agg_diag.GetIndptr();
    const auto& agg_indices = agg_agg_diag.GetIndices();
    int rows = agg_agg_diag.Rows();
    int count = 0;

    for (int i = 0; i < rows; ++i)
    {
        for (int j = agg_indptr[i]; j < agg_indptr[i + 1]; ++j)
        {
            if (agg_indices[j] > i)
            {
                indices[count * 2] = i;
                indices[count * 2 + 1] = agg_indices[j];

                count++;

                indptr[count] = count * 2;
            }
        }
    }

    assert(count == num_faces);

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                        num_faces, num_aggs);
}

SparseMatrix GraphTopology::MakeFaceEdge(const ParMatrix& agg_agg,
                                         const ParMatrix& edge_ext_agg,
                                         const SparseMatrix& agg_edge_ext,
                                         const SparseMatrix& face_edge_ext)
{
    const auto& agg_agg_diag = agg_agg.GetDiag();
    const auto& agg_agg_offd = agg_agg.GetOffd();

    int num_aggs = agg_agg_diag.Rows();
    int num_edges = face_edge_ext.Cols();
    int num_faces_int = face_edge_ext.Rows();
    int num_faces = num_faces_int + agg_agg_offd.nnz();

    std::vector<int> indptr;
    std::vector<int> indices;

    indptr.reserve(num_faces + 1);

    const auto& ext_indptr = face_edge_ext.GetIndptr();
    const auto& ext_indices = face_edge_ext.GetIndices();
    const auto& ext_data = face_edge_ext.GetData();

    indptr.push_back(0);

    for (int i = 0; i < num_faces_int; i++)
    {
        for (int j = ext_indptr[i]; j < ext_indptr[i + 1]; j++)
        {
            if (ext_data[j] > 1)
            {
                indices.push_back(ext_indices[j]);
            }
        }

        indptr.push_back(indices.size());
    }

    const auto& agg_edge_indptr = agg_edge_ext.GetIndptr();
    const auto& agg_edge_indices = agg_edge_ext.GetIndices();

    const auto& agg_offd_indptr = agg_agg_offd.GetIndptr();
    const auto& agg_offd_indices = agg_agg_offd.GetIndices();
    const auto& agg_colmap = agg_agg.GetColMap();

    const auto& edge_offd_indptr = edge_ext_agg.GetOffd().GetIndptr();
    const auto& edge_offd_indices = edge_ext_agg.GetOffd().GetIndices();
    const auto& edge_colmap = edge_ext_agg.GetColMap();

    for (int i = 0; i < num_aggs; ++i)
    {
        for (int j = agg_offd_indptr[i]; j < agg_offd_indptr[i + 1]; ++j)
        {
            int shared = agg_colmap[agg_offd_indices[j]];

            for (int k = agg_edge_indptr[i]; k < agg_edge_indptr[i + 1]; ++k)
            {
                int edge = agg_edge_indices[k];

                if (edge_offd_indptr[edge + 1] > edge_offd_indptr[edge])
                {
                    int edge_loc = edge_offd_indices[edge_offd_indptr[edge]];

                    if (edge_colmap[edge_loc] == shared)
                    {
                        indices.push_back(edge);
                    }
                }
            }

            indptr.push_back(indices.size());
        }
    }

    assert(static_cast<int>(indptr.size()) == num_faces + 1);

    std::vector<double> data(indices.size(), 1);

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                        num_faces, num_edges);
}

SparseMatrix GraphTopology::ExtendFaceAgg(const ParMatrix& agg_agg,
                                          const SparseMatrix& face_agg_int)
{
    const auto& agg_agg_offd = agg_agg.GetOffd();

    int num_aggs = agg_agg.Rows();

    std::vector<int> indptr(face_agg_int.GetIndptr());
    std::vector<int> indices(face_agg_int.GetIndices());

    const auto& agg_offd_indptr = agg_agg_offd.GetIndptr();

    for (int i = 0; i < num_aggs; ++i)
    {
        for (int j = agg_offd_indptr[i]; j < agg_offd_indptr[i + 1]; ++j)
        {
            indices.push_back(i);
            indptr.push_back(indices.size());
        }
    }

    int num_faces = indptr.size() - 1;

    std::vector<double> data(indices.size(), 1);

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                        num_faces, num_aggs);
}



} // namespace smoothg

