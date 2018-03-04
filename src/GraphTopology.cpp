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

    ParMatrix edge_agg_d(comm, edge_starts, agg_starts, edge_ext_agg);
    ParMatrix agg_edge_d = edge_agg_d.Transpose();

    ParMatrix edge_agg_ext = graph.edge_edge_.Mult(edge_agg_d);
    ParMatrix agg_agg = agg_edge_d.Mult(edge_agg_ext);

    agg_edge_ext = 1.0;
    SparseMatrix face_agg_int = MakeFaceAggInt(agg_agg);
    SparseMatrix face_edge_ext = face_agg_int.Mult(agg_edge_ext);

    face_edge_local_ = MakeFaceEdge(agg_agg, edge_agg_ext,
                                    agg_edge_ext, face_edge_ext);

    face_agg_local_ = ExtendFaceAgg(agg_agg, face_agg_int);

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

    swap(lhs.face_face_, rhs.face_face_);
    swap(lhs.face_true_face_, rhs.face_true_face_);
    swap(lhs.face_edge_, rhs.face_edge_);
    swap(lhs.agg_ext_vertex_, rhs.agg_ext_vertex_);
    swap(lhs.agg_ext_edge_, rhs.agg_ext_edge_);
}


} // namespace smoothg

