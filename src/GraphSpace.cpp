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
    : vertex_vdof_(SparseIdentity(graph.NumVertices())),
      edge_edof_(SparseIdentity(graph.NumEdges())), graph_(std::move(graph))

{
    vertex_edof_.MakeRef(graph_.VertexToEdge());
    edof_trueedof_ = make_unique<mfem::HypreParMatrix>();
    edof_trueedof_->MakeRef(graph_.EdgeToTrueEdge());
    if (graph_.HasBoundary())
    {
        edof_bdratt_.MakeRef(graph_.EdgeToBdrAtt());
    }
}

GraphSpace::GraphSpace(Graph graph, mfem::SparseMatrix vertex_vdof,
                       mfem::SparseMatrix vertex_edof, mfem::SparseMatrix edge_edof,
                       std::unique_ptr<mfem::HypreParMatrix> edof_trueedof)
    : vertex_vdof_(std::move(vertex_vdof)), vertex_edof_(std::move(vertex_edof)),
      edge_edof_(std::move(edge_edof)), edof_trueedof_(std::move(edof_trueedof)),
      graph_(std::move(graph))
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
    lhs.vertex_vdof_.Swap(rhs.vertex_edof_);
    lhs.vertex_vdof_.Swap(rhs.edge_edof_);
    std::swap(lhs.edof_trueedof_, rhs.edof_trueedof_);
    lhs.edof_bdratt_.Swap(rhs.edof_bdratt_);

    swap(lhs.graph_, rhs.graph_);
}

} // namespace smoothg

