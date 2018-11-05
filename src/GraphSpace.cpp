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
      vertex_edof_(graph.GetVertexToEdge(), false),
      edge_edof_(SparseIdentity(graph.NumEdges())), graph_(std::move(graph))
{
    edof_trueedof_ = std::make_shared<mfem::HypreParMatrix>();
    edof_trueedof_->MakeRef(graph_.GetEdgeToTrueEdge());
}

GraphSpace::GraphSpace(Graph graph, mfem::SparseMatrix vertex_vdof,
                       mfem::SparseMatrix vertex_edof, mfem::SparseMatrix edge_edof,
                       std::shared_ptr<mfem::HypreParMatrix> edof_trueedof)
    : vertex_vdof_(std::move(vertex_vdof)), vertex_edof_(std::move(vertex_edof)),
      edge_edof_(std::move(edge_edof)), edof_trueedof_(std::move(edof_trueedof)),
      graph_(std::move(graph))
{
}

} // namespace smoothg

