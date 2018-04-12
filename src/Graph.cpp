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

    @brief Contains Graph class
*/

#include "Graph.hpp"

namespace smoothg
{

Graph::Graph(MPI_Comm comm, const SparseMatrix& vertex_edge_global,
             const std::vector<int>& part_global)
{
    assert(static_cast<int>(part_global.size()) == vertex_edge_global.Rows());

    int myid;
    int num_procs;
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &num_procs);

    int num_aggs_global = *std::max_element(std::begin(part_global), std::end(part_global)) + 1;

    SparseMatrix agg_vert = MakeAggVertex(part_global);
    SparseMatrix proc_agg = MakeProcAgg(num_procs, num_aggs_global);

    SparseMatrix proc_vert = proc_agg.Mult(agg_vert);
    SparseMatrix proc_edge = proc_vert.Mult(vertex_edge_global);

    // TODO(gelever1): Check if this must go before the transpose
    proc_edge.SortIndices();

    vertex_map_ = proc_vert.GetIndices(myid);
    edge_map_ = proc_edge.GetIndices(myid);

    vertex_edge_local_ = vertex_edge_global.GetSubMatrix(vertex_map_, edge_map_);
    vertex_edge_local_ = 1.0;

    int nvertices_local = proc_vert.RowSize(myid);
    part_local_.resize(nvertices_local);

    const int agg_begin = proc_agg.GetIndptr()[myid];

    for (int i = 0; i < nvertices_local; ++i)
    {
        part_local_[i] = part_global[vertex_map_[i]] - agg_begin;
    }

    edge_true_edge_ = MakeEdgeTrueEdge(comm, proc_edge, edge_map_);

    ParMatrix edge_true_edge_T = edge_true_edge_.Transpose();
    edge_edge_ = edge_true_edge_.Mult(edge_true_edge_T);
}

Graph::Graph(const Graph& other) noexcept
    : edge_map_(other.edge_map_),
      vertex_map_(other.vertex_map_),
      part_local_(other.part_local_),
      vertex_edge_local_(other.vertex_edge_local_),
      edge_true_edge_(other.edge_true_edge_),
      edge_edge_(other.edge_edge_)
{

}

Graph::Graph(Graph&& other) noexcept
{
    swap(*this, other);
}

Graph& Graph::operator=(Graph other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(Graph& lhs, Graph& rhs) noexcept
{
    std::swap(lhs.edge_map_, rhs.edge_map_);
    std::swap(lhs.vertex_map_, rhs.vertex_map_);
    std::swap(lhs.part_local_, rhs.part_local_);

    swap(lhs.vertex_edge_local_, rhs.vertex_edge_local_);
    swap(lhs.edge_true_edge_, rhs.edge_true_edge_);
    swap(lhs.edge_edge_, rhs.edge_edge_);
}

} // namespace smoothg

