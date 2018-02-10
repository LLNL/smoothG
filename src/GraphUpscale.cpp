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

    @brief Contains GraphUpscale class
*/

#include "GraphUpscale.hpp"


namespace smoothg
{

GraphUpscale::GraphUpscale(MPI_Comm comm,
                 const linalgcpp::SparseMatrix<int>& vertex_edge_global,
                 const std::vector<int>& partitioning_global,
                 double spect_tol, int max_evects,
                 const std::vector<double>& weight_global)
    : Upscale(comm, vertex_edge_global.Rows()),
      global_edges_(vertex_edge_global.Cols()), global_vertices_(vertex_edge_global.Cols())
{
    Init(vertex_edge_global, partitioning_global,
         weight_global, spect_tol, max_evects);
}

GraphUpscale::GraphUpscale(MPI_Comm comm,
                 const linalgcpp::SparseMatrix<int>& vertex_edge_global,
                 double coarse_factor,
                 double spect_tol, int max_evects,
                 const std::vector<double>& weight_global)
    : Upscale(comm, vertex_edge_global.Rows()),
      global_edges_(vertex_edge_global.Cols()), global_vertices_(vertex_edge_global.Cols())
{
    auto edge_vertex = vertex_edge_global.Transpose();
    auto vertex_vertex = vertex_edge_global.Mult(edge_vertex);

    int num_parts = std::max(1.0, (global_vertices_ / (double)(coarse_factor)) + 0.5);

    auto partitioning_global = Partition(vertex_vertex, num_parts);

    Init(vertex_edge_global, partitioning_global,
         weight_global, spect_tol, max_evects);
}

void GraphUpscale::Init(const linalgcpp::SparseMatrix<int>& vertex_edge,
              const std::vector<int>& global_partitioning,
              const std::vector<double>& weight,
              double spect_tol, int max_evects)
{
    DistributeGraph(vertex_edge, global_partitioning);

}

void GraphUpscale::DistributeGraph(const linalgcpp::SparseMatrix<int>& vertex_edge,
                                   const std::vector<int>& global_part)
{
    int num_procs;
    MPI_Comm_size(comm_, &num_procs);

    int num_aggs_global = *std::max(std::begin(global_part), std::end(global_part)) + 1;

    SparseMatrix agg_vert = MakeAggVertex(global_part);
    SparseMatrix proc_agg = MakeProcAgg(num_procs, num_aggs_global);

    SparseMatrix proc_vert = proc_agg.Mult(agg_vert);
    SparseMatrix proc_edge = proc_vert.Mult(vertex_edge);

    // TODO(gelever1): Check if this sort has to go before the transpose
    proc_edge.SortIndices();

    vertex_map_ = proc_vert.GetIndices(myid_);
    edge_map_ = proc_edge.GetIndices(myid_);

    vertex_edge_local_ = vertex_edge.GetSubMatrix(vertex_map_, edge_map_);

    int nvertices_local = proc_vert.RowSize(myid_);

    part_local_.resize(nvertices_local);
    const int agg_begin = proc_agg.GetIndptr()[myid_];

    for (int i = 0; i < nvertices_local; ++i)
    {
        part_local_[i] = global_part[vertex_map_[i]] - agg_begin;
    }

    edge_true_edge_ = MakeEdgeTrueEdge(comm_, proc_edge, edge_map_);

}

} // namespace smoothg
