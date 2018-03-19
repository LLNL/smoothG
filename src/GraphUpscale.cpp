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
                 const linalgcpp::SparseMatrix<double>& vertex_edge_global,
                 const std::vector<int>& partitioning_global,
                 double spect_tol, int max_evects,
                 const std::vector<double>& weight_global)
    : Upscale(comm, vertex_edge_global.Rows()),
      global_edges_(vertex_edge_global.Cols()), global_vertices_(vertex_edge_global.Cols()),
      spect_tol_(spect_tol), max_evects_(max_evects)
{
    Init(vertex_edge_global, partitioning_global, weight_global);
}

GraphUpscale::GraphUpscale(MPI_Comm comm,
                 const SparseMatrix& vertex_edge_global,
                 double coarse_factor,
                 double spect_tol, int max_evects,
                 const std::vector<double>& weight_global)
    : Upscale(comm, vertex_edge_global.Rows()),
      global_edges_(vertex_edge_global.Cols()), global_vertices_(vertex_edge_global.Cols()),
      spect_tol_(spect_tol), max_evects_(max_evects)
{
    SparseMatrix edge_vertex = vertex_edge_global.Transpose();
    SparseMatrix vertex_vertex = vertex_edge_global.Mult(edge_vertex);

    int num_parts = std::max(1.0, (global_vertices_ / (double)(coarse_factor)) + 0.5);

    bool contig = true;
    double ubal = 2.0;
    std::vector<int> partitioning_global = Partition(vertex_vertex, num_parts, contig, ubal);

    Init(vertex_edge_global, partitioning_global, weight_global);
}

void GraphUpscale::Init(const SparseMatrix& vertex_edge,
              const std::vector<int>& global_partitioning,
              const std::vector<double>& weight)
{
    graph_ = Graph(comm_, vertex_edge, global_partitioning);
    mgl_.emplace_back(graph_, weight);
    gt_ = GraphTopology(comm_, graph_);

    coarsener_ = GraphCoarsen(GetFineMatrix(), gt_,
                              max_evects_, spect_tol_);

    mgl_.push_back(coarsener_.Coarsen(gt_, GetFineMatrix()));


}

Vector GraphUpscale::ReadVertexVector(const std::string& filename) const
{
    return ReadVector(filename, graph_.vertex_map_);
}

Vector GraphUpscale::ReadVector(const std::string& filename, const std::vector<int>& local_to_global) const
{
    std::vector<double> global_vect = linalgcpp::ReadText<double>(filename);

    size_t size = local_to_global.size();

    Vector local_vect(size);

    for (size_t i = 0; i < size; ++i)
    {
        local_vect[i] = global_vect[local_to_global[i]];
    }

    return local_vect;
}

} // namespace smoothg
