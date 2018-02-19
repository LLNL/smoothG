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

using Vector = linalgcpp::Vector<double>;
using VectorView = linalgcpp::VectorView<double>;
using BlockVector = linalgcpp::BlockVector<double>;
using SparseMatrix = linalgcpp::SparseMatrix<double>;
using BlockMatrix = linalgcpp::BlockMatrix<double>;
using ParMatrix = parlinalgcpp::ParMatrix;

namespace smoothg
{

GraphUpscale::GraphUpscale(MPI_Comm comm,
                 const linalgcpp::SparseMatrix<double>& vertex_edge_global,
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
                 const SparseMatrix& vertex_edge_global,
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

void GraphUpscale::Init(const SparseMatrix& vertex_edge,
              const std::vector<int>& global_partitioning,
              const std::vector<double>& weight,
              double spect_tol, int max_evects)
{
    DistributeGraph(vertex_edge, global_partitioning);
    MakeFineLevel(weight);

    MakeTopology();
}

void GraphUpscale::DistributeGraph(const SparseMatrix& vertex_edge, const std::vector<int>& global_part)
{
    int num_aggs_global = *std::max_element(std::begin(global_part), std::end(global_part)) + 1;

    SparseMatrix agg_vert = MakeAggVertex(global_part);
    SparseMatrix proc_agg = MakeProcAgg(num_procs_, num_aggs_global);

    SparseMatrix proc_vert = proc_agg.Mult(agg_vert);
    SparseMatrix proc_edge = proc_vert.Mult(vertex_edge);

    // TODO(gelever1): Check if this must go before the transpose
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

void GraphUpscale::MakeFineLevel(const std::vector<double>& global_weight)
{
    int size = edge_map_.size();

    std::vector<double> local_weight(size);

    if (global_weight.size() == edge_true_edge_.Cols())
    {
        for (int i = 0; i < size; ++i)
        {
            assert(std::fabs(global_weight[edge_map_[i]]) > 1e-12);
            local_weight[i] = 1.0 / std::fabs(global_weight[edge_map_[i]]);
        }
    }
    else
    {
        std::fill(std::begin(local_weight), std::end(local_weight), 1.0);
    }

    ParMatrix edge_true_edge_T = edge_true_edge_.Transpose();
    edge_edge_ = edge_true_edge_.Mult(edge_true_edge_T);

    const auto& offd = edge_edge_.GetOffd();

    assert(offd.Rows() == local_weight.size());

    for (int i = 0; i < size; ++i)
    {
        if (offd.RowSize(i))
        {
            local_weight[i] /= 2.0;
        }
    }

    auto DT = vertex_edge_local_.Transpose<double>();
    const auto& indptr(DT.GetIndptr());
    auto& data(DT.GetData());

    int num_vertices = DT.Cols();
    int num_edges = DT.Rows();

    const auto& edge_diag = edge_true_edge_.GetDiag();

    for (int i = 0; i < num_edges; i++)
    {
        const int row_size = DT.RowSize(i);
        assert(row_size == 1 || row_size == 2);

        data[indptr[i]] = 1.;

        if (row_size == 2)
        {
            data[indptr[i] + 1] = -1.;
        }
        else if (edge_diag.RowSize(i) == 0)
        {
            assert(row_size == 1);
            data[indptr[i]] = -1.;
        }
    }

    SparseMatrix M(std::move(local_weight));
    SparseMatrix D(DT.Transpose());
    SparseMatrix W(std::vector<double>(num_vertices, 0.0));

    std::vector<size_t> offsets(3);
    offsets[0] = 0;
    offsets[1] = num_edges;
    offsets[2] = num_edges + num_vertices;

    fine_level_ = BlockMatrix(offsets);

    fine_level_.SetBlock(0, 0, std::move(M));
    fine_level_.SetBlock(0, 1, std::move(DT));
    fine_level_.SetBlock(1, 0, std::move(D));
    fine_level_.SetBlock(1, 1, std::move(W));
}

void GraphUpscale::MakeTopology()
{
    agg_vertex_local_ = MakeAggVertex(part_local_);

    agg_vertex_local_ = 1;
    vertex_edge_local_ = 1;

    SparseMatrix agg_edge_ext = agg_vertex_local_.Mult(vertex_edge_local_);
    agg_edge_ext.SortIndices();

    agg_edge_local_ = RestrictInterior(agg_edge_ext);

    auto edge_ext_agg = agg_edge_ext.Transpose<double>();

    size_t num_vertices = vertex_edge_local_.Rows();
    size_t num_edges = vertex_edge_local_.Cols();
    size_t num_aggs = agg_edge_local_.Rows();

    auto starts3 = parlinalgcpp::GenerateOffsets(comm_, {num_vertices, num_edges, num_aggs});
    const auto& vertex_starts = starts3[0];
    const auto& edge_starts = starts3[1];
    const auto& agg_starts = starts3[2];

    ParMatrix edge_agg_d(comm_, edge_starts, agg_starts, edge_ext_agg);
    ParMatrix agg_edge_d = edge_agg_d.Transpose();

    ParMatrix edge_agg_ext = edge_edge_.Mult(edge_agg_d);
    ParMatrix agg_agg = agg_edge_d.Mult(edge_agg_ext);

    SparseMatrix face_agg_int = MakeFaceAggInt(agg_agg);
    SparseMatrix face_edge_ext = face_agg_int.Mult(agg_edge_ext);

    face_edge_local_ = MakeFaceEdge(agg_agg, edge_edge_,
                                    agg_edge_ext, face_edge_ext);

    face_agg_local_ = ExtendFaceAgg(agg_agg, face_agg_int);

    auto face_starts = parlinalgcpp::GenerateOffsets(comm_, face_agg_local_.Rows());

    auto edge_face = face_edge_local_.Transpose();
    ParMatrix edge_face_d(comm_, edge_starts, face_starts, edge_face);

    face_face_ = parlinalgcpp::RAP(edge_edge_, edge_face_d);
    face_face_ = 1;

    face_true_edge_ = MakeFaceTrueEdge(face_face_);

    ParMatrix vertex_edge_d(comm_, vertex_starts, edge_starts, vertex_edge_local_);
    ParMatrix vertex_edge = vertex_edge_d.Mult(edge_true_edge_);
    ParMatrix edge_vertex = vertex_edge.Transpose();
    ParMatrix agg_edge = agg_edge_d.Mult(edge_true_edge_);

    agg_ext_vertex_ = agg_edge.Mult(edge_vertex);
    agg_ext_vertex_ = 1.0;

    ParMatrix agg_ext_edge_ext = agg_ext_vertex_.Mult(vertex_edge);
    agg_ext_edge_ = RestrictInterior(agg_ext_edge_ext);
}

Vector GraphUpscale::ReadVertexVector(const std::string& filename) const
{
    return ReadVector(filename, vertex_map_);
}

Vector GraphUpscale::ReadVector(const std::string& filename, const std::vector<int>& local_to_global) const
{
    auto global_vect = linalgcpp::ReadText<double>(filename);

    size_t size = local_to_global.size();

    Vector local_vect(size);

    for (size_t i = 0; i < size; ++i)
    {
        local_vect[i] = global_vect[local_to_global[i]];
    }

    return local_vect;
}

} // namespace smoothg
