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
    auto edge_vertex = vertex_edge_global.Transpose();
    auto vertex_vertex = vertex_edge_global.Mult(edge_vertex);

    int num_parts = std::max(1.0, (global_vertices_ / (double)(coarse_factor)) + 0.5);

    bool contig = true;
    double ubal = 2.0;
    auto partitioning_global = Partition(vertex_vertex, num_parts, contig, ubal);

    Init(vertex_edge_global, partitioning_global, weight_global);
}

void GraphUpscale::Init(const SparseMatrix& vertex_edge,
              const std::vector<int>& global_partitioning,
              const std::vector<double>& weight)
{
    DistributeGraph(vertex_edge, global_partitioning);
    MakeFineLevel(weight);

    MakeTopology();
    MakeCoarseSpace();
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

    auto DT = vertex_edge_local_.Transpose();
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

    M_local_ = SparseMatrix(std::move(local_weight));
    D_local_ = SparseMatrix(DT.Transpose());
    W_local_ = SparseMatrix(std::vector<double>(num_vertices, 0.0));
    offsets_ = {0, num_edges, num_edges + num_vertices};

    auto starts = parlinalgcpp::GenerateOffsets(comm_, {D_local_.Rows(), D_local_.Cols()});
    const auto& vertex_starts = starts[0];
    const auto& edge_starts = starts[1];

    ParMatrix M_d(comm_, edge_starts, M_local_);
    ParMatrix D_d(comm_, vertex_starts, edge_starts, D_local_);

    M_global_ = parlinalgcpp::RAP(M_d, edge_true_edge_);
    D_global_ = D_d.Mult(edge_true_edge_);
    W_global_ = ParMatrix(comm_, vertex_starts, W_local_);
    true_offsets_ = {0, M_global_.Rows(), M_global_.Rows() + D_global_.Rows()};
}

void GraphUpscale::MakeTopology()
{
    agg_vertex_local_ = MakeAggVertex(part_local_);

    agg_vertex_local_ = 1;
    vertex_edge_local_ = 1;

    SparseMatrix agg_edge_ext = agg_vertex_local_.Mult(vertex_edge_local_);
    agg_edge_ext.SortIndices();

    agg_edge_local_ = RestrictInterior(agg_edge_ext);

    SparseMatrix edge_ext_agg = agg_edge_ext.Transpose();

    const auto& vertex_starts = D_global_.GetRowStarts();
    const auto& edge_starts = edge_true_edge_.GetRowStarts();

    int num_aggs = agg_edge_local_.Rows();
    auto agg_starts = parlinalgcpp::GenerateOffsets(comm_, num_aggs);

    ParMatrix edge_agg_d(comm_, edge_starts, agg_starts, edge_ext_agg);
    ParMatrix agg_edge_d = edge_agg_d.Transpose();

    ParMatrix edge_agg_ext = edge_edge_.Mult(edge_agg_d);
    ParMatrix agg_agg = agg_edge_d.Mult(edge_agg_ext);

    SparseMatrix face_agg_int = MakeFaceAggInt(agg_agg);
    agg_edge_ext = 1.0;
    SparseMatrix face_edge_ext = face_agg_int.Mult(agg_edge_ext);

    face_edge_local_ = MakeFaceEdge(agg_agg, edge_agg_ext,
                                    agg_edge_ext, face_edge_ext);

    face_agg_local_ = ExtendFaceAgg(agg_agg, face_agg_int);

    auto face_starts = parlinalgcpp::GenerateOffsets(comm_, face_agg_local_.Rows());

    face_edge_ = ParMatrix(comm_, face_starts, edge_starts, face_edge_local_);
    ParMatrix edge_face = face_edge_.Transpose();

    face_face_ = parlinalgcpp::RAP(edge_edge_, edge_face);
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

void GraphUpscale::MakeCoarseSpace()
{
    ParMatrix permute_v = MakeExtPermutation(comm_, agg_ext_vertex_);
    ParMatrix permute_e = MakeExtPermutation(comm_, agg_ext_edge_);

    ParMatrix permute_v_T = permute_v.Transpose();
    ParMatrix permute_e_T = permute_e.Transpose();

    ParMatrix M_ext_global = permute_e.Mult(M_global_.Mult(permute_e_T));
    ParMatrix D_ext_global = permute_v.Mult(D_global_.Mult(permute_e_T));
    ParMatrix W_ext_global = permute_v.Mult(W_global_.Mult(permute_v_T));

    ParMatrix face_edge_true_edge = face_edge_.Mult(edge_true_edge_);
    ParMatrix face_perm_edge = face_edge_true_edge.Mult(permute_e_T);

    const auto& M_ext = M_ext_global.GetDiag();
    const auto& D_ext = D_ext_global.GetDiag();
    const auto& W_ext = W_ext_global.GetDiag();
    const auto& face_edge = face_perm_edge.GetDiag();

    int marker_size = std::max(permute_v.Rows(), permute_e.Rows());
    std::vector<int> col_marker(marker_size, -1);
    std::vector<int> vertex_dof_marker(permute_v.Rows(), -1);
    std::vector<int> row_map;

    int num_aggs = agg_ext_edge_.Rows();
    int num_faces = face_edge_local_.Rows();

    std::vector<DenseMatrix> vertex_targets(num_aggs);
    std::vector<DenseMatrix> agg_ext_sigma(num_aggs);

    linalgcpp::EigenSolver eigen;
    linalgcpp::EigenPair eigen_pair;

    for (int agg = 0; agg < num_aggs; ++agg)
    {
        const auto edge_dofs_ext = GetExtDofs(agg_ext_edge_, agg);
        const auto vertex_dofs_ext = GetExtDofs(agg_ext_vertex_, agg);
        const auto vertex_dofs_local = agg_vertex_local_.GetIndices(agg);

        if (edge_dofs_ext.size() == 0)
        {
            vertex_targets[agg] = DenseMatrix(1, 1, {1.0});
            continue;
        }

        SparseMatrix M_local = M_ext.GetSubMatrix(edge_dofs_ext, edge_dofs_ext, col_marker);
        SparseMatrix D_local = D_ext.GetSubMatrix(vertex_dofs_ext, edge_dofs_ext, col_marker);
        SparseMatrix D_local_T = D_local.Transpose();

        D_local_T.InverseScaleRows(M_local);

        SparseMatrix DMinvDT = D_local.Mult(D_local_T);

        eigen.Solve(DMinvDT, spect_tol_, max_evects_, eigen_pair);

        auto& evals = eigen_pair.first;
        auto& evects = eigen_pair.second;

        DenseMatrix evects_ortho = evects.GetCol(1, evects.Cols());
        agg_ext_sigma[agg] = D_local_T.Mult(evects_ortho);

        DenseMatrix evects_restricted = RestrictLocal(evects, vertex_dof_marker,
                                                      vertex_dofs_ext, vertex_dofs_local);

        vertex_targets[agg] = smoothg::Orthogonalize(evects_restricted);
    }

    DenseMatrix face_sigma;

    for (int face = 0; face < num_faces; ++face)
    {
        auto face_dofs = face_edge.GetIndices(face);
        auto neighbors = face_agg_local_.GetIndices(face);

        int total_vects = 0;

        for (auto agg : neighbors)
        {
            total_vects += agg_ext_sigma[agg].Cols();
        }

        face_sigma.Resize(face_dofs.size(), total_vects);
        int col_count = 0;

        for (auto agg : neighbors)
        {
            const auto edge_dofs_ext = GetExtDofs(agg_ext_edge_, agg);

            DenseMatrix face_restrict = RestrictLocal(agg_ext_sigma[agg], col_marker,
                                                      edge_dofs_ext, face_dofs);

            face_sigma.SetCol(col_count, face_restrict);
            col_count += face_restrict.Cols();
        }

        // TODO(gelever1): SendReduce(face_sigma);

        assert(col_count == total_vects);
    }
        // TODO(gelever1): Collect(face_sigma);

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> vertex_ext_dofs;
        std::vector<int> edge_ext_dofs = face_edge_local_.GetIndices(face);
        std::vector<int> neighbors = face_agg_local_.GetIndices(face);

        for (auto agg : neighbors)
        {
            auto agg_edges = agg_edge_local_.GetIndices(agg);
            edge_ext_dofs.insert(std::end(edge_ext_dofs), std::begin(agg_edges),
                                          std::end(agg_edges));

            auto agg_vertices = agg_vertex_local_.GetIndices(agg);
            vertex_ext_dofs.insert(std::end(vertex_ext_dofs), std::begin(agg_vertices),
                                          std::end(agg_vertices));
        }

        auto D_face = D_local_.GetSubMatrix(vertex_ext_dofs, edge_ext_dofs, col_marker);

        // TODO(gelever1): SendReduce(face, D_face);
    }

    // TODO(gelever1): Collect shared D

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> edge_ext_dofs = face_edge_local_.GetIndices(face);
        std::vector<int> neighbors = face_agg_local_.GetIndices(face);

        for (auto agg : neighbors)
        {
            auto agg_edges = agg_edge_local_.GetIndices(agg);
            edge_ext_dofs.insert(std::end(edge_ext_dofs), std::begin(agg_edges),
                                 std::end(agg_edges));
        }

        auto M_face = M_local_.GetSubMatrix(edge_ext_dofs, edge_ext_dofs, col_marker);
        auto M_diag = M_face.GetDiag();

        // TODO(gelever1): SendReduce(face, M_diag);
    }

    // TODO(gelever1): Collect shared M

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
