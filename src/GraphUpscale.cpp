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
    DistributeGraph(vertex_edge, global_partitioning);
    MakeFineLevel(weight);

    MakeTopology();
    MakeCoarseSpace();
}

void GraphUpscale::DistributeGraph(const SparseMatrix& vertex_edge, const std::vector<int>& global_part)
{
    assert(global_part.size() == vertex_edge.Rows());

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
    vertex_edge_local_ = 1.0;

    int nvertices_local = proc_vert.RowSize(myid_);
    part_local_.resize(nvertices_local);

    const int agg_begin = proc_agg.GetIndptr()[myid_];

    for (int i = 0; i < nvertices_local; ++i)
    {
        part_local_[i] = global_part[vertex_map_[i]] - agg_begin;
    }

    edge_true_edge_ = MakeEdgeTrueEdge(comm_, proc_edge, edge_map_);

    ParMatrix edge_true_edge_T = edge_true_edge_.Transpose();
    edge_edge_ = edge_true_edge_.Mult(edge_true_edge_T);
}

void GraphUpscale::MakeFineLevel(const std::vector<double>& global_weight)
{
    M_local_ = MakeLocalM(edge_true_edge_, edge_edge_, edge_map_, global_weight);
    D_local_ = MakeLocalD(edge_true_edge_, vertex_edge_local_);
    W_local_ = SparseMatrix(std::vector<double>(D_local_.Rows(), 0.0));
    offsets_ = {0, M_local_.Rows(), M_local_.Rows() + D_local_.Rows()};

    auto starts = parlinalgcpp::GenerateOffsets(comm_, {D_local_.Rows(), D_local_.Cols()});
    std::vector<HYPRE_Int>& vertex_starts = starts[0];
    std::vector<HYPRE_Int>& edge_starts = starts[1];

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

    agg_edge_ext = 1.0;
    SparseMatrix face_agg_int = MakeFaceAggInt(agg_agg);
    SparseMatrix face_edge_ext = face_agg_int.Mult(agg_edge_ext);

    face_edge_local_ = MakeFaceEdge(agg_agg, edge_agg_ext,
                                    agg_edge_ext, face_edge_ext);

    face_agg_local_ = ExtendFaceAgg(agg_agg, face_agg_int);

    auto face_starts = parlinalgcpp::GenerateOffsets(comm_, face_agg_local_.Rows());

    face_edge_ = ParMatrix(comm_, face_starts, edge_starts, face_edge_local_);
    ParMatrix edge_face = face_edge_.Transpose();

    face_face_ = parlinalgcpp::RAP(edge_edge_, edge_face);
    face_face_ = 1;

    face_true_face_ = MakeEntityTrueEntity(face_face_);

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

    const SparseMatrix& M_ext = M_ext_global.GetDiag();
    const SparseMatrix& D_ext = D_ext_global.GetDiag();
    const SparseMatrix& W_ext = W_ext_global.GetDiag();
    const SparseMatrix& face_edge = face_perm_edge.GetDiag();

    int marker_size = std::max(permute_v.Rows(), permute_e.Rows());
    std::vector<int> dof_marker(marker_size, -1);

    int num_aggs = agg_ext_edge_.Rows();
    int num_faces = face_edge_local_.Rows();

    std::vector<DenseMatrix> vertex_targets(num_aggs);
    std::vector<DenseMatrix> agg_ext_sigma(num_aggs);

    linalgcpp::EigenSolver eigen;
    linalgcpp::EigenPair eigen_pair;

    for (int agg = 0; agg < num_aggs; ++agg)
    {
        std::vector<int> edge_dofs_ext = GetExtDofs(agg_ext_edge_, agg);
        std::vector<int> vertex_dofs_ext = GetExtDofs(agg_ext_vertex_, agg);
        std::vector<int> vertex_dofs_local = agg_vertex_local_.GetIndices(agg);

        if (edge_dofs_ext.size() == 0)
        {
            vertex_targets[agg] = DenseMatrix(1, 1, {1.0});
            continue;
        }

        SparseMatrix M_local = M_ext.GetSubMatrix(edge_dofs_ext, edge_dofs_ext, dof_marker);
        SparseMatrix D_local = D_ext.GetSubMatrix(vertex_dofs_ext, edge_dofs_ext, dof_marker);
        SparseMatrix D_local_T = D_local.Transpose();

        D_local_T.InverseScaleRows(M_local);

        SparseMatrix DMinvDT = D_local.Mult(D_local_T);

        eigen.Solve(DMinvDT, spect_tol_, max_evects_, eigen_pair);

        auto& evals = eigen_pair.first;
        auto& evects = eigen_pair.second;

        DenseMatrix evects_ortho = evects.GetCol(1, evects.Cols());
        agg_ext_sigma[agg] = D_local_T.Mult(evects_ortho);

        DenseMatrix evects_restricted = RestrictLocal(evects, dof_marker,
                                                      vertex_dofs_ext, vertex_dofs_local);

        vertex_targets[agg] = smoothg::Orthogonalize(evects_restricted);
    }

    SharedEntityComm<DenseMatrix> sec_sigma(face_true_face_);

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> face_dofs = face_edge.GetIndices(face);
        std::vector<int> neighbors = face_agg_local_.GetIndices(face);

        int total_vects = 0;
        int col_count = 0;

        for (auto agg : neighbors)
        {
            total_vects += agg_ext_sigma[agg].Cols();
        }

        DenseMatrix face_sigma(face_dofs.size(), total_vects);

        for (auto agg : neighbors)
        {
            std::vector<int> edge_dofs_ext = GetExtDofs(agg_ext_edge_, agg);

            DenseMatrix face_restrict = RestrictLocal(agg_ext_sigma[agg], dof_marker,
                                                      edge_dofs_ext, face_dofs);

            face_sigma.SetCol(col_count, face_restrict);
            col_count += face_restrict.Cols();
        }

        assert(col_count == total_vects);

        sec_sigma.ReduceSend(face, std::move(face_sigma));
    }

    auto shared_sigma = sec_sigma.Collect();

    SharedEntityComm<SparseMatrix> sec_D(face_true_face_);

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> vertex_ext_dofs;
        std::vector<int> edge_ext_dofs = face_edge_local_.GetIndices(face);
        std::vector<int> neighbors = face_agg_local_.GetIndices(face);

        for (auto agg : neighbors)
        {
            std::vector<int> agg_edges = agg_edge_local_.GetIndices(agg);
            edge_ext_dofs.insert(std::end(edge_ext_dofs), std::begin(agg_edges),
                                          std::end(agg_edges));

            std::vector<int> agg_vertices = agg_vertex_local_.GetIndices(agg);
            vertex_ext_dofs.insert(std::end(vertex_ext_dofs), std::begin(agg_vertices),
                                          std::end(agg_vertices));
        }

        SparseMatrix D_face = D_local_.GetSubMatrix(vertex_ext_dofs, edge_ext_dofs, dof_marker);

        sec_D.ReduceSend(face, std::move(D_face));
    }

    auto shared_D = sec_D.Collect();

    SharedEntityComm<Vector> sec_M(face_true_face_);

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> edge_ext_dofs = face_edge_local_.GetIndices(face);
        std::vector<int> neighbors = face_agg_local_.GetIndices(face);

        for (auto agg : neighbors)
        {
            std::vector<int> agg_edges = agg_edge_local_.GetIndices(agg);
            edge_ext_dofs.insert(std::end(edge_ext_dofs), std::begin(agg_edges),
                                 std::end(agg_edges));
        }

        SparseMatrix M_face = M_local_.GetSubMatrix(edge_ext_dofs, edge_ext_dofs, dof_marker);
        std::vector<double> M_diag_data = M_face.GetDiag();
        Vector M_diag(std::move(M_diag_data));

        sec_M.ReduceSend(face, std::move(M_diag));
    }

    auto shared_M = sec_M.Collect();

    printf("%d: %d %d %d\n", myid_, shared_sigma.size(), shared_D.size(), shared_M.size());
}

Vector GraphUpscale::ReadVertexVector(const std::string& filename) const
{
    return ReadVector(filename, vertex_map_);
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
