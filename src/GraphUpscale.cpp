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
    mixed_mat_fine_ = MixedMatrix(comm_, graph_, weight);
    gt_ = GraphTopology(comm_, graph_);

    MakeCoarseSpace();
}

void GraphUpscale::MakeCoarseSpace()
{
    const SparseMatrix& M_local = mixed_mat_fine_.M_local_;
    const SparseMatrix& D_local = mixed_mat_fine_.D_local_;
    const SparseMatrix& W_local = mixed_mat_fine_.W_local_;

    const ParMatrix& M_global = mixed_mat_fine_.M_global_;
    const ParMatrix& D_global = mixed_mat_fine_.D_global_;
    const ParMatrix& W_global = mixed_mat_fine_.W_global_;

    ParMatrix permute_v = MakeExtPermutation(comm_, gt_.agg_ext_vertex_);
    ParMatrix permute_e = MakeExtPermutation(comm_, gt_.agg_ext_edge_);

    ParMatrix permute_v_T = permute_v.Transpose();
    ParMatrix permute_e_T = permute_e.Transpose();

    ParMatrix M_ext_global = permute_e.Mult(M_global.Mult(permute_e_T));
    ParMatrix D_ext_global = permute_v.Mult(D_global.Mult(permute_e_T));
    ParMatrix W_ext_global = permute_v.Mult(W_global.Mult(permute_v_T));

    ParMatrix face_edge_true_edge = gt_.face_edge_.Mult(graph_.edge_true_edge_);
    ParMatrix face_perm_edge = face_edge_true_edge.Mult(permute_e_T);

    const SparseMatrix& M_ext = M_ext_global.GetDiag();
    const SparseMatrix& D_ext = D_ext_global.GetDiag();
    const SparseMatrix& W_ext = W_ext_global.GetDiag();
    const SparseMatrix& face_edge = face_perm_edge.GetDiag();

    int marker_size = std::max(permute_v.Rows(), permute_e.Rows());
    std::vector<int> dof_marker(marker_size, -1);

    int num_aggs = gt_.agg_ext_edge_.Rows();
    int num_faces = gt_.face_edge_local_.Rows();

    std::vector<DenseMatrix> vertex_targets(num_aggs);
    std::vector<DenseMatrix> agg_ext_sigma(num_aggs);

    linalgcpp::EigenSolver eigen;
    linalgcpp::EigenPair eigen_pair;

    for (int agg = 0; agg < num_aggs; ++agg)
    {
        std::vector<int> edge_dofs_ext = GetExtDofs(gt_.agg_ext_edge_, agg);
        std::vector<int> vertex_dofs_ext = GetExtDofs(gt_.agg_ext_vertex_, agg);
        std::vector<int> vertex_dofs_local = gt_.agg_vertex_local_.GetIndices(agg);

        if (edge_dofs_ext.size() == 0)
        {
            vertex_targets[agg] = DenseMatrix(1, 1, {1.0});
            continue;
        }

        SparseMatrix M_sub = M_ext.GetSubMatrix(edge_dofs_ext, edge_dofs_ext, dof_marker);
        SparseMatrix D_sub = D_ext.GetSubMatrix(vertex_dofs_ext, edge_dofs_ext, dof_marker);
        SparseMatrix D_sub_T = D_sub.Transpose();

        D_sub_T.InverseScaleRows(M_sub);

        SparseMatrix DMinvDT = D_sub.Mult(D_sub_T);

        eigen.Solve(DMinvDT, spect_tol_, max_evects_, eigen_pair);

        auto& evals = eigen_pair.first;
        auto& evects = eigen_pair.second;

        DenseMatrix evects_ortho = evects.GetCol(1, evects.Cols());
        agg_ext_sigma[agg] = D_sub_T.Mult(evects_ortho);

        DenseMatrix evects_restricted = RestrictLocal(evects, dof_marker,
                                                      vertex_dofs_ext, vertex_dofs_local);

        vertex_targets[agg] = smoothg::Orthogonalize(evects_restricted);
    }

    SharedEntityComm<DenseMatrix> sec_sigma(gt_.face_true_face_);

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> face_dofs = face_edge.GetIndices(face);
        std::vector<int> neighbors = gt_.face_agg_local_.GetIndices(face);

        int total_vects = 0;
        int col_count = 0;

        for (auto agg : neighbors)
        {
            total_vects += agg_ext_sigma[agg].Cols();
        }

        DenseMatrix face_sigma(face_dofs.size(), total_vects);

        for (auto agg : neighbors)
        {
            std::vector<int> edge_dofs_ext = GetExtDofs(gt_.agg_ext_edge_, agg);

            DenseMatrix face_restrict = RestrictLocal(agg_ext_sigma[agg], dof_marker,
                                                      edge_dofs_ext, face_dofs);

            face_sigma.SetCol(col_count, face_restrict);
            col_count += face_restrict.Cols();
        }

        assert(col_count == total_vects);

        sec_sigma.ReduceSend(face, std::move(face_sigma));
    }

    auto shared_sigma = sec_sigma.Collect();

    SharedEntityComm<SparseMatrix> sec_D(gt_.face_true_face_);

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> vertex_ext_dofs;
        std::vector<int> edge_ext_dofs = gt_.face_edge_local_.GetIndices(face);
        std::vector<int> neighbors = gt_.face_agg_local_.GetIndices(face);

        for (auto agg : neighbors)
        {
            std::vector<int> agg_edges = gt_.agg_edge_local_.GetIndices(agg);
            edge_ext_dofs.insert(std::end(edge_ext_dofs), std::begin(agg_edges),
                                          std::end(agg_edges));

            std::vector<int> agg_vertices = gt_.agg_vertex_local_.GetIndices(agg);
            vertex_ext_dofs.insert(std::end(vertex_ext_dofs), std::begin(agg_vertices),
                                          std::end(agg_vertices));
        }

        SparseMatrix D_face = D_local.GetSubMatrix(vertex_ext_dofs, edge_ext_dofs, dof_marker);

        sec_D.ReduceSend(face, std::move(D_face));
    }

    auto shared_D = sec_D.Collect();

    SharedEntityComm<Vector> sec_M(gt_.face_true_face_);

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> edge_ext_dofs = gt_.face_edge_local_.GetIndices(face);
        std::vector<int> neighbors = gt_.face_agg_local_.GetIndices(face);

        for (auto agg : neighbors)
        {
            std::vector<int> agg_edges = gt_.agg_edge_local_.GetIndices(agg);
            edge_ext_dofs.insert(std::end(edge_ext_dofs), std::begin(agg_edges),
                                 std::end(agg_edges));
        }

        SparseMatrix M_face = M_local.GetSubMatrix(edge_ext_dofs, edge_ext_dofs, dof_marker);
        std::vector<double> M_diag_data = M_face.GetDiag();
        Vector M_diag(std::move(M_diag_data));

        sec_M.ReduceSend(face, std::move(M_diag));
    }

    auto shared_M = sec_M.Collect();

    printf("%d: %d %d %d\n", myid_, shared_sigma.size(), shared_D.size(), shared_M.size());
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
