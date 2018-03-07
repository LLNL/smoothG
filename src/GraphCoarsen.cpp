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

    @brief GraphCoarsen class
*/

#include "GraphCoarsen.hpp"

namespace smoothg
{

GraphCoarsen::GraphCoarsen(const Graph& graph, const MixedMatrix& mgl, const GraphTopology& gt,
        int max_evects, double spect_tol)
    : max_evects_(max_evects), spect_tol_(spect_tol),
      vertex_targets_(gt.agg_ext_edge_.Rows()),
      edge_targets_(gt.face_edge_.Rows()),
      agg_ext_sigma_(gt.agg_ext_edge_.Rows())
{
    const SparseMatrix& M_local = mgl.M_local_;
    const SparseMatrix& D_local = mgl.D_local_;
    const SparseMatrix& W_local = mgl.W_local_;

    const ParMatrix& M_global = mgl.M_global_;
    const ParMatrix& D_global = mgl.D_global_;
    const ParMatrix& W_global = mgl.W_global_;

    ParMatrix permute_v = MakeExtPermutation(gt.agg_ext_vertex_);
    ParMatrix permute_e = MakeExtPermutation(gt.agg_ext_edge_);

    ParMatrix permute_v_T = permute_v.Transpose();
    ParMatrix permute_e_T = permute_e.Transpose();

    ParMatrix M_ext_global = permute_e.Mult(M_global.Mult(permute_e_T));
    ParMatrix D_ext_global = permute_v.Mult(D_global.Mult(permute_e_T));
    ParMatrix W_ext_global = permute_v.Mult(W_global.Mult(permute_v_T));

    ParMatrix face_edge_true_edge = gt.face_edge_.Mult(graph.edge_true_edge_);
    ParMatrix face_perm_edge = face_edge_true_edge.Mult(permute_e_T);

    const SparseMatrix& M_ext = M_ext_global.GetDiag();
    const SparseMatrix& D_ext = D_ext_global.GetDiag();
    const SparseMatrix& W_ext = W_ext_global.GetDiag();
    const SparseMatrix& face_edge = face_perm_edge.GetDiag();

    int marker_size = std::max(permute_v.Rows(), permute_e.Rows());
    col_marker_.resize(marker_size, -1);

    ComputeVertexTargets(gt, M_ext, D_ext);

    auto shared_sigma = CollectSigma(gt, face_edge);
    auto shared_D = CollectD(gt, D_local);
    auto shared_M = CollectM(gt, M_local);

    printf("%d: %d %d %d\n", permute_v.GetMyId(), shared_sigma.size(), shared_D.size(), shared_M.size());
}

GraphCoarsen::GraphCoarsen(const GraphCoarsen& other) noexcept
    : max_evects_(other.max_evects_),
      spect_tol_(other.spect_tol_),
      coarse_(other.coarse_),
      P_edge_(other.P_edge_),
      P_vertex_(other.P_vertex_),
      vertex_targets_(other.vertex_targets_),
      edge_targets_(other.edge_targets_),
      agg_ext_sigma_(other.agg_ext_sigma_),
      col_marker_(other.col_marker_)
{

}

GraphCoarsen::GraphCoarsen(GraphCoarsen&& other) noexcept
{
    swap(*this, other);
}

GraphCoarsen& GraphCoarsen::operator=(GraphCoarsen other) noexcept
{
    swap(*this, other);

    return *this;
}
    
void swap(GraphCoarsen& lhs, GraphCoarsen& rhs) noexcept
{
    std::swap(lhs.max_evects_, rhs.max_evects_);
    std::swap(lhs.spect_tol_, rhs.spect_tol_);

    swap(lhs.coarse_, rhs.coarse_);
    swap(lhs.P_edge_, rhs.P_edge_);
    swap(lhs.P_vertex_, rhs.P_vertex_);

    swap(lhs.vertex_targets_, rhs.vertex_targets_);
    swap(lhs.edge_targets_, rhs.edge_targets_);
    swap(lhs.agg_ext_sigma_, rhs.agg_ext_sigma_);

    swap(lhs.col_marker_, rhs.col_marker_);
}

void GraphCoarsen::ComputeVertexTargets(const GraphTopology& gt,
        const SparseMatrix& M_ext, const SparseMatrix& D_ext)
{
    int num_aggs = gt.agg_ext_edge_.Rows();

    linalgcpp::EigenSolver eigen;
    linalgcpp::EigenPair eigen_pair;

    for (int agg = 0; agg < num_aggs; ++agg)
    {
        std::vector<int> edge_dofs_ext = GetExtDofs(gt.agg_ext_edge_, agg);
        std::vector<int> vertex_dofs_ext = GetExtDofs(gt.agg_ext_vertex_, agg);
        std::vector<int> vertex_dofs_local = gt.agg_vertex_local_.GetIndices(agg);

        if (edge_dofs_ext.size() == 0)
        {
            vertex_targets_[agg] = DenseMatrix(1, 1, {1.0});
            continue;
        }

        SparseMatrix M_sub = M_ext.GetSubMatrix(edge_dofs_ext, edge_dofs_ext, col_marker_);
        SparseMatrix D_sub = D_ext.GetSubMatrix(vertex_dofs_ext, edge_dofs_ext, col_marker_);
        SparseMatrix D_sub_T = D_sub.Transpose();

        D_sub_T.InverseScaleRows(M_sub);

        SparseMatrix DMinvDT = D_sub.Mult(D_sub_T);

        eigen.Solve(DMinvDT, spect_tol_, max_evects_, eigen_pair);

        auto& evals = eigen_pair.first;
        auto& evects = eigen_pair.second;

        DenseMatrix evects_ortho = evects.GetCol(1, evects.Cols());
        agg_ext_sigma_[agg] = D_sub_T.Mult(evects_ortho);

        DenseMatrix evects_restricted = RestrictLocal(evects, col_marker_,
                                                      vertex_dofs_ext, vertex_dofs_local);

        vertex_targets_[agg] = smoothg::Orthogonalize(evects_restricted);
    }
}

std::vector<std::vector<DenseMatrix>> GraphCoarsen::CollectSigma(const GraphTopology& gt,
        const SparseMatrix& face_edge)
{
    SharedEntityComm<DenseMatrix> sec_sigma(gt.face_true_face_);

    int num_faces = gt.face_edge_local_.Rows();

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> face_dofs = face_edge.GetIndices(face);
        std::vector<int> neighbors = gt.face_agg_local_.GetIndices(face);

        int total_vects = 0;
        int col_count = 0;

        for (auto agg : neighbors)
        {
            total_vects += agg_ext_sigma_[agg].Cols();
        }

        DenseMatrix face_sigma(face_dofs.size(), total_vects);

        for (auto agg : neighbors)
        {
            std::vector<int> edge_dofs_ext = GetExtDofs(gt.agg_ext_edge_, agg);

            DenseMatrix face_restrict = RestrictLocal(agg_ext_sigma_[agg], col_marker_,
                                                      edge_dofs_ext, face_dofs);

            face_sigma.SetCol(col_count, face_restrict);
            col_count += face_restrict.Cols();
        }

        assert(col_count == total_vects);

        sec_sigma.ReduceSend(face, std::move(face_sigma));
    }

    return sec_sigma.Collect();
}

std::vector<std::vector<SparseMatrix>> GraphCoarsen::CollectD(const GraphTopology& gt,
        const SparseMatrix& D_local)
{

    SharedEntityComm<SparseMatrix> sec_D(gt.face_true_face_);

    int num_faces = gt.face_edge_local_.Rows();

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> vertex_ext_dofs;
        std::vector<int> edge_ext_dofs = gt.face_edge_local_.GetIndices(face);
        std::vector<int> neighbors = gt.face_agg_local_.GetIndices(face);

        for (auto agg : neighbors)
        {
            std::vector<int> agg_edges = gt.agg_edge_local_.GetIndices(agg);
            edge_ext_dofs.insert(std::end(edge_ext_dofs), std::begin(agg_edges),
                                          std::end(agg_edges));

            std::vector<int> agg_vertices = gt.agg_vertex_local_.GetIndices(agg);
            vertex_ext_dofs.insert(std::end(vertex_ext_dofs), std::begin(agg_vertices),
                                          std::end(agg_vertices));
        }

        SparseMatrix D_face = D_local.GetSubMatrix(vertex_ext_dofs, edge_ext_dofs, col_marker_);

        sec_D.ReduceSend(face, std::move(D_face));
    }

    return sec_D.Collect();
}

std::vector<std::vector<Vector>> GraphCoarsen::CollectM(const GraphTopology& gt,
        const SparseMatrix& M_local)
{
    SharedEntityComm<Vector> sec_M(gt.face_true_face_);

    int num_faces = gt.face_edge_local_.Rows();

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> edge_ext_dofs = gt.face_edge_local_.GetIndices(face);
        std::vector<int> neighbors = gt.face_agg_local_.GetIndices(face);

        for (auto agg : neighbors)
        {
            std::vector<int> agg_edges = gt.agg_edge_local_.GetIndices(agg);
            edge_ext_dofs.insert(std::end(edge_ext_dofs), std::begin(agg_edges),
                                 std::end(agg_edges));
        }

        SparseMatrix M_face = M_local.GetSubMatrix(edge_ext_dofs, edge_ext_dofs, col_marker_);
        std::vector<double> M_diag_data = M_face.GetDiag();
        Vector M_diag(std::move(M_diag_data));

        sec_M.ReduceSend(face, std::move(M_diag));
    }

    return sec_M.Collect();
}

} // namespace smoothg
