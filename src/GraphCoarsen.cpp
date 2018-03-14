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
    ParMatrix permute_v = MakeExtPermutation(gt.agg_ext_vertex_);
    ParMatrix permute_e = MakeExtPermutation(gt.agg_ext_edge_);

    ParMatrix permute_v_T = permute_v.Transpose();
    ParMatrix permute_e_T = permute_e.Transpose();

    ParMatrix M_ext_global = permute_e.Mult(mgl.M_global_.Mult(permute_e_T));
    ParMatrix D_ext_global = permute_v.Mult(mgl.D_global_.Mult(permute_e_T));

    ParMatrix face_perm_edge = gt.face_edge_.Mult(graph.edge_true_edge_.Mult(permute_e_T));
    const SparseMatrix& face_edge = face_perm_edge.GetDiag();

    int marker_size = std::max(permute_v.Rows(), permute_e.Rows());
    col_marker_.resize(marker_size, -1);

    ComputeVertexTargets(gt, M_ext_global, D_ext_global);

    auto shared_sigma = CollectSigma(gt, face_edge);
    auto shared_M = CollectM(gt, mgl.M_local_);
    auto shared_D = CollectD(gt, mgl.D_local_);

    ComputeEdgeTargets(gt, face_edge, shared_sigma, shared_M, shared_D);
    ScaleEdgeTargets(gt, mgl.D_local_);

    printf("%d: %d %d %d\n", permute_v.GetMyId(), shared_sigma.size(), shared_D.size(), shared_M.size());

    BuildFaceCoarseDof(gt);
    BuildPvertex(gt);
    BuildPedge(gt, mgl);
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
        const ParMatrix& M_ext_global, const ParMatrix& D_ext_global)
{
    const SparseMatrix& M_ext = M_ext_global.GetDiag();
    const SparseMatrix& D_ext = D_ext_global.GetDiag();

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

std::vector<std::vector<std::vector<double>>> GraphCoarsen::CollectM(const GraphTopology& gt,
        const SparseMatrix& M_local)
{
    SharedEntityComm<std::vector<double>> sec_M(gt.face_true_face_);

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
        std::vector<double> M_diag = M_face.GetData();

        sec_M.ReduceSend(face, std::move(M_diag));
    }

    return sec_M.Collect();
}

void GraphCoarsen::ComputeEdgeTargets(const GraphTopology& gt,
                                const SparseMatrix& face_edge,
                                const Vect2D<DenseMatrix>& shared_sigma,
                                const Vect2D<std::vector<double>>& shared_M,
                                const Vect2D<SparseMatrix>& shared_D)
{
    const SparseMatrix& face_shared = gt.face_face_.GetOffd();

    SharedEntityComm<DenseMatrix> sec_face(gt.face_true_face_);
    DenseMatrix collected_sigma;

    int num_faces = gt.face_edge_local_.Rows();

    for (int face = 0; face < num_faces; ++face)
    {
        int num_face_edges = face_edge.RowSize(face);

        if (!sec_face.IsOwnedByMe(face))
        {
            edge_targets_[face].Resize(num_face_edges, 0);
            continue;
        }

        if (num_face_edges == 1)
        {
            edge_targets_[face] = DenseMatrix(1, 1, {1.0});
            continue;
        }

        const auto& face_sigma = shared_sigma[face];
        const auto& face_M = shared_M[face];
        const auto& face_D = shared_D[face];

        linalgcpp::HStack(face_sigma, collected_sigma);

        bool shared = face_shared.RowSize(face) > 0;

        const auto& M_local = shared ? Combine(face_M, num_face_edges) : face_M[0];
        const auto& D_local = shared ? Combine(face_D, num_face_edges) : face_D[0];
        const int split = shared ? face_D[0].Rows() : GetSplit(gt, face);

        GraphEdgeSolver solver(M_local, D_local);
        Vector one_neg_one = MakeOneNegOne(D_local.Rows(), split);

        Vector pv_sol = solver.Mult(one_neg_one);
        VectorView pv_sigma(pv_sol.begin(), num_face_edges);

        edge_targets_[face] = Orthogonalize(collected_sigma, pv_sigma, max_evects_);
    }

    sec_face.Broadcast(edge_targets_);
}

void GraphCoarsen::ScaleEdgeTargets(const GraphTopology& gt, const SparseMatrix& D_local)
{
    int num_faces = gt.face_edge_.Rows();

    for (int face = 0; face < num_faces; ++face)
    {
        int agg = gt.face_agg_local_.GetIndices(face)[0];

        std::vector<int> vertices = gt.agg_vertex_local_.GetIndices(agg);
        std::vector<int> face_dofs = gt.face_edge_local_.GetIndices(face);

        SparseMatrix D_transfer = D_local.GetSubMatrix(vertices, face_dofs, col_marker_);
        DenseMatrix& edge_traces(edge_targets_[face]);

        Vector one(D_transfer.Rows(), 1.0);
        VectorView pv_trace = edge_traces.GetColView(0);

        double oneDpv = one.Mult(D_transfer.Mult(pv_trace));
        double beta = (oneDpv < 0) ? -1.0 : 1.0;

        pv_trace /= oneDpv;

        int num_traces = edge_traces.Cols();

        for (int k = 1; k < num_traces; ++k)
        {
            VectorView trace = edge_traces.GetColView(k);
            double alpha = one.Mult(D_transfer.Mult(trace));

            Vector scaled_pv(pv_trace);
            scaled_pv *= alpha * beta;

            trace -= scaled_pv;
        }
    }
}

std::vector<double> GraphCoarsen::Combine(const std::vector<std::vector<double>>& face_M, int num_face_edges) const
{
    assert(face_M.size() == 2);

    int size = face_M[0].size() + face_M[1].size() - num_face_edges;

    std::vector<double> combined(size);

    for (int i = 0; i < num_face_edges; ++i)
    {
        combined[i] = face_M[0][i] + face_M[1][i];
    }

    std::copy(std::begin(face_M[0]) + num_face_edges,
              std::end(face_M[0]),
              std::begin(combined) + num_face_edges);

    std::copy(std::begin(face_M[1]) + num_face_edges,
              std::end(face_M[1]),
              std::begin(combined) + face_M[0].size());

    return combined;
}

SparseMatrix GraphCoarsen::Combine(const std::vector<SparseMatrix>& face_D, int num_face_edges) const
{
    assert(face_D.size() == 2);

    int rows = face_D[0].Rows() + face_D[1].Rows();
    int cols = face_D[0].Cols() + face_D[1].Cols() - num_face_edges;

    std::vector<int> indptr = face_D[0].GetIndptr();
    indptr.insert(std::end(indptr), std::begin(face_D[1].GetIndptr()) + 1,
                  std::end(face_D[1].GetIndptr()));

    int row_start = face_D[0].Rows() + 1;
    int row_end = indptr.size();
    int nnz_offset = face_D[0].nnz();

    for (int i = row_start; i < row_end; ++i)
    {
        indptr[i] += nnz_offset;
    }

    std::vector<int> indices = face_D[0].GetIndices();
    indices.insert(std::end(indices), std::begin(face_D[1].GetIndices()),
                  std::end(face_D[1].GetIndices()));

    int col_offset = face_D[0].Cols() - num_face_edges;
    int nnz_end = indices.size();

    for (int i = nnz_offset; i < nnz_end; ++i)
    {
        if (indices[i] >= num_face_edges)
        {
            indices[i] += col_offset;
        }
    }

    std::vector<double> data = face_D[0].GetData();
    data.insert(std::end(data), std::begin(face_D[1].GetData()),
                  std::end(face_D[1].GetData()));

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                        rows, cols);
}

Vector GraphCoarsen::MakeOneNegOne(int size, int split) const
{
    assert(size >= 0);
    assert(split >= 0);

    Vector vect(size);

    for (int i = 0; i < split; ++i)
    {
        vect[i] = 1.0 / split;
    }

    for (int i = split; i < size; ++i)
    {
        vect[i] = -1.0 / (size - split);
    }

    return vect;
}

int GraphCoarsen::GetSplit(const GraphTopology& gt, int face) const
{
    std::vector<int> neighbors = gt.face_agg_local_.GetIndices(face);
    assert(neighbors.size() >= 1);
    int agg = neighbors[0];

    return gt.agg_vertex_local_.RowSize(agg);
}

void GraphCoarsen::BuildFaceCoarseDof(const GraphTopology& gt)
{
    int num_faces = gt.face_edge_.Rows();

    std::vector<int> indptr(num_faces + 1);
    indptr[0] = 0;

    for (int i = 0; i < num_faces; ++i)
    {
        indptr[i + 1] = indptr[i] + edge_targets_[i].Cols();
    }

    int num_traces = indptr.back();

    std::vector<int> indices(num_traces);
    std::iota(std::begin(indices), std::end(indices), 0);

    std::vector<double> data(num_traces, 1.0);

    face_cdof_ = SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                              num_faces, num_traces);
}

void GraphCoarsen::BuildPvertex(const GraphTopology& gt)
{
    const SparseMatrix& agg_vertex = gt.agg_vertex_local_;
    int num_aggs = vertex_targets_.size();
    int num_vertices = agg_vertex.Cols();

    std::vector<int> indptr(num_vertices + 1);
    indptr[0] = 0;

    for (int i = 0; i < num_aggs; ++i)
    {
        std::vector<int> vertices = agg_vertex.GetIndices(i);
        int num_coarse_dofs = vertex_targets_[i].Cols();

        for (auto vertex : vertices)
        {
            indptr[vertex + 1] = num_coarse_dofs;
        }
    }

    for (int i = 0; i < num_vertices; ++i)
    {
        indptr[i + 1] += indptr[i];
    }

    int nnz = indptr.back();
    std::vector<int> indices(nnz);
    std::vector<double> data(nnz);

    int coarse_dof_counter = 0;

    for (int i = 0; i < num_aggs; ++i)
    {
        std::vector<int> fine_dofs = agg_vertex.GetIndices(i);
        int num_fine_dofs = fine_dofs.size();
        int num_coarse_dofs = vertex_targets_[i].Cols();

        const DenseMatrix& target_i = vertex_targets_[i];

        for (int j = 0; j < num_fine_dofs; ++j)
        {
            int counter = indptr[fine_dofs[j]];

            for (int k = 0; k < num_coarse_dofs; ++k)
            {
                indices[counter] = coarse_dof_counter + k;
                data[counter] = target_i(j, k);

                counter++;
            }
        }

        coarse_dof_counter += num_coarse_dofs;
    }

    P_vertex_ = SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                             num_vertices, coarse_dof_counter);

}

void GraphCoarsen::BuildPedge(const GraphTopology& gt, const MixedMatrix& mgl)
{
    const SparseMatrix& agg_edge = gt.agg_edge_local_;
    const SparseMatrix& agg_vertex = gt.agg_vertex_local_;
    const SparseMatrix& agg_face = gt.agg_face_local_;
    const SparseMatrix& face_edge = gt.face_edge_local_;

    int num_aggs = agg_edge.Rows();
    int num_edges = agg_edge.Cols();
    int num_vertices = agg_vertex.Cols();

    DenseMatrix bubbles;
    DenseMatrix trace_ext;
    DenseMatrix D_trace;

    for (int i = 0; i < num_aggs; ++i)
    {
        std::vector<int> edge_dofs = agg_edge.GetIndices(i);
        std::vector<int> vertex_dofs = agg_vertex.GetIndices(i);
        std::vector<int> faces = agg_face.GetIndices(i);

        SparseMatrix M = mgl.M_local_.GetSubMatrix(edge_dofs, edge_dofs, col_marker_);
        SparseMatrix D = mgl.D_local_.GetSubMatrix(vertex_dofs, edge_dofs, col_marker_);

        GraphEdgeSolver solver(M, D);

        solver.PartMult(1, vertex_targets_[i], bubbles);

        for (auto face : faces)
        {
            std::vector<int> face_dofs = face_edge.GetIndices(face);

            SparseMatrix D_transfer = mgl.D_local_.GetSubMatrix(vertex_dofs, face_dofs, col_marker_);
            D_trace.Resize(D_transfer.Rows(), edge_targets_[face].Cols());
            D_transfer.Mult(edge_targets_[face], D_trace);

            OrthoConstant(D_trace);

            solver.Mult(D_trace, trace_ext);
        }

    }

}

} // namespace smoothg
