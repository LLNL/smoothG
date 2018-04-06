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

GraphCoarsen::GraphCoarsen(const MixedMatrix& mgl, const GraphTopology& gt,
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

    ParMatrix face_perm_edge = gt.face_edge_.Mult(mgl.edge_true_edge_.Mult(permute_e_T));

    int marker_size = std::max(permute_v.Rows(), permute_e.Rows());
    col_marker_.resize(marker_size, -1);

    ComputeVertexTargets(gt, M_ext_global, D_ext_global);
    ComputeEdgeTargets(gt, mgl, face_perm_edge);

    BuildFaceCoarseDof(gt);
    BuildAggBubbleDof();
    BuildPvertex(gt);
    BuildPedge(gt, mgl);
}

GraphCoarsen::GraphCoarsen(const GraphCoarsen& other) noexcept
    : max_evects_(other.max_evects_),
      spect_tol_(other.spect_tol_),
      P_edge_(other.P_edge_),
      P_vertex_(other.P_vertex_),
      face_cdof_(other.face_cdof_),
      agg_bubble_dof_(other.agg_bubble_dof_),
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

    swap(lhs.P_edge_, rhs.P_edge_);
    swap(lhs.P_vertex_, rhs.P_vertex_);
    swap(lhs.face_cdof_, rhs.face_cdof_);
    swap(lhs.agg_bubble_dof_, rhs.agg_bubble_dof_);

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

        // auto& evals = eigen_pair.first; // TODO(gelever1): verify correct
        auto& evects = eigen_pair.second;

        if (evects.Cols() > 1)
        {
            DenseMatrix evects_ortho = evects.GetCol(1, evects.Cols());
            agg_ext_sigma_[agg] = D_sub_T.Mult(evects_ortho);
        }
        else
        {
            agg_ext_sigma_[agg].Resize(D_sub_T.Rows(), 0);
        }

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
            if (agg_ext_sigma_[agg].Cols() > 0)
            {
                std::vector<int> edge_dofs_ext = GetExtDofs(gt.agg_ext_edge_, agg);

                DenseMatrix face_restrict = RestrictLocal(agg_ext_sigma_[agg], col_marker_,
                        edge_dofs_ext, face_dofs);

                face_sigma.SetCol(col_count, face_restrict);
                col_count += face_restrict.Cols();
            }
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
                                const MixedMatrix& mgl,
                                const ParMatrix& face_perm_edge)
{
    const SparseMatrix& face_edge = face_perm_edge.GetDiag();

    auto shared_sigma = CollectSigma(gt, face_edge);
    auto shared_M = CollectM(gt, mgl.M_local_);
    auto shared_D = CollectD(gt, mgl.D_local_);

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

        const auto& face_M = shared_M[face];
        const auto& face_D = shared_D[face];
        const auto& face_sigma = shared_sigma[face];

        linalgcpp::HStack(face_sigma, collected_sigma);

        bool shared = face_shared.RowSize(face) > 0;

        // TODO(gelever1): resolve this copy.  (Types must match so face_? gets copied and promoted
        // to rvalue).
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

    ScaleEdgeTargets(gt, mgl.D_local_);
}

void GraphCoarsen::ScaleEdgeTargets(const GraphTopology& gt, const SparseMatrix& D_local)
{
    int num_faces = gt.face_edge_.Rows();

    for (int face = 0; face < num_faces; ++face)
    {
        DenseMatrix& edge_traces(edge_targets_[face]);
        if (edge_traces.Cols() < 1)
        {
            continue;
        }

        int agg = gt.face_agg_local_.GetIndices(face)[0];

        std::vector<int> vertices = gt.agg_vertex_local_.GetIndices(agg);
        std::vector<int> face_dofs = gt.face_edge_local_.GetIndices(face);

        SparseMatrix D_transfer = D_local.GetSubMatrix(vertices, face_dofs, col_marker_);

        Vector one(D_transfer.Rows(), 1.0);
        Vector oneD = D_transfer.MultAT(one);
        VectorView pv_trace = edge_traces.GetColView(0);

        double oneDpv = oneD.Mult(pv_trace);
        double beta = (oneDpv < 0) ? -1.0 : 1.0;
        oneDpv *= beta;

        pv_trace /= oneDpv;

        int num_traces = edge_traces.Cols();

        for (int k = 1; k < num_traces; ++k)
        {
            VectorView trace = edge_traces.GetColView(k);
            double alpha = oneD.Mult(trace);

            trace.Sub(alpha * beta, pv_trace);
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

void GraphCoarsen::BuildAggBubbleDof()
{
    int num_aggs = vertex_targets_.size();

    std::vector<int> indptr(num_aggs + 1);
    indptr[0] = 0;

    for (int i = 0; i < num_aggs; ++i)
    {
        assert(vertex_targets_[i].Cols() >= 1);

        indptr[i + 1] = indptr[i] + vertex_targets_[i].Cols() - 1;
    }

    int num_traces = SumCols(edge_targets_);
    int num_bubbles = indptr.back();

    std::vector<int> indices(num_bubbles);
    std::iota(std::begin(indices), std::end(indices), num_traces);

    std::vector<double> data(num_bubbles, 1.0);

    agg_bubble_dof_ = SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                              num_aggs, num_traces + num_bubbles);
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
    const SparseMatrix& agg_face = gt.agg_face_local_;
    const SparseMatrix& agg_edge = gt.agg_edge_local_;
    const SparseMatrix& face_edge = gt.face_edge_local_;
    const SparseMatrix& agg_vertex = gt.agg_vertex_local_;

    int num_aggs = agg_edge.Rows();
    int num_faces = face_edge.Rows();
    int num_edges = agg_edge.Cols();
    int num_coarse_dofs = agg_bubble_dof_.Cols();

    CooMatrix P_edge(num_edges, num_coarse_dofs);

    DenseMatrix bubbles;
    DenseMatrix D_trace;
    DenseMatrix trace_ext;

    for (int agg = 0; agg < num_aggs; ++agg)
    {
        std::vector<int> faces = agg_face.GetIndices(agg);
        std::vector<int> edge_dofs = agg_edge.GetIndices(agg);
        std::vector<int> vertex_dofs = agg_vertex.GetIndices(agg);
        std::vector<int> bubble_dofs = agg_bubble_dof_.GetIndices(agg);

        SparseMatrix M = mgl.M_local_.GetSubMatrix(edge_dofs, edge_dofs, col_marker_);
        SparseMatrix D = mgl.D_local_.GetSubMatrix(vertex_dofs, edge_dofs, col_marker_);

        GraphEdgeSolver solver(M, D);

        for (auto face : faces)
        {
            std::vector<int> face_coarse_dofs = face_cdof_.GetIndices(face);
            std::vector<int> face_fine_dofs = face_edge.GetIndices(face);

            SparseMatrix D_transfer = mgl.D_local_.GetSubMatrix(vertex_dofs, face_fine_dofs, col_marker_);
            D_trace.Resize(D_transfer.Rows(), edge_targets_[face].Cols());
            D_transfer.Mult(edge_targets_[face], D_trace);

            OrthoConstant(D_trace);

            solver.Mult(D_trace, trace_ext);

            P_edge.Add(edge_dofs, face_coarse_dofs, -1.0, trace_ext);
        }

        solver.PartMult(1, vertex_targets_[agg], bubbles);
        P_edge.Add(edge_dofs, bubble_dofs, bubbles);
    }

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> face_fine_dofs = face_edge.GetIndices(face);
        std::vector<int> face_coarse_dofs = face_cdof_.GetIndices(face);

        P_edge.Add(face_fine_dofs, face_coarse_dofs, edge_targets_[face]);
    }

    double zero_tol = 1e-8;
    P_edge.EliminateZeros(zero_tol);

    P_edge_ = P_edge.ToSparse();
}

ParMatrix GraphCoarsen::BuildEdgeTrueEdge(const GraphTopology& gt) const
{
    int num_faces = face_cdof_.Rows();
    int num_traces = face_cdof_.Cols();
    int num_coarse_dofs = P_edge_.Cols();

    const auto& ftf_diag = gt.face_true_face_.GetDiag();

    int num_true_dofs = num_coarse_dofs - num_traces;

    for (int i = 0; i < num_faces; ++i)
    {
        if (ftf_diag.RowSize(i) > 0)
        {
            num_true_dofs += face_cdof_.RowSize(i);
        }
    }

    MPI_Comm comm = gt.face_true_face_.GetComm();
    auto cface_starts = parlinalgcpp::GenerateOffsets(comm, num_coarse_dofs);
    const auto& face_starts = gt.face_true_face_.GetRowStarts();

    SparseMatrix face_cdof_expand(face_cdof_.GetIndptr(), face_cdof_.GetIndices(),
                                  face_cdof_.GetData(), num_faces, num_coarse_dofs);
    ParMatrix face_cdof_d(comm, face_starts, cface_starts, std::move(face_cdof_expand));

    ParMatrix cface_cface = parlinalgcpp::RAP(gt.face_face_, face_cdof_d);

    const SparseMatrix& cface_cface_offd = cface_cface.GetOffd();
    const std::vector<int>& cface_cface_colmap = cface_cface.GetColMap();

    std::vector<int> offd_indptr(num_coarse_dofs + 1);
    offd_indptr[0] = 0;

    int offd_nnz = 0;

    for (int i = 0; i < num_coarse_dofs; ++i)
    {
        if (cface_cface_offd.RowSize(i) > 0)
        {
            offd_nnz++;
        }

        offd_indptr[i + 1] = offd_nnz;
    }

    std::vector<int> offd_indices(offd_nnz);
    std::vector<double> offd_data(offd_nnz, 1.0);

    const auto& face_cdof_indptr = face_cdof_.GetIndptr();
    const auto& face_cdof_indices = face_cdof_.GetIndices();

    int col_count = 0;

    for (int i = 0; i < num_faces; ++i)
    {
        if (gt.face_face_.GetOffd().RowSize(i) > 0)
        {
            int first_dof = face_cdof_indices[face_cdof_indptr[i]];

            std::vector<int> face_cdofs = cface_cface_offd.GetIndices(first_dof);
            assert(static_cast<int>(face_cdofs.size()) == face_cdof_.RowSize(i));

            for (auto cdof : face_cdofs)
            {
                offd_indices[col_count++] = cdof;
            }
        }
    }

    assert(col_count == offd_nnz);
    assert(col_count == static_cast<int>(cface_cface_colmap.size()));

    SparseMatrix d_td_d_diag = SparseIdentity(num_coarse_dofs);
    SparseMatrix d_td_d_offd(std::move(offd_indptr), std::move(offd_indices),
                             std::move(offd_data), num_coarse_dofs, col_count);

    ParMatrix d_td_d(comm, cface_starts, cface_starts,
                     std::move(d_td_d_diag), std::move(d_td_d_offd),
                     cface_cface_colmap);

    return MakeEntityTrueEntity(d_td_d);
}

MixedMatrix GraphCoarsen::Coarsen(const GraphTopology& gt, const MixedMatrix& mgl) const
{
    SparseMatrix P_edge_T = P_edge_.Transpose();
    SparseMatrix P_vertex_T = P_vertex_.Transpose();

    SparseMatrix M_c = P_edge_T.Mult(mgl.M_local_.Mult(P_edge_));
    SparseMatrix D_c = P_vertex_T.Mult(mgl.D_local_.Mult(P_edge_));
    SparseMatrix W_c = P_vertex_T.Mult(mgl.W_local_.Mult(P_vertex_));

    M_c.EliminateZeros(1e-8);
    D_c.EliminateZeros(1e-8);

    ParMatrix edge_true_edge = BuildEdgeTrueEdge(gt);

    return MixedMatrix(std::move(M_c), std::move(D_c), std::move(W_c),
                       std::move(edge_true_edge));
}

Vector GraphCoarsen::Interpolate(const VectorView& coarse_vect) const
{
    return P_vertex_.MultAT(coarse_vect);
}

void GraphCoarsen::Interpolate(const VectorView& coarse_vect, VectorView fine_vect) const
{
    P_vertex_.Mult(coarse_vect, fine_vect);
}

Vector GraphCoarsen::Restrict(const VectorView& fine_vect) const
{
    return P_vertex_.MultAT(fine_vect);
}

void GraphCoarsen::Restrict(const VectorView& fine_vect, VectorView coarse_vect) const
{
    P_vertex_.MultAT(fine_vect, coarse_vect);
}

BlockVector GraphCoarsen::Interpolate(const BlockVector& coarse_vect) const
{
    std::vector<int> fine_offsets = {0, P_edge_.Rows(), P_edge_.Rows() + P_vertex_.Rows()};
    BlockVector fine_vect(fine_offsets);

    Interpolate(coarse_vect, fine_vect);

    return fine_vect;
}

void GraphCoarsen::Interpolate(const BlockVector& coarse_vect, BlockVector& fine_vect) const
{
    P_edge_.Mult(coarse_vect.GetBlock(0), fine_vect.GetBlock(0));
    P_vertex_.Mult(coarse_vect.GetBlock(1), fine_vect.GetBlock(1));
}

BlockVector GraphCoarsen::Restrict(const BlockVector& fine_vect) const
{
    std::vector<int> coarse_offsets = {0, P_edge_.Cols(), P_edge_.Cols() + P_vertex_.Cols()};
    BlockVector coarse_vect(coarse_offsets);

    Restrict(fine_vect, coarse_vect);

    return coarse_vect;
}

void GraphCoarsen::Restrict(const BlockVector& fine_vect, BlockVector& coarse_vect) const
{
    P_edge_.MultAT(fine_vect.GetBlock(0), coarse_vect.GetBlock(0));
    P_vertex_.MultAT(fine_vect.GetBlock(1), coarse_vect.GetBlock(1));
}


} // namespace smoothg
