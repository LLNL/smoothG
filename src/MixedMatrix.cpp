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

    @brief MixedMatrix class
*/

#include "MixedMatrix.hpp"

namespace smoothg
{

MixedMatrix::MixedMatrix(const Graph& graph)
    : edge_true_edge_(graph.edge_true_edge_),
      D_local_(MakeLocalD(graph.edge_true_edge_, graph.vertex_edge_local_)),
      W_local_(graph.W_local_),
      elem_dof_(graph.vertex_edge_local_),
      agg_vertexdof_(SparseIdentity(D_local_.Rows())),
      face_facedof_(SparseIdentity(elem_dof_.Cols()))
{
    int num_vertices = D_local_.Rows();
    int num_edges = D_local_.Cols();

    M_elem_.resize(num_vertices);

    SparseMatrix edge_vertex = D_local_.Transpose();
    std::vector<double> weight_inv = graph.weight_local_;

    for (auto& i : weight_inv)
    {
        assert(std::fabs(i) > 1e-12);
        assert(i == i);
        i = 1.0 / i;
    }

    for (int i = 0; i < num_edges; ++i)
    {
        if (graph.edge_edge_.GetOffd().RowSize(i) == 0)
        {
            weight_inv[i] /= 2.0;
        }
    }

    for (int i = 0; i < num_vertices; ++i)
    {
        std::vector<int> edge_dofs = elem_dof_.GetIndices(i);

        int num_dofs = edge_dofs.size();

        M_elem_[i].SetSize(num_dofs);
        M_elem_[i] = 0.0;

        for (int j = 0; j < num_dofs; ++j)
        {
            M_elem_[i](j, j) = weight_inv[edge_dofs[j]];
        }
    }

    Init();
}

MixedMatrix::MixedMatrix(std::vector<DenseMatrix> M_elem, SparseMatrix elem_dof,
                         SparseMatrix D_local, SparseMatrix W_local,
                         ParMatrix edge_true_edge, SparseMatrix agg_vertexdof,
                         SparseMatrix face_facedof_)
    : edge_true_edge_(std::move(edge_true_edge)),
      D_local_(std::move(D_local)),
      W_local_(std::move(W_local)),
      M_elem_(std::move(M_elem)),
      elem_dof_(std::move(elem_dof)),
      agg_vertexdof_(std::move(agg_vertexdof)),
      face_facedof_(std::move(face_facedof_))
{
    Init();
}

void MixedMatrix::Init()
{
    MPI_Comm comm = edge_true_edge_.GetComm();

    auto starts = parlinalgcpp::GenerateOffsets(comm, {D_local_.Rows(), D_local_.Cols()});
    std::vector<HYPRE_Int>& vertex_starts = starts[0];
    std::vector<HYPRE_Int>& edge_starts = starts[1];

    ParMatrix D_d(comm, vertex_starts, edge_starts, D_local_);
    D_global_ = D_d.Mult(edge_true_edge_);

    if (M_local_.Rows() == D_local_.Cols())
    {
        ParMatrix M_d(comm, edge_starts, M_local_);
        M_global_ = parlinalgcpp::RAP(M_d, edge_true_edge_);
    }

    if (W_local_.Rows() == D_local_.Rows())
    {
        W_global_ = ParMatrix(comm, vertex_starts, W_local_);
    }

    offsets_ = {0, D_local_.Cols(), D_local_.Cols() + D_local_.Rows()};
    true_offsets_ = {0, D_global_.Cols(), D_global_.Cols() + D_global_.Rows()};
}

MixedMatrix::MixedMatrix(const MixedMatrix& other) noexcept
    : edge_true_edge_(other.edge_true_edge_),
      M_local_(other.M_local_),
      D_local_(other.D_local_),
      W_local_(other.W_local_),
      M_global_(other.M_global_),
      D_global_(other.D_global_),
      W_global_(other.W_global_),
      offsets_(other.offsets_),
      true_offsets_(other.true_offsets_),
      M_elem_(other.M_elem_),
      elem_dof_(other.elem_dof_),
      agg_vertexdof_(other.agg_vertexdof_),
      face_facedof_(other.face_facedof_)
{
}

MixedMatrix& MixedMatrix::operator=(MixedMatrix&& other) noexcept
{
    swap(*this, other);

    return *this;
}

MixedMatrix::MixedMatrix(MixedMatrix&& other) noexcept
{
    swap(*this, other);
}

void swap(MixedMatrix& lhs, MixedMatrix& rhs) noexcept
{
    swap(lhs.edge_true_edge_, rhs.edge_true_edge_);

    swap(lhs.M_local_, rhs.M_local_);
    swap(lhs.D_local_, rhs.D_local_);
    swap(lhs.W_local_, rhs.W_local_);

    swap(lhs.M_global_, rhs.M_global_);
    swap(lhs.D_global_, rhs.D_global_);
    swap(lhs.W_global_, rhs.W_global_);

    std::swap(lhs.offsets_, rhs.offsets_);
    std::swap(lhs.true_offsets_, rhs.true_offsets_);

    swap(lhs.M_elem_, rhs.M_elem_);
    swap(lhs.elem_dof_, rhs.elem_dof_);

    swap(lhs.agg_vertexdof_, rhs.agg_vertexdof_);
    std::swap(lhs.face_facedof_, rhs.face_facedof_);
}

int MixedMatrix::Rows() const
{
    return D_local_.Rows() + D_local_.Cols();
}

int MixedMatrix::Cols() const
{
    return D_local_.Rows() + D_local_.Cols();
}

int MixedMatrix::GlobalRows() const
{
    return D_global_.GlobalRows() + D_global_.GlobalCols();
}

int MixedMatrix::GlobalCols() const
{
    return D_global_.GlobalRows() + D_global_.GlobalCols();
}

int MixedMatrix::NNZ() const
{
    return M_local_.nnz() + (2 * D_local_.nnz())
           + W_local_.nnz();
}

int MixedMatrix::GlobalNNZ() const
{
    return M_global_.nnz() + (2 * D_global_.nnz())
           + W_global_.nnz();
}

bool MixedMatrix::CheckW() const
{
    int local_size = W_global_.Rows();
    int global_size;

    MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_MAX, D_global_.GetComm());

    const double zero_tol = 1e-6;

    return global_size > 0 && W_global_.MaxNorm() > zero_tol;
}


ParMatrix MixedMatrix::ToPrimal() const
{
    assert(M_global_.Cols() == D_global_.Cols());
    assert(M_global_.Rows() == D_global_.Cols());

    ParMatrix MinvDT = D_global_.Transpose();
    MinvDT.InverseScaleRows(M_global_.GetDiag().GetDiag());

    ParMatrix A = D_global_.Mult(MinvDT);

    if (CheckW())
    {
        A = ParAdd(A, W_global_);
    }

    return A;
}

void MixedMatrix::AssembleM()
{
    int M_size = D_local_.Cols();
    CooMatrix M_coo(M_size, M_size);

    int num_aggs = M_elem_.size();
    int nnz = 0;

    for (const auto& elem : M_elem_)
    {
        nnz += elem.Rows() * elem.Cols();
    }

    M_coo.Reserve(nnz);

    for (int i = 0; i < num_aggs; ++i)
    {
        std::vector<int> dofs = elem_dof_.GetIndices(i);
        M_coo.Add(dofs, dofs, M_elem_[i]);
    }

    M_coo.EliminateZeros(1e-15);
    M_local_ = M_coo.ToSparse();
    ParMatrix M_d(edge_true_edge_.GetComm(), edge_true_edge_.GetRowStarts(), M_local_);
    M_global_ = parlinalgcpp::RAP(M_d, edge_true_edge_);
}

void MixedMatrix::AssembleM(const std::vector<double>& agg_weight)
{
    assert(agg_weight.size() == M_elem_.size());

    int M_size = D_local_.Cols();
    CooMatrix M_coo(M_size, M_size);

    int num_aggs = M_elem_.size();
    int nnz = 0;

    for (const auto& elem : M_elem_)
    {
        nnz += elem.Rows() * elem.Cols();
    }

    M_coo.Reserve(nnz);

    for (int i = 0; i < num_aggs; ++i)
    {
        double scale = 1.0 / agg_weight[i];
        std::vector<int> dofs = elem_dof_.GetIndices(i);

        M_coo.Add(dofs, dofs, scale, M_elem_[i]);
    }

    M_coo.EliminateZeros(1e-15);
    M_local_ = M_coo.ToSparse();
    ParMatrix M_d(edge_true_edge_.GetComm(), edge_true_edge_.GetRowStarts(), M_local_);
    M_global_ = parlinalgcpp::RAP(M_d, edge_true_edge_);
}

SparseMatrix MixedMatrix::MakeLocalD(const ParMatrix& edge_true_edge,
                                     const SparseMatrix& vertex_edge) const
{
    SparseMatrix edge_vertex = vertex_edge.Transpose();

    std::vector<int> indptr = edge_vertex.GetIndptr();
    std::vector<int> indices = edge_vertex.GetIndices();
    std::vector<double> data = edge_vertex.GetData();

    int num_edges = edge_vertex.Rows();
    int num_vertices = edge_vertex.Cols();

    const SparseMatrix& owned_edges = edge_true_edge.GetDiag();

    for (int i = 0; i < num_edges; i++)
    {
        const int row_edges = edge_vertex.RowSize(i);
        assert(row_edges == 1 || row_edges == 2);

        data[indptr[i]] = 1.;

        if (row_edges == 2)
        {
            data[indptr[i] + 1] = -1.;
        }
        else if (owned_edges.RowSize(i) == 0)
        {
            assert(row_edges == 1);
            data[indptr[i]] = -1.;
        }
    }

    SparseMatrix DT(std::move(indptr), std::move(indices), std::move(data),
                    num_edges, num_vertices);

    return DT.Transpose();
}


} // namespace smoothg
