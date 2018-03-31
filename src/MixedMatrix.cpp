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

MixedMatrix::MixedMatrix(const Graph& graph, const std::vector<double>& global_weight)
    : edge_true_edge_(graph.edge_true_edge_)
{
    M_local_ = MakeLocalM(edge_true_edge_, graph.edge_edge_, graph.edge_map_, global_weight);
    D_local_ = MakeLocalD(edge_true_edge_, graph.vertex_edge_local_);
    W_local_ = SparseMatrix(std::vector<double>(D_local_.Rows(), 0.0));

    Init();
}

MixedMatrix::MixedMatrix(SparseMatrix M_local, SparseMatrix D_local,
                         SparseMatrix W_local, ParMatrix edge_true_edge)
    : M_local_(std::move(M_local)), D_local_(std::move(D_local)),
      W_local_(std::move(W_local)), edge_true_edge_(std::move(edge_true_edge))
{
    Init();
}

void MixedMatrix::Init()
{
    MPI_Comm comm = edge_true_edge_.GetComm();

    auto starts = parlinalgcpp::GenerateOffsets(comm, {D_local_.Rows(), D_local_.Cols()});
    std::vector<HYPRE_Int>& vertex_starts = starts[0];
    std::vector<HYPRE_Int>& edge_starts = starts[1];

    ParMatrix M_d(comm, edge_starts, M_local_);
    ParMatrix D_d(comm, vertex_starts, edge_starts, D_local_);

    M_global_ = parlinalgcpp::RAP(M_d, edge_true_edge_);
    D_global_ = D_d.Mult(edge_true_edge_);
    W_global_ = ParMatrix(comm, vertex_starts, W_local_);

    offsets_ = {0, M_local_.Rows(), M_local_.Rows() + D_local_.Rows()};
    true_offsets_ = {0, M_global_.Rows(), M_global_.Rows() + D_global_.Rows()};
}

MixedMatrix::MixedMatrix(const MixedMatrix& other) noexcept
    : M_local_(other.M_local_),
      D_local_(other.D_local_),
      W_local_(other.W_local_),
      M_global_(other.M_global_),
      D_global_(other.D_global_),
      W_global_(other.W_global_),
      edge_true_edge_(other.edge_true_edge_),
      offsets_(other.offsets_),
      true_offsets_(other.true_offsets_)
{

}

MixedMatrix::MixedMatrix(MixedMatrix&& other) noexcept
{
    swap(*this, other);
}

MixedMatrix& MixedMatrix::operator=(MixedMatrix other) noexcept
{
    swap(*this, other);

    return *this;
}
    
void swap(MixedMatrix& lhs, MixedMatrix& rhs) noexcept
{
    swap(lhs.M_local_, rhs.M_local_);
    swap(lhs.D_local_, rhs.D_local_);
    swap(lhs.W_local_, rhs.W_local_);

    swap(lhs.M_global_, rhs.M_global_);
    swap(lhs.D_global_, rhs.D_global_);
    swap(lhs.W_global_, rhs.W_global_);

    swap(lhs.edge_true_edge_, rhs.edge_true_edge_);

    std::swap(lhs.offsets_, rhs.offsets_);
    std::swap(lhs.true_offsets_, rhs.true_offsets_);
}

SparseMatrix MixedMatrix::MakeLocalM(const ParMatrix& edge_true_edge,
                        const ParMatrix& edge_edge,
                        const std::vector<int>& edge_map,
                        const std::vector<double>& global_weight)
{
    int size = edge_map.size();

    std::vector<double> local_weight(size);

    if (global_weight.size() == edge_true_edge.Cols())
    {
        for (int i = 0; i < size; ++i)
        {
            assert(std::fabs(global_weight[edge_map[i]]) > 1e-12);
            local_weight[i] = 1.0 / std::fabs(global_weight[edge_map[i]]);
        }
    }
    else
    {
        std::fill(std::begin(local_weight), std::end(local_weight), 1.0);
    }

    const SparseMatrix& edge_offd = edge_edge.GetOffd();

    assert(edge_offd.Rows() == local_weight.size());

    for (int i = 0; i < size; ++i)
    {
        if (edge_offd.RowSize(i))
        {
            local_weight[i] /= 2.0;
        }
    }

    return SparseMatrix(std::move(local_weight));
}

SparseMatrix MixedMatrix::MakeLocalD(const ParMatrix& edge_true_edge,
                          const SparseMatrix& vertex_edge)
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

    SparseMatrix DT(std::move(indptr), std::move(indices), std::move(data), num_edges, num_vertices);

    return DT.Transpose();
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
    const double zero_tol = 1e-6;

    return W_global_.MaxNorm() > zero_tol;
}



} // namespace smoothg
