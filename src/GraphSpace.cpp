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

    @brief Implements GraphSpace object.
*/

#include "GraphSpace.hpp"
#include "MatrixUtilities.hpp"

using std::unique_ptr;

namespace smoothg
{

// Construct entities to dofs table in the case when each dof belongs to one
// and only one entity and the enumeration of dofs solely depends on entity
mfem::SparseMatrix BuildEntityToDof(const std::vector<mfem::DenseMatrix>& local_targets)
{
    const unsigned int num_entities = local_targets.size();
    int* I = new int[num_entities + 1]();
    for (unsigned int entity = 0; entity < num_entities; ++entity)
    {
        I[entity + 1] = I[entity] + local_targets[entity].NumCols();
    }

    int nnz = I[num_entities];
    int* J = new int[nnz];
    std::iota(J, J + nnz, 0);

    double* Data = new double[nnz];
    std::fill_n(Data, nnz, 1.);

    return mfem::SparseMatrix(I, J, Data, num_entities, nnz);
}

GraphSpace::GraphSpace(Graph graph)
    : graph_(std::move(graph)),
      vertex_vdof_(SparseIdentity(graph_.NumVertices())),
      edge_edof_(SparseIdentity(graph_.NumEdges())),
      edof_trueedof_(new mfem::HypreParMatrix)
{
    vertex_edof_.MakeRef(graph_.VertexToEdge());
    edof_trueedof_->MakeRef(graph_.EdgeToTrueEdge());
    if (graph_.HasBoundary())
    {
        edof_bdratt_.MakeRef(graph_.EdgeToBdrAtt());
    }
}

GraphSpace::GraphSpace(Graph graph,
                       const std::vector<mfem::DenseMatrix> edge_traces,
                       const std::vector<mfem::DenseMatrix> vertex_targets)
    : graph_(std::move(graph)),
      vertex_vdof_(BuildEntityToDof(vertex_targets)),
      edge_edof_(BuildEntityToDof(edge_traces)),
      vertex_edof_(BuildVertexToEDof()),
      edof_trueedof_(BuildEdofToTrueEdof())
{
    if (graph_.HasBoundary())
    {
        mfem::SparseMatrix edof_edge = smoothg::Transpose(edge_edof_);
        mfem::SparseMatrix tmp = smoothg::Mult(edof_edge, graph_.EdgeToBdrAtt());
        edof_bdratt_.Swap(tmp);
    }
}

GraphSpace::GraphSpace(GraphSpace&& other) noexcept
{
    swap(*this, other);
}

GraphSpace& GraphSpace::operator=(GraphSpace other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(GraphSpace& lhs, GraphSpace& rhs) noexcept
{
    lhs.vertex_vdof_.Swap(rhs.vertex_vdof_);
    lhs.vertex_edof_.Swap(rhs.vertex_edof_);
    lhs.edge_edof_.Swap(rhs.edge_edof_);
    std::swap(lhs.edof_trueedof_, rhs.edof_trueedof_);
    lhs.edof_bdratt_.Swap(rhs.edof_bdratt_);

    swap(lhs.graph_, rhs.graph_);
}

mfem::SparseMatrix GraphSpace::BuildVertexToEDof()
{
    const unsigned int num_vertices = vertex_vdof_.NumRows();
    const mfem::SparseMatrix& vertex_edge = graph_.VertexToEdge();

    int* I = new int[num_vertices + 1]();

    mfem::Array<int> edges;
    for (unsigned int vertex = 0; vertex < num_vertices; vertex++)
    {
        int num_local_edofs = vertex_vdof_.RowSize(vertex) - 1;
        GetTableRow(vertex_edge, vertex, edges);
        for (int& edge : edges)
        {
            num_local_edofs += edge_edof_.RowSize(edge);
        }
        I[vertex + 1] = I[vertex] + num_local_edofs;
    }

    const int nnz = I[num_vertices];
    int* J = new int[nnz];
    double* data = new double[nnz];

    int edof_counter = edge_edof_.NumCols();

    int* J_begin = J;
    double* data_begin = data;

    // data values are chosen for the ease of extended aggregate construction
    for (unsigned int vertex = 0; vertex < num_vertices; vertex++)
    {
        const int num_local_bubbles = vertex_vdof_.RowSize(vertex) - 1;

        int* J_end = J_begin + num_local_bubbles;
        std::iota(J_begin, J_end, edof_counter);
        J_begin = J_end;

        double* data_end = data_begin + num_local_bubbles;
        std::fill(data_begin, data_end, 2.0);
        data_begin = data_end;

        edof_counter += num_local_bubbles;

        GetTableRow(vertex_edge, vertex, edges);
        for (int& edge : edges)
        {
            J_end += edge_edof_.RowSize(edge);
            std::iota(J_begin, J_end, *edge_edof_.GetRowColumns(edge));
            J_begin = J_end;

            data_end += edge_edof_.RowSize(edge);
        }
        std::fill(data_begin, data_end, 1.0);
        data_begin = data_end;
    }

    return mfem::SparseMatrix(I, J, data, num_vertices, edof_counter);
}

unique_ptr<mfem::HypreParMatrix> GraphSpace::BuildEdofToTrueEdof()
{
    const int num_edofs = vertex_edof_.NumCols();
    auto edge_trueedge_edge = AAt(graph_.EdgeToTrueEdge());

    MPI_Comm comm = graph_.GetComm();
    mfem::Array<HYPRE_Int> edof_starts;
    GenerateOffsets(comm, num_edofs, edof_starts);

    unique_ptr<mfem::HypreParMatrix> d_te_d; // dofs sharing the same true edge
    {
        mfem::SparseMatrix edge_edof;
        edge_edof.MakeRef(edge_edof_);
        edge_edof.SetWidth(num_edofs);
        mfem::SparseMatrix edof_edge = smoothg::Transpose(edge_edof);

        auto tmp = ParMult(*edge_trueedge_edge, edge_edof, edof_starts);
        d_te_d = ParMult(edof_edge, *tmp, edof_starts);
    }

    HYPRE_Int* d_te_d_col_map;
    mfem::SparseMatrix d_te_d_offd;
    d_te_d->GetOffd(d_te_d_offd, d_te_d_col_map);

    mfem::SparseMatrix d_td_d_diag = SparseIdentity(num_edofs);
    mfem::SparseMatrix d_td_d_offd(num_edofs, d_te_d_offd.Width());
    {
        mfem::SparseMatrix e_te_e_offd;
        HYPRE_Int* junk_map;
        edge_trueedge_edge->GetOffd(e_te_e_offd, junk_map);

        mfem::Array<int> local_edofs, offd_edofs;
        for (int i = 0; i < e_te_e_offd.NumRows(); i++)
        {
            if (e_te_e_offd.RowSize(i)) // edge is shared by other processors
            {
                GetTableRow(edge_edof_, i, local_edofs);
                GetTableRow(d_te_d_offd, local_edofs[0], offd_edofs);
                assert(local_edofs.Size() == offd_edofs.Size());

                for (int j = 0; j < local_edofs.Size(); ++j)
                {
                    d_td_d_offd.Set(local_edofs[j], offd_edofs[j], 1.0);
                }
            }
        }
        d_td_d_offd.Finalize();
    }

    mfem::HypreParMatrix d_td_d(comm, d_te_d->N(), d_te_d->N(), edof_starts, edof_starts,
                                &d_td_d_diag, &d_td_d_offd, d_te_d_col_map);
    return BuildEntityToTrueEntity(d_td_d);
}

} // namespace smoothg

