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

/**
   @file

   @brief Implements GraphTopology object.
*/

#include "GraphTopology.hpp"
#include "MatrixUtilities.hpp"
#include "MetisGraphPartitioner.hpp"
#include "utilities.hpp"
#include <assert.h>

using std::unique_ptr;

namespace smoothg
{

void GraphTopology::AggregateEdge2AggregateEdgeInt(
    const mfem::SparseMatrix& aggregate_edge,
    mfem::SparseMatrix& aggregate_edge_int)
{
    int* aggregate_edge_i = aggregate_edge.GetI();
    int* aggregate_edge_j = aggregate_edge.GetJ();
    double* aggregate_edge_data = aggregate_edge.GetData();

    int* tmp_i = new int [aggregate_edge.Height() + 1];

    // this removal of entries that have data value != 2 has something to do
    // with orientation of the dofs/elements, as a result of
    // smoothg::Mult(Agg_vertex_, vertex_edge) (or a parallel variant)
    int tmp_nnz = 0;
    for (int i = 0; i < aggregate_edge.Height(); i++)
    {
        tmp_i[i] = tmp_nnz;
        for (int j = aggregate_edge_i[i]; j < aggregate_edge_i[i + 1]; j++)
        {
            if (aggregate_edge_data[j] == 2)
                tmp_nnz++;
        }
    }
    tmp_i[aggregate_edge.Height()] = tmp_nnz;

    int* tmp_j = new int[tmp_nnz];
    tmp_nnz = 0;
    for (int i = 0; i < aggregate_edge.Height(); i++)
        for (int j = aggregate_edge_i[i]; j < aggregate_edge_i[i + 1]; j++)
            if (aggregate_edge_data[j] == 2)
                tmp_j[tmp_nnz++] = aggregate_edge_j[j];

    double* tmp_data = new double[tmp_nnz];
    std::fill_n(tmp_data, tmp_nnz, 1.0);
    mfem::SparseMatrix tmp(tmp_i, tmp_j, tmp_data, aggregate_edge.Height(),
                           aggregate_edge.Width());

    aggregate_edge_int.Swap(tmp);
}

GraphTopology::GraphTopology(const Graph& graph)
    : fine_graph_(&graph), edge_trueedge_edge_(nullptr)
{
}

GraphTopology::GraphTopology(GraphTopology&& graph_topology) noexcept
{
    face_trueface_face_ = std::move(graph_topology.face_trueface_face_);

    Agg_edge_.Swap(graph_topology.Agg_edge_);
    Agg_vertex_.Swap(graph_topology.Agg_vertex_);
    face_edge_.Swap(graph_topology.face_edge_);

    Swap(vertex_start_, graph_topology.GetVertexStart());
    Swap(edge_start_, graph_topology.GetEdgeStart());
    Swap(aggregate_start_, graph_topology.GetAggregateStart());
    Swap(face_start_, graph_topology.GetFaceStart());

    std::swap(fine_graph_, graph_topology.fine_graph_);

    edge_trueedge_edge_ = graph_topology.edge_trueedge_edge_;
}

std::shared_ptr<Graph> GraphTopology::Coarsen(int coarsening_factor)
{
    mfem::Array<int> partitioning;
    PartitionAAT(fine_graph_->VertexToEdge(), partitioning, coarsening_factor);
    return Coarsen(partitioning);
}

std::shared_ptr<Graph> GraphTopology::Coarsen(const mfem::Array<int>& partitioning)
{
    MPI_Comm comm = fine_graph_->GetComm();

    const mfem::SparseMatrix& edge_bdratt = fine_graph_->EdgeToBdrAtt();

    unique_ptr<mfem::HypreParMatrix> edge_trueedge_edge;
    if (edge_trueedge_edge_)
    {
        edge_trueedge_edge = make_unique<mfem::HypreParMatrix>();
        edge_trueedge_edge->MakeRef(*edge_trueedge_edge_);
    }
    else
    {
        edge_trueedge_edge = AAt(fine_graph_->EdgeToTrueEdge());
    }

    int nvertices = fine_graph_->NumVertices();
    int nedges = fine_graph_->NumEdges();
    int nAggs = partitioning.Max() + 1;

    // generate the 'start' array (not true dof)
    mfem::Array<HYPRE_Int>* start[3] = {&vertex_start_, &edge_start_,
                                        &aggregate_start_
                                       };
    HYPRE_Int nloc[3] = {nvertices, nedges, nAggs};
    GenerateOffsets(comm, 3, nloc, start);

    // Construct the relation table aggregate_vertex from partition
    mfem::SparseMatrix tmp = PartitionToMatrix(partitioning, nAggs);
    Agg_vertex_.Swap(tmp);

    auto aggregate_edge = smoothg::Mult(Agg_vertex_, fine_graph_->VertexToEdge());

    // Need to sort the edge indices to prevent index problem in face_edge
    aggregate_edge.SortColumnIndices();

    AggregateEdge2AggregateEdgeInt(aggregate_edge, Agg_edge_);
    mfem::SparseMatrix edge_aggregate(smoothg::Transpose(aggregate_edge));

    // block diagonal edge_aggregate and aggregate_edge
    auto edge_aggregate_d = make_unique<mfem::HypreParMatrix>(
                                comm, edge_start_.Last(), aggregate_start_.Last(),
                                edge_start_, aggregate_start_, &edge_aggregate);
    auto aggregate_edge_d = make_unique<mfem::HypreParMatrix>(
                                comm, aggregate_start_.Last(), edge_start_.Last(),
                                aggregate_start_, edge_start_, &aggregate_edge);

    unique_ptr<mfem::HypreParMatrix> edge_trueedge_Agg(
        ParMult(edge_trueedge_edge.get(), edge_aggregate_d.get()) );
    unique_ptr<mfem::HypreParMatrix> Agg_Agg(
        ParMult(aggregate_edge_d.get(), edge_trueedge_Agg.get()) );

    auto Agg_Agg_d = ((hypre_ParCSRMatrix*) *Agg_Agg)->diag;
    auto Agg_Agg_o = ((hypre_ParCSRMatrix*) *Agg_Agg)->offd;

    // nfaces_int = number of faces interior to this processor
    HYPRE_Int nfaces_int = Agg_Agg_d->num_nonzeros - Agg_Agg_d->num_rows;
    assert( nfaces_int % 2 == 0 );
    nfaces_int /= 2;

    // nfaces_bdr = number of global boundary faces in this processor
    int nfaces_bdr = 0;
    mfem::SparseMatrix aggregate_boundaryattr;
    if (fine_graph_->HasBoundary())
    {
        auto tmp = smoothg::Mult(aggregate_edge, edge_bdratt);
        aggregate_boundaryattr.Swap(tmp);

        nfaces_bdr = aggregate_boundaryattr.NumNonZeroElems();
    }

    // nfaces = number of all coarse faces (interior + shared + boundary)
    HYPRE_Int nfaces = nfaces_int + nfaces_bdr + Agg_Agg_o->num_nonzeros;

    HYPRE_Int* Agg_Agg_d_i = Agg_Agg_d->i;
    HYPRE_Int* Agg_Agg_d_j = Agg_Agg_d->j;
    HYPRE_Int* Agg_Agg_o_i = Agg_Agg_o->i;

    int* face_Agg_i = new int[nfaces + 1];
    int* face_Agg_j = new int[nfaces_int + nfaces];
    double* face_Agg_data = new double[nfaces_int + nfaces];
    std::fill_n(face_Agg_data, nfaces_int + nfaces, 1.);

    face_Agg_i[0] = 0;
    int count = 0;
    for (int i = 0; i < Agg_Agg_d->num_rows - 1; i++)
    {
        for (int j = Agg_Agg_d_i[i]; j < Agg_Agg_d_i[i + 1]; j++)
        {
            if (Agg_Agg_d_j[j] > i)
            {
                face_Agg_j[count * 2] = i;
                face_Agg_j[(count++) * 2 + 1] = Agg_Agg_d_j[j];
                face_Agg_i[count] = count * 2;
            }
        }
    }
    assert(count == nfaces_int);

    // Interior face to aggregate table, to be used to construct face_edge
    mfem::SparseMatrix intface_Agg(face_Agg_i, face_Agg_j, face_Agg_data,
                                   nfaces_int, nAggs, false, false, false);

    // Start to construct face to edge table
    int* face_edge_i = new int[nfaces + 1];
    int face_edge_nnz = 0;

    // Set the entries of aggregate_edge to be 1 so that an edge belonging
    // to an interior face has a entry 2 in face_Agg_edge
    std::fill_n(aggregate_edge.GetData(), aggregate_edge.NumNonZeroElems(), 1.);
    mfem::SparseMatrix intface_Agg_edge = smoothg::Mult(intface_Agg, aggregate_edge);

    int* intface_Agg_edge_i = intface_Agg_edge.GetI();
    int* intface_Agg_edge_j = intface_Agg_edge.GetJ();
    double* intface_Agg_edge_data = intface_Agg_edge.GetData();
    for (int i = 0; i < nfaces_int; i++)
    {
        face_edge_i[i] = face_edge_nnz;
        for (int j = intface_Agg_edge_i[i]; j < intface_Agg_edge_i[i + 1]; j++)
            if (intface_Agg_edge_data[j] == 2)
                face_edge_nnz++;
    }

    // Counting the coarse faces on the global boundary
    int* agg_edge_i = aggregate_edge.GetI();
    int* agg_edge_j = aggregate_edge.GetJ();
    if (fine_graph_->HasBoundary())
    {
        int* agg_bdr_i = aggregate_boundaryattr.GetI();
        int* agg_bdr_j = aggregate_boundaryattr.GetJ();
        for (int i = 0; i < nAggs; i++)
            for (int j = agg_bdr_i[i]; j < agg_bdr_i[i + 1]; j++)
            {
                face_edge_i[count] = face_edge_nnz;
                for (int k = agg_edge_i[i]; k < agg_edge_i[i + 1]; k++)
                    if (edge_bdratt.Elem(agg_edge_j[k], agg_bdr_j[j]))
                        face_edge_nnz++;
                face_Agg_j[nfaces_int + (count++)] = i;
                face_Agg_i[count] = nfaces_int + count;
            }
    }

    // Counting the faces shared between processors
    auto Agg_shareattr_map = ((hypre_ParCSRMatrix*) *Agg_Agg)->col_map_offd;
    auto Agg_Agg_o_j = Agg_Agg_o->j;
    auto edge_shareattr_map = ((hypre_ParCSRMatrix*) *edge_trueedge_Agg)->col_map_offd;
    auto edge_shareattr_i = ((hypre_ParCSRMatrix*) *edge_trueedge_Agg)->offd->i;
    auto edge_shareattr_j = ((hypre_ParCSRMatrix*) *edge_trueedge_Agg)->offd->j;

    int sharedattr, edge, edge_shareattr_loc;
    for (int i = 0; i < Agg_Agg_o->num_rows; i++)
    {
        for (int j = Agg_Agg_o_i[i]; j < Agg_Agg_o_i[i + 1]; j++)
        {
            sharedattr = Agg_shareattr_map[Agg_Agg_o_j[j]];
            face_edge_i[count] = face_edge_nnz;
            for (int k = agg_edge_i[i]; k < agg_edge_i[i + 1]; k++)
            {
                edge = agg_edge_j[k];
                if (edge_shareattr_i[edge + 1] > edge_shareattr_i[edge])
                {
                    edge_shareattr_loc = edge_shareattr_j[edge_shareattr_i[edge]];
                    if (edge_shareattr_map[edge_shareattr_loc] == sharedattr)
                        face_edge_nnz++;
                }
            }
            face_Agg_j[nfaces_int + (count++)] = i;
            face_Agg_i[count] = nfaces_int + count;
        }
    }
    face_edge_i[nfaces] = face_edge_nnz;
    assert(count == nfaces);

    int* face_edge_j = new int [face_edge_nnz];
    face_edge_nnz = 0;

    // Insert edges to the interior coarse faces
    for (int i = 0; i < nfaces_int; i++)
        for (int j = intface_Agg_edge_i[i]; j < intface_Agg_edge_i[i + 1]; j++)
            if (intface_Agg_edge_data[j] == 2)
                face_edge_j[face_edge_nnz++] = intface_Agg_edge_j[j];

    // Insert edges to the coarse faces on the global boundary
    if (fine_graph_->HasBoundary())
    {
        int* agg_bdr_i = aggregate_boundaryattr.GetI();
        int* agg_bdr_j = aggregate_boundaryattr.GetJ();
        for (int i = 0; i < nAggs; i++)
            for (int j = agg_bdr_i[i]; j < agg_bdr_i[i + 1]; j++)
                for (int k = agg_edge_i[i]; k < agg_edge_i[i + 1]; k++)
                    if (edge_bdratt.Elem(agg_edge_j[k], agg_bdr_j[j]))
                        face_edge_j[face_edge_nnz++] = agg_edge_j[k];
    }

    // Insert edges to the faces shared between processors
    for (int i = 0; i < Agg_Agg_o->num_rows; i++)
    {
        for (int j = Agg_Agg_o_i[i]; j < Agg_Agg_o_i[i + 1]; j++)
        {
            sharedattr = Agg_shareattr_map[Agg_Agg_o_j[j]];
            for (int k = agg_edge_i[i]; k < agg_edge_i[i + 1]; k++)
            {
                edge = agg_edge_j[k];
                if (edge_shareattr_i[edge + 1] > edge_shareattr_i[edge])
                {
                    edge_shareattr_loc =
                        edge_shareattr_j[edge_shareattr_i[edge]];
                    if (edge_shareattr_map[edge_shareattr_loc] == sharedattr)
                        face_edge_j[face_edge_nnz++] = edge;
                }
            }
        }
    }
    double* face_edge_data = new double [face_edge_nnz];
    std::fill_n(face_edge_data, face_edge_nnz, 1.0);
    mfem::SparseMatrix face_edge_tmp(face_edge_i, face_edge_j, face_edge_data,
                                     nfaces, nedges);
    face_edge_.Swap(face_edge_tmp);

    // TODO: face_bdratt can be built when counting boundary faces
    std::unique_ptr<mfem::SparseMatrix> face_bdratt;
    if (fine_graph_->HasBoundary())
    {
        face_bdratt.reset(mfem::Mult(face_edge_, edge_bdratt));
    }

    // Complete face to aggregate table
    mfem::SparseMatrix f_A(face_Agg_i, face_Agg_j, face_Agg_data, nfaces, nAggs);
    face_Agg_.Swap(f_A);
    mfem::SparseMatrix Agg_face = smoothg::Transpose(face_Agg_);

    // Build face "dof-true dof-dof" table from local face_edge and
    // the edge "dof-true dof-dof" table
    GenerateOffsets(comm, nfaces, face_start_);

    mfem::SparseMatrix edge_face(smoothg::Transpose(face_edge_));

    // block diagonal edge_face
    mfem::HypreParMatrix edge_face_d(comm, edge_start_.Last(), face_start_.Last(),
                                     edge_start_, face_start_, &edge_face);

    assert(edge_trueedge_edge && edge_face_d);
    face_trueface_face_.reset(smoothg::RAP(*edge_trueedge_edge, edge_face_d));
    assert(face_trueface_face_);
    SetConstantValue(*face_trueface_face_, 1.0);

    // Construct "face to true face" table
    auto face_trueface = BuildEntityToTrueEntity(*face_trueface_face_);

    coarse_graph_ = make_unique<Graph>(Agg_face, *face_trueface, mfem::Vector(),
                                       face_bdratt.get());
    return coarse_graph_;
}

} // namespace smoothg
