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
#include "utilities.hpp"
#include <assert.h>

using std::shared_ptr;
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
    // with orientation of the dofs/elements (may not valid for parallel)
    int tmp_nnz = 0;
    for (int i = 0; i < aggregate_edge.Height(); i++)
    {
        tmp_i[i] = tmp_nnz;
        for (int j = aggregate_edge_i[i]; j < aggregate_edge_i[i + 1]; j++)
            if (aggregate_edge_data[j] == 2)
                tmp_nnz++;
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

// TODO: allow aggregate to be shared by more than one processor
GraphTopology::GraphTopology(
    shared_ptr<mfem::SparseMatrix> vertex_edge,
    shared_ptr<mfem::HypreParMatrix> edge_d_td_,
    const mfem::Array<int>& partition,
    shared_ptr<const mfem::SparseMatrix> edge_boundaryattr)
    : edge_d_td_(edge_d_td_),
      comm_(edge_d_td_->GetComm())
{
    MPI_Comm_size(comm_, &num_procs_);
    MPI_Comm_rank(comm_, &myid_);

    unique_ptr<mfem::HypreParMatrix> edge_td_d( edge_d_td_->Transpose() );

    edge_d_td_d_.reset( ParMult(edge_d_td_.get(), edge_td_d.get()) );

    int nvertices = vertex_edge->Height();
    int nedges = vertex_edge->Width();
    int nAggs = partition.Max() + 1;

    // generate the 'start' array (not true dof)
    mfem::Array<HYPRE_Int>* start[3] = {&vertex_start_, &edge_start_,
                                        &aggregate_start_
                                       };
    HYPRE_Int nloc[3] = {nvertices, nedges, nAggs};
    GenerateOffsets(comm_, 3, nloc, start);

    // Construct the relation table aggregate_vertex from partition
    Agg_vertex_ = PartitionToMatrix(partition, nAggs);

    unique_ptr<mfem::SparseMatrix> aggregate_edge(Mult(*Agg_vertex_,
                                                       *vertex_edge));

    // Need to sort the edge indices to prevent index problem in face_edge
    aggregate_edge->SortColumnIndices();

    Agg_edge_ = make_unique<mfem::SparseMatrix>();
    AggregateEdge2AggregateEdgeInt(*aggregate_edge, *Agg_edge_);
    unique_ptr<mfem::SparseMatrix> edge_aggregate(Transpose(*aggregate_edge));

    // block diagonal edge_aggregate and aggregate_edge
    auto edge_aggregate_d = make_unique<mfem::HypreParMatrix>(
                                comm_, edge_start_.Last(), aggregate_start_.Last(), edge_start_,
                                aggregate_start_, edge_aggregate.get());
    auto aggregate_edge_d = make_unique<mfem::HypreParMatrix>(
                                comm_, aggregate_start_.Last(), edge_start_.Last(),
                                aggregate_start_, edge_start_, aggregate_edge.get());

    unique_ptr<mfem::HypreParMatrix> d_td_d_edge_Agg(
        ParMult(edge_d_td_d_.get(), edge_aggregate_d.get()) );
    unique_ptr<mfem::HypreParMatrix> Agg_Agg(
        ParMult(aggregate_edge_d.get(), d_td_d_edge_Agg.get()) );
    auto Agg_Agg_d = ((hypre_ParCSRMatrix*) *Agg_Agg)->diag;
    auto Agg_Agg_o = ((hypre_ParCSRMatrix*) *Agg_Agg)->offd;

    // nfaces_int = number of faces interior to this processor
    HYPRE_Int nfaces_int = Agg_Agg_d->num_nonzeros - Agg_Agg_d->num_rows;
    assert( nfaces_int % 2 == 0 );
    nfaces_int /= 2;

    // nfaces_bdr = number of global boundary faces in this processor
    int nfaces_bdr = 0;
    unique_ptr<mfem::SparseMatrix> aggregate_boundaryattr;
    if (edge_boundaryattr)
    {
        aggregate_boundaryattr.reset(
            Mult(*aggregate_edge, *edge_boundaryattr) );
        nfaces_bdr = aggregate_boundaryattr->NumNonZeroElems();
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
        for (int j = Agg_Agg_d_i[i]; j < Agg_Agg_d_i[i + 1]; j++)
            if (Agg_Agg_d_j[j] > i)
            {
                face_Agg_j[count * 2] = i;
                face_Agg_j[(count++) * 2 + 1] = Agg_Agg_d_j[j];
                face_Agg_i[count] = count * 2;
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
    std::fill_n(aggregate_edge->GetData(), aggregate_edge->NumNonZeroElems(), 1.);
    mfem::SparseMatrix* intface_Agg_edge = Mult(intface_Agg, *aggregate_edge);

    int* intface_Agg_edge_i = intface_Agg_edge->GetI();
    int* intface_Agg_edge_j = intface_Agg_edge->GetJ();
    double* intface_Agg_edge_data = intface_Agg_edge->GetData();
    for (int i = 0; i < nfaces_int; i++)
    {
        face_edge_i[i] = face_edge_nnz;
        for (int j = intface_Agg_edge_i[i]; j < intface_Agg_edge_i[i + 1]; j++)
            if (intface_Agg_edge_data[j] == 2)
                face_edge_nnz++;
    }

    // Counting the coarse faces on the global boundary
    int* agg_edge_i = aggregate_edge->GetI();
    int* agg_edge_j = aggregate_edge->GetJ();
    if (edge_boundaryattr)
    {
        int* agg_bdr_i = aggregate_boundaryattr->GetI();
        int* agg_bdr_j = aggregate_boundaryattr->GetJ();
        for (int i = 0; i < nAggs; i++)
            for (int j = agg_bdr_i[i]; j < agg_bdr_i[i + 1]; j++)
            {
                face_edge_i[count] = face_edge_nnz;
                for (int k = agg_edge_i[i]; k < agg_edge_i[i + 1]; k++)
                    if (edge_boundaryattr->Elem(agg_edge_j[k], agg_bdr_j[j]))
                        face_edge_nnz++;
                face_Agg_j[nfaces_int + (count++)] = i;
                face_Agg_i[count] = nfaces_int + count;
            }
    }

    // Counting the faces shared between processors
    auto Agg_shareattr_map = ((hypre_ParCSRMatrix*) *Agg_Agg)->col_map_offd;
    auto Agg_Agg_o_j = Agg_Agg_o->j;
    auto edge_shareattr_map =
        ((hypre_ParCSRMatrix*) *d_td_d_edge_Agg)->col_map_offd;
    auto edge_shareattr_i = ((hypre_ParCSRMatrix*) *d_td_d_edge_Agg)->offd->i;
    auto edge_shareattr_j = ((hypre_ParCSRMatrix*) *d_td_d_edge_Agg)->offd->j;

    int sharedattr, edge, edge_shareattr_loc;
    for (int i = 0; i < Agg_Agg_o->num_rows; i++)
        for (int j = Agg_Agg_o_i[i]; j < Agg_Agg_o_i[i + 1]; j++)
        {
            sharedattr = Agg_shareattr_map[Agg_Agg_o_j[j]];
            face_edge_i[count] = face_edge_nnz;
            for (int k = agg_edge_i[i]; k < agg_edge_i[i + 1]; k++)
            {
                edge = agg_edge_j[k];
                if (edge_shareattr_i[edge + 1] > edge_shareattr_i[edge])
                {
                    edge_shareattr_loc =
                        edge_shareattr_j[edge_shareattr_i[edge]];
                    if (edge_shareattr_map[edge_shareattr_loc] == sharedattr)
                        face_edge_nnz++;
                }
            }
            face_Agg_j[nfaces_int + (count++)] = i;
            face_Agg_i[count] = nfaces_int + count;
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
    delete intface_Agg_edge;

    // Insert edges to the coarse faces on the global boundary
    if (edge_boundaryattr)
    {
        int* agg_bdr_i = aggregate_boundaryattr->GetI();
        int* agg_bdr_j = aggregate_boundaryattr->GetJ();
        for (int i = 0; i < nAggs; i++)
            for (int j = agg_bdr_i[i]; j < agg_bdr_i[i + 1]; j++)
                for (int k = agg_edge_i[i]; k < agg_edge_i[i + 1]; k++)
                    if (edge_boundaryattr->Elem(agg_edge_j[k], agg_bdr_j[j]))
                        face_edge_j[face_edge_nnz++] = agg_edge_j[k];
    }

    // Insert edges to the faces shared between processors
    for (int i = 0; i < Agg_Agg_o->num_rows; i++)
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
    double* face_edge_data = new double [face_edge_nnz];
    std::fill_n(face_edge_data, face_edge_nnz, 1.0);
    face_edge_ = make_unique<mfem::SparseMatrix>(face_edge_i, face_edge_j,
                                                 face_edge_data, nfaces, nedges);

    // TODO: face_bdratt can be built when counting boundary faces
    if (edge_boundaryattr)
        face_bdratt_.reset( Mult(*face_edge_, *edge_boundaryattr) );

    // Complete face to aggregate table
    face_Agg_ = make_unique<mfem::SparseMatrix>(face_Agg_i, face_Agg_j,
                                                face_Agg_data, nfaces, nAggs);
    Agg_face_.reset( Transpose(*face_Agg_) );

    // Build face "dof-true dof-dof" table from local face_edge and
    // the edge "dof-true dof-dof" table
    start[0] = &face_start_;
    GenerateOffsets(comm_, 1, &nfaces, start);

    unique_ptr<mfem::SparseMatrix> edge_face(Transpose(*face_edge_));

    // block diagonal edge_face
    auto edge_face_d = make_unique<mfem::HypreParMatrix>(
                           comm_, edge_start_.Last(), face_start_.Last(),
                           edge_start_, face_start_, edge_face.get());

    face_d_td_d_.reset( RAP(edge_d_td_d_.get(), edge_face_d.get()) );
    SetConstantValue(face_d_td_d_.get(), 1.0);

    hypre_ParCSRMatrix* face_shared = *face_d_td_d_;
    HYPRE_Int* face_shared_i = face_shared->offd->i;
    HYPRE_Int* face_shared_j = face_shared->offd->j;
    HYPRE_Int* face_shared_map = face_shared->col_map_offd;
    HYPRE_Int maxface = face_shared->last_row_index;

    // Create a selection matrix to pick one of the processors sharing a true
    // face to own the true face (we pick the processor with a smaller index)
    int* select_i = new int[nfaces + 1];
    int ntruefaces = 0;
    for (int i = 0; i < nfaces; i++)
    {
        select_i[i] = ntruefaces;
        if (face_shared_i[i + 1] == face_shared_i[i])
            ntruefaces++;
        else if (face_shared_map[face_shared_j[face_shared_i[i]]] > maxface)
            ntruefaces++;
    }
    select_i[nfaces] = ntruefaces;
    int* select_j = new int[ntruefaces];
    double* select_data = new double[ntruefaces];
    std::iota(select_j, select_j + ntruefaces, 0);
    std::fill_n(select_data, ntruefaces, 1.);
    mfem::SparseMatrix select(select_i, select_j, select_data,
                              nfaces, ntruefaces);

    // Construct a (block diagonal) global select matrix from local
    start[0] = &trueface_start_;
    GenerateOffsets(comm_, 1, &ntruefaces, start);
    mfem::HypreParMatrix select_d(comm_, face_shared->global_num_rows,
                                  trueface_start_.Last(), face_shared->row_starts,
                                  trueface_start_, &select);

    // Construct face "dof to true dof" table
    face_d_td_.reset( ParMult(face_d_td_d_.get(), &select_d) );

    // Construct extended aggregate to vertex relation tables
    auto vertex_edge_d = make_unique<mfem::HypreParMatrix>(
                             comm_, vertex_start_.Last(), edge_start_.Last(),
                             vertex_start_, edge_start_, vertex_edge.get());
    unique_ptr<mfem::HypreParMatrix> pvertex_edge(
        ParMult(vertex_edge_d.get(), edge_d_td_.get()) );
    unique_ptr<mfem::HypreParMatrix> pedge_vertex( pvertex_edge->Transpose() );

    auto Agg_edge_d = make_unique<mfem::HypreParMatrix>(
                          comm_, aggregate_start_.Last(), edge_start_.Last(),
                          aggregate_start_, edge_start_, aggregate_edge.get());
    unique_ptr<mfem::HypreParMatrix> pAgg_edge(
        ParMult(Agg_edge_d.get(), edge_d_td_.get()) );
    pAggExt_vertex_.reset( ParMult(pAgg_edge.get(), pedge_vertex.get()) );

    // Construct extended aggregate to (interior) edge relation tables
    {
        SetConstantValue(pAggExt_vertex_.get(), 1.);
        unique_ptr<mfem::HypreParMatrix>pAggExt_edge_tmp(
            ParMult(pAggExt_vertex_.get(), pvertex_edge.get()) );

        // Remove the edges on the boundary of extended aggregates (diag part)
        HYPRE_Int* offd_map_tmp;
        mfem::SparseMatrix diag, diag_tmp, offd_tmp;
        pAggExt_edge_tmp->GetDiag(diag_tmp);
        AggregateEdge2AggregateEdgeInt(diag_tmp, diag);

        // Remove the edges on the boundary of extended aggregates (offd part)
        pAggExt_edge_tmp->GetOffd(offd_tmp, offd_map_tmp);
        int* offd_tmp_i = offd_tmp.GetI();
        int* offd_tmp_j = offd_tmp.GetJ();
        double* offd_tmp_data = offd_tmp.GetData();

        int* offd_i = new int[offd_tmp.Height() + 1];
        int offd_nnz(0), offd_ncol(0);
        mfem::Array<int> offd_marker(offd_tmp.Width());
        offd_marker = 0;
        for (int i = 0; i < offd_tmp.Height(); i++)
        {
            offd_i[i] = offd_nnz;
            for (int j = offd_tmp_i[i]; j < offd_tmp_i[i + 1]; j++)
                if (offd_tmp_data[j] == 2)
                {
                    offd_marker[offd_tmp_j[j]]++;
                    offd_nnz++;
                }
        }
        offd_i[offd_tmp.Height()] = offd_nnz;

        // After removing the edges on the boundary of extended aggregates, the
        // number of columns of the offd matrix, as well as the offd_col_map
        // need to be adjusted accordingly
        for (int i = 0; i < offd_marker.Size(); i++)
            if (offd_marker[i])
                offd_ncol++;

        HYPRE_Int* offd_map = new HYPRE_Int[offd_ncol];
        offd_ncol = 0;
        for (int i = 0; i < offd_marker.Size(); i++)
            if (offd_marker[i])
            {
                offd_marker[i] = offd_ncol;
                offd_map[offd_ncol++] = offd_map_tmp[i];
            }

        int* offd_j = new int[offd_nnz];
        offd_nnz = 0;
        for (int i = 0; i < offd_tmp.Height(); i++)
            for (int j = offd_tmp_i[i]; j < offd_tmp_i[i + 1]; j++)
                if (offd_tmp_data[j] == 2)
                    offd_j[offd_nnz++] = offd_marker[offd_tmp_j[j]];

        double* offd_data = new double[offd_nnz];
        std::fill_n(offd_data, offd_nnz, 1.);

        int* diag_i = diag.GetI();
        int* diag_j = diag.GetJ();
        double* diag_data = diag.GetData();

        pAggExt_edge_ = make_unique<mfem::HypreParMatrix>(
                            comm_, aggregate_start_.Last(), edge_d_td_->GetGlobalNumCols(),
                            aggregate_start_, edge_d_td_->ColPart(), diag_i, diag_j,
                            diag_data, offd_i, offd_j, offd_data, offd_ncol, offd_map);

        // set diag not to own i, j, data arrays (pAggExt_edge owns them)
        diag.LoseData();
    }
}

GraphTopology::GraphTopology(const mfem::SparseMatrix& face_edge,
                             const mfem::SparseMatrix& Agg_vertex,
                             const mfem::SparseMatrix& Agg_edge,
                             const mfem::HypreParMatrix& pAggExt_vertex,
                             const mfem::HypreParMatrix& pAggExt_edge,
                             const mfem::SparseMatrix& Agg_face,
                             const mfem::HypreParMatrix& edge_d_td,
                             const mfem::HypreParMatrix& face_d_td,
                             const mfem::HypreParMatrix& face_d_td_d)
{
    face_edge_ = make_unique<mfem::SparseMatrix>(face_edge);
    Agg_vertex_ = make_unique<mfem::SparseMatrix>(Agg_vertex);
    Agg_edge_ = make_unique<mfem::SparseMatrix>(Agg_edge);
    pAggExt_vertex_ = make_unique<mfem::HypreParMatrix>();
    pAggExt_vertex_->MakeRef(pAggExt_vertex);
    pAggExt_edge_ = make_unique<mfem::HypreParMatrix>();
    pAggExt_edge_->MakeRef(pAggExt_edge);
    Agg_face_ = make_unique<mfem::SparseMatrix>(Agg_face);
    edge_d_td_ = make_unique<mfem::HypreParMatrix>();
    edge_d_td_->MakeRef(edge_d_td);
    face_d_td_ = make_unique<mfem::HypreParMatrix>();
    face_d_td_->MakeRef(face_d_td);
    face_d_td_d_ = make_unique<mfem::HypreParMatrix>();
    face_d_td_d_->MakeRef(face_d_td_d);

    int start_size = 3;
    if (!HYPRE_AssumedPartitionCheck())
    {
        MPI_Comm_size(pAggExt_vertex_->GetComm(), &start_size);
        start_size++;
    }

    vertex_start_.SetSize(start_size);
    edge_start_.SetSize(start_size);
    face_start_.SetSize(start_size);
    for (int i = 0; i < start_size; i++)
    {
        vertex_start_[i] = pAggExt_vertex_->ColPart()[i];
        edge_start_[i] = pAggExt_edge_->ColPart()[i];
        face_start_[i] = face_d_td_d_->ColPart()[i];
    }
    face_Agg_.reset( Transpose(*Agg_face_) );
}

} // namespace smoothg
