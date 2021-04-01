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

Graph GraphTopology::Coarsen(const Graph& fine_graph, int coarsening_factor, int num_iso_verts)
{
    mfem::Array<int> partitioning;
    mfem::SparseMatrix vert_edge(fine_graph.VertexToEdge(), false);
    vert_edge = 1.0;

    std::vector<std::vector<int>> iso_verts;
    iso_verts.reserve(num_iso_verts * 20);

    const mfem::SparseMatrix& edge_vert = fine_graph.EdgeToVertex();
    mfem::SparseMatrix vert_vert = smoothg::Mult(vert_edge, edge_vert);
    mfem::SparseMatrix vert_vert2 = smoothg::Mult(vert_vert, vert_vert);
    mfem::SparseMatrix vert_vert3 = smoothg::Mult(vert_vert, vert_vert2);

    const bool isolate_injection_well_cells = true;
    if (isolate_injection_well_cells)
    {
//        mfem::SparseMatrix edge_vert = smoothg::Transpose(vert_edge);

        for (int i = vert_edge.NumRows() - num_iso_verts; i < vert_edge.NumRows(); ++i)
        {
//            assert(vert_vert.RowSize(i) == 2);
//            int i_friend = vert_vert.GetRowColumns(i)[0];
//            i_friend = (i_friend == i) ? vert_vert.GetRowColumns(i)[1] : i_friend;
//            iso_verts.push_back(std::vector<int>(1, i_friend));

            for (int j = 0; j < vert_vert.RowSize(i); ++j)
            {
                const int i_friend = vert_vert.GetRowColumns(i)[j];
                if (i_friend != i) { iso_verts.push_back(std::vector<int>(1, i_friend)); }
            }
        }
    }

    const bool isolate_production_well_cells = true;
    if (isolate_production_well_cells && fine_graph.EdgeToBdrAtt().NumCols() > 1)
    {
        int num_well_edges = (fine_graph.EdgeToBdrAtt().NumCols()-1) + num_iso_verts;
        for (int i = edge_vert.NumRows() - num_well_edges; i < edge_vert.NumRows() - num_iso_verts; ++i)
        {
            assert(edge_vert.RowSize(i) == 1);
            int production_well_cell = edge_vert.GetRowColumns(i)[0];
            iso_verts.push_back(std::vector<int>(1, production_well_cell));

//            for (int j = 0; j < vert_vert.RowSize(production_well_cell); ++j)
//            {
//                const int i_friend = vert_vert.GetRowColumns(production_well_cell)[j];
//                if (i_friend != production_well_cell) { iso_verts.push_back(std::vector<int>(1, i_friend)); }
//            }
        }
    }

    for (int i = vert_edge.NumRows() - num_iso_verts; i < vert_edge.NumRows(); ++i)
    {
        iso_verts.push_back(std::vector<int>(1, i));
    }

    PartitionAAT(vert_edge, partitioning, coarsening_factor, false, std::move(iso_verts));
    return Coarsen(fine_graph, partitioning);
}

Graph GraphTopology::Coarsen(const Graph& fine_graph, const mfem::Array<int>& partitioning)
{
    MPI_Comm comm = fine_graph.GetComm();

    const mfem::SparseMatrix& edge_bdratt = fine_graph.EdgeToBdrAtt();
    const auto& edge_trueedge_edge = fine_graph.EdgeToTrueEdgeToEdge();

    // generate the 'start' array
    int nAggs = partitioning.Max() + 1;

    mfem::Array<HYPRE_Int> agg_starts;
    GenerateOffsets(comm, nAggs, agg_starts);

    // Construct the relation table aggregate_vertex from partition
    mfem::SparseMatrix tmp = PartitionToMatrix(partitioning, nAggs);
    Agg_vertex_.Swap(tmp);

    auto aggregate_edge = smoothg::Mult(Agg_vertex_, fine_graph.VertexToEdge());

    // Need to sort the edge indices to prevent index problem in face_edge
    aggregate_edge.SortColumnIndices();

    mfem::SparseMatrix edge_agg(smoothg::Transpose(aggregate_edge));

    auto edge_trueedge_Agg = ParMult(edge_trueedge_edge, edge_agg, agg_starts);
    auto Agg_Agg = ParMult(aggregate_edge, *edge_trueedge_Agg, agg_starts);

    auto Agg_Agg_d = ((hypre_ParCSRMatrix*) *Agg_Agg)->diag;
    auto Agg_Agg_o = ((hypre_ParCSRMatrix*) *Agg_Agg)->offd;

    // nfaces_int = number of faces interior to this processor
    HYPRE_Int nfaces_int = Agg_Agg_d->num_nonzeros - Agg_Agg_d->num_rows;
    assert( nfaces_int % 2 == 0 );
    nfaces_int /= 2;

    // nfaces_bdr = number of global boundary faces in this processor
    int nfaces_bdr = 0;
    mfem::SparseMatrix aggregate_boundaryattr;
    if (fine_graph.HasBoundary())
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
            if (intface_Agg_edge_data[j] == 2.0)
                face_edge_nnz++;
    }

    // Counting the coarse faces on the global boundary
    int* agg_edge_i = aggregate_edge.GetI();
    int* agg_edge_j = aggregate_edge.GetJ();
    if (fine_graph.HasBoundary())
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
            if (intface_Agg_edge_data[j] == 2.0)
                face_edge_j[face_edge_nnz++] = intface_Agg_edge_j[j];

    // Insert edges to the coarse faces on the global boundary
    if (fine_graph.HasBoundary())
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

    mfem::SparseMatrix tmp_face_edge(face_edge_i, face_edge_j, face_edge_data,
                                     nfaces, fine_graph.NumEdges());

    // Construct "face to true face" table
    mfem::Array<int> face_starts;
    GenerateOffsets(comm, nfaces, face_starts);
    mfem::SparseMatrix edge_face = smoothg::Transpose(tmp_face_edge);
    auto e_te_f = ParMult(edge_trueedge_edge, edge_face, face_starts);
    auto face_trueface_face = ParMult(tmp_face_edge, *e_te_f, face_starts);
    auto tmp_face_trueface = BuildEntityToTrueEntity(*face_trueface_face);
    *tmp_face_trueface = 1.0;

    // Reorder shared faces so that their "true face" numbering is increasing
    auto face_reorder_map = EntityReorderMap(*tmp_face_trueface, *face_trueface_face);

    auto face_trueface = ParMult(face_reorder_map, *tmp_face_trueface, face_starts);
    face_trueface->CopyRowStarts();
    face_trueface->CopyColStarts();

    mfem::SparseMatrix tmp_face_Agg(face_Agg_i, face_Agg_j, face_Agg_data, nfaces, nAggs);
    auto face_Agg = smoothg::Mult(face_reorder_map, tmp_face_Agg);
    face_Agg.SortColumnIndices();

    auto reordered_face_edge = smoothg::Mult(face_reorder_map, tmp_face_edge);
    face_edge_.Swap(reordered_face_edge);

    std::unique_ptr<mfem::SparseMatrix> face_bdratt;
    if (fine_graph.HasBoundary())
    {
        face_bdratt.reset(mfem::Mult(face_edge_, edge_bdratt));
    }

    return Graph(std::move(face_Agg), std::move(face_trueface),
                 agg_starts, face_starts, face_bdratt.get());
}

} // namespace smoothg
