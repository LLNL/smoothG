/*BHEADER**********************************************************************
 *
 * Copyright (c) 2017,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * LLNL-CODE-XXXXXX. All Rights reserved.
 *
 * This file is part of smoothG.  See file COPYRIGHT for details.
 * For more information and source code availability see XXXXX.
 *
 * smoothG is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 ***********************************************************************EHEADER*/

/** @file

    @brief Implements Graph object.
*/

#include "Graph.hpp"
#include "MetisGraphPartitioner.hpp"

#if SMOOTHG_USE_PARMETIS
#include "ParMetisGraphPartitioner.hpp"
#endif

using std::shared_ptr;
using std::unique_ptr;

namespace smoothg
{

Graph::Graph(MPI_Comm comm,
             const mfem::SparseMatrix& vertex_edge_global,
             const mfem::Array<int>& partition_global)
    : comm_(comm)
{
    MPI_Comm_size(comm_, &num_procs_);
    MPI_Comm_rank(comm_, &myid_);

    Distribute(vertex_edge_global, partition_global);
}

Graph::Graph(MPI_Comm comm,
             const mfem::SparseMatrix& vertex_edge_global,
             const int coarsening_factor,
             const bool do_parmetis_partition)
    : comm_(comm)
{
    MPI_Comm_size(comm_, &num_procs_);
    MPI_Comm_rank(comm_, &myid_);

    int num_vertices_global = vertex_edge_global.Height();
    int num_parts_global = (num_vertices_global / (double)(coarsening_factor)) + 0.5;
    num_parts_global = std::max(1, num_parts_global);

    mfem::Array<int> partition_global;
    if (do_parmetis_partition)
    {
#if SMOOTHG_USE_PARMETIS
        PartitionByIndex(num_vertices_global, num_procs_, partition_global);
        Distribute(vertex_edge_global, partition_global);

        mfem::Array<int> partition_distributed;
        smoothg::ParMetisGraphPartitioner parallel_partitioner;
        parallel_partitioner.doPartition(
            *this, num_parts_global, partition_distributed);
        Redistribute(partition_distributed);
#else
        std::cout << "ParMetis needs to enabled!\n";
        std::abort();
#endif
    }
    else
    {
        auto edge_vert = smoothg::Transpose(vertex_edge_global);
        auto vert_vert = smoothg::Mult(vertex_edge_global, edge_vert);

        // TODO(gelever1) : should processor 0 partition and distribute or assume all processors will
        // obtain the same global partition from metis?
        smoothg::MetisGraphPartitioner partitioner;
        partitioner.setUnbalanceTol(2);
        partitioner.doPartition(vert_vert, num_parts_global, partition_global);
        Distribute(vertex_edge_global, partition_global);
    }
}

void Graph::Distribute(const mfem::SparseMatrix& vertex_edge_global,
                       const mfem::Array<int>& partition_global)
{
    MFEM_VERIFY(HYPRE_AssumedPartitionCheck(),
                "this method can not be used without assumed partition");

    // Get the number of local aggregates by dividing the total by num_procs
    int nAggs_global = partition_global.Size() ? partition_global.Max() + 1 : 0;
    int nAggs_local = nAggs_global / num_procs_;
    int nAgg_leftover = nAggs_global % num_procs_;

    // Construct the relation table aggregate_vertex from global partition
    auto Agg_vert = PartitionToMatrix(partition_global, nAggs_global);

    // Construct the relation table proc_aggregate
    int* proc_Agg_i = new int[num_procs_ + 1];
    int* proc_Agg_j = new int[nAggs_global];
    double* proc_Agg_data = new double[nAggs_global];
    std::fill_n(proc_Agg_data, nAggs_global, 1.);
    std::iota(proc_Agg_j, proc_Agg_j + nAggs_global, 0);

    // For proc id < nAgg_leftover, nAggs_local have one more (from leftover)
    nAggs_local++;
    for (int id = 0; id <= nAgg_leftover; id++)
        proc_Agg_i[id] = id * nAggs_local;
    nAggs_local--;
    for (int id = nAgg_leftover + 1; id <= num_procs_; id++)
        proc_Agg_i[id] = proc_Agg_i[id - 1] + nAggs_local;
    mfem::SparseMatrix proc_Agg(proc_Agg_i, proc_Agg_j, proc_Agg_data,
                                num_procs_, nAggs_global);

    // Compute edge_proc relation (for constructing edge to true edge later)
    mfem::SparseMatrix proc_vert = smoothg::Mult(proc_Agg, Agg_vert);
    mfem::SparseMatrix proc_edge = smoothg::Mult(proc_vert, vertex_edge_global);
    proc_edge.SortColumnIndices();
    mfem::SparseMatrix edge_proc(smoothg::Transpose(proc_edge) );

    // Construct vertex local to global index array
    int nvertices_local = proc_vert.RowSize(myid_);
    mfem::Array<int> vert_loc2glo_tmp;
    vert_loc2glo_tmp.MakeRef(proc_vert.GetRowColumns(myid_), nvertices_local);
    vert_loc2glo_tmp.Copy(vert_local2global_);

    // Construct edge local to global index array
    int nedges_local = proc_edge.RowSize(myid_);
    mfem::Array<int> edge_local2global_tmp;
    edge_local2global_tmp.MakeRef(proc_edge.GetRowColumns(myid_), nedges_local);
    edge_local2global_tmp.Copy(edge_local2global_);

    // Construct local partitioning array for local vertices
    partition_local_.SetSize(nvertices_local);
    int vert_global;
    int Agg_begin = proc_Agg_i[myid_];
    for (int i = 0; i < nvertices_local; i++)
    {
        vert_global = vert_local2global_[i];
        partition_local_[i] = partition_global[vert_global] - Agg_begin;
    }

    // Count number of true edges in each processor
    int ntedges_global = vertex_edge_global.Width();
    mfem::Array<int> tedge_couters(num_procs_ + 1);
    tedge_couters = 0;
    for (int i = 0; i < ntedges_global; i++)
        tedge_couters[edge_proc.GetRowColumns(i)[0] + 1]++;
    int ntedges_local = tedge_couters[myid_ + 1];
    tedge_couters.PartialSum();
    assert(tedge_couters.Last() == ntedges_global);

    // Renumber true edges so that the new numbering is contiguous in processor
    mfem::Array<int> tedge_old2new(ntedges_global);
    for (int i = 0; i < ntedges_global; i++)
        tedge_old2new[i] = tedge_couters[edge_proc.GetRowColumns(i)[0]]++;

    // Construct edge to true edge table
    int* e_te_diag_i = new int[nedges_local + 1];
    int* e_te_diag_j = new int[ntedges_local];
    double* e_te_diag_data = new double[ntedges_local];
    e_te_diag_i[0] = 0;
    std::fill_n(e_te_diag_data, ntedges_local, 1.0);

    assert(nedges_local - ntedges_local >= 0);
    int* e_te_offd_i = new int[nedges_local + 1];
    int* e_te_offd_j = new int[nedges_local - ntedges_local];
    double* e_te_offd_data = new double[nedges_local - ntedges_local];
    HYPRE_Int* e_te_col_map = new HYPRE_Int[nedges_local - ntedges_local];
    e_te_offd_i[0] = 0;
    std::fill_n(e_te_offd_data, nedges_local - ntedges_local, 1.0);

    for (int i = num_procs_ - 1; i > 0; i--)
        tedge_couters[i] = tedge_couters[i - 1];
    tedge_couters[0] = 0;

    mfem::Array<mfem::Pair<HYPRE_Int, int> > offdmap_pair(
        nedges_local - ntedges_local);

    int tedge_new;
    int tedge_begin = tedge_couters[myid_];
    int tedge_end = tedge_couters[myid_ + 1];
    int diag_counter(0), offd_counter(0);
    for (int i = 0; i < nedges_local; i++)
    {
        tedge_new = tedge_old2new[edge_local2global_[i]];
        if ( (tedge_new >= tedge_begin) && (tedge_new < tedge_end) )
        {
            e_te_diag_j[diag_counter++] = tedge_new - tedge_begin;
        }
        else
        {
            offdmap_pair[offd_counter].two = offd_counter;
            offdmap_pair[offd_counter++].one = tedge_new;
        }
        e_te_diag_i[i + 1] = diag_counter;
        e_te_offd_i[i + 1] = offd_counter;
    }
    assert(offd_counter == nedges_local - ntedges_local);

    // Entries of the offd_col_map for edge_e_te_ should be in ascending order
    mfem::SortPairs<HYPRE_Int, int>(offdmap_pair, offd_counter);

    for (int i = 0; i < offd_counter; i++)
    {
        e_te_offd_j[offdmap_pair[i].two] = i;
        e_te_col_map[i] = offdmap_pair[i].one;
    }

    // Generate the "start" array for edge and true edge
    mfem::Array<HYPRE_Int> edge_starts, tedge_starts;
    mfem::Array<HYPRE_Int>* starts[2] = {&edge_starts, &tedge_starts};
    HYPRE_Int size[2] = {nedges_local, ntedges_local};
    GenerateOffsets(comm_, 2, size, starts);

    edge_e_te_ = make_unique<mfem::HypreParMatrix>(
                     comm_, edge_starts.Last(), ntedges_global, edge_starts, tedge_starts,
                     e_te_diag_i, e_te_diag_j, e_te_diag_data,
                     e_te_offd_i, e_te_offd_j, e_te_offd_data, offd_counter, e_te_col_map);
    edge_e_te_->CopyRowStarts();
    edge_e_te_->CopyColStarts();

    // Extract local submatrix of the global vertex to edge relation table
    mfem::Array<int> map(ntedges_global);
    map = -1;

    auto tmp = ExtractRowAndColumns(vertex_edge_global, vert_local2global_,
                                    edge_local2global_, map);
    vertex_edge_local_.Swap(tmp);

    // Compute vertex_trueedge
    GenerateOffsets(comm_, vertex_edge_local_.Height(), vertex_starts_);
    vertex_trueedge_.reset(
        edge_e_te_->LeftDiagMult(vertex_edge_local_, vertex_starts_) );
}

unique_ptr<mfem::HypreParMatrix> Graph::NewEntityToOldTrueEntity(
    const mfem::HypreParMatrix& local_distributed)
{
    mfem::SparseMatrix diag, offd;
    HYPRE_Int* colmap;
    local_distributed.GetDiag(diag);
    local_distributed.GetOffd(offd, colmap);
    HYPRE_Int* distributed_starts = local_distributed.GetColStarts();

    // Find the first offd index that is greater than all diag indices
    // We will fill the permutation matrix in such a way that new local
    // distributed entities are ordered by the global indices of the
    // corresponding old distributed entities (this is only needed for edges)
    int order_preserve_offset = 0;
    for (; order_preserve_offset < offd.Width(); order_preserve_offset++)
    {
        if (colmap[order_preserve_offset] > distributed_starts[0])
        {
            break;
        }
    }

    // Store indices of distributed entities owned by this proc and other procs
    auto distributed_diag = FindNonZeroColumns(diag);
    auto distributed_offd = FindNonZeroColumns(offd);

    int ndistributed_diag = distributed_diag.size();
    int ndistributed_offd = distributed_offd.size();
    int ndistributed_new = ndistributed_diag + ndistributed_offd;

    // Start to form the NewEntity_OldTrueEntity table for distributed entities
    int* perm_diag_i = new int[ndistributed_new + 1];
    int* perm_diag_j = new int[ndistributed_diag];
    double* perm_diag_data = new double[ndistributed_diag];
    int* perm_offd_i = new int[ndistributed_new + 1];
    int* perm_offd_j = new int[ndistributed_offd];
    double* perm_offd_data = new double[ndistributed_offd];

    // diag part of the NewEntity_OldTrueEntity table
    std::fill_n(perm_diag_i, order_preserve_offset, 0);
    std::iota(perm_diag_i + order_preserve_offset,
              perm_diag_i + order_preserve_offset + ndistributed_diag + 1, 0);
    std::fill_n(perm_diag_i + order_preserve_offset + ndistributed_diag + 1,
                ndistributed_offd - order_preserve_offset, ndistributed_diag);

    ndistributed_diag = 0;
    for (auto i = distributed_diag.begin(); i != distributed_diag.end(); i++)
        perm_diag_j[ndistributed_diag++] = *i;

    std::fill_n(perm_diag_data, ndistributed_diag, 1.0);

    // offd part of the NewEntity_OldTrueEntity table
    std::iota(perm_offd_i, perm_offd_i + order_preserve_offset + 1, 0);
    std::fill_n(perm_offd_i + order_preserve_offset,
                ndistributed_diag, order_preserve_offset);
    std::iota(perm_offd_i + ndistributed_diag + order_preserve_offset,
              perm_offd_i + ndistributed_new + 1, order_preserve_offset);

    std::iota(perm_offd_j, perm_offd_j + ndistributed_offd, 0);
    std::fill_n(perm_offd_data, ndistributed_offd, 1.0);

    // Copy the column map of local_distributed
    HYPRE_Int* perm_colmap = new HYPRE_Int[ndistributed_offd];
    ndistributed_offd = 0;
    for (auto i = distributed_offd.begin(); i != distributed_offd.end(); i++)
        perm_colmap[ndistributed_offd++] = colmap[*i];

    mfem::Array<int> newdistributed_starts;
    GenerateOffsets(comm_, ndistributed_new, newdistributed_starts);
    HYPRE_Int ndistributed_old_global = local_distributed.GetGlobalNumCols();

    // Construct the NewEntity_OldTrueEntity table for distributed entities
    auto out = make_unique<mfem::HypreParMatrix>(
                   comm_, newdistributed_starts.Last(), ndistributed_old_global,
                   newdistributed_starts, distributed_starts, perm_diag_i,
                   perm_diag_j, perm_diag_data, perm_offd_i, perm_offd_j,
                   perm_offd_data, ndistributed_offd, perm_colmap);
    out->CopyRowStarts();

    return out;
}

unique_ptr<mfem::HypreParMatrix> Graph::DistributedPartitionToParMatrix(
    const mfem::Array<int>& partition_distributed)
{
    mfem::Array<int> vertex_starts, Agg_starts;

    int nparts_global;
    int part_local_max = partition_distributed.Max() + 1;
    MPI_Allreduce(&part_local_max, &nparts_global, 1, MPI_INT, MPI_MAX, comm_);

    int nAggs_local =
        nparts_global / num_procs_ + (myid_ < (nparts_global % num_procs_));
    int nvertices_local = partition_distributed.Size();

    mfem::Array<int>* starts[2] = {&vertex_starts, &Agg_starts};
    int sizes[2] = {nvertices_local, nAggs_local};
    GenerateOffsets(comm_, 2, sizes, starts);

    // Use map since the keys are sorted
    std::map<unsigned, unsigned> col_map_inv;
    int nverts_in_diag_Agg = 0;
    int Agg_begin = Agg_starts[0];
    int Agg_end = Agg_starts[1];
    int global_Agg_id;
    for (int i = 0; i < nvertices_local; i++)
    {
        global_Agg_id = partition_distributed[i];
        if (Agg_begin <= global_Agg_id && global_Agg_id < Agg_end)
            nverts_in_diag_Agg++;
        else
            col_map_inv[global_Agg_id] = 1;
    }
    int nverts_in_offd_Agg = nvertices_local - nverts_in_diag_Agg;

    // Construct and sort the offd col map for hypre_ParCSRmatrix
    HYPRE_Int* col_map = new HYPRE_Int[col_map_inv.size()];
    int nAggs_offd = 0;
    for (auto i = col_map_inv.begin(); i != col_map_inv.end(); i++)
    {
        col_map_inv[i->first] = nAggs_offd;
        col_map[nAggs_offd++] = i->first;
    }

    int* diag_i = new int[nvertices_local + 1];
    int* diag_j = new int[nverts_in_diag_Agg];
    int* offd_i = new int[nvertices_local + 1];
    int* offd_j = new int[nverts_in_offd_Agg];
    diag_i[0] = 0;
    offd_i[0] = 0;
    nverts_in_diag_Agg = 0;
    nverts_in_offd_Agg = 0;
    for (int i = 0; i < nvertices_local; i++)
    {
        global_Agg_id = partition_distributed[i];
        if (Agg_begin <= global_Agg_id && global_Agg_id < Agg_end)
        {
            diag_i[i + 1] = diag_i[i] + 1;
            offd_i[i + 1] = offd_i[i];
            diag_j[nverts_in_diag_Agg++] = global_Agg_id - Agg_begin;
        }
        else
        {
            diag_i[i + 1] = diag_i[i];
            offd_i[i + 1] = offd_i[i] + 1;
            offd_j[nverts_in_offd_Agg++] = col_map_inv[global_Agg_id];
        }
    }

    double* diag_data = new double[nverts_in_diag_Agg];
    double* offd_data = new double[nverts_in_offd_Agg];
    std::fill_n(diag_data, nverts_in_diag_Agg, 1.0);
    std::fill_n(offd_data, nverts_in_offd_Agg, 1.0);

    auto vertex_Agg_tmp = make_unique<mfem::HypreParMatrix>(
                              comm_, vertex_starts.Last(), Agg_starts.Last(),
                              vertex_starts, Agg_starts, diag_i, diag_j, diag_data,
                              offd_i, offd_j, offd_data, nAggs_offd, col_map);
    vertex_Agg_tmp->CopyRowStarts();
    vertex_Agg_tmp->CopyColStarts();

    return vertex_Agg_tmp;
}

unique_ptr<mfem::HypreParMatrix>
Graph::RedistributeVertices(const mfem::HypreParMatrix& vertex_Agg_tmp)
{
    // Construct newvertex_vertex (permutation) matrix based on Agg_vertex_tmp
    unique_ptr<mfem::HypreParMatrix> Agg_vertex_tmp(
        vertex_Agg_tmp.Transpose() );
    auto vert_perm = NewEntityToOldTrueEntity(*Agg_vertex_tmp);
    vert_perm->CopyColStarts();

    // Compute the block "diagonal" Par_CSRmatrix vertex_Agg
    unique_ptr<mfem::HypreParMatrix> vertex_Agg(
        ParMult(vert_perm.get(), &vertex_Agg_tmp) );

    // The off diagonal part of vertex_Agg (after permuatation) should be all 0
    assert(((hypre_ParCSRMatrix*) *vertex_Agg)->offd->num_nonzeros == 0);

    // Store the partition vector under the new vertex numbering
    partition_local_.SetSize(vertex_Agg->Height());
    std::copy_n(((hypre_ParCSRMatrix*) *vertex_Agg)->diag->j,
                partition_local_.Size(), partition_local_.GetData());

    // Update vert_local2global_ after the redistribution of vertices
    mfem::Vector vert_local2global(vert_local2global_.Size());
    for (int i = 0; i < vert_local2global.Size(); i++)
        vert_local2global[i] = vert_local2global_[i];

    mfem::Vector vert_local2global_new(vert_perm->Height());
    vert_perm->Mult(vert_local2global, vert_local2global_new);

    vert_local2global_.SetSize(vert_local2global_new.Size());
    for (int i = 0; i < vert_local2global_.Size(); i++)
        vert_local2global_[i] = (int)(vert_local2global_new[i]);

    return vert_perm;
}

void Graph::UpdateEdgeLocalToGlobalMap(
    const mfem::HypreParMatrix& newedge_oldtrueedge)
{
    mfem::SparseMatrix edge_e_te_diag;
    edge_e_te_->GetDiag(edge_e_te_diag);
    mfem::Vector edge_local2global(edge_local2global_.Size());
    for (int i = 0; i < edge_local2global.Size(); i++)
        edge_local2global[i] = edge_local2global_[i];

    mfem::Vector edge_truelocal2global(edge_e_te_diag.Width());
    edge_truelocal2global = 0.0;
    edge_e_te_diag.MultTranspose(edge_local2global, edge_truelocal2global);

    mfem::Vector edge_local2global_new(newedge_oldtrueedge.Height());
    newedge_oldtrueedge.Mult(edge_truelocal2global, edge_local2global_new);

    edge_local2global_.SetSize(edge_local2global_new.Size());
    for (int i = 0; i < edge_local2global_.Size(); i++)
        edge_local2global_[i] = (int)(edge_local2global_new[i]);
}

void Graph::RedistributeEdges(const mfem::HypreParMatrix& vertex_permutation)
{
    // Permute vertices in the old vertex_trueedge table
    unique_ptr<mfem::HypreParMatrix> newvertex_oldtrueedge(
        ParMult(&vertex_permutation, vertex_trueedge_.get()) );

    // Update vertex_starts_
    for (int i = 0; i < vertex_starts_.Size() - 1; i++)
        vertex_starts_[i] = vertex_permutation.RowPart()[i];
    vertex_starts_.Last() = vertex_permutation.GetGlobalNumRows();

    // Construct newedge_oldtrueedge matrix based on newvertex_oldtrueedge
    auto newedge_oldtrueedge = NewEntityToOldTrueEntity(*newvertex_oldtrueedge);
    unique_ptr<mfem::HypreParMatrix> oldtrueedge_newedge(
        newedge_oldtrueedge->Transpose() );

    // Update edge local to global map after the redistribution of edges
    UpdateEdgeLocalToGlobalMap(*newedge_oldtrueedge);

    // Permute edges and obtain local vertex_edge after redistribution of edges
    unique_ptr<mfem::HypreParMatrix> pnewvertex_newedge(
        ParMult(newvertex_oldtrueedge.get(), oldtrueedge_newedge.get()) );
    mfem::SparseMatrix newvertex_newedge;
    pnewvertex_newedge->GetDiag(newvertex_newedge);
    vertex_edge_local_ = newvertex_newedge;

    // Construct new "edge to true edge" table
    unique_ptr<mfem::HypreParMatrix> newedge_e_te_e(
        ParMult(newedge_oldtrueedge.get(), oldtrueedge_newedge.get()));

    int ntrueedges_old = edge_e_te_->GetGlobalNumCols();
    edge_e_te_ = BuildEntityToTrueEntity(*newedge_e_te_e);

    // Global number of true edge should be the same after redistribution
    assert(edge_e_te_->GetGlobalNumCols() == ntrueedges_old);
}

void Graph::LocalDepthFirstSearch(const mfem::SparseMatrix& vert_vert,
                                  const mfem::Array<int>& local_vertex_map,
                                  const int vertex,
                                  const int Agg)
{
    const int* vertex_neighbors = vert_vert.GetRowColumns(vertex);
    int vertex_neighbor;

    for (int i = 0; i < vert_vert.RowSize(vertex); i++)
    {
        vertex_neighbor = vertex_neighbors[i];
        if ( (partition_local_[vertex_neighbor] == -1) &&
             (local_vertex_map[vertex_neighbor] == -1) )
        {
            partition_local_[vertex_neighbor] = Agg;
            LocalDepthFirstSearch(vert_vert, local_vertex_map,
                                  vertex_neighbor, Agg);
        }
    }
}

void Graph::SeparateNoncontigousPartitions()
{
    auto edge_vert = smoothg::Transpose(vertex_edge_local_);
    auto vert_vert = smoothg::Mult(vertex_edge_local_, edge_vert);

    int nAggs = partition_local_.Size() > 0 ? partition_local_.Max() + 1 : 0;
    auto Agg_vertex = PartitionToMatrix(partition_local_, nAggs);

    mfem::Array<int> local_vertices;
    int vertex;

    mfem::Array<int> vertex_map(vert_vert.Width());
    vertex_map = 0;

    partition_local_ = -1;
    int new_Agg_counter = 0;
    for (int i = 0; i < nAggs; i++)
    {
        GetTableRow(Agg_vertex, i, local_vertices);
        for (int j = 0; j < local_vertices.Size(); j++)
            vertex_map[local_vertices[j]] = -1;
        for (int j = 0; j < local_vertices.Size(); j++)
        {
            vertex = local_vertices[j];
            if (partition_local_[vertex] == -1)
            {
                partition_local_[vertex] = new_Agg_counter;
                LocalDepthFirstSearch(vert_vert, vertex_map,
                                      vertex, new_Agg_counter++);
            }
        }
        for (int j = 0; j < local_vertices.Size(); j++)
            vertex_map[local_vertices[j]] = 0;
    }
}

void Graph::Redistribute(const mfem::Array<int>& partition_distributed)
{
    auto vert_Agg_tmp = DistributedPartitionToParMatrix(partition_distributed);
    auto vert_permutation = RedistributeVertices(*vert_Agg_tmp);
    RedistributeEdges(*vert_permutation);
    SeparateNoncontigousPartitions();
}

} // namespace smoothg
