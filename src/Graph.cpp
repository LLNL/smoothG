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
             const mfem::Vector& edge_weight_global)
    : comm_(comm)
{
    MPI_Comm_size(comm_, &num_procs_);
    MPI_Comm_rank(comm_, &myid_);

    Distribute(vertex_edge_global, edge_weight_global, partition_global);
}


void Graph::Distribute(const mfem::SparseMatrix& vertex_edge_global,
                       const mfem::Vector& edge_weight_global,
                       int coarsening_factor, bool do_parmetis_partition)
{
    int num_vertices_global = vertex_edge_global.Height();
    int num_parts_global = (num_vertices_global / (double)(coarsening_factor)) + 0.5;
    num_parts_global = std::max(1, num_parts_global);

    mfem::Array<int> partition_global;
    if (do_parmetis_partition)
    {
#if SMOOTHG_USE_PARMETIS
        PartitionByIndex(num_vertices_global, num_procs_, partition_global);
        Distribute(vertex_edge_global, edge_weight_global, partition_global);

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
        // TODO(gelever1) : should processor 0 partition and distribute or assume all processors will
        // obtain the same global partition from metis?
        auto vert_vert = smoothg::AAT(vertex_edge_global);
        Partition(vert_vert, partition_global, num_procs_);
        Distribute(vertex_edge_global, edge_weight_global, partition_global);
    }
}

void Graph::Distribute(const mfem::SparseMatrix& vertex_edge_global,
                       const mfem::Vector& edge_weight_global,
                       const mfem::Array<int>& partition_global)
{
    DistributeVertexEdge(vertex_edge_global, partition_global);
    DistributeEdgeWeight(edge_weight_global);
}

void Graph::DistributeVertexEdge(const mfem::SparseMatrix& vertex_edge_global,
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

    edge_trueedge_ = make_unique<mfem::HypreParMatrix>(
                         comm_, edge_starts.Last(), ntedges_global, edge_starts, tedge_starts,
                         e_te_diag_i, e_te_diag_j, e_te_diag_data,
                         e_te_offd_i, e_te_offd_j, e_te_offd_data, offd_counter, e_te_col_map);
    edge_trueedge_->CopyRowStarts();
    edge_trueedge_->CopyColStarts();

    // Extract local submatrix of the global vertex to edge relation table
    mfem::Array<int> map(ntedges_global);
    map = -1;

    auto tmp = ExtractRowAndColumns(vertex_edge_global, vert_local2global_,
                                    edge_local2global_, map);
    vertex_edge_local_.Swap(tmp);

    // Compute vertex_trueedge
    GenerateOffsets(comm_, vertex_edge_local_.Height(), vertex_starts_);
    vertex_trueedge_.reset(
        edge_trueedge_->LeftDiagMult(vertex_edge_local_, vertex_starts_) );
}

void Graph::DistributeEdgeWeight(const mfem::Vector& edge_weight_global)
{
    edge_weight_local_.SetSize(vertex_edge_local_.Width());
    edge_weight_global.GetSubVector(edge_local2global_, edge_weight_local_);
}

} // namespace smoothg
