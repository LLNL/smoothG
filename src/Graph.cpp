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
{
    Distribute(comm, vertex_edge_global, edge_weight_global);
    const mfem::SparseMatrix* edge_bdratt = nullptr;
    Init(*edge_trueedge_, edge_bdratt);
}

Graph::Graph(const mfem::SparseMatrix& vertex_edge_local,
             const mfem::HypreParMatrix& edge_trueedge,
             const mfem::Vector& edge_weight_local,
             const mfem::SparseMatrix* edge_bdratt)
    : vertex_edge_local_(vertex_edge_local)
{
    if (edge_weight_local.Size() > 0)
    {
        SplitEdgeWeight(edge_weight_local);
    }
    else
    {
        mfem::Vector unit_edge_weight(vertex_edge_local_.Width());
        unit_edge_weight = 1.0;
        FixSharedEdgeWeight(edge_trueedge, unit_edge_weight);
        SplitEdgeWeight(unit_edge_weight);
    }

    Init(edge_trueedge, edge_bdratt);
}

Graph::Graph(const mfem::SparseMatrix& vertex_edge_local,
             const mfem::HypreParMatrix& edge_trueedge,
             const std::vector<mfem::Vector>& split_edge_weight,
             const mfem::SparseMatrix* edge_bdratt)
    : vertex_edge_local_(vertex_edge_local), split_edge_weight_(split_edge_weight)
{
    Init(edge_trueedge, edge_bdratt);
}

Graph::Graph(mfem::SparseMatrix edge_vertex_local,
             std::unique_ptr<mfem::HypreParMatrix> edge_trueedge,
             const mfem::Array<int>& vertex_starts,
             const mfem::Array<int>& edge_starts,
             const mfem::SparseMatrix* edge_bdratt)
    : vertex_edge_local_(smoothg::Transpose(edge_vertex_local)),
      edge_trueedge_(std::move(edge_trueedge)),
      vertex_weight_(vertex_edge_local_.NumRows()),
      edge_vertex_local_(std::move(edge_vertex_local)),
      edge_trueedge_edge_(AAt(*edge_trueedge_))
{
    vertex_starts.Copy(vertex_starts_);
    edge_starts.Copy(edge_starts_);

    if (edge_bdratt != nullptr)
    {
        mfem::SparseMatrix tmp(*edge_bdratt);
        edge_bdratt_.Swap(tmp);
    }

    vertex_trueedge_ = ParMult(vertex_edge_local_, *edge_trueedge_, vertex_starts_);
    edge_vertex_local_.SortColumnIndices();

    vertex_weight_ = 1.0;
}

Graph::Graph(const Graph& other) noexcept
    : vertex_edge_local_(other.vertex_edge_local_),
      edge_trueedge_(Copy(*other.edge_trueedge_)),
      split_edge_weight_(other.split_edge_weight_),
      vertex_weight_(other.vertex_weight_),
      edge_bdratt_(other.edge_bdratt_),
      edge_vertex_local_(other.edge_vertex_local_),
      edge_trueedge_edge_(Copy(*other.edge_trueedge_edge_)),
      vertex_trueedge_(Copy(*other.vertex_trueedge_))
{
    other.vert_loc_to_glo_.Copy(vert_loc_to_glo_);
    other.edge_loc_to_glo_.Copy(edge_loc_to_glo_);
    other.vertex_starts_.Copy(vertex_starts_);
    other.edge_starts_.Copy(edge_starts_);
}

Graph::Graph(Graph&& other) noexcept
{
    swap(*this, other);
}

Graph& Graph::operator=(Graph other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(Graph& lhs, Graph& rhs) noexcept
{
    lhs.vertex_edge_local_.Swap(rhs.vertex_edge_local_);
    std::swap(lhs.split_edge_weight_, rhs.split_edge_weight_);
    std::swap(lhs.edge_trueedge_, rhs.edge_trueedge_);
    lhs.edge_bdratt_.Swap(rhs.edge_bdratt_);

    lhs.edge_vertex_local_.Swap(rhs.edge_vertex_local_);
    std::swap(lhs.edge_trueedge_edge_, rhs.edge_trueedge_edge_);
    std::swap(lhs.vertex_trueedge_, rhs.vertex_trueedge_);
    mfem::Swap(lhs.vert_loc_to_glo_, rhs.vert_loc_to_glo_);
    mfem::Swap(lhs.edge_loc_to_glo_, rhs.edge_loc_to_glo_);
    mfem::Swap(lhs.vertex_starts_, rhs.vertex_starts_);
    mfem::Swap(lhs.edge_starts_, rhs.edge_starts_);
    mfem::Swap(lhs.vertex_weight_, rhs.vertex_weight_);
}

void Graph::Init(const mfem::HypreParMatrix& edge_trueedge,
                 const mfem::SparseMatrix* edge_bdratt)
{
    if (edge_bdratt != nullptr)
    {
        mfem::SparseMatrix tmp(*edge_bdratt);
        edge_bdratt_.Swap(tmp);
    }

    edge_vertex_local_ = smoothg::Transpose(vertex_edge_local_);
    edge_vertex_local_.SortColumnIndices();

    edge_starts_.SetSize(3);
    edge_starts_[0] = edge_trueedge.GetRowStarts()[0];
    edge_starts_[1] = edge_trueedge.GetRowStarts()[1];
    edge_starts_[2] = edge_trueedge.M();

    if (edge_trueedge_edge_ == nullptr)
    {
        edge_trueedge_edge_ = AAt(edge_trueedge);
    }

    if (edge_trueedge_ == nullptr)
    {
        ReorderEdges(edge_trueedge);
    }

    GenerateOffsets(GetComm(), vertex_edge_local_.Height(), vertex_starts_);

    vertex_trueedge_ = ParMult(vertex_edge_local_, *edge_trueedge_, vertex_starts_);

    vertex_weight_.SetSize(vertex_edge_local_.NumRows());
    vertex_weight_ = 1.0;
}

void Graph::Distribute(MPI_Comm comm,
                       const mfem::SparseMatrix& vertex_edge_global,
                       const mfem::Vector& edge_weight_global)
{
    DistributeVertexEdge(comm, vertex_edge_global);
    mfem::Vector edge_weight_local = DistributeEdgeWeight(edge_weight_global);
    SplitEdgeWeight(edge_weight_local);
}

void Graph::DistributeVertexEdge(MPI_Comm comm,
                                 const mfem::SparseMatrix& vert_edge_global)
{
    MFEM_VERIFY(HYPRE_AssumedPartitionCheck(),
                "this method can not be used without assumed partition");

    int num_procs;
    int myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    mfem::SparseMatrix vert_vert = AAt(vert_edge_global);
    mfem::Array<int> partition;
    Partition(vert_vert, partition, num_procs);

    // Construct processor to vertex/edge from global partition
    mfem::SparseMatrix proc_vert = PartitionToMatrix(partition, num_procs);
    mfem::SparseMatrix proc_edge = smoothg::Mult(proc_vert, vert_edge_global);
    proc_edge.SortColumnIndices(); // TODO: this may not be needed once SEC is fixed

    // Construct vertex/edge local to global index array
    GetTableRowCopy(proc_vert, myid, vert_loc_to_glo_);
    GetTableRowCopy(proc_edge, myid, edge_loc_to_glo_);

    // Extract local submatrix of the global vertex to edge relation table
    auto tmp = ExtractRowAndColumns(vert_edge_global, vert_loc_to_glo_, edge_loc_to_glo_);
    vertex_edge_local_.Swap(tmp);

    MakeEdgeTrueEdge(comm, myid, proc_edge);
}

void Graph::MakeEdgeTrueEdge(MPI_Comm comm, int myid, const mfem::SparseMatrix& proc_edge)
{
    const int num_procs = proc_edge.Height();
    const int nedges_local = proc_edge.RowSize(myid);

    mfem::SparseMatrix edge_proc = smoothg::Transpose(proc_edge);

    // Count number of true edges in each processor
    int ntedges_global = proc_edge.Width();
    mfem::Array<int> tedge_couters(num_procs + 1);
    tedge_couters = 0;
    for (int i = 0; i < ntedges_global; i++)
        tedge_couters[edge_proc.GetRowColumns(i)[0] + 1]++;
    int ntedges_local = tedge_couters[myid + 1];
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

    for (int i = num_procs - 1; i > 0; i--)
        tedge_couters[i] = tedge_couters[i - 1];
    tedge_couters[0] = 0;

    mfem::Array<mfem::Pair<HYPRE_Int, int> > offdmap_pair(
        nedges_local - ntedges_local);

    int tedge_new;
    int tedge_begin = tedge_couters[myid];
    int tedge_end = tedge_couters[myid + 1];
    int diag_counter(0), offd_counter(0);
    for (int i = 0; i < nedges_local; i++)
    {
        tedge_new = tedge_old2new[edge_loc_to_glo_[i]];
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
    GenerateOffsets(comm, 2, size, starts);

    edge_trueedge_ = make_unique<mfem::HypreParMatrix>(
                         comm, edge_starts.Last(), ntedges_global, edge_starts, tedge_starts,
                         e_te_diag_i, e_te_diag_j, e_te_diag_data,
                         e_te_offd_i, e_te_offd_j, e_te_offd_data, offd_counter, e_te_col_map);
    edge_trueedge_->CopyRowStarts();
    edge_trueedge_->CopyColStarts();
}

mfem::Vector Graph::DistributeEdgeWeight(const mfem::Vector& edge_weight_global)
{
    mfem::Vector edge_weight_local(vertex_edge_local_.Width());
    if (edge_weight_global.Size())
    {
        edge_weight_global.GetSubVector(edge_loc_to_glo_, edge_weight_local);
    }
    else
    {
        edge_weight_local = 1.0;
    }
    FixSharedEdgeWeight(*edge_trueedge_, edge_weight_local);

    return edge_weight_local;
}

void Graph::FixSharedEdgeWeight(const mfem::HypreParMatrix& edge_trueedge,
                                mfem::Vector& edge_weight_local)
{
    if (edge_trueedge_edge_ == nullptr)
    {
        edge_trueedge_edge_ = AAt(edge_trueedge);
    }
    mfem::SparseMatrix edge_is_shared = GetOffd(*edge_trueedge_edge_);

    assert(edge_is_shared.Height() == edge_weight_local.Size());
    for (int edge = 0; edge < edge_is_shared.Height(); ++edge)
    {
        if (edge_is_shared.RowSize(edge))
        {
            edge_weight_local[edge] *= 2.0;
        }
    }
}

void Graph::SplitEdgeWeight(const mfem::Vector& edge_weight_local)
{
    const mfem::SparseMatrix edge_vert = smoothg::Transpose(vertex_edge_local_);
    split_edge_weight_.resize(edge_vert.Width());

    mfem::Array<int> edges;
    for (int vert = 0; vert < edge_vert.Width(); vert++)
    {
        GetTableRow(vertex_edge_local_, vert, edges);
        split_edge_weight_[vert].SetSize(edges.Size());
        for (int i = 0; i < edges.Size(); i++)
        {
            const int edge = edges[i];
            double ratio = edge_vert.RowSize(edge) == 2 ? 2.0 : 1.0;
            split_edge_weight_[vert][i] = edge_weight_local[edge] * ratio;
        }
    }
}

mfem::Vector Graph::ReadVertexVector(const std::string& filename) const
{
    assert(vert_loc_to_glo_.Size() == vertex_edge_local_.Height());
    return ReadVector(filename, vertex_starts_.Last(), vert_loc_to_glo_);
}

void Graph::ReorderEdges(const mfem::HypreParMatrix& edge_trueedge)
{
    auto reorder_map = EntityReorderMap(edge_trueedge, *edge_trueedge_edge_);
    edge_trueedge_ = ParMult(reorder_map, edge_trueedge, edge_starts_);
    edge_trueedge_->CopyColStarts();
    edge_trueedge_edge_ = AAt(*edge_trueedge_);

    if (HasBoundary())
    {
        auto tmp = smoothg::Mult(reorder_map, edge_bdratt_);
        edge_bdratt_.Swap(tmp);
    }

    auto edge_vertex_local_tmp = smoothg::Mult(reorder_map, edge_vertex_local_);
    edge_vertex_local_.Swap(edge_vertex_local_tmp);

    auto vertex_edge_local_tmp = smoothg::Transpose(edge_vertex_local_);
    vertex_edge_local_.Swap(vertex_edge_local_tmp);

    mfem::Array<int> reordered_edges, original_edges;
    for (int vert = 0; vert < NumVertices(); ++vert)
    {
        GetTableRow(vertex_edge_local_, vert, reordered_edges);
        GetTableRow(vertex_edge_local_tmp, vert, original_edges);
        mfem::Vector edge_weight_local(reordered_edges.Size());
        for (int i = 0; i < reordered_edges.Size(); ++i)
        {
            int reordered_edge = reordered_edges[i];
            int original_edge = reorder_map.GetRowColumns(reordered_edge)[0];
            int original_local = original_edges.Find(original_edge);
            assert(original_local != -1);
            edge_weight_local[i] = split_edge_weight_[vert][original_local];
        }
        mfem::Swap(split_edge_weight_[vert], edge_weight_local);
    }
}

mfem::Vector Graph::ReadVector(const std::string& filename, int global_size,
                               const mfem::Array<int>& local_to_global) const
{
    assert(global_size > 0);

    std::ifstream file(filename);
    assert(file.is_open());

    mfem::Vector global_vect(global_size);
    mfem::Vector local_vect;

    global_vect.Load(file, global_size);
    global_vect.GetSubVector(local_to_global, local_vect);

    return local_vect;
}

void Graph::WriteVertexVector(const mfem::Vector& vec_loc, const std::string& filename) const
{
    assert(vert_loc_to_glo_.Size() == vertex_edge_local_.Height());
    WriteVector(vec_loc, filename, vertex_starts_.Last(), vert_loc_to_glo_);
}

void Graph::WriteVector(const mfem::Vector& vect, const std::string& filename,
                        int global_size, const mfem::Array<int>& local_to_global) const
{
    assert(global_size > 0);
    assert(vect.Size() <= global_size);

    int num_procs;
    int myid;
    MPI_Comm_size(GetComm(), &num_procs);
    MPI_Comm_rank(GetComm(), &myid);

    mfem::Vector global_local(global_size);
    global_local = 0.0;
    global_local.SetSubVector(local_to_global, vect);

    mfem::Vector global_global(global_size);
    MPI_Scan(global_local.GetData(), global_global.GetData(), global_size,
             MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (myid == num_procs - 1)
    {
        std::ofstream out_file(filename);
        out_file.precision(16);
        out_file << std::scientific;
        global_global.Print(out_file, 1);
    }
}

void Graph::SetNewLocalWeight(std::vector<mfem::Vector> split_edge_weight)
{
    split_edge_weight_ = std::move(split_edge_weight);
}

} // namespace smoothg
