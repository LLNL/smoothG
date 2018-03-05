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

   @brief Implements MetisGraphPartitioner object.
*/

#include "MetisGraphPartitioner.hpp"
#include <assert.h>

namespace smoothg
{

MetisGraphPartitioner::MetisGraphPartitioner(PartType _part_type)
    : part_type_(_part_type), unbalance_tol_(1.001)
{
    METIS_SetDefaultOptions(options_);
    options_[METIS_OPTION_CONTIG] = 1;
}

void MetisGraphPartitioner::doPartition(const mfem::SparseMatrix& wtable,
                                        int& num_partitions,
                                        mfem::Array<int>& partitioning,
                                        bool use_edge_weight)
{
    const int nvertices = wtable.Size();

    partitioning.SetSize(nvertices);

    mfem::SparseMatrix sub_table;
    mfem::Array<int> sub_part;
    mfem::Array<int> sub_to_global;

    if (pre_isolated_vertices_.Size() > 0)
    {
        IsolatePreProcess(wtable, sub_table, sub_to_global);
        sub_part.SetSize(sub_to_global.Size());
    }
    else
    {
        sub_table.MakeRef(wtable);
        sub_part.MakeRef(partitioning);
    }

    int sub_nvertices = sub_table.Size();

    auto adjacency = getAdjacency(sub_table);
    int* i_ptr = adjacency.GetI();
    int* j_ptr = adjacency.GetJ();

    int* edge_weight_ptr = nullptr;
    mfem::Array<int> adj_weight_int;
    if (use_edge_weight)
    {
        adj_weight_int.SetSize(adjacency.NumNonZeroElems());
        mfem::Vector adj_weight(adjacency.GetData(), adj_weight_int.Size());
        double adj_wt_min = adj_weight.Min();
        for (int i = 0; i < adj_weight.Size(); i++)
            adj_weight_int[i] = floor(log2(adj_weight[i] / adj_wt_min)) + 1;
        edge_weight_ptr = adj_weight_int.GetData();
    }

    int* vertex_weight_ptr = nullptr;

    int ncon = 1;
    int edgecut;

    switch (part_type_)
    {
        // This function should be used to partition a graph into a small
        // number of partitions (less than 8).
        case PartType::RECURSIVE:
        {
            options_[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
            CallMetis = METIS_PartGraphRecursive;
            break;
        }
        case PartType::KWAY:
        {
            options_[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
            CallMetis = METIS_PartGraphKway;
            break;
        }
        case PartType::TOTALCOMMUNICATION:
        {
            options_[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
            CallMetis = METIS_PartGraphKway;
            break;
        }
        default:
            mfem::mfem_error("invalid options");
    }

    if (num_partitions > 1)
    {
        int err = CallMetis(&sub_nvertices, &ncon, i_ptr, j_ptr,
                            vertex_weight_ptr, nullptr, edge_weight_ptr,
                            &num_partitions, nullptr, &unbalance_tol_,
                            options_, &edgecut, sub_part);

        assert(err == METIS_OK);
    }
    else
    {
        sub_part = 0;
    }

    if (pre_isolated_vertices_.Size() > 0)
    {
        for (int i = 0; i < sub_part.Size(); ++i)
        {
            partitioning[sub_to_global[i]] = sub_part[i];
        }

        for (auto vertex : pre_isolated_vertices_)
        {
            partitioning[vertex] = num_partitions++;
        }
    }

    if (post_isolated_vertices_.Size() > 0)
    {
        IsolatePostProcess(wtable, num_partitions, partitioning);
    }

    removeEmptyParts(partitioning, num_partitions);
}

void MetisGraphPartitioner::SetPreIsolateVertices(int index)
{
    pre_isolated_vertices_.Append(index);
}

void MetisGraphPartitioner::SetPreIsolateVertices(const mfem::Array<int>& indices)
{
    pre_isolated_vertices_.Append(indices);
}

void MetisGraphPartitioner::SetPostIsolateVertices(int index)
{
    post_isolated_vertices_.Append(index);
}

void MetisGraphPartitioner::SetPostIsolateVertices(const mfem::Array<int>& indices)
{
    post_isolated_vertices_.Append(indices);
}

mfem::SparseMatrix MetisGraphPartitioner::getAdjacency(
    const mfem::SparseMatrix& wtable)
{
    const int nvertices = wtable.Size();
    const int nedges = wtable.NumNonZeroElems();

    int* adj_i = new int[nvertices + 1];
    int* adj_j = new int[nedges];
    double* adj_weight = new double[nedges];

    const int* w_i = wtable.GetI();
    const int* w_j = wtable.GetJ();
    const double* w_data = wtable.GetData();

    int nnz = 0;

    for (int i = 0; i < nvertices; ++i)
    {
        adj_i[i] = nnz;
        for (int j = w_i[i]; j < w_i[i + 1]; ++j)
        {
            if (w_j[j] != i)
            {
                adj_j[nnz] = w_j[j];
                adj_weight[nnz] = w_data[j];
                ++nnz;
            }
        }
    }

    adj_i[nvertices] = nnz;

    return mfem::SparseMatrix(adj_i, adj_j, adj_weight,
                              nvertices, nvertices);
}

void MetisGraphPartitioner::removeEmptyParts(mfem::Array<int>& partitioning,
                                             int& num_partitions) const
{
    int shift = 0;
    std::vector<int> count(num_partitions, 0);
    std::vector<int> shifter(num_partitions, 0);

    for (const int& part : partitioning)
        ++count[part];

    for (int i = 0; i < num_partitions; ++i)
    {
        if (!count[i])
            ++shift;
        shifter[i] = shift;
    }

    for (int& part : partitioning)
        part -= shifter[part];

    num_partitions -= shift;
}

void MetisGraphPartitioner::IsolatePreProcess(const mfem::SparseMatrix& wtable,
                                              mfem::SparseMatrix& sub_table,
                                              mfem::Array<int>& sub_to_global)
{
    std::vector<int> indices(wtable.Height());
    std::iota(std::begin(indices), std::end(indices), 0);
    indices.erase(std::remove_if(std::begin(indices), std::end(indices),
    [this](int x) { return pre_isolated_vertices_.Find(x) != -1; }),
    std::end(indices));

    mfem::Array<int> indices_m(indices.data(), indices.size());
    indices_m.Copy(sub_to_global);

    mfem::Array<int> col_map(wtable.Height());
    col_map = -1;

    mfem::SparseMatrix sub_mat = ExtractRowAndColumns(wtable, indices_m,
                                                      indices_m, col_map);
    sub_table.Swap(sub_mat);
}

void MetisGraphPartitioner::IsolatePostProcess(const mfem::SparseMatrix& wtable,
                                               int& num_partitions,
                                               mfem::Array<int>& partitioning)
{
    // do a post-processing of partitioning to put critical vertices in their own partitions
    {
        int c_elem = num_partitions;
        for (int i(0); i < post_isolated_vertices_.Size(); ++i)
        {
            int num = post_isolated_vertices_[i];
            partitioning[num] = c_elem;
            c_elem++;
        }
        num_partitions = c_elem;
    }

    // removing cells may have made some partitions disconnected, so now
    // we fix that
    connectedComponents(partitioning, wtable);
    num_partitions = partitioning.Max() + 1;
}

int MetisGraphPartitioner::connectedComponents(mfem::Array<int>& partitioning,
                                               const mfem::SparseMatrix& conn)
{
    MFEM_ASSERT(partitioning.Size() == conn.Height(), "Wrong sized input!");
    MFEM_ASSERT(partitioning.Size() == conn.Width(), "Wrong sized input!");
    int num_nodes = conn.Height();
    int num_part(partitioning.Max() + 1);

    mfem::Array<int> component(num_nodes);
    component = -1;
    mfem::Array<int> offset_comp(num_part + 1);
    offset_comp = 0;
    mfem::Array<int> num_comp(offset_comp.GetData() + 1, num_part);
    int i, j, k;
    const int* i_table, *j_table;

    i_table = conn.GetI();
    j_table = conn.GetJ();

    mfem::Array<int> vertex_stack(num_nodes);
    int stack_p, stack_top_p, node;

    stack_p = 0;
    stack_top_p = 0;  // points to the first unused element in the stack
    for (node = 0; node < num_nodes; node++)
    {
        if (partitioning[node] < 0)
            continue;

        if (component[node] >= 0)
            continue;

        component[node] = num_comp[partitioning[node]]++;
        vertex_stack[stack_top_p++] = node;

        for ( ; stack_p < stack_top_p; stack_p++)
        {
            i = vertex_stack[stack_p];
            if (partitioning[i] < 0)
                continue;

            for (j = i_table[i]; j < i_table[i + 1]; j++)
            {
                k = j_table[j];
                if (partitioning[k] == partitioning[i] )
                {
                    if (component[k] < 0)
                    {
                        component[k] = component[i];
                        vertex_stack[stack_top_p++] = k;
                    }
                    MFEM_ASSERT(component[k] == component[i], "Impossible topology!");
                }
            }
        }
    }
    offset_comp.PartialSum();
    for (int i(0); i < num_nodes; ++i)
        partitioning[i] = offset_comp[partitioning[i]] + component[i];

    MFEM_ASSERT(partitioning.Max() + 1 == offset_comp.Last(),
                "Partitioning inconsistent with components!");
    return offset_comp.Last();
}

void Partition(const mfem::SparseMatrix& w_table, mfem::Array<int>& partitioning,
               int num_parts, bool use_edge_weight)
{
    MetisGraphPartitioner partitioner;
    partitioner.setUnbalanceTol(2);
    partitioner.doPartition(w_table, num_parts, partitioning, use_edge_weight);
}

void PartitionAAT(const mfem::SparseMatrix& vertex_edge,
                  mfem::Array<int>& partitioning, int coarsening_factor)
{
    const mfem::SparseMatrix edge_vert = smoothg::Transpose(vertex_edge);
    const mfem::SparseMatrix vert_vert = smoothg::Mult(vertex_edge, edge_vert);

    const int nvertices = vert_vert.Height();
    int num_partitions = (nvertices / (double)(coarsening_factor)) + 0.5;
    num_partitions = std::max(1, num_partitions);
    Partition(vert_vert, partitioning, num_partitions);
}

} // namespace smoothg
