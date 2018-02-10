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

    @brief Implementations of some utility routines for linear algebra.

    These are implemented with and operate on linalgcpp data structures.
*/

#include "Utilities.hpp"

namespace smoothg
{
parlinalgcpp::ParMatrix MakeEdgeTrueEdge(MPI_Comm comm, const linalgcpp::SparseMatrix<int>& proc_edge,
                                         const std::vector<int>& edge_map)
{
    int myid;
    int num_procs;

    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    auto edge_proc = proc_edge.Transpose();

    size_t num_edges_local = proc_edge.RowSize(myid);
    size_t num_tedges_global = proc_edge.Cols();

    std::vector<int> tedge_counter(num_procs + 1, 0);

    for (size_t i = 0; i < num_tedges_global; ++i)
    {
        tedge_counter[edge_proc.GetIndices(i)[0] + 1]++;
    }

    size_t num_tedges_local = tedge_counter[myid + 1];
    size_t num_edge_diff = num_edges_local - num_tedges_local;
    std::partial_sum(std::begin(tedge_counter), std::end(tedge_counter),
                     std::begin(tedge_counter));

    assert(tedge_counter.back() == static_cast<int>(num_tedges_global));

    std::vector<int> edge_perm(num_tedges_global);

    for (size_t i = 0; i < num_tedges_global; ++i)
    {
        edge_perm[i] = tedge_counter[edge_proc.GetIndices(i)[0]]++;
    }

    for (int i = num_procs - 1; i > 0; i--)
    {
        tedge_counter[i] = tedge_counter[i - 1];
    }
    tedge_counter[0] = 0;

    std::vector<int> diag_indptr(num_edges_local + 1);
    std::vector<int> diag_indices(num_tedges_local);
    std::vector<double> diag_data(num_tedges_local, 1.0);

    std::vector<int> offd_indptr(num_edges_local + 1);
    std::vector<int> offd_indices(num_edge_diff);
    std::vector<double> offd_data(num_edge_diff, 1.0);
    std::vector<HYPRE_Int> col_map(num_edge_diff);
    std::vector<std::pair<HYPRE_Int, int>> offd_map(num_edge_diff);

    diag_indptr[0] = 0;
    offd_indptr[0] = 0;

    int tedge_begin = tedge_counter[myid];
    int tedge_end = tedge_counter[myid + 1];

    int diag_counter = 0;
    int offd_counter = 0;

    for (size_t i = 0; i < num_edges_local; ++i)
    {
        int tedge = edge_perm[edge_map[i]];

        if ((tedge>= tedge_begin) && (tedge < tedge_end))
        {
            diag_indices[diag_counter++] = tedge - tedge_begin;
        }
        else
        {
            offd_map[offd_counter].first = offd_counter;
            offd_map[offd_counter++].second = tedge;
        }

        diag_indptr[i + 1] = diag_counter;
        offd_indptr[i + 1] = offd_counter;
    }

    assert(offd_counter == static_cast<int>(num_edge_diff));

    auto compare = [] (const auto& lhs, const auto& rhs)
    {
        return lhs.first < rhs.first;
    };

    std::sort(std::begin(offd_map), std::end(offd_map), compare);

    for (int i = 0; i < offd_counter; ++i)
    {
        offd_indices[offd_map[i].second] = i;
        col_map[i] = offd_map[i].first;
    }

    auto starts = parlinalgcpp::GenerateOffsets(comm, {num_edges_local, num_tedges_local});

    linalgcpp::SparseMatrix<double> diag(std::move(diag_indptr), std::move(diag_indices), std::move(diag_data),
                                         num_edges_local, num_tedges_local);

    linalgcpp::SparseMatrix<double> offd(std::move(offd_indptr), std::move(offd_indices), std::move(offd_data),
                                         num_edges_local, num_edge_diff);
                                         
    return parlinalgcpp::ParMatrix(comm, starts[0], starts[1],
                                   std::move(diag), std::move(offd),
                                   std::move(col_map));
}


} // namespace smoothg
