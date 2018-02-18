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
            offd_map[offd_counter].first = tedge;
            offd_map[offd_counter].second = offd_counter;
            offd_counter++;
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

linalgcpp::SparseMatrix<int> RestrictInterior(const linalgcpp::SparseMatrix<int>& mat)
{
    int rows = mat.Rows();
    int cols = mat.Cols();

    const auto& mat_indptr(mat.GetIndptr());
    const auto& mat_indices(mat.GetIndices());
    const auto& mat_data(mat.GetData());

    std::vector<int> indptr(rows + 1);
    std::vector<int> indices;

    indices.reserve(mat.nnz());

    for (int i = 0; i < rows; ++i)
    {
        indptr[i] = indices.size();

        for (int j = mat_indptr[i]; j < mat_indptr[i + 1]; ++j)
        {
            if (mat_data[j] > 1)
            {
                indices.push_back(mat_indices[j]);
            }
        }
    }

    indptr[rows] = indices.size();

    std::vector<int> data(indices.size(), 1);

    return linalgcpp::SparseMatrix<int>(std::move(indptr), std::move(indices), std::move(data), rows, cols);
}

linalgcpp::SparseMatrix<int> MakeFaceAggInt(const parlinalgcpp::ParMatrix& agg_agg)
{
    const auto& agg_agg_diag = agg_agg.GetDiag();
    const auto& agg_agg_offd = agg_agg.GetOffd();

    int num_aggs = agg_agg_diag.Rows();
    int num_faces_int = agg_agg_diag.nnz() - agg_agg_diag.Rows();

    assert(num_faces_int % 2 == 0);
    num_faces_int /= 2;

    int num_faces = num_faces_int + agg_agg_offd.nnz();
    std::vector<int> indptr(num_faces + 1);
    std::vector<int> indices(num_faces * 2);
    std::vector<int> data(num_faces * 2, 1);

    indptr[0] = 0;

    const auto& agg_indptr = agg_agg_diag.GetIndptr();
    const auto& agg_indices = agg_agg_diag.GetIndices();
    int rows = agg_agg_diag.Rows();
    int count = 0;

    for (int i = 0; i < rows; ++i)
    {
        for (int j = agg_indptr[i]; j < agg_indptr[i + 1]; ++j)
        {
            if (agg_indices[j] > i)
            {
                indices[count * 2] = i;
                indices[count * 2 + 1] = agg_indices[j];

                count++;

                indptr[count] = count * 2;
            }
        }
    }

    assert(count == num_faces_int);

    return linalgcpp::SparseMatrix<int>(std::move(indptr), std::move(indices), std::move(data),
            num_faces, num_aggs);
}

linalgcpp::SparseMatrix<int> MakeFaceEdge(const parlinalgcpp::ParMatrix& agg_agg,
                                          const parlinalgcpp::ParMatrix& edge_edge,
                                          const linalgcpp::SparseMatrix<int>& agg_edge_ext,
                                          const linalgcpp::SparseMatrix<int>& face_edge_ext)
{
    const auto& agg_agg_diag = agg_agg.GetDiag();
    const auto& agg_agg_offd = agg_agg.GetOffd();

    int num_aggs = agg_agg_diag.Rows();
    int num_edges = face_edge_ext.Cols();
    int num_faces_int = face_edge_ext.Rows();
    int num_faces = num_faces_int + agg_agg_offd.nnz();

    std::vector<int> indptr;
    std::vector<int> indices;

    indptr.reserve(num_faces + 1);

    const auto& ext_indptr = face_edge_ext.GetIndptr();
    const auto& ext_indices = face_edge_ext.GetIndices();
    const auto& ext_data = face_edge_ext.GetData();

    indptr.push_back(0);

    for (int i = 0; i < num_faces_int; i++)
    {
        for (int j = ext_indptr[i]; j < ext_indptr[i + 1]; j++)
        {
            if (ext_data[j] > 1)
            {
                indices.push_back(ext_indices[j]);
            }
        }

        indptr.push_back(indices.size());
    }

    const auto& agg_edge_indptr = agg_edge_ext.GetIndptr();
    const auto& agg_edge_indices = agg_edge_ext.GetIndices();

    const auto& agg_offd_indptr = agg_agg_offd.GetIndptr();
    const auto& agg_offd_indices = agg_agg_offd.GetIndices();
    const auto& agg_offd_data = agg_agg_offd.GetData();
    const auto& agg_colmap = agg_agg.GetColMap();

    const auto& edge_offd_indptr = edge_edge.GetOffd().GetIndptr();
    const auto& edge_offd_indices = edge_edge.GetOffd().GetIndices();
    const auto& edge_offd_data = edge_edge.GetOffd().GetData();
    const auto& edge_colmap = edge_edge.GetColMap();

    for (int i = 0; i < num_aggs; ++i)
    {
        for (int j = agg_offd_indptr[i]; j < agg_offd_indptr[i + 1]; ++j)
        {
            int shared = agg_colmap[agg_offd_indices[j]];

            for (int k = agg_edge_indptr[i]; k < agg_edge_indptr[i + 1]; k++)
            {
                int edge = agg_edge_indices[k];

                if (edge_offd_indptr[edge + 1] > edge_offd_indptr[edge])
                {
                    int edge_loc = edge_offd_indices[edge_offd_indptr[edge]];

                    if (edge_colmap[edge_loc] == shared)
                    {
                        indices.push_back(edge);
                    }
                }
            }

            indptr.push_back(indices.size());
        }
    }

    assert(indptr.size() == num_faces + 1);

    std::vector<int> data(indices.size(), 1);

    return linalgcpp::SparseMatrix<int>(std::move(indptr), std::move(indices), std::move(data),
                                        num_faces, num_edges);
}

linalgcpp::SparseMatrix<int> ExtendFaceAgg(const parlinalgcpp::ParMatrix& agg_agg,
                                           const linalgcpp::SparseMatrix<int>& face_agg_int)
{
    const auto& agg_agg_diag = agg_agg.GetDiag();
    const auto& agg_agg_offd = agg_agg.GetOffd();

    int num_aggs = agg_agg.Rows();

    std::vector<int> indptr(face_agg_int.GetIndptr());
    std::vector<int> indices(face_agg_int.GetIndices());

    const auto& agg_offd_indptr = agg_agg_offd.GetIndptr();

    for (int i = 0; i < num_aggs; ++i)
    {
        for (int j = agg_offd_indptr[i]; j < agg_offd_indptr[i + 1]; ++j)
        {
            indices.push_back(i);
            indptr.push_back(indices.size());
        }
    }

    size_t num_faces = indptr.size() - 1;

    std::vector<int> data(indices.size(), 1);

    return linalgcpp::SparseMatrix<int>(std::move(indptr), std::move(indices), std::move(data),
                                        num_faces, num_aggs);
}

} // namespace smoothg
