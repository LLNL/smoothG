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
parlinalgcpp::ParMatrix MakeEdgeTrueEdge(MPI_Comm comm, const linalgcpp::SparseMatrix<double>& proc_edge,
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

linalgcpp::SparseMatrix<double> RestrictInterior(const linalgcpp::SparseMatrix<double>& mat)
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

    std::vector<double> data(indices.size(), 1);

    return linalgcpp::SparseMatrix<double>(std::move(indptr), std::move(indices), std::move(data), rows, cols);
}

parlinalgcpp::ParMatrix RestrictInterior(const parlinalgcpp::ParMatrix& mat)
{
    int num_rows = mat.Rows();

    const auto& diag_ext = mat.GetDiag();
    const auto& offd_ext = mat.GetOffd();
    const auto& colmap_ext = mat.GetColMap();

    const auto& offd_indptr = offd_ext.GetIndptr();
    const auto& offd_indices = offd_ext.GetIndices();
    const auto& offd_data = offd_ext.GetData();
    const int num_offd = offd_ext.Cols();

    std::vector<int> indptr(num_rows + 1);
    std::vector<int> offd_marker(num_offd, -1);

    int offd_nnz = 0;

    for (int i = 0; i < num_rows; ++i)
    {
        indptr[i] = offd_nnz;

        for (int j = offd_indptr[i]; j < offd_indptr[i + 1]; ++j)
        {
            if (offd_data[j] > 1)
            {
                offd_marker[offd_indices[j]] = 1;
                offd_nnz++;
            }
        }
    }

    indptr[num_rows] = offd_nnz;

    int offd_num_cols = std::count_if(std::begin(offd_marker), std::end(offd_marker),
                                      [](auto i) { return i > 0; });

    std::vector<HYPRE_Int> col_map(offd_num_cols);
    int count = 0;

    for (int i = 0; i < num_offd; ++i)
    {
        if (offd_marker[i] > 0)
        {
            offd_marker[i] = count;
            col_map[count] = colmap_ext[i];

            count++;
        }
    }

    assert(count == offd_num_cols);

    std::vector<int> indices(offd_nnz);
    std::vector<double> data(offd_nnz, 1.0);

    count = 0;

    for (int i = 0; i < num_rows; i++)
    {
        for (int j = offd_indptr[i]; j < offd_indptr[i + 1]; ++j)
        {
            if (offd_data[j] > 1)
            {
                indices[count++] = offd_marker[offd_indices[j]];
            }
        }
    }

    assert(count == offd_nnz);

    linalgcpp::SparseMatrix<double> diag = RestrictInterior(diag_ext);
    linalgcpp::SparseMatrix<double> offd(std::move(indptr), std::move(indices), std::move(data),
                                         num_rows, offd_num_cols);

    return parlinalgcpp::ParMatrix(mat.GetComm(), mat.GetRowStarts(), mat.GetColStarts(),
                                   std::move(diag), std::move(offd), std::move(col_map));
}

linalgcpp::SparseMatrix<double> MakeFaceAggInt(const parlinalgcpp::ParMatrix& agg_agg)
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
    std::vector<double> data(num_faces * 2, 1);

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

    return linalgcpp::SparseMatrix<double>(std::move(indptr), std::move(indices), std::move(data),
            num_faces, num_aggs);
}

linalgcpp::SparseMatrix<double> MakeFaceEdge(const parlinalgcpp::ParMatrix& agg_agg,
                                          const parlinalgcpp::ParMatrix& edge_edge,
                                          const linalgcpp::SparseMatrix<double>& agg_edge_ext,
                                          const linalgcpp::SparseMatrix<double>& face_edge_ext)
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
    const auto& agg_colmap = agg_agg.GetColMap();

    const auto& edge_offd_indptr = edge_edge.GetOffd().GetIndptr();
    const auto& edge_offd_indices = edge_edge.GetOffd().GetIndices();
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

    std::vector<double> data(indices.size(), 1);

    return linalgcpp::SparseMatrix<double>(std::move(indptr), std::move(indices), std::move(data),
                                        num_faces, num_edges);
}

linalgcpp::SparseMatrix<double> ExtendFaceAgg(const parlinalgcpp::ParMatrix& agg_agg,
                                           const linalgcpp::SparseMatrix<double>& face_agg_int)
{
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

    std::vector<double> data(indices.size(), 1);

    return linalgcpp::SparseMatrix<double>(std::move(indptr), std::move(indices), std::move(data),
                                        num_faces, num_aggs);
}

parlinalgcpp::ParMatrix MakeFaceTrueEdge(const parlinalgcpp::ParMatrix& face_face)
{
    const auto& offd = face_face.GetOffd();

    const auto& offd_indptr = offd.GetIndptr();
    const auto& offd_indices = offd.GetIndices();
    const auto& offd_colmap = face_face.GetColMap();

    HYPRE_Int last_row = face_face.GetColStarts()[1];

    int num_faces = face_face.Rows();
    std::vector<int> select_indptr(num_faces + 1);

    int num_true_faces = 0;

    for (int i = 0; i < num_faces; ++i)
    {
        select_indptr[i] = num_true_faces;

        int row_size = offd.RowSize(i);

        if (row_size == 0 || offd_colmap[offd_indices[offd_indptr[i]]] > last_row )
        {
            assert(row_size == 0 || row_size == 1);
            num_true_faces++;
        }
    }

    select_indptr[num_faces] = num_true_faces;

    std::vector<double> select_data(num_true_faces, 1.0);
    std::vector<int> select_indices(num_true_faces);
    std::iota(std::begin(select_indices), std::end(select_indices), 0);

    linalgcpp::SparseMatrix<double> select(std::move(select_indptr), std::move(select_indices), std::move(select_data),
                                           num_faces, num_true_faces);

    MPI_Comm comm = face_face.GetComm();
    auto face_true_starts = parlinalgcpp::GenerateOffsets(comm, num_true_faces);
    parlinalgcpp::ParMatrix select_d(comm, face_face.GetRowStarts(), face_true_starts, select);

    return face_face.Mult(select_d);
}

parlinalgcpp::ParMatrix MakeExtPermutation(MPI_Comm comm, const parlinalgcpp::ParMatrix& parmat)
{
    const auto& diag = parmat.GetDiag();
    const auto& offd = parmat.GetOffd();
    const auto& colmap = parmat.GetColMap();

    int num_diag = diag.Cols();
    int num_offd = offd.Cols();
    int num_ext = num_diag + num_offd;

    const auto& mat_starts = parmat.GetColStarts();
    auto ext_starts = parlinalgcpp::GenerateOffsets(comm, num_ext);

    linalgcpp::SparseMatrix<double> perm_diag = SparseIdentity(num_ext, num_diag);
    linalgcpp::SparseMatrix<double> perm_offd = SparseIdentity(num_ext, num_offd, num_diag);

    return parlinalgcpp::ParMatrix(comm, ext_starts, mat_starts, std::move(perm_diag), std::move(perm_offd), colmap);
}

linalgcpp::SparseMatrix<double> SparseIdentity(int size)
{
    assert(size >= 0);

    return linalgcpp::SparseMatrix<double>(std::vector<double>(size, 1.0));
}

linalgcpp::SparseMatrix<double> SparseIdentity(int rows, int cols, int row_offset, int col_offset)
{
    assert(rows >= 0);
    assert(cols >= 0);
    assert(row_offset <= rows);
    assert(row_offset >= 0);
    assert(col_offset <= cols);
    assert(col_offset >= 0);

    const int diag_size = std::min(rows - row_offset, cols - col_offset);

    std::vector<int> indptr(rows + 1);

    std::fill(std::begin(indptr), std::begin(indptr) + row_offset, 0);
    std::iota(std::begin(indptr) + row_offset, std::begin(indptr) + row_offset + diag_size, 0);
    std::fill(std::begin(indptr) + row_offset + diag_size, std::begin(indptr) + rows + 1, diag_size);

    std::vector<int> indices(diag_size);
    std::iota(std::begin(indices), std::begin(indices) + diag_size, col_offset);

    std::vector<double> data(diag_size, 1.0);

    return linalgcpp::SparseMatrix<double>(std::move(indptr), std::move(indices), std::move(data), rows, cols);
}

std::vector<int> GetExtDofs(const parlinalgcpp::ParMatrix& mat_ext, int row)
{
    const auto& diag = mat_ext.GetDiag();
    const auto& offd = mat_ext.GetOffd();

    auto diag_dofs = diag.GetIndices(row);
    auto offd_dofs = offd.GetIndices(row);

    int num_diag = diag_dofs.size();

    for (auto i : offd_dofs)
    {
        diag_dofs.push_back(i + num_diag);
    }

    return diag_dofs;
}

} // namespace smoothg
