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

int MyId(MPI_Comm comm)
{
    int myid;
    MPI_Comm_rank(comm, &myid);

    return myid;
}

SparseMatrix MakeLocalM(const ParMatrix& edge_true_edge,
                        const ParMatrix& edge_edge,
                        const std::vector<int>& edge_map,
                        const std::vector<double>& global_weight)
{
    int size = edge_map.size();

    std::vector<double> local_weight(size);

    if (global_weight.size() == edge_true_edge.Cols())
    {
        for (int i = 0; i < size; ++i)
        {
            assert(std::fabs(global_weight[edge_map[i]]) > 1e-12);
            local_weight[i] = 1.0 / std::fabs(global_weight[edge_map[i]]);
        }
    }
    else
    {
        std::fill(std::begin(local_weight), std::end(local_weight), 1.0);
    }

    const SparseMatrix& edge_offd = edge_edge.GetOffd();

    assert(edge_offd.Rows() == local_weight.size());

    for (int i = 0; i < size; ++i)
    {
        if (edge_offd.RowSize(i))
        {
            local_weight[i] /= 2.0;
        }
    }

    return SparseMatrix(std::move(local_weight));
}

SparseMatrix MakeLocalD(const ParMatrix& edge_true_edge,
                          const SparseMatrix& vertex_edge)
{
    SparseMatrix edge_vertex = vertex_edge.Transpose();

    std::vector<int> indptr = edge_vertex.GetIndptr();
    std::vector<int> indices = edge_vertex.GetIndices();
    std::vector<double> data = edge_vertex.GetData();

    int num_edges = edge_vertex.Rows();
    int num_vertices = edge_vertex.Cols();

    const SparseMatrix& owned_edges = edge_true_edge.GetDiag();

    for (int i = 0; i < num_edges; i++)
    {
        const int row_edges = edge_vertex.RowSize(i);
        assert(row_edges == 1 || row_edges == 2);

        data[indptr[i]] = 1.;

        if (row_edges == 2)
        {
            data[indptr[i] + 1] = -1.;
        }
        else if (owned_edges.RowSize(i) == 0)
        {
            assert(row_edges == 1);
            data[indptr[i]] = -1.;
        }
    }

    SparseMatrix DT(std::move(indptr), std::move(indices), std::move(data), num_edges, num_vertices);

    return DT.Transpose();
}

ParMatrix MakeEdgeTrueEdge(MPI_Comm comm, const SparseMatrix& proc_edge,
                                         const std::vector<int>& edge_map)
{
    int myid;
    int num_procs;

    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    auto edge_proc = proc_edge.Transpose();

    int num_edges_local = proc_edge.RowSize(myid);
    int num_tedges_global = proc_edge.Cols();

    std::vector<int> tedge_counter(num_procs + 1, 0);

    for (int i = 0; i < num_tedges_global; ++i)
    {
        tedge_counter[edge_proc.GetIndices(i)[0] + 1]++;
    }

    int num_tedges_local = tedge_counter[myid + 1];
    int num_edge_diff = num_edges_local - num_tedges_local;
    std::partial_sum(std::begin(tedge_counter), std::end(tedge_counter),
            std::begin(tedge_counter));

    assert(tedge_counter.back() == static_cast<int>(num_tedges_global));

    std::vector<int> edge_perm(num_tedges_global);

    for (int i = 0; i < num_tedges_global; ++i)
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

    for (int i = 0; i < num_edges_local; ++i)
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

    SparseMatrix diag(std::move(diag_indptr), std::move(diag_indices), std::move(diag_data),
            num_edges_local, num_tedges_local);

    SparseMatrix offd(std::move(offd_indptr), std::move(offd_indices), std::move(offd_data),
            num_edges_local, num_edge_diff);

    return ParMatrix(comm, starts[0], starts[1],
            std::move(diag), std::move(offd),
            std::move(col_map));
}

SparseMatrix RestrictInterior(const SparseMatrix& mat)
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

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data), rows, cols);
}

ParMatrix RestrictInterior(const ParMatrix& mat)
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

    SparseMatrix diag = RestrictInterior(diag_ext);
    SparseMatrix offd(std::move(indptr), std::move(indices), std::move(data),
            num_rows, offd_num_cols);

    return ParMatrix(mat.GetComm(), mat.GetRowStarts(), mat.GetColStarts(),
            std::move(diag), std::move(offd), std::move(col_map));
}

SparseMatrix MakeFaceAggInt(const ParMatrix& agg_agg)
{
    const auto& agg_agg_diag = agg_agg.GetDiag();
    const auto& agg_agg_offd = agg_agg.GetOffd();

    int num_aggs = agg_agg_diag.Rows();
    int num_faces = agg_agg_diag.nnz() - agg_agg_diag.Rows();

    assert(num_faces % 2 == 0);
    num_faces /= 2;

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

    assert(count == num_faces);

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
            num_faces, num_aggs);
}

SparseMatrix MakeFaceEdge(const ParMatrix& agg_agg,
        const ParMatrix& edge_ext_agg,
        const SparseMatrix& agg_edge_ext,
        const SparseMatrix& face_edge_ext)
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

    const auto& edge_offd_indptr = edge_ext_agg.GetOffd().GetIndptr();
    const auto& edge_offd_indices = edge_ext_agg.GetOffd().GetIndices();
    const auto& edge_colmap = edge_ext_agg.GetColMap();

    for (int i = 0; i < num_aggs; ++i)
    {
        for (int j = agg_offd_indptr[i]; j < agg_offd_indptr[i + 1]; ++j)
        {
            int shared = agg_colmap[agg_offd_indices[j]];

            for (int k = agg_edge_indptr[i]; k < agg_edge_indptr[i + 1]; ++k)
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

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
            num_faces, num_edges);
}

SparseMatrix ExtendFaceAgg(const ParMatrix& agg_agg,
        const SparseMatrix& face_agg_int)
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

    int num_faces = indptr.size() - 1;

    std::vector<double> data(indices.size(), 1);

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
            num_faces, num_aggs);
}

ParMatrix MakeEntityTrueEntity(const ParMatrix& entity_entity)
{
    const auto& offd = entity_entity.GetOffd();

    const auto& offd_indptr = offd.GetIndptr();
    const auto& offd_indices = offd.GetIndices();
    const auto& offd_colmap = entity_entity.GetColMap();

    HYPRE_Int last_row = entity_entity.GetColStarts()[1];

    int num_entities = entity_entity.Rows();
    std::vector<int> select_indptr(num_entities + 1);

    int num_true_entities = 0;

    for (int i = 0; i < num_entities; ++i)
    {
        select_indptr[i] = num_true_entities;

        int row_size = offd.RowSize(i);

        if (row_size == 0 || offd_colmap[offd_indices[offd_indptr[i]]] >= last_row )
        {
            assert(row_size == 0 || row_size == 1);
            num_true_entities++;
        }
    }

    select_indptr[num_entities] = num_true_entities;

    std::vector<int> select_indices(num_true_entities);
    std::iota(std::begin(select_indices), std::end(select_indices), 0);

    std::vector<double> select_data(num_true_entities, 1.0);

    SparseMatrix select(std::move(select_indptr), std::move(select_indices), std::move(select_data),
            num_entities, num_true_entities);

    MPI_Comm comm = entity_entity.GetComm();
    auto true_starts = parlinalgcpp::GenerateOffsets(comm, num_true_entities);

    ParMatrix select_d(comm, entity_entity.GetRowStarts(), true_starts, std::move(select));

    return entity_entity.Mult(select_d);
}

ParMatrix MakeExtPermutation(const ParMatrix& parmat)
{
    MPI_Comm comm = parmat.GetComm();

    const auto& diag = parmat.GetDiag();
    const auto& offd = parmat.GetOffd();
    const auto& colmap = parmat.GetColMap();

    int num_diag = diag.Cols();
    int num_offd = offd.Cols();
    int num_ext = num_diag + num_offd;

    const auto& mat_starts = parmat.GetColStarts();
    auto ext_starts = parlinalgcpp::GenerateOffsets(comm, num_ext);

    SparseMatrix perm_diag = SparseIdentity(num_ext, num_diag);
    SparseMatrix perm_offd = SparseIdentity(num_ext, num_offd, num_diag);

    return ParMatrix(comm, ext_starts, mat_starts, std::move(perm_diag), std::move(perm_offd), colmap);
}

SparseMatrix SparseIdentity(int size)
{
    assert(size >= 0);

    return SparseMatrix(std::vector<double>(size, 1.0));
}

SparseMatrix SparseIdentity(int rows, int cols, int row_offset, int col_offset)
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

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data), rows, cols);
}

std::vector<int> GetExtDofs(const ParMatrix& mat_ext, int row)
{
    const auto& diag = mat_ext.GetDiag();
    const auto& offd = mat_ext.GetOffd();

    auto diag_dofs = diag.GetIndices(row);
    auto offd_dofs = offd.GetIndices(row);

    int diag_size = diag.Cols();

    for (auto i : offd_dofs)
    {
        diag_dofs.push_back(i + diag_size);
    }

    return diag_dofs;
}

void SetMarker(std::vector<int>& marker, const std::vector<int>& indices)
{
    const int size = indices.size();

    for (int i = 0; i < size; ++i)
    {
        assert(indices[i] < marker.size());

        marker[indices[i]] = i;
    }
}

void ClearMarker(std::vector<int>& marker, const std::vector<int>& indices)
{
    const int size = indices.size();

    for (int i = 0; i < size; ++i)
    {
        assert(indices[i] < marker.size());

        marker[indices[i]] = -1;
    }
}

DenseMatrix Orthogonalize(DenseMatrix& mat, int max_keep)
{
    VectorView vect = mat.GetColView(0);

    return Orthogonalize(mat, vect, max_keep);
}

DenseMatrix Orthogonalize(DenseMatrix& mat, const VectorView& vect_view, int max_keep)
{
    if (mat.Rows() == 0 || mat.Cols() == 0)
    {
        return mat;
    }

    // If the view is of mat, deflate will destroy it,
    // so copy is needed
    Vector vect(vect_view);
    Normalize(vect);

    Deflate(mat, vect);

    auto singular_values = mat.SVD();

    const double tol = singular_values.front() * 1e-8;
    int keep = 0;

    if (max_keep < 0)
    {
        max_keep = mat.Cols();
    }

    max_keep -= 1;

    while (keep < max_keep && singular_values[keep] > tol)
    {
        keep++;
    }

    DenseMatrix out(mat.Rows(), keep + 1);

    out = -1.0;

    out.SetCol(0, vect);

    for (int i = 0; i < keep; ++i)
    {
        auto col = mat.GetColView(i);
        out.SetCol(i + 1, col);
    }

    return out;
}

void OrthoConstant(DenseMatrix& mat)
{
    int cols = mat.Cols();

    for (int i = 0; i < cols; ++i)
    {
        VectorView col = mat.GetColView(i);
        SubAvg(col);
    }
}

void OrthoConstant(VectorView& vect)
{
    SubAvg(vect);
}

void OrthoConstant(MPI_Comm comm, VectorView& vect, int global_size)
{
    double local_sum = vect.Sum();
    double global_sum = 0.0;

    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

    vect -= global_sum / global_size;
}

void Deflate(DenseMatrix& A, const VectorView& v)
{
    int rows = A.Rows();
    int cols = A.Cols();

    assert(v.size() == rows);

    Vector v_T_A = A.MultAT(v);

    for (int j = 0; j < cols; ++j)
    {
        double vt_A_j = v_T_A[j];

        for (int i = 0; i < rows; ++i)
        {
            A(i, j) -= v[i] * vt_A_j;
        }
    }
}

DenseMatrix RestrictLocal(const DenseMatrix& ext_mat,
                          std::vector<int>& global_marker,
                          const std::vector<int>& ext_indices,
                          const std::vector<int>& local_indices)
{
    SetMarker(global_marker, ext_indices);

    int local_size = local_indices.size();

    std::vector<int> row_map(local_size);

    for (int i = 0; i < local_size; ++i)
    {
        assert(global_marker[local_indices[i]] >= 0);
        row_map[i] = global_marker[local_indices[i]];
    }

    ClearMarker(global_marker, ext_indices);

    return ext_mat.GetRow(row_map);
}

} // namespace smoothg
