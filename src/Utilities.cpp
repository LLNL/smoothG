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

        if ((tedge >= tedge_begin) && (tedge < tedge_end))
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

    auto compare = [] (const std::pair<HYPRE_Int, int>& lhs,
                       const std::pair<HYPRE_Int, int>& rhs)
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
    [](int i) { return i > 0; });

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
        assert(indices[i] < static_cast<int>(marker.size()));

        marker[indices[i]] = i;
    }
}

void ClearMarker(std::vector<int>& marker, const std::vector<int>& indices)
{
    const int size = indices.size();

    for (int i = 0; i < size; ++i)
    {
        assert(indices[i] < static_cast<int>(marker.size()));

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
    assert(mat.Rows() == vect_view.size());

    // If the view is of mat, deflate will destroy it,
    // so copy is needed
    Vector vect(vect_view);
    Normalize(vect);

    if (mat.Rows() == 0 || mat.Cols() == 0)
    {
        DenseMatrix out(mat.Rows(), 1);
        out.SetCol(0, vect);

        return out;
    }

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

void OrthoConstant(VectorView vect)
{
    SubAvg(vect);
}

void OrthoConstant(MPI_Comm comm, VectorView vect, int global_size)
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

double DivError(MPI_Comm comm, const SparseMatrix& D, const VectorView& numer,
                const VectorView& denom)
{
    Vector sigma_diff(denom);
    sigma_diff -= numer;

    Vector Dfine = D.Mult(denom);
    Vector Ddiff = D.Mult(sigma_diff);

    const double error = parlinalgcpp::ParL2Norm(comm, Ddiff) /
                         parlinalgcpp::ParL2Norm(comm, Dfine);

    return error;
}

double CompareError(MPI_Comm comm, const VectorView& numer, const VectorView& denom)
{
    Vector diff(denom);
    diff -= numer;

    const double error = parlinalgcpp::ParL2Norm(comm, diff) /
                         parlinalgcpp::ParL2Norm(comm, denom);

    return error;
}

void ShowErrors(const std::vector<double>& error_info, std::ostream& out, bool pretty)
{
    assert(error_info.size() >= 3);

    std::map<std::string, double> values =
    {
        {"finest-p-error", error_info[0]},
        {"finest-u-error", error_info[1]},
        {"finest-div-error", error_info[2]}
    };

    if (error_info.size() > 3)
    {
        values.emplace(std::make_pair("operator-complexity", error_info[3]));
    }

    PrintJSON(values, out, pretty);
}

std::vector<double> ComputeErrors(MPI_Comm comm, const SparseMatrix& M,
                                  const SparseMatrix& D,
                                  const BlockVector& upscaled_sol,
                                  const BlockVector& fine_sol)
{
    BlockVector M_scaled_up_sol(upscaled_sol);
    BlockVector M_scaled_fine_sol(fine_sol);

    const std::vector<double>& M_data = M.GetData();

    const int num_edges = upscaled_sol.GetBlock(0).size();

    for (int i = 0; i < num_edges; ++i)
    {
        assert(M_data[i] >= 0);

        M_scaled_up_sol[i] *= std::sqrt(M_data[i]);
        M_scaled_fine_sol[i] *= std::sqrt(M_data[i]);
    }

    std::vector<double> info(3);

    info[0] = CompareError(comm, M_scaled_up_sol.GetBlock(1), M_scaled_fine_sol.GetBlock(1));  // vertex
    info[1] = CompareError(comm, M_scaled_up_sol.GetBlock(0), M_scaled_fine_sol.GetBlock(0));  // edge
    info[2] = DivError(comm, D, upscaled_sol.GetBlock(0), fine_sol.GetBlock(0));   // div

    return info;
}

void PrintJSON(const std::map<std::string, double>& values, std::ostream& out,
               bool pretty)
{
    const std::string new_line = pretty ? "\n" : "";
    const std::string indent = pretty ? "  " : "";
    std::stringstream ss;

    out << "{" << new_line;

    for (const auto& pair : values)
    {
        ss.str("");
        ss << indent << "\"" << std::right << pair.first << "\": "
           << std::left << std::setprecision(16) << pair.second;

        if (&pair != &(*values.rbegin()))
        {
            ss << std::left << ",";
        }

        ss << new_line;

        out << ss.str();
    }

    out << "}" << new_line;
}

SparseMatrix MakeProcAgg(int num_procs, int num_aggs_global)
{
    int num_aggs_local = num_aggs_global / num_procs;
    int num_left = num_aggs_global % num_procs;

    std::vector<int> indptr(num_procs + 1);
    std::vector<int> indices(num_aggs_global);
    std::vector<double> data(num_aggs_global, 1.0);

    std::iota(std::begin(indices), std::end(indices), 0);

    for (int i = 0; i <= num_left; ++i)
    {
        indptr[i] = i * (num_aggs_local + 1);
    }

    for (int i = num_left + 1; i <= num_procs; ++i)
    {
        indptr[i] = indptr[i - 1] + num_aggs_local;
    }

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                        num_procs, num_aggs_global);
}

SparseMatrix MakeAggVertex(const std::vector<int>& partition)
{
    assert(partition.size() > 0);

    const int num_parts = *std::max_element(std::begin(partition), std::end(partition)) + 1;
    const int num_vert = partition.size();

    std::vector<int> indptr(num_vert + 1);
    std::vector<double> data(num_vert, 1);

    std::iota(std::begin(indptr), std::end(indptr), 0);

    SparseMatrix vertex_agg(std::move(indptr), partition, std::move(data), num_vert, num_parts);

    return vertex_agg.Transpose();
}

double PowerIterate(MPI_Comm comm, const linalgcpp::Operator& A, VectorView result,
                    int max_iter, double tol, bool verbose)
{
    using parlinalgcpp::ParMult;
    using parlinalgcpp::ParL2Norm;

    int myid;
    MPI_Comm_rank(comm, &myid);

    Vector temp(result.size());

    double rayleigh = 0.0;
    double old_rayleigh = 0.0;

    for (int i = 0; i < max_iter; ++i)
    {
        A.Mult(result, temp);

        rayleigh = ParMult(comm, temp, result) / ParMult(comm, result, result);
        temp /= ParL2Norm(comm, temp);

        swap(temp, result);

        if (verbose && myid == 0)
        {
            std::cout << std::scientific;
            std::cout << " i: " << i << " ray: " << rayleigh;
            std::cout << " inverse: " << (1.0 / rayleigh);
            std::cout << " rate: " << (std::fabs(rayleigh - old_rayleigh) / rayleigh) << "\n";
        }

        if (std::fabs(rayleigh - old_rayleigh) / std::fabs(rayleigh) < tol)
        {
            break;
        }

        old_rayleigh = rayleigh;
    }

    return rayleigh;
}

void BroadCast(MPI_Comm comm, SparseMatrix& mat)
{
    int myid;
    MPI_Comm_rank(comm, &myid);

    int sizes[3];

    if (myid == 0)
    {
        sizes[0] = mat.Rows();
        sizes[1] = mat.Cols();
        sizes[2] = mat.nnz();
    }

    MPI_Bcast(sizes, 3, MPI_INT, 0, comm);

    bool master = (myid == 0);

    std::vector<int> indptr(master ? 0 : sizes[0] + 1);
    std::vector<int> indices(master ? 0 : sizes[2]);
    std::vector<double> data(master ? 0 : sizes[2]);

    int* I_ptr = master ? mat.GetIndptr().data() : indptr.data();
    int* J_ptr = master ? mat.GetIndices().data() : indices.data();
    double* Data_ptr = master ? mat.GetData().data() : data.data();

    MPI_Bcast(I_ptr, sizes[0] + 1, MPI_INT, 0, comm);
    MPI_Bcast(J_ptr, sizes[2], MPI_INT, 0, comm);
    MPI_Bcast(Data_ptr, sizes[2], MPI_DOUBLE, 0, comm);

    if (myid != 0)
    {
        mat = SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                           sizes[0], sizes[1]);
    }
}

void ExtractSubMatrix(const SparseMatrix& A, const std::vector<int>& rows,
                      const std::vector<int>& cols, const std::vector<int>& colMapper,
                      DenseMatrix& A_sub)
{
    const int num_row = rows.size();
    const int num_col = cols.size();

    const auto& A_i = A.GetIndptr();
    const auto& A_j = A.GetIndices();
    const auto& A_data = A.GetData();

    A_sub.SetSize(num_row, num_col, 0.0);

    for (int i = 0; i < num_row; ++i)
    {
        const int row = rows[i];

        for (int j = A_i[row]; j < A_i[row + 1]; ++j)
        {
            const int col = colMapper[A_j[j]];

            if (col >= 0)
            {
                A_sub(i, col) = A_data[j];
            }
        }
    }
}

void MultScalarVVt(double a, const VectorView& v, DenseMatrix& aVVt)
{
    int n = v.size();
    aVVt.SetSize(n, n);

    for (int i = 0; i < n; i++)
    {
        double avi = a * v[i];

        for (int j = 0; j < i; j++)
        {
            double avivj = avi * v[j];

            aVVt(i, j) = avivj;
            aVVt(j, i) = avivj;
        }

        aVVt(i, i) = avi * v[i];
    }
}

SparseMatrix AssembleElemMat(const SparseMatrix& elem_dof, const std::vector<DenseMatrix>& elems)
{
    int num_elem = elem_dof.Rows();
    int num_dof = elem_dof.Cols();

    CooMatrix coo(num_dof);

    for (int i = 0; i < num_elem; ++i)
    {
        std::vector<int> dofs = elem_dof.GetIndices(i);

        coo.Add(dofs, dofs, elems[i]);
    }

    return coo.ToSparse();
}

} // namespace smoothg
