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

    These are implemented with and operate on MFEM data structures.
*/

#include "MatrixUtilities.hpp"
#include <assert.h>
#include "utilities.hpp"

using std::unique_ptr;

namespace smoothg
{

void Print(const mfem::DenseMatrix& mat, const std::string& label, std::ostream& out)
{
    const int old_precision = out.precision();
    out.precision(5);

    out << label << "\n";

    for (int i = 0; i < mat.Height(); ++i)
    {
        for (int j = 0; j < mat.Width(); ++j)
        {
            out << std::setw(5) << mat(i, j) << " ";
        }

        out << "\n";
    }

    out.precision(old_precision);
}

void Print(const mfem::SparseMatrix& mat, const std::string& label, std::ostream& out)
{
    mfem::DenseMatrix dense(mat.Height(), mat.Width());
    dense = 0.0;

    for (int i = 0; i < mat.Height(); ++i)
    {
        for (int j = mat.GetI()[i]; j < mat.GetI()[i + 1]; ++j)
        {
            dense(i, mat.GetJ()[j]) = mat.GetData()[j];
        }
    }

    Print(dense, label, out);
}


mfem::SparseMatrix Transpose(const mfem::SparseMatrix& A)
{
    std::unique_ptr<mfem::SparseMatrix> A_t_ptr(mfem::Transpose(A));
    mfem::SparseMatrix A_t;
    A_t.Swap(*A_t_ptr);

    return A_t;
}

mfem::SparseMatrix Mult(const mfem::SparseMatrix& A, const mfem::SparseMatrix& B)
{
    std::unique_ptr<mfem::SparseMatrix> C_ptr(mfem::Mult(A, B));
    mfem::SparseMatrix C;
    C.Swap(*C_ptr);

    return C;
}

mfem::SparseMatrix Threshold(const mfem::SparseMatrix& mat, double tol)
{
    // TODO(gelever1): Perform this more better

    mfem::SparseMatrix out(mat.Height(), mat.Width());

    for (int i = 0; i < mat.Height(); ++i)
    {
        for (int j = mat.GetI()[i]; j < mat.GetI()[i + 1]; ++j)
        {
            const double val = mat.GetData()[j];
            // if (std::fabs(val) > tol)
            // Keep diagonal always??
            if (std::fabs(val) > tol || i == mat.GetJ()[j])
            {
                out.Add(i, mat.GetJ()[j], val);
            }
        }
    }

    out.Finalize();

    return out;
}

mfem::SparseMatrix TableToMatrix(const mfem::Table& table)
{
    const int height = table.Size();
    const int width = table.Width();
    const int nnz = table.Size_of_connections();

    int* i = new int[height + 1];
    int* j = new int[nnz];
    double* data = new double[nnz];

    std::copy_n(table.GetI(), height + 1, i);
    std::copy_n(table.GetJ(), nnz, j);
    std::fill_n(data, nnz, 1.);

    return mfem::SparseMatrix(i, j, data, height, width);
}

mfem::Table MatrixToTable(const mfem::SparseMatrix& mat)
{
    const int nrows = mat.Height();
    const int nnz = mat.NumNonZeroElems();

    int* i = new int[nrows + 1];
    int* j = new int[nnz];

    std::copy_n(mat.GetI(), nrows + 1, i);
    std::copy_n(mat.GetJ(), nnz, j);

    mfem::Table table;
    table.SetIJ(i, j, nrows);
    return table;
}

mfem::HypreParMatrix* RAP(const mfem::HypreParMatrix& R, const mfem::HypreParMatrix& A,
                          const mfem::HypreParMatrix& P)
{
    // TODO(gelever1): Remove const cast once mfem version update
    unique_ptr<mfem::HypreParMatrix> RT(const_cast<mfem::HypreParMatrix&>(R).Transpose());
    assert(RT);
    unique_ptr<mfem::HypreParMatrix> AP(mfem::ParMult(const_cast<mfem::HypreParMatrix*>(&A),
                                                      const_cast<mfem::HypreParMatrix*>(&P)));

    mfem::HypreParMatrix* rap = mfem::ParMult(RT.get(), AP.get());
    assert(rap);

    rap->CopyRowStarts();
    //rap->CopyColStarts();
    hypre_ParCSRMatrixSetNumNonzeros(*rap);

    return rap;
}

mfem::HypreParMatrix* RAP(const mfem::HypreParMatrix& A, const mfem::HypreParMatrix& P)
{
    return RAP(P, A, P);
}

void BroadCast(MPI_Comm comm, mfem::SparseMatrix& mat)
{
    int myid;
    MPI_Comm_rank(comm, &myid);

    int sizes[3];
    if (myid == 0)
    {
        sizes[0] = mat.Height();
        sizes[1] = mat.Width();
        sizes[2] = mat.NumNonZeroElems();
    }
    MPI_Bcast(sizes, 3, MPI_INT, 0, comm);

    int* I;
    int* J;
    double* Data;

    if (myid == 0)
    {
        I = mat.GetI();
        J = mat.GetJ();
        Data = mat.GetData();
    }
    else
    {
        I = new int[sizes[0] + 1];
        J = new int[sizes[2]];
        Data = new double[sizes[2]];
    }

    MPI_Bcast(I, sizes[0] + 1, MPI_INT, 0, comm);
    MPI_Bcast(J, sizes[2], MPI_INT, 0, comm);
    MPI_Bcast(Data, sizes[2], MPI_DOUBLE, 0, comm);

    if (myid != 0)
    {
        mfem::SparseMatrix tmp(I, J, Data, sizes[0], sizes[1]);
        mat.Swap(tmp);
    }
}


void MultSparseDense(const mfem::SparseMatrix& A, mfem::DenseMatrix& B,
                     mfem::DenseMatrix& C)
{
    C.SetSize(A.Height(), B.Width());
    MFEM_ASSERT(A.Width() == B.Height(), "incompatible dimensions");

    mfem::Vector column_in, column_out;
    for (int j = 0; j < B.Width(); ++j)
    {
        B.GetColumnReference(j, column_in);
        C.GetColumnReference(j, column_out);
        A.Mult(column_in, column_out);
    }
}

void MultSparseDenseTranspose(const mfem::SparseMatrix& A, mfem::DenseMatrix& B,
                              mfem::DenseMatrix& C)
{
    MFEM_ASSERT(C.Width() == A.Height() && C.Height() == B.Width() &&
                A.Width() == B.Height(), "incompatible dimensions");

    const double* A_data = A.GetData();
    const int* A_i = A.GetI();
    const int* A_j = A.GetJ();

    const int A_height = A.Height();
    const int B_width = B.Width();

    for (int k = 0; k < B_width; ++k)
    {
        for (int i = 0; i < A_height; ++i)
        {
            double val = 0.0;

            for (int j = A_i[i]; j < A_i[i + 1]; ++j)
            {
                val += A_data[j] * B(A_j[j], k);
            }

            C(k, i) = val;
        }
    }
}

void Mult_a_VVt(const double a, const mfem::Vector& v, mfem::DenseMatrix& aVVt)
{
    int n = v.Size();

    assert(aVVt.Height() == n && aVVt.Width() == n);

    for (int i = 0; i < n; i++)
    {
        double avi = a * v(i);
        for (int j = 0; j < i; j++)
        {
            double avivj = avi * v(j);
            aVVt(i, j) = avivj;
            aVVt(j, i) = avivj;
        }
        aVVt(i, i) = avi * v(i);
    }
}

void SetConstantValue(mfem::HypreParMatrix& pmat, double c)
{
    pmat = c;
}

mfem::SparseMatrix PartitionToMatrix(
    const mfem::Array<int>& partition, int nparts)
{
    int nvertices = partition.Size();
    int* aggregate_vertex_i = new int[nparts + 1]();
    int* aggregate_vertex_j = new int[nvertices];
    double* aggregate_vertex_data = new double[nvertices];

    std::fill_n(aggregate_vertex_data, nvertices, 1.);

    for (int i = 0; i < nvertices; ++i)
        aggregate_vertex_i[partition[i] + 1]++;
    for (int i = 1; i < nparts; ++i)
        aggregate_vertex_i[i + 1] += aggregate_vertex_i[i];
    for (int i = 0; i < nvertices; ++i)
        aggregate_vertex_j[aggregate_vertex_i[partition[i]]++] = i;
    assert(aggregate_vertex_i[nparts - 1] == aggregate_vertex_i[nparts]);
    for (int i = nparts - 1; i > 0; --i)
        aggregate_vertex_i[i] = aggregate_vertex_i[i - 1];
    aggregate_vertex_i[0] = 0;

    return mfem::SparseMatrix(aggregate_vertex_i, aggregate_vertex_j, aggregate_vertex_data,
                              nparts, nvertices);
}

mfem::SparseMatrix SparseIdentity(int size)
{
    int* I = new int[size + 1];
    std::iota(I, I + size + 1, 0);
    int* J = new int[size];
    std::iota(J, J + size, 0);
    double* Data = new double[size];
    std::fill_n(Data, size, 1.0);

    return mfem::SparseMatrix(I, J, Data, size, size);
}

mfem::SparseMatrix SparseIdentity(int rows, int cols, int row_offset, int col_offset)
{
    assert(rows >= 0);
    assert(cols >= 0);
    assert(row_offset <= rows);
    assert(row_offset >= 0);
    assert(col_offset <= cols);
    assert(col_offset >= 0);

    const int diag_size = std::min(rows - row_offset, cols - col_offset);

    int* I = new int[rows + 1];
    std::fill(I, I + row_offset, 0);
    std::iota(I + row_offset, I + row_offset + diag_size, 0);
    std::fill(I + row_offset + diag_size, I + rows + 1, diag_size);

    int* J = new int[diag_size];
    std::iota(J, J + diag_size, col_offset);
    double* Data = new double[diag_size];
    std::fill_n(Data, diag_size, 1.0);

    return mfem::SparseMatrix(I, J, Data, rows, cols);
}

void Add(const double a, mfem::SparseMatrix& mat, const double b,
         const mfem::Vector& vec, const bool invert_vec)
{
    assert(mat.Height() == vec.Size());
    assert(mat.Width() == vec.Size());

    if (a != 1.0)
        mat *= a;

    if (invert_vec)
    {
        for (int i = 0; i < vec.Size() ; i++)
        {
            mat(i, i) += (b / vec(i));
        }
    }
    else
    {
        for (int i = 0; i < vec.Size() ; i++)
        {
            mat(i, i) += (b * vec(i));
        }
    }
}

void Add(mfem::SparseMatrix& mat, const mfem::Vector& vec, const bool invert_vec)
{
    smoothg::Add(1.0, mat, 1.0, vec, invert_vec);
}

mfem::SparseMatrix Mult_AtDA(const mfem::SparseMatrix& A, const mfem::Vector& D)
{
    std::unique_ptr<mfem::SparseMatrix> AtDA_ptr(mfem::Mult_AtDA(A, D));
    mfem::SparseMatrix AtDA;
    AtDA.Swap(*AtDA_ptr);

    return AtDA;
}

mfem::SparseMatrix VectorToMatrix(const mfem::Vector& vect)
{
    if (vect.Size() == 0)
    {
        return mfem::SparseMatrix();
    }

    const int size = vect.Size();

    int* I = new int[size + 1];
    int* J = new int[size];
    double* Data = new double[size];

    for (int i = 0; i < size; ++i)
    {
        Data[i] = vect[i];
    }

    std::iota(I, I + size + 1, 0);
    std::iota(J, J + size, 0);

    return mfem::SparseMatrix(I, J, Data, size, size);
}

// Modified from MFEM
mfem::HypreParMatrix* ParAdd(const mfem::HypreParMatrix& A_ref, const mfem::HypreParMatrix& B_ref)
{
    hypre_ParCSRMatrix* A(const_cast<mfem::HypreParMatrix&>(A_ref));
    hypre_ParCSRMatrix* B(const_cast<mfem::HypreParMatrix&>(B_ref));

    MPI_Comm            comm   = hypre_ParCSRMatrixComm(A);
    hypre_CSRMatrix*    A_diag = hypre_ParCSRMatrixDiag(A);
    hypre_CSRMatrix*    A_offd = hypre_ParCSRMatrixOffd(A);
    HYPRE_Int*          A_cmap = hypre_ParCSRMatrixColMapOffd(A);
    HYPRE_Int           A_cmap_size = hypre_CSRMatrixNumCols(A_offd);
    hypre_CSRMatrix*    B_diag = hypre_ParCSRMatrixDiag(B);
    hypre_CSRMatrix*    B_offd = hypre_ParCSRMatrixOffd(B);
    HYPRE_Int*          B_cmap = hypre_ParCSRMatrixColMapOffd(B);
    HYPRE_Int           B_cmap_size = hypre_CSRMatrixNumCols(B_offd);
    hypre_ParCSRMatrix* C;
    hypre_CSRMatrix*    C_diag;
    hypre_CSRMatrix*    C_offd;
    HYPRE_Int*          C_cmap;
    HYPRE_Int           im;
    HYPRE_Int           cmap_differ;

    /* Check if A_cmap and B_cmap are the same. */
    cmap_differ = 0;
    if (A_cmap_size != B_cmap_size)
    {
        cmap_differ = 1; /* A and B have different cmap_size */
    }
    else
    {
        for (im = 0; im < A_cmap_size; im++)
        {
            if (A_cmap[im] != B_cmap[im])
            {
                cmap_differ = 1; /* A and B have different cmap arrays */
                break;
            }
        }
    }

    if ( cmap_differ == 0 )
    {
        /* A and B have the same column mapping for their off-diagonal blocks so
           we can sum the diagonal and off-diagonal blocks separately and reduce
           temporary memory usage. */

        /* Add diagonals, off-diagonals, copy cmap. */
        C_diag = hypre_CSRMatrixAdd(A_diag, B_diag);
        if (!C_diag)
        {
            return NULL; /* error: A_diag and B_diag have different dimensions */
        }
        C_offd = hypre_CSRMatrixAdd(A_offd, B_offd);
        if (!C_offd)
        {
            hypre_CSRMatrixDestroy(C_diag);
            return NULL; /* error: A_offd and B_offd have different dimensions */
        }
        /* copy A_cmap -> C_cmap */
        C_cmap = hypre_TAlloc(HYPRE_Int, A_cmap_size);
        for (im = 0; im < A_cmap_size; im++)
        {
            C_cmap[im] = A_cmap[im];
        }

        C = hypre_ParCSRMatrixCreate(comm,
                                     hypre_ParCSRMatrixGlobalNumRows(A),
                                     hypre_ParCSRMatrixGlobalNumCols(A),
                                     hypre_ParCSRMatrixRowStarts(A),
                                     hypre_ParCSRMatrixColStarts(A),
                                     hypre_CSRMatrixNumCols(C_offd),
                                     hypre_CSRMatrixNumNonzeros(C_diag),
                                     hypre_CSRMatrixNumNonzeros(C_offd));

        /* In C, destroy diag/offd (allocated by Create) and replace them with
        C_diag/C_offd. */
        hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(C));
        hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(C));
        hypre_ParCSRMatrixDiag(C) = C_diag;
        hypre_ParCSRMatrixOffd(C) = C_offd;

        hypre_ParCSRMatrixColMapOffd(C) = C_cmap;
    }
    else
    {
        /* A and B have different column mappings for their off-diagonal blocks so
        we need to use the column maps to create full-width CSR matricies. */

        int  ierr = 0;
        hypre_CSRMatrix* csr_A;
        hypre_CSRMatrix* csr_B;
        hypre_CSRMatrix* csr_C_temp;

        /* merge diag and off-diag portions of A */
        csr_A = hypre_MergeDiagAndOffd(A);

        /* merge diag and off-diag portions of B */
        csr_B = hypre_MergeDiagAndOffd(B);

        /* add A and B */
        csr_C_temp = hypre_CSRMatrixAdd(csr_A, csr_B);

        /* delete CSR versions of A and B */
        ierr += hypre_CSRMatrixDestroy(csr_A);
        ierr += hypre_CSRMatrixDestroy(csr_B);

        /* create a new empty ParCSR matrix to contain the sum */
        C = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                     hypre_ParCSRMatrixGlobalNumRows(A),
                                     hypre_ParCSRMatrixGlobalNumCols(A),
                                     hypre_ParCSRMatrixRowStarts(A),
                                     hypre_ParCSRMatrixColStarts(A),
                                     0, 0, 0);

        /* split C into diag and off-diag portions */
        /* FIXME: GenerateDiagAndOffd() uses an int array of size equal to the
           number of columns in csr_C_temp which is the global number of columns
           in A and B. This does not scale well. */
        ierr += GenerateDiagAndOffd(csr_C_temp, C,
                                    hypre_ParCSRMatrixFirstColDiag(A),
                                    hypre_ParCSRMatrixLastColDiag(A));

        /* delete CSR version of C */
        ierr += hypre_CSRMatrixDestroy(csr_C_temp);

        assert(ierr == 0);
    }

    /* hypre_ParCSRMatrixSetNumNonzeros(A); */

    /* Make sure that the first entry in each row is the diagonal one. */
    hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(C));

    /* C owns diag, offd, and cmap. */
    hypre_ParCSRMatrixSetDataOwner(C, 1);
    /* C does not own row and column starts. */
    hypre_ParCSRMatrixSetRowStartsOwner(C, 0);
    hypre_ParCSRMatrixSetColStartsOwner(C, 0);

    return new mfem::HypreParMatrix(C);
}

double MaxNorm(const mfem::HypreParMatrix& A)
{
    mfem::SparseMatrix diag;
    mfem::SparseMatrix offd;

    HYPRE_Int* junk_map;
    A.GetDiag(diag);
    A.GetOffd(offd, junk_map);

    double local_max = std::max(diag.MaxNorm(), offd.MaxNorm());

    double global_max;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, A.GetComm());

    return global_max;
}

mfem::SparseMatrix ExtractRowAndColumns(
    const mfem::SparseMatrix& A, const mfem::Array<int>& rows,
    const mfem::Array<int>& cols, mfem::Array<int>& colMapper,
    bool colMapper_not_filled)
{
    if (rows.Size() == 0 || cols.Size() == 0)
    {
        mfem::SparseMatrix out(rows.Size(), cols.Size());
        out.Finalize();
        return out;
    }

    const int* i_A = A.GetI();
    const int* j_A = A.GetJ();
    const double* a_A = A.GetData();

    assert(rows.Size() && rows.Max() < A.Height());
    assert(cols.Size() && cols.Max() < A.Width());
    assert(colMapper.Size() >= A.Width());

    if (colMapper_not_filled)
    {
        for (int jcol(0); jcol < cols.Size(); ++jcol)
            colMapper[cols[jcol]] = jcol;
    }

    const int nrow_sub = rows.Size();
    const int ncol_sub = cols.Size();

    int* i_sub = new int[nrow_sub + 1];
    i_sub[0] = 0;

    // Find the number of nnz.
    int nnz = 0;
    for (int i = 0; i < nrow_sub; ++i)
    {
        const int current_row = rows[i];

        for (int j = i_A[current_row]; j < i_A[current_row + 1]; ++j)
        {
            if (colMapper[j_A[j]] >= 0)
                ++nnz;
        }

        i_sub[i + 1] = nnz;
    }

    // Allocate memory
    int* j_sub = new int[nnz];
    double* a_sub = new double[nnz];

    // Fill in the matrix
    int count = 0;
    for (int i(0); i < nrow_sub; ++i)
    {
        const int current_row = rows[i];

        for (int j = i_A[current_row]; j < i_A[current_row + 1]; ++j)
        {
            if (colMapper[j_A[j]] >= 0)
            {
                j_sub[count] = colMapper[j_A[j]];
                a_sub[count] = a_A[j];
                count++;
            }
        }
    }

    assert(count == nnz);

    // Restore colMapper so it can be reused other times!
    if (colMapper_not_filled)
    {
        for (int jcol(0); jcol < cols.Size(); ++jcol)
            colMapper[cols[jcol]] = -1;
    }

    return mfem::SparseMatrix(i_sub, j_sub, a_sub,
                              nrow_sub, ncol_sub);
}

void ExtractSubMatrix(
    const mfem::SparseMatrix& A, const mfem::Array<int>& rows,
    const mfem::Array<int>& cols, const mfem::Array<int>& colMapper,
    mfem::DenseMatrix& A_sub)
{
    const int nrow = rows.Size();
    const int ncol = cols.Size();

    const int* A_i = A.GetI();
    const int* A_j = A.GetJ();
    const double* A_data = A.GetData();

    A_sub.SetSize(nrow, ncol);
    A_sub = 0.;

    for (int i = 0; i < nrow; ++i)
    {
        const int row = rows[i];

        for (int j = A_i[row]; j < A_i[row + 1]; ++j)
        {
            const int col = colMapper[A_j[j]];

            if (col >= 0)
                A_sub(i, col) = A_data[j];
        }
    }
}

void ExtractColumns(
    const mfem::DenseMatrix& A, const mfem::Array<int>& ref_to_col,
    const mfem::Array<int>& subcol_to_ref, mfem::DenseMatrix& A_sub,
    const int row_offset)
{
    const int A_width = A.Width();
    const int A_height = A.Height();
    const int A_sub_height = A_sub.Height();

    assert((A_height + row_offset) <= A_sub_height);

    for (int j = 0; j < subcol_to_ref.Size(); ++j)
    {
        int A_col = ref_to_col[subcol_to_ref[j]];
        assert(A_col >= 0);
        assert(A_col < A_width);
        std::copy_n(A.Data() + A_col * A_height, A_height,
                    A_sub.Data() + j * A_sub_height + row_offset);
    }
}

void Full(const mfem::SparseMatrix& Asparse, mfem::DenseMatrix& Adense)
{
    const int nrow = Asparse.Size();
    const int ncol = Asparse.Width();

    Adense.SetSize(nrow, ncol);
    Adense = 0.;

    const int* A_i = Asparse.GetI();
    const int* A_j = Asparse.GetJ();
    const double* A_data = Asparse.GetData();

    int jcol = 0;
    int end;

    for (int irow(0); irow < nrow; ++irow)
        for (end = A_i[irow + 1]; jcol != end; ++jcol)
            Adense(irow, A_j[jcol]) = A_data[jcol];
}

void FullTranspose(const mfem::SparseMatrix& Asparse, mfem::DenseMatrix& AdenseT)
{
    const int nrow = Asparse.Size();
    const int ncol = Asparse.Width();

    AdenseT.SetSize(ncol, nrow);
    AdenseT = 0.;

    const int* A_i = Asparse.GetI();
    const int* A_j = Asparse.GetJ();
    const double* A_data = Asparse.GetData();

    const int* j_it = A_j;
    const double* a_it = A_data;
    const int* end;

    for (int irow(0); irow < nrow; ++irow)
        for (end = A_j + A_i[irow + 1]; j_it != end; ++j_it, ++a_it)
            AdenseT(*j_it, irow) = *a_it;
}

void Concatenate(const mfem::Vector& a, const mfem::DenseMatrix& b,
                 mfem::DenseMatrix& C)
{
    int nrow_a = a.Size();
    int nrow_C = C.Height();
    int ncol_C = C.Width();

    assert(nrow_a == b.Height());
    assert(nrow_a == nrow_C);
    assert(ncol_C <= b.Width() + 1);

    double* a_data = a.GetData();
    double* C_data = C.Data();

    assert(a_data);
    assert(C_data);

    C_data = std::copy(a_data, a_data + nrow_a, C_data);
    double* b_data = b.Data();
    int hw = nrow_C * (ncol_C - 1);
    if (hw > 0)
        std::copy(b_data, b_data + hw, C_data);
}

// Note: v is assumed to be a unit vector
void Deflate(mfem::DenseMatrix& a, const mfem::Vector& v)
{
    int nrow = a.Height();
    int ncol = a.Width();

    mfem::DenseMatrix v_row_vec(v.GetData(), 1, nrow);
    mfem::DenseMatrix scale(1, ncol);
    mfem::Mult(v_row_vec, a, scale);

    // a -= vv^Ta
    double* ad = a.Data();
    double* vd;
    double* sd = scale.Data();
    for (int j = 0; j < ncol; j++)
    {
        vd = v.GetData();
        for (int i = 0; i < nrow; i++)
            *(ad++) -= *(vd++) * (*sd);
        sd++;
    }
}

void orthogonalize_from_constant(mfem::Vector& vec)
{
    vec -= vec.Sum() / vec.Size();
}

/// @todo MPI_COMM_WORLD should be more generic
void par_orthogonalize_from_constant(mfem::Vector& vec, int globalsize)
{
    double localsum = vec.Sum();
    double globalsum;
    MPI_Allreduce(&localsum, &globalsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    vec -= globalsum / globalsize;
}

void par_orthogonalize_from_constant(mfem::HypreParVector& vec)
{
    par_orthogonalize_from_constant(vec, vec.GlobalSize());
}

std::vector<mfem::Vector> get_blocks(
    const std::vector<std::unique_ptr<mfem::BlockVector>>& blockvecs,
    int block_num)
{
    std::vector<mfem::Vector> vecs;
    for (auto const& bv : blockvecs)
    {
        vecs.emplace_back(bv->GetBlock(block_num));
    }
    return vecs;
}

mfem::DenseMatrix get_sq_differences_matrix(
    const std::vector<mfem::Vector>& vecs,
    const mfem::SparseMatrix* inner_prod_mat,
    bool diag_sq_norms)
{
    unsigned int numvecs = vecs.size();
    int vecsize = vecs[0].Size();

#ifdef SMOOTHG_DEBUG
    if (vecs.size() < 2)
    {
        throw std::runtime_error("get_sq_differences_matrix: Must use a vecs "
                                 "vector of size at least 2.");
    }
    for (unsigned int i = 1; i < numvecs; i++)
    {
        if (vecs[i].Size() != vecsize)
        {
            throw std::runtime_error("get_sq_differences_matrix: Not all "
                                     "vectors have the same size.");
        }
    }
#endif // SMOOTHG_DEBUG

    mfem::Vector diff(vecsize);
    mfem::DenseMatrix sq_diff_mat(numvecs, numvecs);
    sq_diff_mat = 0.0;
    for (unsigned int i = 0; i < numvecs; i++)
    {
        for (unsigned int j = 0; j < i; j++)
        {
            subtract(vecs[i], vecs[j], diff);
            if (inner_prod_mat)
                sq_diff_mat(i, j) = inner_prod_mat->InnerProduct(diff, diff);
            else
                sq_diff_mat(i, j) = diff * diff;
        }
        if (diag_sq_norms)
        {
            if (inner_prod_mat)
                sq_diff_mat(i, i) = inner_prod_mat->InnerProduct(vecs[i], vecs[i]);
            else
                sq_diff_mat(i, i) = vecs[i] * vecs[i];
        }
    }
    return sq_diff_mat;
}

void GenerateOffsets(MPI_Comm comm, int N, HYPRE_Int loc_sizes[],
                     mfem::Array<HYPRE_Int>* offsets[])
{
    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    if (HYPRE_AssumedPartitionCheck())
    {
        mfem::Array<HYPRE_Int> temp(N);
        MPI_Scan(loc_sizes, temp.GetData(), N, HYPRE_MPI_INT, MPI_SUM, comm);
        for (int i = 0; i < N; i++)
        {
            offsets[i]->SetSize(3);
            (*offsets[i])[0] = temp[i] - loc_sizes[i];
            (*offsets[i])[1] = temp[i];
        }
        MPI_Bcast(temp.GetData(), N, HYPRE_MPI_INT, num_procs - 1, comm);
        for (int i = 0; i < N; i++)
        {
            (*offsets[i])[2] = temp[i];
            // check for overflow (TODO: define our own verify)
            MFEM_VERIFY((*offsets[i])[0] >= 0 && (*offsets[i])[1] >= 0,
                        "overflow in offsets");
        }
    }
    else
    {
        mfem::Array<HYPRE_Int> temp(N * num_procs);
        MPI_Allgather(loc_sizes, N, HYPRE_MPI_INT, temp.GetData(), N,
                      HYPRE_MPI_INT, comm);
        for (int i = 0; i < N; i++)
        {
            mfem::Array<HYPRE_Int>& offs = *offsets[i];
            offs.SetSize(num_procs + 1);
            offs[0] = 0;
            for (int j = 0; j < num_procs; j++)
            {
                offs[j + 1] = offs[j] + temp[i + N * j];
            }
            // Check for overflow (TODO: define our own verify)
            MFEM_VERIFY(offs[myid] >= 0 && offs[myid + 1] >= 0,
                        "overflow in offsets");
        }
    }
}

void GenerateOffsets(MPI_Comm comm, int local_size, mfem::Array<HYPRE_Int>& offsets)
{
    mfem::Array<HYPRE_Int>* start[1] = {&offsets};
    const int N = 1;

    GenerateOffsets(comm, N, &local_size, start);
}

// This constructor assumes M to be diagonal
LocalGraphEdgeSolver::LocalGraphEdgeSolver(const mfem::SparseMatrix& M,
                                           const mfem::SparseMatrix& D)
{
    double* M_data = M.GetData();
    Init(M_data, D);
}

// This constructor takes the diagonal of M (as a Vector) as input
LocalGraphEdgeSolver::LocalGraphEdgeSolver(const mfem::Vector& M,
                                           const mfem::SparseMatrix& D)
{
    double* M_data = M.GetData();
    Init(M_data, D);
}

void LocalGraphEdgeSolver::Init(double* M_data, const mfem::SparseMatrix& D)
{
    mfem::SparseMatrix DT(smoothg::Transpose(D));
    MinvDT_.Swap(DT);

    // Compute M^{-1}D^T (assuming M is diagonal)
    int* DT_i = MinvDT_.GetI();
    double* DT_data = MinvDT_.GetData();

    for (int i = 0; i < MinvDT_.Height(); i++)
    {
        const double scale = M_data[i];
        for (int j = DT_i[i]; j < DT_i[i + 1]; j++)
            DT_data[j] /= scale;
    }

    // TODO(gelever1): change all the swaps once mfem version > PR #352
    unique_ptr<mfem::SparseMatrix> tmp(mfem::Mult(D, MinvDT_));
    A_.Swap(*tmp);

    // Eliminate the first unknown so that A_ is invertible
    A_.EliminateRowCol(0);
    solver_ = make_unique<mfem::UMFPackSolver>(A_);
}

void LocalGraphEdgeSolver::Mult(const mfem::Vector& rhs,
                                mfem::Vector& sol_sigma)
{
    // Set rhs(0)=0 so that the modified system after
    // the elimination is consistent with the original one
    mfem::Vector& rhs_copy = const_cast<mfem::Vector&>(rhs);
    double rhs_0 = rhs_copy(0);
    rhs_copy(0) = 0.;

    mfem::Vector sol_u_tmp(A_.Size());
    sol_u_tmp = 0.;
    solver_->Mult(rhs_copy, sol_u_tmp);

    // Set rhs(0) back to its original vale (rhs is const Vector)
    rhs_copy(0) = rhs_0;

    // SparseMatrix::Mult asserts that sol.Size() should equal MinvDT_->Height()
    MinvDT_.Mult(sol_u_tmp, sol_sigma);
}

void LocalGraphEdgeSolver::Mult(const mfem::Vector& rhs,
                                mfem::Vector& sol_sigma, mfem::Vector& sol_u)
{
    // Set rhs(0)=0 so that the modified system after
    // the elimination is consistent with the original one
    mfem::Vector& rhs_copy = const_cast<mfem::Vector&>(rhs);
    double rhs_0 = rhs_copy(0);
    rhs_copy(0) = 0.;

    sol_u = 0.;
    solver_->Mult(rhs_copy, sol_u);
    orthogonalize_from_constant(sol_u);

    // Set rhs(0) back to its original vale (rhs is const Vector)
    rhs_copy(0) = rhs_0;

    // SparseMatrix::Mult asserts that sol.Size() should equal MinvDT_->Height()
    MinvDT_.Mult(sol_u, sol_sigma);
}

double InnerProduct(const mfem::Vector& weight, const mfem::Vector& u,
                    const mfem::Vector& v)
{
    double out = 0.;
    double* w_data = weight.GetData();
    double* u_data = u.GetData();
    double* v_data = v.GetData();
    for (int i = 0; i < weight.Size(); i++)
        out += (*w_data++) * (*u_data++) * (*v_data++);
    return out;
}

double InnerProduct(const mfem::Vector& u, const mfem::Vector& v)
{
    double out = 0.;
    double* u_data = u.GetData();
    double* v_data = v.GetData();
    for (int i = 0; i < u.Size(); i++)
        out += (*u_data++) * (*v_data++);
    return out;
}

std::unique_ptr<mfem::HypreParMatrix> BuildEntityToTrueEntity(
    const mfem::HypreParMatrix& entity_trueentity_entity)
{
    hypre_ParCSRMatrix* entity_shared = entity_trueentity_entity;
    HYPRE_Int* entity_shared_i = entity_shared->offd->i;
    HYPRE_Int* entity_shared_j = entity_shared->offd->j;
    HYPRE_Int* entity_shared_map = entity_shared->col_map_offd;
    HYPRE_Int max_entity = entity_shared->last_row_index;

    // Diagonal part
    int nentities = entity_trueentity_entity.Width();
    int* select_i = new int[nentities + 1];
    int ntrueentities = 0;
    for (int i = 0; i < nentities; i++)
    {
        select_i[i] = ntrueentities;
        int j_offset = entity_shared_i[i];
        if (entity_shared_i[i + 1] == j_offset)
            ntrueentities++;
        else if (entity_shared_map[entity_shared_j[j_offset]] > max_entity)
            ntrueentities++;
    }
    select_i[nentities] = ntrueentities;
    int* select_j = new int[ntrueentities];
    double* select_data = new double[ntrueentities];
    std::iota(select_j, select_j + ntrueentities, 0);
    std::fill_n(select_data, ntrueentities, 1.);
    mfem::SparseMatrix select_diag(select_i, select_j, select_data,
                                   nentities, ntrueentities);

    // Construct a "block diagonal" global select matrix from local
    auto comm = entity_trueentity_entity.GetComm();
    mfem::Array<int> trueentity_starts;
    GenerateOffsets(comm, ntrueentities, trueentity_starts);

    mfem::HypreParMatrix select(
        comm, entity_shared->global_num_rows, trueentity_starts.Last(),
        entity_shared->row_starts, trueentity_starts, &select_diag);

    auto out = mfem::ParMult(&entity_trueentity_entity, &select);
    out->CopyRowStarts();
    out->CopyColStarts();

    return unique_ptr<mfem::HypreParMatrix>(out);
}

} // namespace smoothg
