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

void MultSparseDense(const mfem::SparseMatrix& A, mfem::DenseMatrix& B,
                     mfem::DenseMatrix& C)
{
    MFEM_ASSERT(C.Height() == A.Height() && C.Width() == B.Width() &&
                A.Width() == B.Height(), "incompatible dimensions");
    //    auto B_ref = const_cast<DenseMatrix&>(B);
    mfem::Vector column_in, column_out;
    for (int j = 0; j < B.Width(); ++j)
    {
        B.GetColumnReference(j, column_in);
        C.GetColumnReference(j, column_out);
        A.Mult(column_in, column_out);
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

void SetConstantValue(mfem::HypreParMatrix* pmat, double c)
{
    hypre_CSRMatrix* Diag = ((hypre_ParCSRMatrix*) *pmat)->diag;
    hypre_CSRMatrix* Offd = ((hypre_ParCSRMatrix*) *pmat)->offd;
    double* Diag_data = hypre_CSRMatrixData(Diag);
    double* Offd_data = hypre_CSRMatrixData(Offd);
    std::fill_n(Diag_data, Diag->num_nonzeros, c);
    std::fill_n(Offd_data, Offd->num_nonzeros, c);
}

unique_ptr<mfem::SparseMatrix> PartitionToMatrix(
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

    return make_unique<mfem::SparseMatrix>(
               aggregate_vertex_i, aggregate_vertex_j, aggregate_vertex_data,
               nparts, nvertices);
}

unique_ptr<mfem::SparseMatrix> SparseIdentity(int size)
{
    int* I = new int[size + 1];
    std::iota(I, I + size + 1, 0);
    int* J = new int[size];
    std::iota(J, J + size, 0);
    double* Data = new double[size];
    std::fill_n(Data, size, 1.0);

    return make_unique<mfem::SparseMatrix>(I, J, Data, size, size);
}

unique_ptr<mfem::SparseMatrix> ExtractRowAndColumns(
    const mfem::SparseMatrix& A, const mfem::Array<int>& rows,
    const mfem::Array<int>& cols, mfem::Array<int>& colMapper,
    bool colMapper_not_filled)
{
    if (rows.Size() == 0 || cols.Size() == 0)
    {
        auto out = make_unique<mfem::SparseMatrix>(rows.Size(), cols.Size());
        out->Finalize();
        return out;
    }

    const int* i_A = A.GetI();
    const int* j_A = A.GetJ();
    const double* a_A = A.GetData();

    assert(rows.Size() && rows.Max() < A.Height());
    assert(cols.Size() && cols.Max() < A.Width());
    assert(colMapper.Size() >= A.Width());

    if (colMapper_not_filled)
        for (int jcol(0); jcol < cols.Size(); ++jcol)
            colMapper[cols[jcol]] = jcol;

    const int nrow_sub = rows.Size();
    const int ncol_sub = cols.Size();

    int* i_sub = new int[nrow_sub + 1];
    i_sub[0] = 0;

    // Find the number of nnz.
    int currentRow(-1);
    int nnz(0);
    for (int i(0); i < nrow_sub; ++i)
    {
        currentRow = rows[i];
        for (const int* it = j_A + i_A[currentRow]; it != j_A + i_A[currentRow + 1]; ++it)
            if ( colMapper[*it] >= 0)
                ++nnz;

        i_sub[i + 1] = nnz;
    }

    // Allocate memory
    int* j_sub = new int[nnz];
    double* a_sub = new double[nnz];

    // Fill in the matrix
    const double* it_a;
    int* it_j_sub = j_sub;
    double* it_a_sub = a_sub;
    for (int i(0); i < nrow_sub; ++i)
    {
        currentRow = rows[i];
        it_a = a_A + i_A[currentRow];
        for (const int* it = j_A + i_A[currentRow]; it != j_A + i_A[currentRow + 1]; ++it, ++it_a)
            if ( colMapper[*it] >= 0)
            {
                *(it_j_sub++) = colMapper[*it];
                *(it_a_sub++) = *it_a;
            }
    }

    // Restore colMapper so it can be reused other times!
    if (colMapper_not_filled)
        for (int jcol(0); jcol < cols.Size(); ++jcol)
            colMapper[cols[jcol]] = -1;

    return make_unique<mfem::SparseMatrix>(i_sub, j_sub, a_sub,
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

    int irow, j;
    for (int i = 0; i < nrow; i++)
    {
        irow = rows[i];
        const double* a_it = A_data + A_i[irow];
        for (const int* j_it = A_j + A_i[irow]; j_it != A_j + A_i[irow + 1];
             ++j_it, ++a_it)
        {
            j = colMapper[*j_it];
            if (j >= 0)
                A_sub(i, j) = *a_it;
        }
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
    MinvDT_.reset( Transpose(D) );

    // Compute M^{-1}D^T (assuming M is diagonal)
    int* DT_i = MinvDT_->GetI();
    double* DT_data = MinvDT_->GetData();
    double scale;
    for (int i = 0; i < MinvDT_->Height(); i++)
    {
        scale = M_data[i];
        for (int j = DT_i[i]; j < DT_i[i + 1]; j++)
            DT_data[j] /= scale;
    }
    A_.reset( mfem::Mult(D, *MinvDT_) );

    // Eliminate the first unknown so that A_ is invertible
    A_->EliminateRowCol(0);
    solver_ = make_unique<mfem::UMFPackSolver>(*A_);
}

void LocalGraphEdgeSolver::Mult(const mfem::Vector& rhs, mfem::
                                Vector& sol_sigma)
{
    // Set rhs(0)=0 so that the modified system after
    // the elimination is consistent with the original one
    mfem::Vector& rhs_copy = const_cast<mfem::Vector&>(rhs);
    double rhs_0 = rhs_copy(0);
    rhs_copy(0) = 0.;

    mfem::Vector sol_u_tmp(A_->Size());
    sol_u_tmp = 0.;
    solver_->Mult(rhs_copy, sol_u_tmp);

    // Set rhs(0) back to its original vale (rhs is const Vector)
    rhs_copy(0) = rhs_0;

    // SparseMatrix::Mult asserts that sol.Size() should equal MinvDT_->Height()
    MinvDT_->Mult(sol_u_tmp, sol_sigma);
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
    MinvDT_->Mult(sol_u, sol_sigma);
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

} // namespace smoothg
