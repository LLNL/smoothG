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
   \file

   \brief A class to manage shared entity communication

   This particular file contains specific instantiations for data types.
   You need to reimplement each of these routines for each datatype
   you want to communicate.
*/

#include "sharedentitycommunication.hpp"

namespace smoothg
{

template <>
void SharedEntityCommunication<mfem::DenseMatrix>::SetSizeSpecifier()
{
    size_specifier_ = 2;
}

template <>
void SharedEntityCommunication<mfem::DenseMatrix>::PackSendSizes(
    const mfem::DenseMatrix& mat, int* sizes)
{
    sizes[0] = mat.Height();
    sizes[1] = mat.Width();
}

template <>
void SharedEntityCommunication<mfem::DenseMatrix>::CopyData(
    mfem::DenseMatrix& copyto, const mfem::DenseMatrix& copyfrom)
{
    // todo: should just use operator=?
    copyto.SetSize(copyfrom.Height(), copyfrom.Width());
    memcpy(copyto.Data(), copyfrom.Data(),
           copyfrom.Height() * copyfrom.Width() * sizeof(double));
}

template <>
void SharedEntityCommunication<mfem::DenseMatrix>::SendData(
    const mfem::DenseMatrix& mat, int recipient, int tag, MPI_Request* request)
{
    MPI_Isend(mat.Data(), mat.Height() * mat.Width(), MPI_DOUBLE,
              recipient, tag, comm_, request);
}

template <>
void SharedEntityCommunication<mfem::DenseMatrix>::ReceiveData(
    mfem::DenseMatrix& mat, int* sizes, int sender, int tag,
    MPI_Request* request)
{
    const int rows = sizes[0];
    const int columns = sizes[1];
    mat.SetSize(rows, columns);
    MPI_Irecv(mat.Data(),
              rows * columns,
              MPI_DOUBLE,
              sender,
              tag,
              comm_,
              request);
}

template class SharedEntityCommunication<mfem::DenseMatrix>;

template <>
void SharedEntityCommunication<mfem::SparseMatrix>::SetSizeSpecifier()
{
    size_specifier_ = 3;
}

template <>
void SharedEntityCommunication<mfem::SparseMatrix>::PackSendSizes(
    const mfem::SparseMatrix& mat, int* sizes)
{
    sizes[0] = mat.Height();
    sizes[1] = mat.Width();
    sizes[2] = mat.NumNonZeroElems();
}

template <>
void SharedEntityCommunication<mfem::SparseMatrix>::CopyData(
    mfem::SparseMatrix& copyto, const mfem::SparseMatrix& copyfrom)
{
    mfem::SparseMatrix copyto_tmp(copyfrom, true);
    copyto_tmp.Finalize();
    copyto.Swap(copyto_tmp);
}

template <>
void SharedEntityCommunication<mfem::SparseMatrix>::SendData(
    const mfem::SparseMatrix& mat, int recipient, int tag, MPI_Request* request)
{
    MPI_Isend(mat.GetI(), mat.Height() + 1, MPI_INT,
              recipient, tag, comm_, request);
    MPI_Isend(mat.GetJ(), mat.NumNonZeroElems(), MPI_INT,
              recipient, tag + 1, comm_, request);
    MPI_Isend(mat.GetData(), mat.NumNonZeroElems(), MPI_DOUBLE,
              recipient, tag + 2, comm_, request);
}

template <>
void SharedEntityCommunication<mfem::SparseMatrix>::ReceiveData(
    mfem::SparseMatrix& mat, int* sizes, int sender, int tag,
    MPI_Request* request)
{
    const int rows = sizes[0];
    const int columns = sizes[1];
    const int nnz = sizes[2];

    int* mat_i = new int[rows + 1];
    int* mat_j = new int[nnz];
    double* mat_data = new double[nnz];

    MPI_Irecv(mat_i, rows + 1, MPI_INT, sender, tag, comm_, request);
    MPI_Irecv(mat_j, nnz, MPI_INT, sender, tag + 1, comm_, request);
    MPI_Irecv(mat_data, nnz, MPI_DOUBLE, sender, tag + 2, comm_, request);

    mfem::SparseMatrix mat_tmp(mat_i, mat_j, mat_data, rows, columns);
    mat.Swap(mat_tmp);
}

template class SharedEntityCommunication<mfem::SparseMatrix>;

template <>
void SharedEntityCommunication<mfem::Vector>::SetSizeSpecifier()
{
    size_specifier_ = 1;
}

template <>
void SharedEntityCommunication<mfem::Vector>::PackSendSizes(
    const mfem::Vector& vec, int* sizes)
{
    sizes[0] = vec.Size();
}

template <>
void SharedEntityCommunication<mfem::Vector>::CopyData(
    mfem::Vector& copyto, const mfem::Vector& copyfrom)
{
    copyto.SetSize(copyfrom.Size());
    copyto = copyfrom;
}

template <>
void SharedEntityCommunication<mfem::Vector>::SendData(
    const mfem::Vector& vec, int recipient, int tag, MPI_Request* request)
{
    MPI_Isend(vec.GetData(), vec.Size(), MPI_DOUBLE,
              recipient, tag, comm_, request);
}

template <>
void SharedEntityCommunication<mfem::Vector>::ReceiveData(
    mfem::Vector& vec, int* sizes, int sender, int tag, MPI_Request* request)
{
    const int size = sizes[0];
    vec.SetSize(size);
    MPI_Irecv(vec.GetData(),
              size,
              MPI_DOUBLE,
              sender,
              tag,
              comm_,
              request);
}

template class SharedEntityCommunication<mfem::Vector>;

} // namespace smoothg
