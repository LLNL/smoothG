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

   @brief A class to manage shared entity communication

   This implements a kind of general reduction algorithm, beyond what you
   can do with matrix-matrix multiplies or MPI_Reduce. In particular, for the
   spectral method we want to do reduction where the operation is some kind of
   SVD, which requires something more complicated.

   The complicated logic on the Reduce side is because we are leaving the actual
   reduce operation to the user, so you can think of it more as a "collect"
   operation onto the master, where the user is responsible to do what is
   necessary.

   This is "fairly" generic but not completely, if you want to use for a
   datatype other than mfem::DenseMatrix or mfem::Vector you need to implement:
   SetSizeSpecifer(), PackSizes(), SendData(), ReceiveData(), and CopyData()
   routines yourself.

   Significant improvements to handling the "tags" argument to honor the
   MPI_TAG_UB constraint are due to Alex Druinsky from Lawrence Berkeley
   (adruinksy@lbl.gov).
*/

#include "SharedEntityComm.hpp"

namespace smoothg
{

template<>
void SharedEntityComm<Vector>::SetSizeSpecifier()
{
    size_specifier_ = 1;
}

template<>
void SharedEntityComm<DenseMatrix>::SetSizeSpecifier()
{
    size_specifier_ = 2;
}

template<>
void SharedEntityComm<SparseMatrix>::SetSizeSpecifier()
{
    size_specifier_ = 3;
}

template<>
std::vector<int>
SharedEntityComm<Vector>::PackSendSize(const Vector& vect) const
{
    return std::vector<int>{vect.size()};
}

template<>
std::vector<int>
SharedEntityComm<DenseMatrix>::PackSendSize(const DenseMatrix& mat) const
{
    return std::vector<int>{mat.Rows(), mat.Cols()};
}

template<>
std::vector<int>
SharedEntityComm<SparseMatrix>::PackSendSize(const SparseMatrix& mat) const
{
    return std::vector<int>{mat.Rows(), mat.Cols(), mat.nnz()};
}

template<>
void
SharedEntityComm<Vector>::SendData(const Vector& vect,
        int recipient, int tag, MPI_Request& request) const
{
    MPI_Isend(std::begin(vect), vect.size(), MPI_DOUBLE,
              recipient, tag, comm_, &request);
}

template<>
void
SharedEntityComm<DenseMatrix>::SendData(const DenseMatrix& mat,
        int recipient, int tag, MPI_Request& request) const
{
    MPI_Isend(mat.GetData(), mat.Rows() * mat.Cols(), MPI_DOUBLE,
              recipient, tag, comm_, &request);
}

template<>
void
SharedEntityComm<SparseMatrix>::SendData(const SparseMatrix& mat,
        int recipient, int tag, MPI_Request& request) const
{
    MPI_Isend(mat.GetIndptr().data(), mat.Rows() + 1, MPI_INT,
              recipient, tag, comm_, &request);
    MPI_Isend(mat.GetIndices().data(), mat.nnz(), MPI_INT,
              recipient, tag + 1, comm_, &request);
    MPI_Isend(mat.GetData().data(), mat.nnz(), MPI_DOUBLE,
              recipient, tag + 2, comm_, &request);
}

template<>
Vector SharedEntityComm<Vector>::ReceiveData(const std::vector<int>& sizes, int sender, int tag, MPI_Request& request) const
{
    const int size = sizes[0];

    Vector vect(size);

    MPI_Irecv(std::begin(vect), size, MPI_DOUBLE,
              sender, tag, comm_, &request);

    return vect;
}

template<>
DenseMatrix SharedEntityComm<DenseMatrix>::ReceiveData(const std::vector<int>& sizes, int sender, int tag, MPI_Request& request) const
{
    const int rows = sizes[0];
    const int cols = sizes[1];

    const int size = rows * cols;

    std::vector<double> data(size);

    MPI_Irecv(data.data(), size, MPI_DOUBLE,
              sender, tag, comm_, &request);

    return DenseMatrix(rows, cols, std::move(data));
}

template<>
SparseMatrix SharedEntityComm<SparseMatrix>::ReceiveData(const std::vector<int>& sizes, int sender, int tag, MPI_Request& request) const
{
    const int rows = sizes[0];
    const int cols = sizes[1];
    const int nnz = sizes[2];

    std::vector<int> indptr(rows + 1);
    std::vector<int> indices(nnz);
    std::vector<double> data(nnz);

    MPI_Irecv(indptr.data(), rows + 1, MPI_INT, sender, tag, comm_, &request);
    MPI_Irecv(indices.data(), nnz, MPI_INT, sender, tag + 1, comm_, &request);
    MPI_Irecv(data.data(), nnz, MPI_DOUBLE, sender, tag + 2, comm_, &request);

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data), rows, cols);
}


} // namespace smoothg
