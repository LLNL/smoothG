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

#include "mfem.hpp"

#include "smoothG_config.h"
#include "../src/arpackeig.hpp"
#include "../src/utilities.hpp"

using namespace smoothg;

mfem::SparseMatrix* build_sparse_fd_matrix(int num_dof)
{
    mfem::SparseMatrix* out = new mfem::SparseMatrix(num_dof, num_dof);

    out->Add(0, 0, 2.0);
    out->Add(0, 1, -1.0);
    for (int i = 1; i < num_dof - 1; ++i)
    {
        out->Add(i, i - 1, -1.0);
        out->Add(i, i, 2.0);
        out->Add(i, i + 1, -1.0);
    }
    out->Add(num_dof - 1, num_dof - 2, -1.0);
    out->Add(num_dof - 1, num_dof - 1, 2.0);

    out->Finalize();
    return out;
}

/**
   Test eigenvalues of very simple 1D finite difference stencil.
*/
#if SMOOTHG_USE_ARPACK
int test_fd_sparse()
{
    const int num_dof = 10;
    const int num_ev = 5;

    std::cout << "Building test matrix..." << std::endl;
    mfem::SparseMatrix* Amatrix = build_sparse_fd_matrix(num_dof);

    mfem::Vector eigenvalues;
    mfem::DenseMatrix eigenvectors;
    SparseEigensolver eigensolver;

    eigensolver.Compute(*Amatrix, eigenvalues, eigenvectors, num_ev);

    std::cout << "Eigenvalues:" << std::endl;
    for (int i = num_ev - 1; i >= 0; --i)
        std::cout << i << ": " << 1. / eigenvalues[i] << std::endl;
    std::cout << "Eigenvector corresponding to smallest eigenvalue:" << std::endl;
    for (int i = 0; i < num_dof; ++i)
        std::cout << i << ": " << eigenvectors(i, num_ev - 1) << std::endl;

    const double expected_smallest_ev = 0.08101405; // from python...
    double error = std::fabs(1. / eigenvalues[num_ev - 1] - expected_smallest_ev);
    std::cout << "Eigenvalue error: " << error << std::endl;
    bool success;
    if (error < 1.e-6) // tolerance in SparseEigensolver
    {
        success = true;
        std::cout << "Basic FD test of ARPACK passes." << std::endl;
    }
    else
    {
        success = false;
        std::cout << "Basic FD test of ARPACK FAILS!" << std::endl;
    }

    std::cout << "Destroying MFEM test matrix..."
              << std::endl;
    delete Amatrix;

    if (success)
        return 0;
    else
        return 1;
}
#endif

mfem::DenseMatrix* build_dense_fd_matrix(int num_dof)
{
    mfem::DenseMatrix* out_p = new mfem::DenseMatrix(num_dof, num_dof);
    mfem::DenseMatrix& out = *out_p;

    out(0, 0) = 2.0;
    out(0, 1) = -1.0;
    for (int i = 1; i < num_dof - 1; ++i)
    {
        out(i, i - 1) = -1.0;
        out(i, i) = 2.0;
        out(i, i + 1) = -1.0;
    }
    out(num_dof - 1, num_dof - 2) = -1.0;
    out(num_dof - 1, num_dof - 1) = 2.0;

    return out_p;
}

int test_fd_dense()
{
    const int num_dof = 10;
    const int num_ev = 5;

    std::cout << "Building test matrix..." << std::endl;
    mfem::DenseMatrix* Amatrix = build_dense_fd_matrix(num_dof);

    mfem::Vector eigenvalues;
    mfem::DenseMatrix eigenvectors;
    Eigensolver eigensolver;

    eigensolver.Compute(*Amatrix, eigenvalues, eigenvectors, 1.0, num_ev);

    std::cout << "Eigenvalues:" << std::endl;
    for (int i = 0; i < num_ev; ++i)
        std::cout << i << ": " << eigenvalues[i] << std::endl;
    std::cout << "Eigenvector corresponding to smallest eigenvalue:" << std::endl;
    for (int i = 0; i < num_dof; ++i)
        std::cout << i << ": " << eigenvectors(i, 0) << std::endl;

    const double expected_smallest_ev = 0.08101405; // from python...
    double error = std::fabs(eigenvalues[0] - expected_smallest_ev);
    std::cout << "Eigenvalue error: " << error << std::endl;
    bool success;
    if (error < 1.e-6) // tolerance in SparseEigensolver
    {
        success = true;
        std::cout << "Basic FD test of dense Eigensolver passes." << std::endl;
    }
    else
    {
        success = false;
        std::cout << "Basic FD test of dense Eigensolver FAILS!" << std::endl;
    }

    std::cout << "Destroying MFEM test matrix..."
              << std::endl;
    delete Amatrix;

    if (success)
        return 0;
    else
        return 1;
}

int main(int argc, char* argv[])
{
    int out = 0;
#if SMOOTHG_USE_ARPACK
    out += test_fd_sparse();
#endif
    out += test_fd_dense();
    return out;
}
