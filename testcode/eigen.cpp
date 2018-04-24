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

#include "smoothG.hpp"
#include "smoothG_config.h"

using namespace smoothg;

// build tridiagonal matrix with constant diagonal, sub- and super-diagonal
SparseMatrix sparse_tridiag_matrix(int num_dof, double diag, double subd)
{
    CooMatrix out(num_dof, num_dof);

    out.Add(0, 0, diag);
    out.Add(0, 1, subd);

    for (int i = 1; i < num_dof - 1; ++i)
    {
        out.Add(i, i - 1, subd);
        out.Add(i, i, diag);
        out.Add(i, i + 1, subd);
    }

    out.Add(num_dof - 1, num_dof - 2, subd);
    out.Add(num_dof - 1, num_dof - 1, diag);

    return out.ToSparse();
}

/**
   Test eigenvalues of very simple 1D finite difference stencil.
*/
#if SMOOTHG_USE_ARPACK
SparseMatrix build_sparse_fd_matrix(int num_dof)
{
    return sparse_tridiag_matrix(num_dof, 2.0, -1.0);
}

SparseMatrix build_sparse_fe_stiffness_matrix(int num_dof)
{
    double h = 1.0 / (num_dof + 1);
    return sparse_tridiag_matrix(num_dof, 2.0 / h, -1.0 / h);
}

SparseMatrix build_sparse_fe_mass_matrix(int num_dof)
{
    double h = 1.0 / (num_dof + 1);
    return sparse_tridiag_matrix(num_dof, h * 2.0 / 3.0, h / 6.0);
}

int test_fd_sparse()
{
    const int num_dof = 10;
    const int num_ev = 5;

    std::cout << "Building sparse FD test matrix..." << std::endl;
    SparseMatrix Amatrix = build_sparse_fd_matrix(num_dof);

    std::vector<double> eigenvalues;
    DenseMatrix eigenvectors;
    LocalEigenSolver eigensolver(num_ev, 1.0);

    eigensolver.Compute(Amatrix, eigenvalues, eigenvectors);

    std::cout << "Eigenvalues:" << std::endl;

    for (int i = 0; i < num_ev; ++i)
    {
        std::cout << i << ": " << eigenvalues[i] << std::endl;
    }

    std::cout << "Eigenvector corresponding to smallest eigenvalue:" << std::endl;

    for (int i = 0; i < num_dof; ++i)
    {
        std::cout << i << ": " << eigenvectors(i, 0) << std::endl;
    }

    const double expected_smallest_ev = 0.08101405; // from python...
    double error = std::fabs(eigenvalues[0] - expected_smallest_ev);
    std::cout << "Eigenvalue error: " << error << std::endl;

    bool failed = (error > 1e-6);

    std::cout << "Basic FD test of ARPACK " << (failed ? "FAILS!" : "passes.") << "\n";

    return failed;
}

int test_fe_sparse()
{
    const int num_dof = 10;
    const int num_ev = 5;

    std::cout << "Building sparse FE test matrices...\n";

    SparseMatrix Amatrix = build_sparse_fe_stiffness_matrix(num_dof);
    SparseMatrix Mmatrix = build_sparse_fe_mass_matrix(num_dof);

    std::vector<double> eigenvalues;
    DenseMatrix eigenvectors;
    LocalEigenSolver eigensolver(num_ev, 1.0);

    eigensolver.Compute(Amatrix, Mmatrix, eigenvalues, eigenvectors);

    std::cout << "Eigenvalues:\n";

    for (int i = 0; i < num_ev; ++i)
    {
        std::cout << i << ": " << eigenvalues[i] << "\n";
    }

    std::cout << "Eigenvector corresponding to smallest eigenvalue:" << "\n";

    for (int i = 0; i < num_dof; ++i)
    {
        std::cout << i << ": " << eigenvectors(i, 0) << "\n";
    }

    const double expected_smallest_ev = 9.93687142; // from python...
    double error = std::fabs(eigenvalues[0] - expected_smallest_ev);
    std::cout << "Eigenvalue error: " << error << "\n";

    bool failed = (error > 1e-6);

    std::cout << "Basic FE test of ARPACK " << (failed ? "FAILS!" : "passes.") << "\n";

    return failed;
}
#endif

DenseMatrix dense_tridiag_matrix(int num_dof, double diag, double subd)
{
    return sparse_tridiag_matrix(num_dof, diag, subd).ToDense();
}

DenseMatrix build_dense_fd_matrix(int num_dof)
{
    return dense_tridiag_matrix(num_dof, 2.0, -1.0);
}

DenseMatrix build_dense_fe_stiffness_matrix(int num_dof)
{
    double h = 1.0 / (num_dof + 1);
    return dense_tridiag_matrix(num_dof, 2.0 / h, -1.0 / h);
}

DenseMatrix build_dense_fe_mass_matrix(int num_dof)
{
    double h = 1.0 / (num_dof + 1);
    return dense_tridiag_matrix(num_dof, h * 2.0 / 3.0, h / 6.0);
}

int test_fd_dense()
{
    const int num_dof = 10;
    const int num_ev = 5;

    std::cout << "Building dense FD test matrix...\n";
    DenseMatrix Amatrix = build_dense_fd_matrix(num_dof);

    std::vector<double> eigenvalues;
    DenseMatrix eigenvectors;
    LocalEigenSolver eigensolver(num_ev, 1.0);

    eigensolver.Compute(Amatrix, eigenvalues, eigenvectors);

    std::cout << "Eigenvalues:\n";

    for (int i = 0; i < num_ev; ++i)
    {
        std::cout << i << ": " << eigenvalues[i] << "\n";
    }

    std::cout << "Eigenvector corresponding to smallest eigenvalue:\n";

    for (int i = 0; i < num_dof; ++i)
    {
        std::cout << i << ": " << eigenvectors(i, 0) << "\n";
    }

    const double expected_smallest_ev = 0.08101405; // from python...
    double error = std::fabs(eigenvalues[0] - expected_smallest_ev);
    std::cout << "Eigenvalue error: " << error << "\n";

    bool failed = (error > 1e-6);

    std::cout << "Basic FD test of dense Eigensolver " << (failed ? "FAILS!" : "passes.") << "\n";

    return failed;
}

int test_fe_dense()
{
    const int num_dof = 10;
    const int num_ev = 5;

    std::cout << "Building dense FE test matrices..." << std::endl;
    DenseMatrix Amatrix = build_dense_fe_stiffness_matrix(num_dof);
    DenseMatrix Mmatrix = build_dense_fe_mass_matrix(num_dof);

    std::vector<double> eigenvalues;
    DenseMatrix eigenvectors;
    LocalEigenSolver eigensolver(num_ev, 1.0);

    eigensolver.Compute(Amatrix, Mmatrix, eigenvalues, eigenvectors);

    std::cout << "Eigenvalues:\n";

    for (int i = 0; i < num_ev; ++i)
    {
        std::cout << i << ": " << eigenvalues[i] << "\n";
    }

    std::cout << "Eigenvector corresponding to smallest eigenvalue:\n";

    for (int i = 0; i < num_dof; ++i)
    {
        std::cout << i << ": " << eigenvectors(i, 0) << "\n";
    }

    const double expected_smallest_ev = 9.93687142; // from python...
    double error = std::fabs(eigenvalues[0] - expected_smallest_ev);
    std::cout << "Eigenvalue error: " << error << "\n";

    bool failed = (error > 1e-6);

    std::cout << "Basic FD test of dense Eigensolver " << (failed ? "FAILS!" : "passes.") << "\n";
    
    return failed;
}

int main(int argc, char* argv[])
{
    int out = 0;
#if SMOOTHG_USE_ARPACK
    out += test_fd_sparse();
    out += test_fe_sparse();
#endif
    out += test_fd_dense();
    out += test_fe_dense();
    return out;
}
