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

    @brief Contains implementation of NonlinearMG
*/

#include "NonlinearSolver.hpp"

namespace smoothg
{

NonlinearSolver::NonlinearSolver(MPI_Comm comm, int size, std::string tag)
    : comm_(comm), size_(size), tag_(tag), residual_(size)
{
    MPI_Comm_rank(comm_, &myid_);
}

double NonlinearSolver::ResidualNorm(const mfem::Vector& sol, const mfem::Vector& rhs)
{
    residual_ = 0.0;
    Mult(sol, residual_);

    residual_ -= rhs;

    mfem::Vector true_resid = AssembleTrueVector(residual_);

    return mfem::ParNormlp(true_resid, 2, comm_);
}

void NonlinearSolver::Solve(const mfem::Vector& rhs, mfem::Vector& sol)
{
    if (max_num_iter_ == 1)
    {
        IterationStep(rhs, sol);
    }
    else
    {
        mfem::StopWatch chrono;
        chrono.Start();

        mfem::Vector zero_vec(sol);
        zero_vec = 0.0;
        double norm = ResidualNorm(zero_vec, rhs);

        converged_ = false;
        for (iter_ = 0; iter_ < max_num_iter_; iter_++)
        {
            double resid = ResidualNorm(sol, rhs);
            double rel_resid = resid / norm;

            if (myid_ == 0 && print_level_ > 0)
            {
                std::cout << tag_ << " iter " << iter_ << ":  rel resid = "
                          << rel_resid << "  abs resid = " << resid << "\n";
            }

            if (resid < atol_ || rel_resid < rtol_)
            {
                converged_ = true;
                break;
            }

            IterationStep(rhs, sol);
        }

        if (myid_ == 0 && !converged_ && print_level_ >= 0)
        {
            std::cout << "Warning: " << tag_ << " solver reached maximum "
                      << "number of iterations!\n";
        }
        else if (myid_ == 0 && print_level_ >= 0)
        {
            std::cout << tag_ << " solver took " << iter_ << " iterations in "
                      << chrono.RealTime() << " seconds.\n\n";
        }
    }
}

NonlinearMG::NonlinearMG(MPI_Comm comm, int size, int num_levels, Cycle cycle)
    : NonlinearSolver(comm, size, "Nonlinear MG"), cycle_(cycle),
      num_levels_(num_levels), rhs_(num_levels_), sol_(num_levels_), help_(num_levels_)
{ }

void NonlinearMG::Mult(const mfem::Vector& x, mfem::Vector& Rx)
{
    Mult(0, x, Rx);
}

void NonlinearMG::IterationStep(const mfem::Vector& rhs, mfem::Vector& sol)
{
    rhs_[0] = rhs;
    sol_[0] = sol;

    FAS_Cycle(0);

    sol = sol_[0];
}

void NonlinearMG::FAS_Cycle(int level)
{
    if (level == num_levels_ - 1)
    {
        Solve(level, rhs_[level], sol_[level]);
    }
    else
    {
        // Pre-smoothing
        if (cycle_ == V_CYCLE)
        {
            Smoothing(level, rhs_[level], sol_[level]);
        }

        // Compute FAS coarser level rhs
        // f_{l+1} = P^T( f_l - A_l(x_l) ) + A_{l+1}(pi x_l)
        Mult(level, sol_[level], help_[level]);
        help_[level] -= rhs_[level];

        Restrict(level, help_[level], help_[level + 1]);

        Project(level, sol_[level], sol_[level + 1]);

        Mult(level + 1, sol_[level + 1], rhs_[level + 1]);
        rhs_[level + 1] -= help_[level + 1];

        // Store projected coarse solution pi x_l
        mfem::Vector coarse_sol = sol_[level + 1];

        // Go to coarser level (sol_[level + 1] will be updated)
        FAS_Cycle(level + 1);

        // Compute correction x_l += P( x_{l+1} - pi x_l )
        coarse_sol -= sol_[level + 1];
        Interpolate(level + 1, coarse_sol, help_[level]);
        sol_[level] -= help_[level];

        // Post-smoothing
        Smoothing(level, rhs_[level], sol_[level]);
    }
}

} // namespace smoothg
