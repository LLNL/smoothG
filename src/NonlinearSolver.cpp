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

NonlinearSolver::NonlinearSolver(MPI_Comm comm, int size, NLSolverParameters param)
    : comm_(comm), size_(size), tag_("Nonlinear"), residual_(size), param_(param)
{
    MPI_Comm_rank(comm_, &myid_);
}

double NonlinearSolver::ResidualNorm(const mfem::Vector& sol, const mfem::Vector& rhs)
{
    residual_ = 0.0;
    Mult(sol, residual_);
    residual_ -= rhs;
    SetZeroAtMarker(GetEssDofs(), residual_);
    return mfem::ParNormlp(AssembleTrueVector(residual_), 2, comm_);
}

void NonlinearSolver::Solve(const mfem::Vector& rhs, mfem::Vector& sol)
{
    mfem::StopWatch chrono;
    chrono.Start();

    mfem::Vector zero_vec(sol);
    zero_vec = 0.0;
    rhs_norm_ = prev_resid_norm_ = ResidualNorm(zero_vec, rhs);
    adjusted_tol_ = std::max(param_.atol, param_.rtol * rhs_norm_);

    converged_ = false;
    for (iter_ = 0; iter_ < param_.max_num_iter + 1; iter_++)
    {
        if (check_converge_)
        {
            resid_norm_ = ResidualNorm(sol, rhs);

            if (myid_ == 0 && param_.print_level > 0)
            {
                double rel_resid = resid_norm_ / rhs_norm_;
                std::cout << tag_ << " iter " << iter_ << ": rel resid = "
                          << rel_resid << ", abs resid = " << resid_norm_ << ".\n";
            }

            converged_ = (resid_norm_ < adjusted_tol_);

            if (converged_ || iter_ == param_.max_num_iter) { break; }

            UpdateLinearSolveTol();
            prev_resid_norm_ = resid_norm_;
        }

        IterationStep(rhs, sol);
    }

    timing_ = chrono.RealTime();
    if (!converged_ && myid_ == 0 && param_.print_level >= 0)
    {
        std::cout << "Warning: " << tag_ << " solver reached maximum number "
                  << "of iterations and took " << timing_ << " seconds!\n\n";
    }
    else if (myid_ == 0 && param_.print_level >= 0)
    {
        std::cout << tag_ << " solver took " << iter_ << " iterations in "
                  << timing_ << " seconds.\n\n";
    }
}

void NonlinearSolver::UpdateLinearSolveTol()
{
    double exponent = linearization_ == Newton ? (1.0 + std::sqrt(5)) / 2 : 1.0;
    double ref_norm = linearization_ == Newton ? prev_resid_norm_ : rhs_norm_;
    double tol = std::pow(resid_norm_ / ref_norm, exponent);
    linear_tol_ = std::max(std::min(tol, linear_tol_), 1e-8);
}

NonlinearMG::NonlinearMG(MPI_Comm comm, int size, int num_levels, FASParameters param)
    : NonlinearSolver(comm, size, param),
      cycle_(param.cycle), num_levels_(num_levels), rhs_(num_levels_),
      sol_(num_levels_), help_(num_levels_), resid_norms_(num_levels)
{
    tag_ = "Nonlinear MG";
}

void NonlinearMG::Mult(const mfem::Vector& x, mfem::Vector& Rx)
{
    Mult(0, x, Rx);
}

void NonlinearMG::IterationStep(const mfem::Vector& rhs, mfem::Vector& sol)
{
    rhs_[0].SetDataAndSize(rhs.GetData(), rhs.Size());
    sol_[0].SetDataAndSize(sol.GetData(), sol.Size());

    FAS_Cycle(0);
}

void NonlinearMG::FAS_Cycle(int level)
{
    if (level == num_levels_ - 1)
    {
        Solve(level, rhs_[level], sol_[level]);
        SetZeroAtMarker(GetEssDofs(level), sol_[level]);
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
        SetZeroAtMarker(GetEssDofs(level), help_[level]);

        mfem::Vector true_resid = AssembleTrueVector(level, help_[level]);

        resid_norms_[level] = resid_norm_ = mfem::ParNormlp(true_resid, 2, comm_);
        if (level == 0)
        {
            UpdateLinearSolveTol();

            if (resid_norm_ < adjusted_tol_)
            {
                converged_ = true;
                if (myid_ == 0)
                {
                    std::cout << "V cycle terminated after pre-smoothing\n";
                }
                return;
            }

            prev_resid_norm_ = resid_norm_;
        }

        if (level || resid_norm_ > rhs_norm_ * (linearization_ == Picard ? 1e-8 : 1e-4))
        {
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

            if (cycle_ == V_CYCLE)
            {
                BackTracking(level, rhs_[level], resid_norms_[level],
                             sol_[level], help_[level]);
            }
        }

        // Post-smoothing
        if (cycle_ == V_CYCLE || cycle_ == FMG)
        {
            Smoothing(level, rhs_[level], sol_[level]);
        }
    }
}

} // namespace smoothg
