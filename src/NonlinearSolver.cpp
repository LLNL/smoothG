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

NonlinearSolver::NonlinearSolver(MPI_Comm comm, NLSolverParameters param)
    : comm_(comm), tag_("Nonlinear"), linear_tol_(param.init_linear_tol),
      param_(param)
{
    MPI_Comm_rank(comm_, &myid_);
}

void NonlinearSolver::Solve(const mfem::Vector& rhs, mfem::Vector& sol)
{
    mfem::StopWatch chrono;
    chrono.Start();

    if (param_.check_converge)
    {
        mfem::Vector zero_vec(sol);
        zero_vec = 0.0;
        rhs_norm_ = prev_resid_norm_ = ResidualNorm(zero_vec, rhs);
        adjusted_tol_ = std::max(param_.atol, param_.rtol * rhs_norm_);
    }

    converged_ = false;
    for (iter_ = 0; iter_ < param_.max_num_iter + 1; iter_++)
    {
        if (param_.check_converge)
        {
            resid_norm_ = ResidualNorm(sol, rhs);

            if (myid_ == 0 && param_.print_level > 0)
            {
                double rel_resid = resid_norm_ / rhs_norm_;
                std::cout << tag_ << " iter " << iter_ << ": rel resid = "
                          << rel_resid << ", abs resid = " << resid_norm_ << ".\n";
            }

            converged_ = (resid_norm_ < adjusted_tol_);

            UpdateLinearSolveTol();
            prev_resid_norm_ = resid_norm_;
        }

        if (converged_ || iter_ == param_.max_num_iter) { break; }

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
    double exponent = param_.linearization == Newton ? (1.0 + std::sqrt(5)) / 2 : 1.0;
    double ref_norm = param_.linearization == Newton ? prev_resid_norm_ : rhs_norm_;
    double tol = std::pow(resid_norm_ / ref_norm, exponent);
    linear_tol_ = std::max(std::min(tol, linear_tol_), 1e-8);
}

FAS::FAS(MPI_Comm comm, FASParameters param)
    : NonlinearSolver(comm, param.nl_solve), rhs_(param.num_levels),
      sol_(rhs_.size()), help_(rhs_.size()), resid_norms_(rhs_.size()), param_(param)
{
    tag_ = "FAS";
}

void FAS::IterationStep(const mfem::Vector& rhs, mfem::Vector& sol)
{
    rhs_[0].SetDataAndSize(rhs.GetData(), rhs.Size());
    sol_[0].SetDataAndSize(sol.GetData(), sol.Size());

    MG_Cycle(0);
}

double FAS::ResidualNorm(const mfem::Vector& sol, const mfem::Vector& rhs)
{
    return ResidualNorm(0, ComputeResidual(0, sol, rhs));
}

void FAS::MG_Cycle(int level)
{
    if (level == param_.num_levels - 1)
    {
        Smoothing(level, rhs_[level], sol_[level]);
    }
    else
    {
        // Pre-smoothing
        if (param_.cycle == V_CYCLE)
        {
            Smoothing(level, rhs_[level], sol_[level]);
        }

        // Compute FAS coarser level rhs
        // f_{l+1} = P^T( f_l - A_l(x_l) ) + A_{l+1}(pi x_l)
        help_[level] = ComputeResidual(level, sol_[level], rhs_[level]);
        resid_norms_[level] = resid_norm_ = ResidualNorm(level, help_[level]);

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

        if (level || resid_norm_ > rhs_norm_ * param_.coarse_correct_tol)
        {
            Restrict(level, help_[level], help_[level + 1]);

            Project(level, sol_[level], sol_[level + 1]);

            rhs_[level + 1] = ComputeResidual(level + 1, sol_[level + 1], help_[level + 1]);

            // Store projected coarse solution pi x_l
            mfem::Vector coarse_sol = sol_[level + 1];

            // Go to coarser level (sol_[level + 1] will be updated)
            MG_Cycle(level + 1);

            // Compute correction x_l += P( x_{l+1} - pi x_l )
            coarse_sol -= sol_[level + 1];

            Interpolate(level + 1, coarse_sol, help_[level]);
            sol_[level] -= help_[level];

            if (param_.cycle == V_CYCLE)
            {
                BackTracking(level, rhs_[level], resid_norms_[level],
                             sol_[level], help_[level]);
            }
        }

        // Post-smoothing
        Smoothing(level, rhs_[level], sol_[level]);
    }
}

} // namespace smoothg
