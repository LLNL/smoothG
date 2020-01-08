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
      update_is_needed_(true), param_(param)
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
        rhs_norm_ = prev_resid_norm_ = Norm(Residual(zero_vec, rhs));
        adjusted_tol_ = std::max(param_.atol, param_.rtol * rhs_norm_);
    }

    converged_ = false;
    for (iter_ = 0; iter_ < param_.max_num_iter + 1; iter_++)
    {
        if (param_.check_converge)
        {
            if (update_is_needed_) { resid_norm_ = Norm(Residual(sol, rhs)); }
            converged_ = (resid_norm_ < adjusted_tol_);
            UpdateLinearSolveTol();
            prev_resid_norm_ = resid_norm_;

            if (myid_ == 0 && param_.print_level > 0)
            {
                std::cout << tag_ << " iter " << iter_ << ": abs resid = " << resid_norm_
                          << ", rel resid = " << resid_norm_ / rhs_norm_ << ".\n";
            }
        }

        if (converged_ || iter_ == param_.max_num_iter) { break; }

        IterationStep(rhs, sol);
    }

    timing_ = chrono.RealTime();
    if (!converged_ && myid_ == 0 && param_.print_level > -1)
    {
        std::cout << "Warning: " << tag_ << " reached maximum number "
                  << "of iterations and took " << timing_ << " seconds!\n\n";
    }
    else if (myid_ == 0 && param_.print_level > -1)
    {
        std::cout << tag_ << " took " << iter_ << " iterations in "
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

void NonlinearSolver::BackTracking(const mfem::Vector& rhs, double prev_resid_norm,
                                   mfem::Vector& x, mfem::Vector& dx)
{
    if (param_.num_backtrack == 0) { return; }

    update_is_needed_ = false;
    int k = 0;
    resid_norm_ = Norm(Residual(x, rhs));

    while (k < param_.num_backtrack && resid_norm_ > prev_resid_norm)
    {
        dx *= 0.5;
        x += dx;

        const double backtracking_resid_norm = resid_norm_;
        resid_norm_ = Norm(Residual(x, rhs));

        if (resid_norm_ > 0.9 * backtracking_resid_norm)
        {
            x -= dx;
            resid_norm_ = backtracking_resid_norm;
            break;
        }

        if (myid_ == 0  && param_.print_level > 1)
        {
            if (k == 0) { std::cout << "  backtracking: || R(u) ||"; }
            std::cout << " -> " << backtracking_resid_norm;
        }
        k++;
    }

    if (k > 0 && myid_ == 0 && param_.print_level > 1)
    {
        std::cout << "\n";
    }
}

FAS::FAS(MPI_Comm comm, FASParameters param)
    : NonlinearSolver(comm, param.nl_solve), rhs_(param.num_levels),
      sol_(rhs_.size()), help_(rhs_.size()), solvers_(rhs_.size()), param_(param)
{
    tag_ = "FAS";
}

void FAS::Smoothing(int level, const mfem::Vector& in, mfem::Vector& out)
{
    solvers_[level]->SetLinearRelTol(linear_tol_);
    solvers_[level]->Solve(in, out);
}

void FAS::MG_Cycle(int l)
{
    if (param_.cycle == V_CYCLE || l == param_.num_levels - 1)
    {
        Smoothing(l, rhs_[l], sol_[l]); // Pre-smoothing
    }

    if (l == param_.num_levels - 1) { return; } // terminate if coarsest level

    // Compute FAS coarser level rhs
    // f_{l+1} = P^T( f_l - A_l(x_l) ) + A_{l+1}(pi x_l)
    help_[l] = solvers_[l]->Residual(sol_[l], rhs_[l]);
    double resid_norm_l = resid_norm_ = solvers_[l]->Norm(help_[l]);

    if (l == 0 && param_.cycle == V_CYCLE)
    {
        if (resid_norm_ < adjusted_tol_)
        {
            update_is_needed_ = false;
            if (myid_ == 0)
            {
                std::cout << "V cycle terminated after pre-smoothing\n";
            }
            return;
        }
        UpdateLinearSolveTol();
        prev_resid_norm_ = resid_norm_;
    }

    if (l || resid_norm_ > rhs_norm_ * param_.coarse_correct_tol)
    {
        Restrict(l, help_[l], help_[l + 1]);
        Project(l, sol_[l], sol_[l + 1]);
        rhs_[l + 1] = solvers_[l + 1]->Residual(sol_[l + 1], help_[l + 1]);

        // Store projected coarse solution pi x_l
        mfem::Vector coarse_sol = sol_[l + 1];

        MG_Cycle(l + 1); // Go to coarser level (sol_[l+1] will be updated)

        // Compute correction x_l += P( x_{l+1} - pi x_l )
        coarse_sol -= sol_[l + 1];
        Interpolate(l + 1, coarse_sol, help_[l]);
        sol_[l] -= help_[l];

        if (param_.cycle == V_CYCLE)
        {
            solvers_[l]->BackTracking(rhs_[l], resid_norm_l, sol_[l], help_[l]);
        }
    }

    Smoothing(l, rhs_[l], sol_[l]); // Post-smoothing
}

void FAS::IterationStep(const mfem::Vector& rhs, mfem::Vector& sol)
{
    rhs_[0].SetDataAndSize(rhs.GetData(), rhs.Size());
    sol_[0].SetDataAndSize(sol.GetData(), sol.Size());
    MG_Cycle(0);
}

} // namespace smoothg
