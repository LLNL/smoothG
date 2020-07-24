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
    : comm_(comm), tag_("Nonlinear"), converged_(false),
      linear_iter_(0), linear_tol_(param.init_linear_tol), param_(param)
{
    MPI_Comm_rank(comm_, &myid_);
}

void NonlinearSolver::Solve(const mfem::Vector& rhs, mfem::Vector& sol)
{
    mfem::StopWatch chrono;
    chrono.Start();

    mfem::Vector sol_change(sol.Size());
    iter_ = 0;

    if (param_.check_converge)
    {
        sol_change = 0.0;
        rhs_norm_ = ResidualNorm(sol_change, rhs);
        adjusted_tol_ = std::max(param_.atol, param_.rtol * rhs_norm_);
    }

    if (param_.num_backtrack) { resid_norm_ = ResidualNorm(sol, rhs); }

    for (; iter_ < param_.max_num_iter + 1; iter_++)
    {
        if (param_.check_converge)
        {
            if (!param_.num_backtrack) { resid_norm_ = ResidualNorm(sol, rhs); }
            if (myid_ == 0 && param_.print_level > 0)
            {
                std::cout << tag_ << " iter " << iter_ << ": abs resid = " << resid_norm_
                          << ", rel resid = " << resid_norm_ / rhs_norm_ << ".\n";
            }
            if ((converged_ = (resid_norm_ < adjusted_tol_))) { break; }

            if (iter_ && sol.Size() < 30000)
            {
                const double x_norm = mfem::ParNormlp(sol, 2, comm_);
                const double dx_norm = mfem::ParNormlp(sol_change, 2, comm_);

                if ((converged_ = ((dx_norm / x_norm) < param_.rtol)))
                {
                    std::cout << tag_ << " iter " << iter_ << ": x_norm = " << x_norm
                              << ", dx_norm = " << dx_norm << ".\n";

                    break;
                }
            }

            if (iter_ > 0) { UpdateLinearSolveTol(); }
        }

        if (iter_ == param_.max_num_iter) { break; }

        prev_resid_norm_ = resid_norm_;
        Step(rhs, sol, sol_change);
        BackTracking(rhs, prev_resid_norm_, sol, sol_change);
    }

    timing_ = chrono.RealTime();

    if (myid_ == 0 && param_.print_level > -1 && !param_.check_converge)
    {
        std::cout << "No convergence info: check_converge flag is false!\n";
    }
    else if (myid_ == 0 && param_.print_level > -1 && param_.check_converge)
    {
        auto msg = converged_ ? " converged" : " DID NOT converge";
        std::cout << tag_ << msg << " in " << iter_ << " iterations:"
                  << "\n    Absolute residual: " << resid_norm_
                  << "\n    Relative residual: " << resid_norm_ / rhs_norm_
                  << "\n    Time elapsed:      " << timing_ << " seconds\n\n";
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

    resid_norm_ = ResidualNorm(x, rhs);

    int k = 0;
    while (k < param_.num_backtrack && resid_norm_ > prev_resid_norm)
    {
        dx *= 0.5;
        x -= dx;

        const double backtracking_resid_norm = resid_norm_;
        resid_norm_ = ResidualNorm(x, rhs);

        if (resid_norm_ > 0.9 * backtracking_resid_norm)
        {
            x += dx;
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

mfem::Vector FAS::Residual(const mfem::Vector& x, const mfem::Vector& y)
{
    return solvers_[0]->Residual(x, y);
}

double FAS::ResidualNorm(const mfem::Vector& x, const mfem::Vector& y)
{
//    if (iter_ > 0 && (param_.fine.check_converge || param_.fine.num_backtrack))
//    {
//        return solvers_[0]->GetResidualNorm();
//    }
    return solvers_[0]->ResidualNorm(x, y);
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

    if (l == 0 && param_.cycle == V_CYCLE)
    {
        resid_norm_ = solvers_[l]->ResidualNorm(sol_[l], rhs_[l]);
        if (resid_norm_ < adjusted_tol_)
        {
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
        // Compute FAS coarser level rhs
        // f_{l+1} = P^T( f_l - A_l(x_l) ) + A_{l+1}(pi x_l)
        help_[l] = solvers_[l]->Residual(sol_[l], rhs_[l]);
        Restrict(l, help_[l], help_[l + 1]);
        Project(l, sol_[l], sol_[l + 1]);
        rhs_[l + 1] = solvers_[l + 1]->Residual(sol_[l + 1], help_[l + 1]);

        // Store projected coarse solution pi x_l

        mfem::Vector coarse_sol = sol_[l + 1];
        if (param_.cycle == FMG)
        {
            coarse_sol = 0.0;
        }
        double resid_norm_l = Norm(l, help_[l]);

        MG_Cycle(l + 1); // Go to coarser level (sol_[l+1] will be updated)

        // Compute correction x_l += P( x_{l+1} - pi x_l )
        sol_[l + 1] -= coarse_sol;
        Interpolate(l + 1, sol_[l + 1], help_[l]);
        if (param_.cycle == V_CYCLE)
        {
            sol_[l] += help_[l];
        }
        else
        {
            sol_[l] = help_[l];
        }

        if (param_.cycle == V_CYCLE)
        {
            solvers_[l]->BackTracking(rhs_[l], resid_norm_l, sol_[l], help_[l]);
        }
    }

    Smoothing(l, rhs_[l], sol_[l]); // Post-smoothing
}

void FAS::Step(const mfem::Vector& rhs, mfem::Vector& x, mfem::Vector& dx)
{
//    if (param_.nl_solve.num_backtrack > 0)
    { dx.Set(-1.0, x); }

    rhs_[0].SetDataAndSize(rhs.GetData(), rhs.Size());
    sol_[0].SetDataAndSize(x.GetData(), x.Size());
    MG_Cycle(0);

//    if (param_.nl_solve.num_backtrack > 0)
    { dx += x; }
}

} // namespace smoothg
