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

NonlinearSolver::NonlinearSolver(MPI_Comm comm, int size, SolveType solve_type,
                                 std::string tag, double initial_linear_tol)
    : comm_(comm), size_(size), solve_type_(solve_type), tag_(tag), residual_(size),
      linear_tol_criterion_(NonlinearResidual), linear_tol_(initial_linear_tol)
{
    MPI_Comm_rank(comm_, &myid_);
}

double NonlinearSolver::ResidualNorm(const mfem::Vector& sol, const mfem::Vector& rhs)
{
    residual_ = 0.0;
    Mult(sol, residual_);

    residual_ -= rhs;

    for (int i = 0; i < GetEssDofs().Size(); ++i)
    {
        if (GetEssDofs()[i])
            residual_[i] = 0.0;
    }

    mfem::Vector true_resid = AssembleTrueVector(residual_);

    return mfem::ParNormlp(true_resid, 2, comm_);
}

void NonlinearSolver::Solve(const mfem::Vector& rhs, mfem::Vector& sol)
{
//    if (max_num_iter_ == 1)
//    {
//        IterationStep(rhs, sol);
//    }
//    else
    {
        mfem::StopWatch chrono;
        chrono.Start();

        mfem::Vector zero_vec(sol);
        zero_vec = 0.0;
        rhs_norm_ = prev_resid_norm_ = ResidualNorm(zero_vec, rhs);

        adjusted_tol_ = std::max(atol_, rtol_ * rhs_norm_);

        converged_ = false;
        for (iter_ = 0; iter_ < max_num_iter_ + 1; iter_++)
        {
            resid_norm_ = ResidualNorm(sol, rhs);

            if (myid_ == 0 && print_level_ > 0)
            {
                double rel_resid = resid_norm_ / rhs_norm_;
                std::cout << tag_ << " iter " << iter_ << ": rel resid = "
                          << rel_resid << ", abs resid = " << resid_norm_ << ".\n";
            }

            converged_ = (resid_norm_ < adjusted_tol_);

            if (converged_ || iter_ == max_num_iter_)
            {
                break;
            }

            UpdateLinearSolveTol();
            prev_resid_norm_ = resid_norm_;

            IterationStep(rhs, sol);
        }

        timing_ = chrono.RealTime();
        if (!converged_ && myid_ == 0 && print_level_ >= 0)
        {
            std::cout << "Warning: " << tag_ << " solver reached maximum "
                      << "number of iterations!\n";
        }
        else if (myid_ == 0 && print_level_ >= 0)
        {
            std::cout << tag_ << " solver took " << iter_ << " iterations in "
                      << timing_ << " seconds.\n\n";
        }
    }
}

void NonlinearSolver::UpdateLinearSolveTol()
{
    double tol;
    if (linear_tol_criterion_ == TaylorResidual)
    {
        tol = std::fabs(resid_norm_ - linear_resid_norm_) / prev_resid_norm_;
    }
    else // NonlinearResidual
    {
//        double exponent = solve_type_ == Newton ? (1.0 + std::sqrt(5)) / 2 : 1.0;//
        double ref_norm = solve_type_ == Newton ? prev_resid_norm_ : rhs_norm_;
        tol = std::pow(resid_norm_ / ref_norm, 1.0);
    }

    linear_tol_ = std::max(std::min(tol, linear_tol_), 1e-8);
}

NonlinearMG::NonlinearMG(MPI_Comm comm, int size, int num_levels, NLMGParameter param)
    : NonlinearSolver(comm, size, param.solve_type, "Nonlinear MG", param.initial_linear_tol),
      cycle_(param.cycle), num_levels_(num_levels), rhs_(num_levels_),
      sol_(num_levels_), help_(num_levels_), residual_norms_(num_levels)
{ }

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

        {
            for (int i = 0; i < GetEssDofs(level).Size(); ++i)
            {
                if (GetEssDofs(level)[i])
                    sol_[level][i] = 0.0;
            }
        }
    }
    else
    {
        // Pre-smoothing
        if (cycle_ == V_CYCLE || cycle_ == CASCADIC)
        {
            Smoothing(level, rhs_[level], sol_[level]);
        }

        // Compute FAS coarser level rhs
        // f_{l+1} = P^T( f_l - A_l(x_l) ) + A_{l+1}(pi x_l)
        Mult(level, sol_[level], help_[level]);
        help_[level] -= rhs_[level];

        {
//            if (level == 0)
            {
                for (int i = 0; i < GetEssDofs(level).Size(); ++i)
                {
                    if (GetEssDofs(level)[i])
                        help_[level][i] = 0.0;
                }
            }

            mfem::Vector true_resid = AssembleTrueVector(level, help_[level]);

            if (level == 0)
            {
                resid_norm_ = mfem::ParNormlp(true_resid, 2, comm_);
                UpdateLinearSolveTol();
                prev_resid_norm_ = resid_norm_;

                if (resid_norm_ < adjusted_tol_)
                {
                    converged_ = true;
                    if (level == 0 && myid_ == 0)
                    {
                        std::cout<<"V cycle terminated after pre-smoothing\n";
                    }
                    return;
                }
            }
            else
            {
                residual_norms_[level] = mfem::ParNormlp(true_resid, 2, comm_);
            }

//            if (myid_==0)
//            {
//                double resid_norm = level ? residual_norms_[level] : resid_norm_;
//                std::cout<<"level "<<level<<": pre-smooth resid = " << resid_norm <<" "<<resid_norm /rhs_norm_<<"\n";
//            }
        }

//if (solve_type_ == Picard || (level || resid_norm_ > rhs_norm_*1e-2))

        if (level || resid_norm_ > rhs_norm_*(solve_type_ == Picard ? 1e-8 : 1e-4))
            //if (iter_ == 0)
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
                BackTracking(level, rhs_[level], level ? residual_norms_[level] : prev_resid_norm_,
                             sol_[level], help_[level]);
            }

//            {
//                Mult(level, sol_[level], help_[level]);
//                help_[level] -= rhs_[level];

//                for (int i = 0; i < GetEssDofs(level).Size(); ++i)
//                {
//                    if (GetEssDofs(level)[i])
//                        help_[level][i] = 0.0;
//                }

//                mfem::Vector true_resid = AssembleTrueVector(level, help_[level]);
//                residual_norms_[level] = mfem::ParNormlp(true_resid, 2, comm_);

//                if (myid_==0)
//                {
//                    std::cout<<"level "<<level<<": coarse grid correction resid = " << residual_norms_[level] <<"\n";
//                }
//            }
        }

        // Post-smoothing
        if (cycle_ == V_CYCLE || cycle_ == FMG)
        {
            Smoothing(level, rhs_[level], sol_[level]);
        }
    }
}

} // namespace smoothg
