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

    @brief Contains class NonlinearMG
*/

#ifndef __NONLINEARMG_HPP__
#define __NONLINEARMG_HPP__

#include "utilities.hpp"
#include "Hierarchy.hpp"

namespace smoothg
{

/// Linearization type
enum SolveType { Newton, Picard };

/// Respectively modified from choice 1 and 2 in Eisenstat and Walker, SISC 1996
enum EisenstatWalker { TaylorResidual, NonlinearResidual };

/// Iterative solver for nonlinear problems
class NonlinearSolver
{
public:
    NonlinearSolver(MPI_Comm comm, int size, SolveType solve_type,
                    std::string tag, double initial_linear_tol);

    // Solve R(sol) = rhs
    void Solve(const mfem::Vector& rhs, mfem::Vector& sol);

    // Compute the residual Rx = R(x).
    virtual void Mult(const mfem::Vector& x, mfem::Vector& Rx) = 0;

    ///@name Set solver parameters
    ///@{
    void SetPrintLevel(int print_level) { print_level_ = print_level; }
    void SetMaxIter(int max_num_iter) { max_num_iter_ = max_num_iter; }
    void SetRelTol(double rtol) { rtol_ = rtol; }
    void SetAbsTol(double atol) { atol_ = atol; }
    void SetLinearTolCriterion(EisenstatWalker criterion)
    {
        linear_tol_criterion_ = criterion;
    }
    ///@}

    ///@name Get results of iterative solve
    ///@{
    int GetNumIterations() const { return iter_; }
    double GetTiming() const { return timing_; }
    bool IsConverged() const { return converged_; }
    ///@}
protected:
    double ResidualNorm(const mfem::Vector& sol, const mfem::Vector& rhs);

    virtual void IterationStep(const mfem::Vector& x, mfem::Vector& y) = 0;

    virtual mfem::Vector AssembleTrueVector(const mfem::Vector& vec_dof) const = 0;

    void UpdateLinearSolveTol();

    virtual const mfem::Array<int>& GetEssDofs() const = 0;

    // default nonlinear solver options
    int print_level_ = 0;
    int max_num_iter_ = 50;
    double rtol_ = 1e-8;
    double atol_ = 1e-10;

    int iter_;
    double timing_;
    bool converged_;

    MPI_Comm comm_;
    int myid_;
    int size_;
    SolveType solve_type_;
    std::string tag_;

    mfem::Vector residual_;

    double adjusted_tol_;  // max(atol_, rtol_ * || rhs ||)
    double rhs_norm_;
    double resid_norm_;
    double prev_resid_norm_;

    EisenstatWalker linear_tol_criterion_;
    double linear_tol_;
    double linear_resid_norm_;
};

enum Cycle { V_CYCLE, FMG, CASCADIC };

struct NLMGParameter
{
    Cycle cycle = V_CYCLE;
    SolveType solve_type = Newton;
    int max_num_backtrack = 4;
    double diff_tol = 5;
    double coarse_diff_tol = 5;
    int num_relax_fine = 1;
    int num_relax_middle = 1;
    int num_relax_coarse = 1;
    double initial_linear_tol = 1e-8;

    void RegisterInOptionsParser(mfem::OptionsParser& args)
    {
        args.AddOption(&max_num_backtrack, "--num-backtrack", "--max-num-backtrack",
                       "Maximum number of backtracking steps.");
        args.AddOption(&diff_tol, "--diff-tol", "--diff-tol",
                       "Tolerance for solution change in fine level.");
        args.AddOption(&coarse_diff_tol, "--coarse-diff-tol", "--coarse-diff-tol",
                       "Tolerance for solution change in coarse level.");
        args.AddOption(&num_relax_fine, "--num-relax-fine", "--num-relax-fine",
                       "Number of relaxation in fine level.");
        args.AddOption(&num_relax_middle, "--num-relax-middle", "--num-relax-middle",
                       "Number of relaxation in intermediate levels.");
        args.AddOption(&num_relax_coarse, "--num-relax-coarse", "--num-relax-coarse",
                       "Number of relaxation in coarse level.");
        args.AddOption(&initial_linear_tol, "--init-linear-tol", "--init-linear-tol",
                       "Initial tol for linear solve inside nonlinear iterations.");
    }
};

/**
   @brief Nonlinear multigrid using full approximation scheme and nonlinear relaxation.

       Solve a nonlinear problem using FAS
       Vectors here are in "dof" numbering, NOT "truedof" numbering.
*/
class NonlinearMG : public NonlinearSolver
{
public:
    // the time dependent operators gets updated during solving
    NonlinearMG(MPI_Comm comm, int size, int num_levels, NLMGParameter param);

    virtual void Mult(const mfem::Vector& x, mfem::Vector& Rx);
protected:
    void FAS_Cycle(int level);

    virtual void IterationStep(const mfem::Vector& rhs, mfem::Vector& sol);

    virtual mfem::Vector AssembleTrueVector(const mfem::Vector& vec_dof) const = 0;

    /// Evaluates the action of the operator out = A[level](in)
    virtual void Mult(int level, const mfem::Vector& in, mfem::Vector& out) = 0;

    /// Solves the (possibly nonlinear) problem A[level](sol) = rhs
    virtual void Solve(int level, const mfem::Vector& rhs, mfem::Vector& sol) = 0;

    /// Restrict a vector from level to level+1 (coarser level)
    virtual void Restrict(int level, const mfem::Vector& fine, mfem::Vector& coarse) const = 0;

    /// Interpolate a vector from level to level-1 (finer level)
    virtual void Interpolate(int level, const mfem::Vector& coarse, mfem::Vector& fine) const = 0;

    /// Project a vector from level to level+1 (coarser level)
    virtual void Project(int level, const mfem::Vector& fine, mfem::Vector& coarse) const = 0;

    /// Relaxation on each level
    virtual void Smoothing(int level, const mfem::Vector& in, mfem::Vector& out) = 0;

    virtual void BackTracking(int level, const mfem::Vector& rhs, double prev_resid_norm,
                              mfem::Vector& x, mfem::Vector& dx) = 0;


    virtual mfem::Vector AssembleTrueVector(int level, const mfem::Vector& vec_dof) const = 0;
    virtual const mfem::Array<int>& GetEssDofs(int level) const = 0;

    Cycle cycle_;
    int num_levels_;
    std::vector<mfem::Vector> rhs_;
    std::vector<mfem::Vector> sol_;
    mutable std::vector<mfem::Vector> help_;

    std::vector<double> residual_norms_;
};

} // namespace smoothg

#endif /* __NONLINEARMG_HPP__ */
