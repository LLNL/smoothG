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

    @brief Contains class NonlinearSolver and FAS
*/

#ifndef __NONLINEARSOLVER_HPP__
#define __NONLINEARSOLVER_HPP__

#include "utilities.hpp"
#include "Hierarchy.hpp"

namespace smoothg
{

/// Linearization method
enum Linearization { Newton, Picard };

/// Parameter list for abstract nonlinear solver
struct NLSolverParameters
{
    int print_level = 0;
    int max_num_iter = 50;
    double rtol = 1e-8;
    double atol = 1e-10;

    bool check_converge = true;
    Linearization linearization = Newton;
    int num_backtrack = 0;
    double diff_tol = -1.0;
    double init_linear_tol = 1e-8;
};

/**
   @brief Abstract iterative solver class for nonlinear problems.

       This class takes care of nonlinear iterations; computation of nonlinear
       residual and iteration step need to be defined in derived class.
*/
class NonlinearSolver
{
public:
    /// Constructor
    NonlinearSolver(MPI_Comm comm, NLSolverParameters param);

    /// Solve A(sol) = rhs
    void Solve(const mfem::Vector& rhs, mfem::Vector& sol);

    /// Reduce dx (change in solution) if || A(x) - rhs || > prev_resid_norm
    virtual void BackTracking(const mfem::Vector& rhs,  double prev_resid_norm,
                              mfem::Vector& x, mfem::Vector& dx);

    /// @return residual = A(x) - y
    virtual mfem::Vector Residual(const mfem::Vector& x, const mfem::Vector& y) = 0;

    /// @return residual norm || A(x) - y ||
    virtual double ResidualNorm(const mfem::Vector& x, const mfem::Vector& y) = 0;

    ///@name Set solver parameters
    ///@{
    void SetPrintLevel(int print_level) { param_.print_level = print_level; }
    void SetMaxIter(int max_num_iter) { param_.max_num_iter = max_num_iter; }
    void SetRelTol(double rtol) { param_.rtol = rtol; }
    void SetAbsTol(double atol) { param_.atol = atol; }
    virtual void SetLinearRelTol(double tol) { linear_tol_ = tol; }
    ///@}

    ///@name Get results of iterative solve
    ///@{
    bool IsConverged() const { return converged_; }
    int GetNumIterations() const { return iter_; }
    double GetTiming() const { return timing_; }
    double GetResidualNorm() const { return resid_norm_; }
    int GetNumLinearIterations() const { return linear_iter_; }
    ///@}
protected:
    /// Nonlinear iteration step: x is updated, dx stores the change in x
    virtual void Step(const mfem::Vector& rhs, mfem::Vector& x, mfem::Vector& dx) = 0;

    /// Update linear tolerance based on choice 2 in Eisenstat & Walker, SISC 1996
    void UpdateLinearSolveTol();

    MPI_Comm comm_;
    int myid_;
    std::string tag_;

    int iter_;
    double timing_;
    bool converged_;

    int linear_iter_;

    double adjusted_tol_;  // max(atol_, rtol_ * || rhs ||)
    double rhs_norm_;
    double resid_norm_;
    double prev_resid_norm_;
    double linear_tol_;

    NLSolverParameters param_;
};

enum Cycle { V_CYCLE, FMG };

struct FASParameters
{
    int num_levels = 1;             // number of multigrid levels
    Cycle cycle = V_CYCLE;          // multigrid cycle type
    double coarse_correct_tol;      // no coarse correction if rel resid < tol
    NLSolverParameters nl_solve;    // for FAS itself as a nonlinear solver
    NLSolverParameters fine;        // for finest level nonlinear solve
    NLSolverParameters mid;         // for intermediate levels nonlinear solves
    NLSolverParameters coarse;      // for coarsest level nonlinear solve
};

/**
   @brief Nonlinear multigrid solver using full approximation scheme.

       Abstract class for FAS. Solver in each level and operations like
       interpolation, restriction, projection, etc. need to be provided.
*/
class FAS : public NonlinearSolver
{
public:
    /// Constructor
    FAS(MPI_Comm comm, FASParameters param);

    mfem::Vector Residual(const mfem::Vector& x, const mfem::Vector& y) override;

    double ResidualNorm(const mfem::Vector& x, const mfem::Vector& y) override;

    int GetNumCoarsestIterations() const { return coarsest_nonlinear_iter_; }
protected:
    virtual double Norm(int level, const mfem::Vector& vec) const = 0;

    /// Restrict a vector from level to level+1 (coarser level)
    virtual void Restrict(int level, const mfem::Vector& fine, mfem::Vector& coarse) const = 0;

    /// Interpolate a vector from level to level-1 (finer level)
    virtual void Interpolate(int level, const mfem::Vector& coarse, mfem::Vector& fine) const = 0;

    /// Project a vector from level to level+1 (coarser level)
    virtual void Project(int level, const mfem::Vector& fine, mfem::Vector& coarse) const = 0;

    void Smoothing(int level, const mfem::Vector& in, mfem::Vector& out);
    void MG_Cycle(int level);
    void Step(const mfem::Vector& rhs, mfem::Vector& x, mfem::Vector& dx) override;

    int coarsest_nonlinear_iter_ = 0;

    std::vector<mfem::Vector> rhs_;
    std::vector<mfem::Vector> sol_;
    std::vector<mfem::Vector> help_;
    std::vector<std::unique_ptr<NonlinearSolver>> solvers_;
    FASParameters param_;
};

} // namespace smoothg

#endif /* __NONLINEARSOLVER_HPP__ */
