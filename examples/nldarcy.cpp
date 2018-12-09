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

/**
   @file singlephase.cpp
   @brief This is an example for upscaling a graph Laplacian coming from a finite
   volume discretization of a single phase flow and transport model in parallel.

   A simple way to run the example:

   mpirun -n 4 ./singlephase
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "mfem.hpp"
//#include "spe10.hpp"
#include "well.hpp"

#include "../src/picojson.h"
#include "../src/smoothG.hpp"

using namespace smoothg;

enum Level { Fine = 0, Coarse };

/**
   @brief Nonlinear elliptic problem

   Given \f$f \in L^2(\Omega)\f$, \f$k(p)\f$ a differentiable function of p,
   find \f$p\f$ such that \f$-div(k_0k(p)\nabla p) = f\f$.
*/

class SingleLevelSolver : public NonlinearSolver
{
public:
    /**
       @todo take Kappa(p) as input
    */
    SingleLevelSolver(const DarcyProblem& darcy_problem, FiniteVolumeMLMC& up,
                      Level level, SolveType solve_type);

    // Solve A(sol) = rhs
    virtual void Solve(const mfem::Vector& rhs, mfem::Vector& sol);

    // Compute the residual Rx = R(x) = A(x) - f.
    virtual void Mult(const mfem::Vector& x, mfem::Vector& Rx);

private:
    void PicardSolve(const mfem::BlockVector& rhs, mfem::BlockVector& x);
    void PicardStep(const mfem::BlockVector& rhs, mfem::BlockVector& x);
    void NewtonSolve(const mfem::BlockVector& rhs, mfem::BlockVector& x);

    Level level_;
    FiniteVolumeMLMC& up_;
    SolveType solve_type_;

    mfem::Array<int> offsets_;
    unique_ptr<mfem::Operator> solver_;

    unique_ptr<mfem::BlockVector> block_f_;   // block_f = (0, -f)
    mfem::Vector kp_;                         // kp_ = Kappa(p)
    mfem::Vector p_fine_;                     // fine representation of coarse p
};

// nonlinear elliptic hierarchy
class NonlinearEllipticHierarchy : public Hierarchy
{
public:
    NonlinearEllipticHierarchy(const DarcyProblem& darcy_problem,
                               FiniteVolumeMLMC& up,
                               SolveType solve_type);
    NonlinearEllipticHierarchy() = delete;

    // Compute resid = R(in)
    virtual void Mult(int level, const mfem::Vector& x, mfem::Vector& Rx) const;
    virtual void Solve(int level, const mfem::Vector& rhs, mfem::Vector& sol);
    virtual void Restrict(int level, const mfem::Vector& fine, mfem::Vector& coarse) const;
    virtual void Interpolate(int level, const mfem::Vector& coarse, mfem::Vector& fine) const;
    virtual void Project(int level, const mfem::Vector& fine, mfem::Vector& coarse) const;
    virtual void Smoothing(int level, const mfem::Vector& in, mfem::Vector& out) const;
    virtual const int DomainSize(int level) const { return solvers_[level]->Size(); }
    virtual const int RangeSize(int level) const { return solvers_[level]->Size(); }
private:
    FiniteVolumeMLMC& up_;
    std::vector<unique_ptr<SingleLevelSolver> > solvers_;
    std::vector<mfem::Array<int> > offsets_;
    mfem::Vector PuTPu_diag_;
};

void Kappa(const mfem::Vector& p, mfem::Vector& kp);

mfem::Array<int> well_vertices;
int main(int argc, char* argv[])
{
    int num_procs, myid;

    // 1. Initialize MPI
    mpi_session session(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // program options from command line
    mfem::OptionsParser args(argc, argv);
    const char* permFile = "spe_perm.dat";
    args.AddOption(&permFile, "-p", "--perm",
                   "SPE10 permeability file data.");
    int nDimensions = 2;
    args.AddOption(&nDimensions, "-d", "--dim",
                   "Dimension of the physical space.");
    int slice = 0;
    args.AddOption(&slice, "-s", "--slice", "Slice of SPE10 data to take for 2D run.");
    int max_evects = 4;
    args.AddOption(&max_evects, "-m", "--max-evects",
                   "Maximum eigenvectors per aggregate.");
    double spect_tol = 1.0;
    args.AddOption(&spect_tol, "-t", "--spect-tol",
                   "Spectral tolerance for eigenvalue problems.");
    bool hybridization = false;
    args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                   "--no-hybridization", "Enable hybridization.");
    bool dual_target = true;
    args.AddOption(&dual_target, "-du", "--dual-target", "-no-du",
                   "--no-dual-target", "Use dual graph Laplacian in trace generation.");
    bool scaled_dual = true;
    args.AddOption(&scaled_dual, "-sd", "--scaled-dual", "-no-sd",
                   "--no-scaled-dual", "Scale dual graph Laplacian by (inverse) edge weight.");
    bool energy_dual = true;
    args.AddOption(&energy_dual, "-ed", "--energy-dual", "-no-ed",
                   "--no-energy-dual", "Use energy matrix in trace generation.");
    int nref_s = 4;
    args.AddOption(&nref_s, "-nsr", "--nref-ser", "Number of serial refinements.");
    int nref_p = 0;
    args.AddOption(&nref_p, "-npr", "--nref-par", "Number of parallel refinements.");
    int vis_step = 0;
    args.AddOption(&vis_step, "-vs", "--vis-step", "Step size for visualization.");
    int coarsening_factor = 100;
    args.AddOption(&coarsening_factor, "-cf", "--coarsen-factor", "Coarsening factor");
    double correlation_length = 0.1;
    args.AddOption(&correlation_length, "-cl", "--correlation-length", "Correlation length");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(std::cout);
        }
        MPI_Finalize();
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(std::cout);
    }

    const int nbdr = nDimensions == 2 ? 4 : 6;;
    mfem::Array<int> ess_attr(nbdr);
    ess_attr = 0;

    // Setting up finite volume discretization problem
    //    LognormalProblem fv_problem(nDimensions, nref_s, nref_p, correlation_length);
    SPE10Problem fv_problem("spe_perm.dat", nDimensions, 5, slice, ess_attr, 15, 0);

    auto& vertex_edge = fv_problem.GetVertexEdge();
    auto& edge_d_td = fv_problem.GetEdgeToTrueEdge();
    auto& weight = fv_problem.GetWeight();
    auto& local_weight = fv_problem.GetLocalWeight();
    auto& edge_bdr_att = fv_problem.GetEdgeBoundaryAttributeTable();

    mfem::Array<int> partition;
    int nparts = std::max(vertex_edge.Height() / coarsening_factor, 1);
    bool adaptive_part = false;
    bool use_edge_weight = false;
    PartitionVerticesByMetis(vertex_edge, weight, well_vertices, nparts,
                             partition, adaptive_part, use_edge_weight);

    // Create Upscaler and Solve
    FiniteVolumeMLMC fvupscale(comm, vertex_edge, weight, partition, edge_d_td,
                               edge_bdr_att, ess_attr, spect_tol, max_evects,
                               dual_target, scaled_dual, energy_dual, hybridization, false);
    fvupscale.PrintInfo();
    fvupscale.ShowSetupTime();
    fvupscale.MakeFineSolver();

    mfem::BlockVector rhs(fvupscale.GetFineBlockVector());
    rhs = 0.0;
    //    rhs.GetBlock(1) = (-1.0 * fv_problem.CellVolume());

    mfem::StopWatch ch;
    ch.Clear(); ch.Start();

    mfem::BlockVector sol_picard(fvupscale.GetFineBlockVector());
    sol_picard = 0.0;

    SingleLevelSolver sls(fv_problem, fvupscale, Fine, Picard);
    sls.SetPrintLevel(1);
    sls.Solve(rhs, sol_picard);

    if (myid == 0)
    {
        std::cout << "Picard iteration took " << sls.GetNumIterations()
                  << " iterations in " << ch.RealTime() << " seconds.\n";
    }

    ch.Clear(); ch.Start();

    mfem::BlockVector sol_nlmg(fvupscale.GetFineBlockVector());
    sol_nlmg = 0.0;
    NonlinearEllipticHierarchy hierarchy(fv_problem, fvupscale, Picard);
    NonlinearMG nlmg(hierarchy, V_CYCLE);
    nlmg.SetPrintLevel(1);
    nlmg.Solve(rhs, sol_nlmg);

    if (myid == 0)
    {
        std::cout << "Nonlinear MG took " << nlmg.GetNumIterations()
                  << " iterations in " << ch.RealTime() << " seconds.\n";
    }

    double p_err = CompareError(comm, sol_picard.GetBlock(1), sol_nlmg.GetBlock(1));
    if (myid == 0)
    {
        std::cout << "Relative errors: " << p_err << "\n";
    }

    if (vis_step > 0)
    {
        mfem::socketstream sout;
        fv_problem.VisSetup(sout, sol_picard.GetBlock(1), 0.0, 0.0, "");
    }

    return EXIT_SUCCESS;
}

SingleLevelSolver::SingleLevelSolver(const DarcyProblem& darcy_problem, FiniteVolumeMLMC& up,
                                     Level level, SolveType solve_type)
    : NonlinearSolver(up.GetComm(), up.GetNumTotalDofs(level)),
      level_(level), up_(up), solve_type_(solve_type)
{
    if (level_ == Fine)
    {
        up_.FineBlockOffsets(offsets_);
        block_f_ = make_unique<mfem::BlockVector>(offsets_);
        block_f_->GetBlock(1) = darcy_problem.GetVertexRHS();
        solver_ = make_unique<UpscaleFineBlockSolve>(up_);
    }
    else
    {
        up_.CoarseBlockOffsets(offsets_);
        block_f_ = make_unique<mfem::BlockVector>(offsets_);
        block_f_->GetBlock(1) = up_.Restrict(darcy_problem.GetVertexRHS());
        solver_ = make_unique<UpscaleCoarseBlockSolve>(up_);
    }
    block_f_->GetBlock(0) = 0.0;
    kp_.SetSize(block_f_->BlockSize(1));
    p_fine_.SetSize(up.GetNumVertexDofs(Fine));
}

void SingleLevelSolver::Mult(const mfem::Vector& x, mfem::Vector& Rx)
{
    assert(size_ == Rx.Size());
    assert(size_ == x.Size());
    mfem::BlockVector block_x(x.GetData(), offsets_);
    mfem::BlockVector block_Rx(Rx.GetData(), offsets_);

    block_Rx = 0.0;

    if (level_ == Fine)
    {
        Kappa(block_x.GetBlock(1), kp_);
    }
    else
    {
        up_.Interpolate(block_x.GetBlock(1), p_fine_);
        kp_.SetSize(p_fine_.Size());
        Kappa(p_fine_, kp_);
    }
    up_.RescaleCoefficient(level_, kp_);
    up_.Mult(level_, block_x, block_Rx);
    block_Rx -= *block_f_;
}

void SingleLevelSolver::Solve(const mfem::Vector& rhs, mfem::Vector& sol)
{
    assert(solve_type_ == Newton || solve_type_ == Picard);

    mfem::BlockVector block_sol(sol.GetData(), offsets_);
    mfem::BlockVector block_rhs(rhs.GetData(), offsets_);
    if (solve_type_ == Picard)
    {
        PicardSolve(block_rhs, block_sol);
    }
    else
    {
        NewtonSolve(block_rhs, block_sol);
    }
}

void SingleLevelSolver::PicardSolve(const mfem::BlockVector& rhs, mfem::BlockVector& x)
{
    mfem::BlockVector adjusted_source(*block_f_);

    adjusted_source += rhs;
    if (max_num_iter_ == 1)
    {
        PicardStep(adjusted_source, x);
    }
    else
    {
        double norm = mfem::ParNormlp(adjusted_source, 2, comm_);
        converged_ = false;
        for (iter_ = 0; iter_ < max_num_iter_; iter_++)
        {
            double resid = ResidualNorm(x, rhs);
            double rel_resid = resid / norm;

            if (myid_ == 0 && print_level_ > 0)
            {
                std::cout << "Picard iter " << iter_ << ":  rel resid = "
                          << rel_resid << "  abs resid = " << resid << "\n";
            }

            if (resid < atol_ || rel_resid < rtol_)
            {
                converged_ = true;
                break;
            }

            PicardStep(adjusted_source, x);
        }

        if (myid_ == 0 && !converged_ && print_level_ >= 0)
        {
            std::cout << "Warning: Picard iteration reached maximum number of iterations!\n";
        }
    }
}

void SingleLevelSolver::PicardStep(const mfem::BlockVector& rhs, mfem::BlockVector& x)
{
    if (level_ == Fine)
    {
        Kappa(x.GetBlock(1), kp_);
    }
    else
    {
        up_.Interpolate(x.GetBlock(1), p_fine_);
        kp_.SetSize(p_fine_.Size());
        Kappa(p_fine_, kp_);
    }
    up_.RescaleCoefficient(level_, kp_);

    if (level_ == Fine)
        up_.SetMaxIter(max_num_iter_ * 15);
    solver_->Mult(rhs, x);
}

void SingleLevelSolver::NewtonSolve(const mfem::BlockVector& rhs, mfem::BlockVector& x)
{
    // TBD...
}

NonlinearEllipticHierarchy::NonlinearEllipticHierarchy(
    const DarcyProblem& darcy_problem, FiniteVolumeMLMC& up, SolveType solve_type)
    : Hierarchy(up.GetComm(), 2), up_(up), solvers_(2), offsets_(2)
{
    solvers_[0] = make_unique<SingleLevelSolver>(darcy_problem, up_, Fine, solve_type);
    solvers_[1] = make_unique<SingleLevelSolver>(darcy_problem, up_, Coarse, solve_type);
    solvers_[0]->SetPrintLevel(-1);
    solvers_[1]->SetPrintLevel(-1);
    solvers_[0]->SetMaxIter(1);
    solvers_[1]->SetMaxIter(1);
    up_.FineBlockOffsets(offsets_[0]);
    up_.CoarseBlockOffsets(offsets_[1]);
}

void NonlinearEllipticHierarchy::Mult(int level, const mfem::Vector& x, mfem::Vector& Rx) const
{
    solvers_[level]->Mult(x, Rx);
}

void NonlinearEllipticHierarchy::Solve(int level, const mfem::Vector& rhs, mfem::Vector& sol)
{
    solvers_[level]->Solve(rhs, sol);
}

void NonlinearEllipticHierarchy::Restrict(
    int level, const mfem::Vector& fine, mfem::Vector& coarse) const
{
    assert(level == 0);
    mfem::BlockVector block_fine(fine.GetData(), offsets_[level]);
    mfem::BlockVector block_coarse(coarse.GetData(), offsets_[level + 1]);
    up_.Restrict(block_fine, block_coarse);
}

void NonlinearEllipticHierarchy::Interpolate(
    int level, const mfem::Vector& coarse, mfem::Vector& fine) const
{
    assert(level == 1);
    mfem::BlockVector block_fine(fine.GetData(), offsets_[level - 1]);
    mfem::BlockVector block_coarse(coarse.GetData(), offsets_[level]);
    up_.Interpolate(block_coarse, block_fine);
}

void NonlinearEllipticHierarchy::Project(
    int level, const mfem::Vector& fine, mfem::Vector& coarse) const
{
    assert(level == 0);
    mfem::BlockVector block_fine(fine.GetData(), offsets_[level]);
    mfem::BlockVector block_coarse(coarse.GetData(), offsets_[level + 1]);
    up_.Project(block_fine, block_coarse);
}

void NonlinearEllipticHierarchy::Smoothing(
    int level, const mfem::Vector& in, mfem::Vector& out) const
{
    solvers_[level]->Solve(in, out);
}

// Kappa(p) = exp(- \alpha p)
void Kappa(const mfem::Vector& p, mfem::Vector& kp)
{
    assert(kp.Size() == p.Size());
    for (int i = 0; i < p.Size(); i++)
    {
        kp(i) = std::exp(-3e-6 * p(i));
        assert(kp(i) > 0.0);
    }
}
