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
   @file nldarcy.cpp
   @brief nonlinear Darcy's problem.
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "pde.hpp"

#include "../src/picojson.h"

using namespace smoothg;

using std::unique_ptr;

enum Soil { Loam, Sand };

/// storing constants needed to evaluate nonlinear permeability multiplier
struct Kappa
{
    double alpha;
    double beta;
    double K_s; // cm/day
    mfem::Vector Z;

    Kappa(double alpha) : alpha(alpha), beta(0.0), K_s(0.0) { }
    Kappa(const Soil& soil, const mfem::Vector& Z_in);
    mfem::Vector Eval(const mfem::Vector& p) const;
    mfem::Vector dKinv_dp(const mfem::Vector& p) const;
};

/**
   @brief Solver for nonlinear elliptic problem associated with div (K(p) grad p)

   Given \f$f \in L^2(\Omega)\f$, \f$k(p)\f$ a differentiable function of p,
   find \f$p\f$ such that \f$-div(k_0k(p)\nabla p) = f\f$.
*/
class LevelSolver : public NonlinearSolver
{
public:
    /// Constructor
    LevelSolver(const MixedMatrix& mixed_system, int level, Kappa kappa,
                const mfem::Array<int>& ess_attr, NLSolverParameters param);

    void Mult(const mfem::Vector& x, mfem::Vector& Ax) override;

    /// Set relative tolerance of solver for linearized problem
    void SetLinearRelTol(double tol) { linear_solver_->SetRelTol(tol); }

    /// Shrink delta x (change in solution) in case of bad step
    void BackTracking(const mfem::Vector& rhs,  double prev_resid_norm,
                      mfem::Vector& x, mfem::Vector& dx, bool interlopate = false);
private:
    void PicardStep(const mfem::BlockVector& rhs, mfem::BlockVector& x);
    void NewtonStep(const mfem::BlockVector& rhs, mfem::BlockVector& x);
    void Build_dMdp(const mfem::Vector& flux, const mfem::Vector& p);
    void IterationStep(const mfem::Vector& rhs, mfem::Vector& sol) override;
    mfem::Vector AssembleTrueVector(const mfem::Vector& vec) const override;
    const mfem::Array<int>& GetEssDofs() const override { return mixed_system_.GetEssDofs(); }

    int level_;
    const MixedMatrix& mixed_system_;
    const mfem::Array<int>& offsets_;

    unique_ptr<MixedLaplacianSolver> linear_solver_;

    mfem::Vector p_;         // projected pressure in piecewise 1 basis
    mfem::Vector kp_;        // kp_ = Kappa(p)
    mfem::Vector dkinv_dp_;  // dkinv_dp_ = d ( Kappa(p)^{-1} ) / dp

    std::vector<mfem::DenseMatrix> dMdp_;
    Kappa kappa_;
};

// FAS for nonlinear elliptic problem
class EllipticNLMG : public FAS
{
public:
    EllipticNLMG(const Hierarchy& hierarchy_, const Kappa& kappa,
                 const mfem::Array<int>& ess_attr, FASParameters param);
private:
    void Mult(int level, const mfem::Vector& x, mfem::Vector& Ax) override;
    void Restrict(int level, const mfem::Vector& fine, mfem::Vector& coarse) const override;
    void Interpolate(int level, const mfem::Vector& coarse, mfem::Vector& fine) const override;
    void Project(int level, const mfem::Vector& fine, mfem::Vector& coarse) const override;
    void Smoothing(int level, const mfem::Vector& in, mfem::Vector& out) override;
    void BackTracking(int level, const mfem::Vector& rhs, double prev_resid_norm,
                      mfem::Vector& x, mfem::Vector& dx) override;
    mfem::Vector AssembleTrueVector(int level, const mfem::Vector& vec) const override;
    const mfem::Array<int>& GetEssDofs(int level) const override;

    const Hierarchy& hierarchy_;
    std::vector<LevelSolver> solvers_;
};

void SetOptions(FASParameters& param, bool use_vcycle, bool use_newton,
                int num_backtrack, double diff_tol);

int main(int argc, char* argv[])
{
    int num_procs, myid;
    picojson::object serialize;

    // 1. Initialize MPI
    mpi_session session(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // program options from command line
    mfem::OptionsParser args(argc, argv);

    UpscaleParameters upscale_param;
    upscale_param.RegisterInOptionsParser(args);

    const char* problem_name = "spe10";
    args.AddOption(&problem_name, "-mp", "--model-problem",
                   "Model problem (spe10, egg, lognormal, richard)");
    const char* perm_file = "spe_perm.dat";
    args.AddOption(&perm_file, "-p", "--perm", "SPE10 permeability file data.");
    int dim = 2;
    args.AddOption(&dim, "-d", "--dim",
                   "Dimension of the physical space.");
    int slice = 0;
    args.AddOption(&slice, "-s", "--slice",
                   "Slice of SPE10 data to take for 2D run.");
    int num_sr = 0;
    args.AddOption(&num_sr, "-nsr", "--num-serial-refine",
                   "Number of serial refinement");
    int num_pr = 0;
    args.AddOption(&num_pr, "-npr", "--num-parallel-refine",
                   "Number of parallel refinement");
    double correlation = 0.1;
    args.AddOption(&correlation, "-cl", "--correlation-length",
                   "Correlation length");
    double alpha = 1.0;
    args.AddOption(&alpha, "-alpha", "--alpha", "alpha");
    bool use_newton = true;
    args.AddOption(&use_newton, "-newton", "--use-newton", "-picard",
                   "--use-picard", "Use Newton or Picard iteration.");
    bool use_vcycle = true;
    args.AddOption(&use_vcycle, "-VCycle", "--use-VCycle", "-FMG",
                   "--use-FMG", "Use V-cycle or FMG-cycle.");
    bool visualization = false;
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization", "Enable visualization.");
    int num_backtrack = 4;
    args.AddOption(&num_backtrack, "--num-backtrack", "--num-backtrack",
                   "Maximum number of backtracking steps.");
    double diff_tol = -1.0;
    args.AddOption(&diff_tol, "--diff-tol", "--diff-tol",
                   "Tolerance for coefficient change.");
    FASParameters mg_param;
    args.AddOption(&mg_param.fine.max_num_iter, "--num-relax-fine", "--num-relax-fine",
                   "Number of relaxation in fine level.");
    args.AddOption(&mg_param.mid.max_num_iter, "--num-relax-mid", "--num-relax-mid",
                   "Number of relaxation in intermediate levels.");
    args.AddOption(&mg_param.coarse.max_num_iter, "--num-relax-coarse", "--num-relax-coarse",
                   "Number of relaxation in coarse level.");
    args.AddOption(&mg_param.nl_solve.init_linear_tol, "--init-linear-tol", "--init-linear-tol",
                   "Initial tol for linear solve inside nonlinear iterations.");
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
    mg_param.num_levels = upscale_param.max_levels;
    SetOptions(mg_param, use_vcycle, use_newton, num_backtrack, diff_tol);

    // Setting up finite volume discretization problem
    double use_metis = true;
    std::string problem(problem_name);
    mfem::Array<int> ess_attr(problem == "egg" ? 3 : (dim == 3 ? 6 : 4));
    ess_attr = 1;

    unique_ptr<DarcyProblem> fv_problem;
    Kappa kappa(alpha);
    if (problem == "spe10")
    {
        ess_attr[dim - 2] = 0;
        fv_problem.reset(new SPE10Problem(perm_file, dim, 5, slice, use_metis, ess_attr));

    }
    else if (problem == "egg")
    {
        ess_attr[1] = 0;
        fv_problem.reset(new EggModel(num_sr, num_pr, ess_attr));
    }
    else if (problem == "lognormal")
    {
        ess_attr = 0;
        fv_problem.reset(new LognormalModel(dim, num_sr, num_pr, correlation, ess_attr));
    }
    else if (problem == "richard")
    {
        ess_attr[0] = 0;
        fv_problem.reset(new Richards(num_sr, ess_attr));
        kappa = Kappa(Loam, fv_problem->ComputeZ());
    }
    else
    {
        mfem::mfem_error("Unknown model problem!");
    }

    Graph graph = fv_problem->GetFVGraph(true);

    mfem::Array<int> partitioning;
    if (upscale_param.max_levels > 1)
    {
        mfem::Array<int> coarsening_factors(problem == "egg" ? 3 : dim);
        coarsening_factors = 1;
        coarsening_factors[0] = upscale_param.coarse_factor;
        fv_problem->Partition(use_metis, coarsening_factors, partitioning);
        upscale_param.num_iso_verts = fv_problem->NumIsoVerts();
    }

    // Create hierarchy
    Hierarchy hierarchy(graph, upscale_param, &partitioning, &ess_attr);
    hierarchy.PrintInfo();

    mfem::BlockVector rhs(hierarchy.BlockOffsets(0));
    rhs.GetBlock(0) = fv_problem->GetEdgeRHS();
    rhs.GetBlock(1) = fv_problem->GetVertexRHS();

    mfem::BlockVector sol_nlmg(rhs);
    sol_nlmg = 0.0;

    EllipticNLMG nlmg(hierarchy, kappa, ess_attr, mg_param);
    nlmg.SetPrintLevel(1);
    nlmg.SetRelTol(1e-6);
    nlmg.SetMaxIter(200);
    nlmg.Solve(rhs, sol_nlmg);

    serialize["nonlinear-iterations"] = picojson::value((double)nlmg.GetNumIterations());

    if (visualization)
    {
        sol_nlmg.GetBlock(1).Add(problem == "richard" ? -1.0 : 0.0, kappa.Z);

        mfem::socketstream sout;
        fv_problem->VisSetup(sout, sol_nlmg.GetBlock(0), 0.0, 0.0, "", false, false);
        fv_problem->VisSetup(sout, sol_nlmg.GetBlock(1), 0.0, 0.0, "", false, true);
        if (problem == "richard")
            sout << "keys ]]]]]]]]]]]]]]]]]]]]]]]]]]]]fmm\n";
    }

    if (myid == 0)
    {
        std::cout << picojson::value(serialize).serialize(true) << std::endl;
    }

    return EXIT_SUCCESS;
}

void SetOptions(FASParameters& param, bool use_vcycle, bool use_newton,
                int num_backtrack, double diff_tol)
{
    param.cycle = use_vcycle ? V_CYCLE : FMG;
    param.nl_solve.linearization = use_newton ? Newton : Picard;
    param.coarse_correct_tol = use_newton ? 1e-4 : 1e-8;
    param.fine.check_converge = false;
    param.fine.linearization = param.nl_solve.linearization;
    param.mid.linearization = param.nl_solve.linearization;
    param.coarse.linearization = param.nl_solve.linearization;
    param.fine.num_backtrack = num_backtrack;
    param.mid.num_backtrack = num_backtrack;
    param.coarse.num_backtrack = num_backtrack;
    param.fine.diff_tol = diff_tol;
    param.mid.diff_tol = diff_tol;
    param.coarse.diff_tol = diff_tol;
}

LevelSolver::LevelSolver(const MixedMatrix& mixed_system, int level, Kappa kappa,
                         const mfem::Array<int>& ess_attr, NLSolverParameters param)
    : NonlinearSolver(mixed_system.GetComm(), mixed_system.BlockOffsets()[2], param),
      level_(level), mixed_system_(mixed_system), offsets_(mixed_system_.BlockOffsets()),
      p_(mixed_system.GetGraph().NumVertices()), kp_(p_.Size()), dkinv_dp_(p_.Size()),
      kappa_(std::move(kappa))
{
    tag_ = param.linearization ? "Picard" : "Newton";

    if (level > 0) // Hybridization solver
    {
        linear_solver_.reset(new HybridSolver(mixed_system_, &ess_attr));
    }
    else // L2-H1 block diagonal preconditioner
    {
        linear_solver_.reset(new MinresBlockSolverFalse(mixed_system_, &ess_attr));
    }
    linear_solver_->SetPrintLevel(-1);
    linear_solver_->SetMaxIter(200);
}

void LevelSolver::Mult(const mfem::Vector& x, mfem::Vector& Ax)
{
    assert(size_ == Ax.Size());
    assert(size_ == x.Size());
    mfem::BlockVector block_x(x.GetData(), offsets_);
    mfem::BlockVector block_Ax(Ax.GetData(), offsets_);

    p_ = mixed_system_.PWConstProject(block_x.GetBlock(1));
    kp_ = kappa_.Eval(p_);
    mixed_system_.Mult(kp_, block_x, block_Ax);
}

mfem::Vector LevelSolver::AssembleTrueVector(const mfem::Vector& vec) const
{
    return mixed_system_.AssembleTrueVector(vec);
}

void LevelSolver::IterationStep(const mfem::Vector& rhs, mfem::Vector& sol)
{
    mfem::BlockVector block_sol(sol.GetData(), offsets_);
    mfem::BlockVector block_rhs(rhs.GetData(), offsets_);

    if (param_.linearization == Picard)
    {
        PicardStep(block_rhs, block_sol);
    }
    else
    {
        NewtonStep(block_rhs, block_sol);
    }
}

void LevelSolver::PicardStep(const mfem::BlockVector& rhs, mfem::BlockVector& x)
{
    mfem::BlockVector delta_x(x);
    prev_resid_norm_ = ResidualNorm(x, rhs); // kp_ is updated in ResidualNorm

    linear_solver_->UpdateElemScaling(kp_);
    linear_solver_->Solve(rhs, x);
    delta_x -= x;

    BackTracking(rhs, prev_resid_norm_, x, delta_x);
}

void LevelSolver::NewtonStep(const mfem::BlockVector& rhs, mfem::BlockVector& x)
{
    Mult(x, residual_);  // p_ and kp_ is updated in Mult
    residual_ -= rhs;
    SetZeroAtMarker(GetEssDofs(), residual_);

    Build_dMdp(x.GetBlock(0), p_);
    linear_solver_->UpdateJacobian(kp_, dMdp_);

    mfem::BlockVector block_residual(residual_.GetData(), offsets_);
    mfem::BlockVector delta_x = linear_solver_->Solve(block_residual);
    x -= delta_x;

    mfem::Vector true_resid = AssembleTrueVector(block_residual);
    prev_resid_norm_ = mfem::ParNormlp(true_resid, 2, comm_);

    BackTracking(rhs, prev_resid_norm_, x, delta_x);
}

void LevelSolver::BackTracking(const mfem::Vector& rhs, double prev_resid_norm,
                               mfem::Vector& x, mfem::Vector& dx, bool interlopate)
{
    mfem::BlockVector block_dx(dx.GetData(), offsets_);

    if (!interlopate && param_.diff_tol > 0) // constrain change in p
    {
        auto delta_p = mixed_system_.PWConstProject(block_dx.GetBlock(1));

        auto max_p_change = AbsMax(delta_p, comm_);
        auto relative_change = max_p_change * kappa_.alpha / std::log(param_.diff_tol);

        if (relative_change > 1)
        {
            dx /= relative_change;
            x.Add(relative_change - 1.0, dx);
        }
    }

    if (param_.num_backtrack == 0) { return; }

    int k = 0;
    resid_norm_ = ResidualNorm(x, rhs);

    while (k < param_.num_backtrack && resid_norm_ > prev_resid_norm)
    {
        double backtracking_resid_norm = resid_norm_;

        dx *= 0.5;
        x += dx;

        resid_norm_ = ResidualNorm(x, rhs);

        if (resid_norm_ > 0.9 * backtracking_resid_norm)
        {
            x -= dx;
            break;
        }

        if (myid_ == 0  && param_.print_level > 1)
        {
            if (k == 0)
            {
                std::cout << "  Level " << level_ << " backtracking: || R(u) ||";
            }
            std::cout << " -> " << backtracking_resid_norm;
        }
        k++;
    }

    if (k > 0 && myid_ == 0 && param_.print_level > 1)
    {
        std::cout << "\n";
    }
}

void LevelSolver::Build_dMdp(const mfem::Vector& flux, const mfem::Vector& p)
{
    auto& vert_edof = mixed_system_.GetGraphSpace().VertexToEDof();
    auto& vert_vdof = mixed_system_.GetGraphSpace().VertexToVDof();
    auto& MB = dynamic_cast<const ElementMBuilder&>(mixed_system_.GetMBuilder());
    auto& M_el = MB.GetElementMatrices();
    auto& proj_pwc = const_cast<mfem::SparseMatrix&>(mixed_system_.GetPWConstProj());

    dMdp_.resize(M_el.size());
    mfem::Array<int> local_edofs, local_vdofs, vert(1);
    mfem::Vector sigma_loc, Msigma_vec;
    mfem::DenseMatrix proj_pwc_loc;

    dkinv_dp_ = kappa_.dKinv_dp(p);

    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        GetTableRow(vert_vdof, i, local_vdofs);
        vert[0] = i;

        flux.GetSubVector(local_edofs, sigma_loc);
        Msigma_vec.SetSize(local_edofs.Size());
        M_el[i].Mult(sigma_loc, Msigma_vec);
        mfem::DenseMatrix Msigma_loc(Msigma_vec.GetData(), M_el[i].Size(), 1);

        proj_pwc_loc.SetSize(1, local_vdofs.Size());
        proj_pwc_loc = 0.0;
        proj_pwc.GetSubMatrix(vert, local_vdofs, proj_pwc_loc);
        proj_pwc_loc *= dkinv_dp_[i];

        dMdp_[i].SetSize(local_edofs.Size(), local_vdofs.Size());
        mfem::Mult(Msigma_loc, proj_pwc_loc, dMdp_[i]);
    }
}

EllipticNLMG::EllipticNLMG(const Hierarchy& hierarchy, const Kappa& kappa,
                           const mfem::Array<int>& ess_attr, FASParameters param)
    : FAS(hierarchy.GetComm(), hierarchy.BlockOffsets(0)[2], param), hierarchy_(hierarchy)
{
    std::vector<mfem::Vector> help(param.num_levels);
    solvers_.reserve(param.num_levels);
    for (int l = 0; l < param.num_levels; ++l)
    {
        Kappa kappa_l(kappa);
        if (kappa.Z.Size())
        {
            help[l] = l ? hierarchy.Project(l - 1, help[l - 1]) : kappa.Z;
            kappa_l.Z = hierarchy.PWConstProject(l, help[l]);
        }

        auto& matrix_l = hierarchy.GetMatrix(l);
        auto& param_l = l ? (l < param.num_levels - 1 ? param.mid : param.coarse) : param.fine;
        solvers_.emplace_back(matrix_l, l, std::move(kappa_l), ess_attr, param_l);
        solvers_[l].SetPrintLevel(param_.cycle == V_CYCLE ? -1 : 0);

        if (l > 0)
        {
            rhs_[l].SetSize(matrix_l.NumTotalDofs());
            sol_[l].SetSize(matrix_l.NumTotalDofs());
            rhs_[l] = 0.0;
            sol_[l] = 0.0;
        }
        help_[l].SetSize(matrix_l.NumTotalDofs());
        help_[l] = 0.0;

        if (myid_ == 0 && param.nl_solve.print_level > -1)
        {
            std::cout << "\nMG level " << l << " parameters:\n  Number of "
                      << "smoothing steps: " << param_l.max_num_iter << "\n"
                      << "  Kappa change tol: " << param_l.diff_tol << "\n"
                      << "  Max number of residual-based backtracking: "
                      << param_l.num_backtrack << "\n";
        }
    }
}

void EllipticNLMG::Mult(int level, const mfem::Vector& x, mfem::Vector& Ax)
{
    solvers_[level].Mult(x, Ax);
}

void EllipticNLMG::Restrict(int level, const mfem::Vector& fine, mfem::Vector& coarse) const
{
    mfem::BlockVector block_fine(fine.GetData(), hierarchy_.BlockOffsets(level));
    coarse = hierarchy_.Restrict(level, block_fine);
}

void EllipticNLMG::Interpolate(int level, const mfem::Vector& coarse, mfem::Vector& fine) const
{
    mfem::BlockVector block_coarse(coarse.GetData(), hierarchy_.BlockOffsets(level));
    fine = hierarchy_.Interpolate(level, block_coarse);
}

void EllipticNLMG::Project(int level, const mfem::Vector& fine, mfem::Vector& coarse) const
{
    mfem::BlockVector block_fine(fine.GetData(), hierarchy_.BlockOffsets(level));
    coarse = hierarchy_.Project(level, block_fine);
}

void EllipticNLMG::Smoothing(int level, const mfem::Vector& in, mfem::Vector& out)
{
    solvers_[level].SetLinearRelTol(linear_tol_);
    solvers_[level].Solve(in, out);
}

void EllipticNLMG::BackTracking(int level, const mfem::Vector& rhs, double prev_resid_norm,
                                mfem::Vector& x, mfem::Vector& dx)
{
    solvers_[level].BackTracking(rhs, prev_resid_norm, x, dx, true);
}

mfem::Vector EllipticNLMG::AssembleTrueVector(int level, const mfem::Vector& vec) const
{
    return hierarchy_.GetMatrix(level).AssembleTrueVector(vec);
}

const mfem::Array<int>& EllipticNLMG::GetEssDofs(int level) const
{
    return hierarchy_.GetMatrix(level).GetEssDofs();
}

Kappa::Kappa(const Soil& soil, const mfem::Vector& Z_in)
    : alpha(soil == Loam ? 124.6 : 1.175e6), beta(soil == Loam ? 1.77 : 4.74),
      K_s(soil == Loam ? 1.067 : 816.), Z(Z_in) { }

mfem::Vector Kappa::Eval(const mfem::Vector& p) const
{
    mfem::Vector out(p.Size());

    if (Z.Size() == 0)    // Kappa(p) = exp(\alpha p)
    {
        for (int i = 0; i < p.Size(); i++)
        {
            out[i] = std::exp(alpha * p[i]);
            assert(out[i] > 0.0);
        }
    }
    else    // Kappa(p) = K\alpha / (\alpha + |p(x, y, z) - z|^\beta)
    {
        const double alpha_K_s = K_s * alpha;

        mfem::Vector out(p.Size());
        assert(Z.Size() == p.Size());
        for (int i = 0; i < p.Size(); i++)
        {
            out[i] = alpha_K_s / (alpha + std::pow(std::fabs(p[i] - Z[i]), beta));
            assert(out[i] > 0.0);
        }
        return out;
    }
    return out;
}

mfem::Vector Kappa::dKinv_dp(const mfem::Vector& p) const
{
    mfem::Vector out(p.Size());

    if (Z.Size() == 0)
    {
        for (int i = 0; i < p.Size(); i++)
        {
            double exp_ap = std::exp(alpha * p[i]);
            assert(exp_ap > 0.0);
            out[i] = -(alpha * exp_ap) / (exp_ap * exp_ap);
        }
    }
    else
    {
        const double b_over_a_K_s = beta / (K_s * alpha);
        const double beta_minus_1 = beta - 1.0;

        mfem::Vector out(p.Size());
        assert(Z.Size() == p.Size());
        for (int i = 0; i < p.Size(); i++)
        {
            double p_head = p[i] - Z[i];
            double sign = p_head < 0.0 ? -1.0 : 1.0;
            out[i] = sign * b_over_a_K_s * std::pow(std::fabs(p_head), beta_minus_1);
        }
    }
    return out;
}
