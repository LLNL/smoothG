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

using namespace smoothg;

using std::unique_ptr;

/**
   @brief Nonlinear elliptic problem

   Given \f$f \in L^2(\Omega)\f$, \f$k(p)\f$ a differentiable function of p,
   find \f$p\f$ such that \f$-div(k_0k(p)\nabla p) = f\f$.
*/

class LevelSolver : public NonlinearSolver
{
public:
    /**
       @todo take Kappa(p) as input
    */
    LevelSolver(Hierarchy& hierarchy, int level,
                mfem::Vector Z_vector, SolveType solve_type);

    // Compute Ax = A(x).
    virtual void Mult(const mfem::Vector& x, mfem::Vector& Ax);

    double GetLinearResidualNorm() const { return linear_resid_norm_; }
private:
    void EvalCoef(const mfem::Vector& sol_block1);
    void EvalCoefDerivative(const mfem::Vector& sol_block1);
    void PicardStep(const mfem::BlockVector& rhs, mfem::BlockVector& x);
    void NewtonStep(const mfem::BlockVector& rhs, mfem::BlockVector& x);
    void Build_dMdp(const mfem::BlockVector& iterate);

    virtual void IterationStep(const mfem::Vector& rhs, mfem::Vector& sol);

    virtual mfem::Vector AssembleTrueVector(const mfem::Vector& vec) const;

    double LinearResidualNorm(const mfem::Vector& x, const mfem::Vector& y) const;

    virtual const mfem::Array<int>& GetEssDofs() const
    {
        return hierarchy_.GetMatrix(level_).GetEssDofs();
    }

    int level_;
    Hierarchy& hierarchy_;
    SolveType solve_type_;

    const mfem::Array<int>& offsets_;
    mfem::Vector p_;         // coefficient vector in piecewise 1 basis
    mfem::Vector kp_;        // kp_ = Kappa(p)
    mfem::Vector dkinv_dp_;  // dkinv_dp_ = d ( Kappa(p)^{-1} ) / dp

    std::vector<mfem::DenseMatrix> dMdp_;
    mfem::Vector Z_vector_;
};

// nonlinear elliptic hierarchy
class EllipticNLMG : public NonlinearMG
{
public:
    EllipticNLMG(Hierarchy& hierarchy_, const mfem::Vector& Z_fine,
                 Cycle cycle, SolveType solve_type);
    void Solve(const mfem::Vector& rhs, mfem::Vector& sol)
    {
        NonlinearSolver::Solve(rhs, sol);
    }

private:
    virtual void Mult(int level, const mfem::Vector& x, mfem::Vector& Ax);
    virtual void Solve(int level, const mfem::Vector& rhs, mfem::Vector& sol);
    virtual void Restrict(int level, const mfem::Vector& fine, mfem::Vector& coarse) const;
    virtual void Interpolate(int level, const mfem::Vector& coarse, mfem::Vector& fine) const;
    virtual void Project(int level, const mfem::Vector& fine, mfem::Vector& coarse) const;
    virtual void Smoothing(int level, const mfem::Vector& in, mfem::Vector& out);
    virtual mfem::Vector AssembleTrueVector(const mfem::Vector& vec) const;
    virtual int LevelSize(int level) const;

    const mfem::Array<int>& Offsets(int level) const;

    virtual const mfem::Array<int>& GetEssDofs() const
    {
        return hierarchy_.GetMatrix(0).GetEssDofs();
    }

    Hierarchy& hierarchy_;
    std::vector<LevelSolver> solvers_;
};

double alpha;
// Kappa(p) = exp(\alpha p)
void Kappa(const mfem::Vector& p, mfem::Vector& kp);
void dKinv_dp(const mfem::Vector& p, mfem::Vector& dkinv_dp);

// Kappa(p) = K\alpha / (\alpha + |p(x, y, z) - z|^\beta)
void Kappa(const mfem::Vector& p, const mfem::Vector& Z_vec, mfem::Vector& kp);
void dKinv_dp(const mfem::Vector& p, const mfem::Vector& Z_vec, mfem::Vector& dkinv_dp);

int main(int argc, char* argv[])
{
    int num_procs, myid;

    // 1. Initialize MPI
    mpi_session session(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // program options from command line
    UpscaleParameters upscale_param;
    upscale_param.spect_tol = 1.0;
    mfem::OptionsParser args(argc, argv);
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
    int num_sr = 3;
    args.AddOption(&num_sr, "-nsr", "--num-serial-refine",
                   "Number of serial refinement");
    int num_pr = 0;
    args.AddOption(&num_pr, "-npr", "--num-parallel-refine",
                   "Number of parallel refinement");
    double correlation = 0.1;
    args.AddOption(&correlation, "-cl", "--correlation-length",
                   "Correlation length");
    bool use_newton = false;
    args.AddOption(&use_newton, "-newton", "--use-newton", "-picard",
                   "--use-picard", "Use Newton or Picard iteration.");
    bool visualization = false;
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization", "Enable visualization.");
    // Read upscaling options from command line into upscale_param object
    upscale_param.RegisterInOptionsParser(args);
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

    // Setting up finite volume discretization problem
    std::string problem(problem_name);
    mfem::Array<int> ess_attr(problem == "egg" ? 3 : (dim == 3 ? 6 : 4));
    ess_attr = 0;

    mfem::Vector Z_fine;
    unique_ptr<DarcyProblem> fv_problem;
    if (problem == "spe10")
    {
        fv_problem.reset(new SPE10Problem(perm_file, dim, 5, slice, 0, ess_attr));
        alpha = -5.5e-6;
    }
    else if (problem == "egg")
    {
        fv_problem.reset(new EggModel(num_sr, num_pr, ess_attr));
        alpha = 3.;//-5e-1;
    }
    else if (problem == "lognormal")
    {
        fv_problem.reset(new LognormalModel(dim, num_sr, num_pr, correlation, ess_attr));
        alpha = -8e0;
    }
    else if (problem == "richard")
    {
        ess_attr = 1;
        ess_attr[0] = 0;
        fv_problem.reset(new Richards(num_sr, ess_attr));
        Z_fine = fv_problem->GetZVector();
        alpha = 124.6;          // Loam
//        alpha = 1.175e6;    // sand
    }
    else
    {
        mfem::mfem_error("Unknown model problem!");
    }

    Graph graph = fv_problem->GetFVGraph(true);

    // Create hierarchy
    Hierarchy hierarchy(std::move(graph), upscale_param, nullptr, &ess_attr);
    hierarchy.PrintInfo();

    if (upscale_param.hybridization)
    {
        hierarchy.SetRelTol(1e-12);
        hierarchy.SetAbsTol(1e-15);
    }

    mfem::BlockVector rhs(hierarchy.GetMatrix(0).BlockOffsets());
    if (problem == "richard")
    {
        rhs.GetBlock(0) = fv_problem->GetEdgeRHS();
        rhs.GetBlock(1) = fv_problem->GetVertexRHS();
    }
    else
    {
        rhs.GetBlock(0) = 0.0;
        rhs.GetBlock(1) = -fv_problem->CellVolume();
    }

    mfem::BlockVector sol_picard(rhs);
    sol_picard = 0.0;
    LevelSolver sls(hierarchy, 0, Z_fine, use_newton ? Newton : Picard);
    sls.SetPrintLevel(1);
    sls.SetMaxIter(2000);
    sls.Solve(rhs, sol_picard);

    mfem::BlockVector sol_nlmg(rhs);
    sol_nlmg = 0.0;
    EllipticNLMG nlmg(hierarchy, Z_fine, V_CYCLE, use_newton ? Newton : Picard);
    nlmg.SetPrintLevel(1);
    nlmg.SetMaxIter(2000);
    nlmg.Solve(rhs, sol_nlmg);

    double p_err = CompareError(comm, sol_nlmg.GetBlock(1), sol_picard.GetBlock(1));
    if (myid == 0)
    {
        std::cout << "Relative errors: " << p_err << "\n";
    }

    if (visualization)
    {
        if (problem == "richard")
        {
            sol_picard.GetBlock(1) -= Z_fine;
            sol_nlmg.GetBlock(1) -= Z_fine;
        }

        mfem::socketstream sout;
        fv_problem->VisSetup(sout, sol_picard.GetBlock(1), 0.0, 0.0, "");
        if (problem == "richard")
            sout << "keys ]]]]]]]]]]]]]]]]]]]]]]]]]]]]fmm\n";
        fv_problem->VisSetup(sout, sol_nlmg.GetBlock(1), 0.0, 0.0, "");
        if (problem == "richard")
            sout << "keys ]]]]]]]]]]]]]]]]]]]]]]]]]]]]fmm\n";
    }

    return EXIT_SUCCESS;
}

/// @todo take MixedMatrix only
LevelSolver::LevelSolver(Hierarchy& hierarchy, int level,
                         mfem::Vector Z_vector, SolveType solve_type)
    : NonlinearSolver(hierarchy.GetComm(), hierarchy.BlockOffsets(level)[2],
                      solve_type == Picard ? "Picard" : "Newton"),
      level_(level), hierarchy_(hierarchy), solve_type_(solve_type),
      offsets_(hierarchy_.BlockOffsets(level)), p_(hierarchy_.NumVertices(level)),
      kp_(p_.Size()), dkinv_dp_(p_.Size()), Z_vector_(std::move(Z_vector))
{
    hierarchy_.SetPrintLevel(0);
    hierarchy_.SetAbsTol(0);
    hierarchy_.SetMaxIter(100);
}

void LevelSolver::Mult(const mfem::Vector& x, mfem::Vector& Ax)
{
    assert(size_ == Ax.Size());
    assert(size_ == x.Size());
    mfem::BlockVector block_x(x.GetData(), offsets_);
    mfem::BlockVector block_Ax(Ax.GetData(), offsets_);

    EvalCoef(block_x.GetBlock(1));
    hierarchy_.GetMatrix(level_).Mult(kp_, block_x, block_Ax);
}

mfem::Vector LevelSolver::AssembleTrueVector(const mfem::Vector& vec) const
{
    return hierarchy_.GetMatrix(level_).AssembleTrueVector(vec);
}

void LevelSolver::IterationStep(const mfem::Vector& rhs, mfem::Vector& sol)
{
    mfem::BlockVector block_sol(sol.GetData(), offsets_);
    mfem::BlockVector block_rhs(rhs.GetData(), offsets_);

    if (max_num_iter_ > 1)
    {
        hierarchy_.SetRelTol(linear_tol_);
    }

    if (solve_type_ == Picard)
    {
        PicardStep(block_rhs, block_sol);
    }
    else
    {
        NewtonStep(block_rhs, block_sol);
    }
}

void LevelSolver::EvalCoef(const mfem::Vector& sol_block1)
{
    p_ = hierarchy_.PWConstProject(level_, sol_block1);
    if (Z_vector_.Size())
        Kappa(p_, Z_vector_, kp_);
    else
        Kappa(p_, kp_);
}

void LevelSolver::EvalCoefDerivative(const mfem::Vector& sol_block1)
{
//    p_ = hierarchy_.PWConstProject(level_, sol_block1);
    if (Z_vector_.Size())
        dKinv_dp(p_, Z_vector_, dkinv_dp_);
    else
        dKinv_dp(p_, dkinv_dp_);
}

void LevelSolver::PicardStep(const mfem::BlockVector& rhs, mfem::BlockVector& x)
{
    EvalCoef(x.GetBlock(1));
    hierarchy_.RescaleCoefficient(level_, kp_);
    hierarchy_.Solve(level_, rhs, x);

    if (linear_tol_criterion_ == TaylorResidual)
    {
        linear_resid_norm_ = LinearResidualNorm(x, rhs);
    }
}

void LevelSolver::NewtonStep(const mfem::BlockVector& rhs, mfem::BlockVector& x)
{
    Mult(x, residual_);
    residual_ -= rhs;
    for (int i = 0; i < x.BlockSize(0); ++i)
    {
        if (GetEssDofs()[i])
            residual_[i] = 0.0;
    }

    Build_dMdp(x);

    hierarchy_.UpdateJacobian(level_, kp_, dMdp_);

    mfem::BlockVector block_residual(residual_.GetData(), offsets_);
    mfem::BlockVector delta_x = hierarchy_.Solve(level_, block_residual);
    x -= delta_x;

    if (linear_tol_criterion_ == TaylorResidual)
    {
        linear_resid_norm_ = LinearResidualNorm(delta_x, block_residual);
    }
}

double LevelSolver::LinearResidualNorm(const mfem::Vector& x, const mfem::Vector& y) const
{
    const MixedMatrix& mixed_system = hierarchy_.GetMatrix(level_);

    mfem::BlockVector block_x(x.GetData(), offsets_);
    mfem::BlockVector block_y(y.GetData(), offsets_);

    mfem::BlockVector linear_resid(offsets_);
    mixed_system.Mult(kp_, block_x, linear_resid);
    linear_resid -= block_y;

    if (solve_type_ == Newton)
    {
        auto& vert_vdof = mixed_system.GetGraphSpace().VertexToVDof();
        auto& vert_edof = mixed_system.GetGraphSpace().VertexToEDof();

        mfem::Array<int> local_edofs, local_vdofs;
        mfem::Vector x_loc;
        mfem::Vector y_loc;
        for (int i = 0; i < vert_vdof.NumRows(); ++i)
        {
            GetTableRow(vert_vdof, i, local_vdofs);
            GetTableRow(vert_edof, i, local_edofs);

            block_x.GetBlock(1).GetSubVector(local_vdofs, x_loc);

            y_loc.SetSize(local_edofs.Size());
            dMdp_[i].Mult(x_loc, y_loc);

            for (int j = 0; j < local_edofs.Size(); ++j)
            {
                linear_resid[local_edofs[j]] += y_loc[j];
            }
        }
    }

    mfem::Vector true_linear_resid = AssembleTrueVector(linear_resid);

    return mfem::ParNormlp(true_linear_resid, 2, comm_);
}

void LevelSolver::Build_dMdp(const mfem::BlockVector& iterate)
{
    auto& mixed_system = hierarchy_.GetMatrix(level_);
    auto& vert_edof = mixed_system.GetGraphSpace().VertexToEDof();
    auto& vert_vdof = mixed_system.GetGraphSpace().VertexToVDof();

    auto& MB = dynamic_cast<const ElementMBuilder&>(mixed_system.GetMBuilder());
    auto& M_el = MB.GetElementMatrices();

    auto& proj_pwc = const_cast<mfem::SparseMatrix&>(mixed_system.GetPWConstProj());

    dMdp_.resize(M_el.size());
    mfem::Array<int> local_edofs, local_vdofs, vert(1);
    mfem::Vector sigma_loc, Msigma_vec;
    mfem::DenseMatrix proj_pwc_loc;

    EvalCoefDerivative(iterate.GetBlock(1)); // dkinv_dp_ is updated

    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        GetTableRow(vert_vdof, i, local_vdofs);
        vert[0] = i;

        iterate.GetBlock(0).GetSubVector(local_edofs, sigma_loc);
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

EllipticNLMG::EllipticNLMG(Hierarchy& hierarchy, const mfem::Vector& Z_fine,
                           Cycle cycle, SolveType solve_type)
    : NonlinearMG(hierarchy.GetComm(), hierarchy.BlockOffsets(0)[2],
                  hierarchy.NumLevels(), cycle),
      hierarchy_(hierarchy)
{
    std::vector<mfem::Vector> help(hierarchy.NumLevels());

    solvers_.reserve(num_levels_);
    for (int level = 0; level < num_levels_; ++level)
    {
        mfem::Vector Z_vec;
        if (Z_fine.Size())
        {
            if (level == 0)
            {
                help[level].SetDataAndSize(Z_fine.GetData(), Z_fine.Size());
            }
            else
            {
                help[level] = hierarchy.Project(level - 1, help[level - 1]);
            }
            Z_vec = hierarchy.PWConstProject(level, help[level]);
        }

        if (level > 0)
        {
            rhs_[level].SetSize(LevelSize(level));
            sol_[level].SetSize(LevelSize(level));
            rhs_[level] = 0.0;
            sol_[level] = 0.0;
        }
        help_[level].SetSize(LevelSize(level));
        help_[level] = 0.0;

        solvers_.emplace_back(hierarchy_, level, std::move(Z_vec), solve_type);
        solvers_[level].SetPrintLevel(-1);
        solvers_[level].SetMaxIter(1);
    }
}

void EllipticNLMG::Mult(int level, const mfem::Vector& x, mfem::Vector& Ax)
{
    solvers_[level].Mult(x, Ax);
}

void EllipticNLMG::Solve(int level, const mfem::Vector& rhs, mfem::Vector& sol)
{
    Smoothing(level, rhs, sol);
}

void EllipticNLMG::Restrict(int level, const mfem::Vector& fine, mfem::Vector& coarse) const
{
    mfem::BlockVector block_fine(fine.GetData(), Offsets(level));
    coarse = hierarchy_.Restrict(level, block_fine);
}

void EllipticNLMG::Interpolate(int level, const mfem::Vector& coarse, mfem::Vector& fine) const
{
    mfem::BlockVector block_coarse(coarse.GetData(), Offsets(level));
    fine = hierarchy_.Interpolate(level, block_coarse);
}

void EllipticNLMG::Project(int level, const mfem::Vector& fine, mfem::Vector& coarse) const
{
    mfem::BlockVector block_fine(fine.GetData(), Offsets(level));
    coarse = hierarchy_.Project(level, block_fine);
}

void EllipticNLMG::Smoothing(int level, const mfem::Vector& in, mfem::Vector& out)
{
    hierarchy_.SetRelTol((level ? 1e-2 : 1.0) * linear_tol_);

    solvers_[level].Solve(in, out);

    if (level == 0)
    {
        linear_resid_norm_ = solvers_[0].GetLinearResidualNorm();
    }
}

const mfem::Array<int>& EllipticNLMG::Offsets(int level) const
{
    return hierarchy_.BlockOffsets(level);
}

mfem::Vector EllipticNLMG::AssembleTrueVector(const mfem::Vector& vec) const
{
    return hierarchy_.GetMatrix(0).AssembleTrueVector(vec);
}

int EllipticNLMG::LevelSize(int level) const
{
    return hierarchy_.GetMatrix(level).NumTotalDofs();
}

// SPE10: -7-6    Egg: -5e-1     Lognormal: -8e0 (cl = 0.1)
void Kappa(const mfem::Vector& p, mfem::Vector& kp)
{
    assert(kp.Size() == p.Size());
    for (int i = 0; i < p.Size(); i++)
    {
        kp[i] = std::exp(alpha * p[i]);
        assert(kp[i] > 0.0);
    }
}

void dKinv_dp(const mfem::Vector& p, mfem::Vector& dkinv_dp)
{
    assert(dkinv_dp.Size() == p.Size());
    for (int i = 0; i < p.Size(); i++)
    {
        double exp_ap = std::exp(alpha * p[i]);
        assert(exp_ap > 0.0);
        dkinv_dp[i] = -alpha / exp_ap;
    }
}

// Loam
double beta = 1.55;
double K_s = 1.067;//* 0.01; // cm/day

// Sand
//double beta = 4.74;
//double K_s = 816.0;// * 0.01; // cm/day

void Kappa(const mfem::Vector& p, const mfem::Vector& Z_vec, mfem::Vector& kp)
{
    double alpha_K_s = K_s * alpha;

    assert(kp.Size() == p.Size());
    assert(Z_vec.Size() == p.Size());
    for (int i = 0; i < p.Size(); i++)
    {
        kp[i] = alpha_K_s / (alpha + std::pow(std::fabs(p[i] - Z_vec[i]), beta));
        assert(kp[i] > 0.0);
    }
}

void dKinv_dp(const mfem::Vector& p, const mfem::Vector& Z_vec, mfem::Vector& dkinv_dp)
{
    double b_over_a_K_s = beta / (K_s * alpha);

    assert(dkinv_dp.Size() == p.Size());
    assert(Z_vec.Size() == p.Size());
    for (int i = 0; i < p.Size(); i++)
    {
        double p_head = p[i] - Z_vec[i];
        double sign = p_head < 0.0 ? -1.0 : 1.0;
        dkinv_dp[i] = sign * b_over_a_K_s * std::pow(std::fabs(p_head), beta - 1.0);
    }
}
