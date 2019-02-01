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

class SingleLevelSolver : public NonlinearSolver
{
public:
    /**
       @todo take Kappa(p) as input
    */
    SingleLevelSolver(Hierarchy& hierarchy, int level, SolveType solve_type);

    // Compute Ax = A(x).
    virtual void Mult(const mfem::Vector& x, mfem::Vector& Ax);

private:
    void PicardStep(const mfem::BlockVector& rhs, mfem::BlockVector& x);
    void NewtonStep(const mfem::BlockVector& rhs, mfem::BlockVector& x);

    virtual void IterationStep(const mfem::Vector& rhs, mfem::Vector& sol);

    virtual mfem::Vector AssembleTrueVector(const mfem::Vector& vec) const;

    int level_;
    Hierarchy& hierarchy_;
    SolveType solve_type_;

    const mfem::Array<int>& offsets_;
    mfem::Vector p_;         // coefficient vector in piecewise 1 basis
    mfem::Vector kp_;        // kp_ = Kappa(p)

    mfem::Vector Z_vector_level_;
};

// nonlinear elliptic hierarchy
class EllipticNLMG : public NonlinearMG
{
public:
    EllipticNLMG(Hierarchy& hierarchy_, Cycle cycle, SolveType solve_type);
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

    Hierarchy& hierarchy_;
    std::vector<SingleLevelSolver> solvers_;
};

void Kappa(const mfem::Vector& p, mfem::Vector& kp);

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
    args.AddOption(&perm_file, "-p", "--perm",
                   "SPE10 permeability file data.");
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
    unique_ptr<DarcyProblem> fv_problem;

    if (problem == "spe10")
    {
        fv_problem.reset(new SPE10Problem(perm_file, dim, 5, slice, 0, ess_attr));
    }
    else if (problem == "egg")
    {
        fv_problem.reset(new EggModel(num_sr, num_pr, ess_attr));
    }
    else if (problem == "lognormal")
    {
        fv_problem.reset(new LognormalModel(dim, num_sr, num_pr, correlation, ess_attr));
    }
    else if (problem == "richard")
    {
        ess_attr = 1;
        ess_attr[0] = 0;
        fv_problem.reset(new Richards(num_sr, ess_attr));
    }
    else
    {
        mfem::mfem_error("Unknown model problem!");
    }

    Graph graph = fv_problem->GetFVGraph();

    // Create hierarchy
    Hierarchy hierarchy(std::move(graph), upscale_param, nullptr, &ess_attr);
    hierarchy.PrintInfo();

    if (myid == 0)
    {
        std::cout << "\n";
    }

    mfem::BlockVector rhs(hierarchy.GetMatrix(0).BlockOffsets());
    rhs.GetBlock(0) = 0.0;//fv_problem.GetEdgeRHS();;
    rhs.GetBlock(1) = fv_problem->CellVolume();//fv_problem.GetVertexRHS();

    mfem::BlockVector sol_picard(rhs);
    sol_picard = 0.0;

    SingleLevelSolver sls(hierarchy, 0, Picard);
    sls.SetPrintLevel(1);
    sls.Solve(rhs, sol_picard);

    mfem::BlockVector sol_nlmg(rhs);
    sol_nlmg = 0.0;

    EllipticNLMG nlmg(hierarchy, V_CYCLE, Picard);
    nlmg.SetPrintLevel(1);
    nlmg.Solve(rhs, sol_nlmg);

    double p_err = CompareError(comm, sol_picard.GetBlock(1), sol_nlmg.GetBlock(1));
    if (myid == 0)
    {
        std::cout << "Relative errors: " << p_err << "\n";
    }

    if (visualization)
    {
        mfem::socketstream sout;
        fv_problem->VisSetup(sout, sol_picard.GetBlock(1), 0.0, 0.0, "");
    }

    return EXIT_SUCCESS;
}

/// @todo take MixedMatrix only
SingleLevelSolver::SingleLevelSolver(Hierarchy& hierarchy, int level, SolveType solve_type)
    : NonlinearSolver(hierarchy.GetMatrix(level).GetComm(),
                      hierarchy.GetMatrix(level).NumTotalDofs(), "Picard"),
      level_(level), hierarchy_(hierarchy), solve_type_(solve_type),
      offsets_(hierarchy_.GetMatrix(level).BlockOffsets()),
      p_(hierarchy_.GetMatrix(level).GetGraph().NumVertices()),
      kp_(p_.Size())
{ }

void SingleLevelSolver::Mult(const mfem::Vector& x, mfem::Vector& Ax)
{
    assert(size_ == Ax.Size());
    assert(size_ == x.Size());
    mfem::BlockVector block_x(x.GetData(), offsets_);
    mfem::BlockVector block_Ax(Ax.GetData(), offsets_);

    p_ = hierarchy_.PWConstProject(level_, block_x.GetBlock(1));
    Kappa(p_, kp_);
    hierarchy_.GetMatrix(level_).Mult(kp_, block_x, block_Ax);
}

mfem::Vector SingleLevelSolver::AssembleTrueVector(const mfem::Vector& vec) const
{
    return hierarchy_.GetMatrix(level_).AssembleTrueVector(vec);
}

void SingleLevelSolver::IterationStep(const mfem::Vector& rhs, mfem::Vector& sol)
{
    mfem::BlockVector block_sol(sol.GetData(), offsets_);
    mfem::BlockVector block_rhs(rhs.GetData(), offsets_);
    if (solve_type_ == Picard)
    {
        PicardStep(block_rhs, block_sol);
    }
    else
    {
        NewtonStep(block_rhs, block_sol);
    }
}

void SingleLevelSolver::PicardStep(const mfem::BlockVector& rhs, mfem::BlockVector& x)
{
    p_ = hierarchy_.PWConstProject(level_, x.GetBlock(1));
    Kappa(p_, kp_);

    hierarchy_.RescaleCoefficient(level_, kp_);
    hierarchy_.Solve(level_, rhs, x);
}

void SingleLevelSolver::NewtonStep(const mfem::BlockVector& rhs, mfem::BlockVector& x)
{
    // TBD...
}

EllipticNLMG::EllipticNLMG(Hierarchy& hierarchy, Cycle cycle, SolveType solve_type)
    : NonlinearMG(hierarchy.GetMatrix(0).GetComm(), hierarchy.GetMatrix(0).NumTotalDofs(),
                  hierarchy.NumLevels(), cycle),
      hierarchy_(hierarchy)
{
    //    std::vector<Vector> Z_vectors_help(up_.NumLevels());
    //    std::vector<Vector> Z_vectors(up_.NumLevels());

    hierarchy_.SetPrintLevel(-1);
    hierarchy_.SetMaxIter(10);

    solvers_.reserve(num_levels_);
    for (int level = 0; level < num_levels_; ++level)
    {
        //        if (i == 0)
        //        {
        //            Z_vectors[i] = Z_vector_glo;
        //            Z_vectors_help[i] = Z_vector_glo;
        //        }
        //        else
        //        {
        //            Z_vectors_help[i] = up_.GetCoarsener(i-1).Restrict(Z_vectors_help[i-1]);
        //            Z_vectors[i].SetSize(up_.GetMatrix(i).GetElemDof().Rows());
        //            up_.Project_PW_One(i, Z_vectors_help[i], Z_vectors[i]);
        //        }

        rhs_[level].SetSize(LevelSize(level));
        sol_[level].SetSize(LevelSize(level));
        help_[level].SetSize(LevelSize(level));
        rhs_[level] = 0.0;
        sol_[level] = 0.0;
        help_[level] = 0.0;

        solvers_.emplace_back(hierarchy_, level, solve_type);
        solvers_[level].SetPrintLevel(-1);
        solvers_[level].SetMaxIter(1);
    }
}

void EllipticNLMG::Mult(
    int level, const mfem::Vector& x, mfem::Vector& Ax)
{
    solvers_[level].Mult(x, Ax);
}

void EllipticNLMG::Solve(
    int level, const mfem::Vector& rhs, mfem::Vector& sol)
{
    solvers_[level].Solve(rhs, sol);
}

void EllipticNLMG::Restrict(
    int level, const mfem::Vector& fine, mfem::Vector& coarse) const
{
    mfem::BlockVector block_fine(fine.GetData(), Offsets(level));
    mfem::BlockVector block_coarse(coarse.GetData(), Offsets(level));
    coarse = hierarchy_.Restrict(level, block_fine);
}

void EllipticNLMG::Interpolate(
    int level, const mfem::Vector& coarse, mfem::Vector& fine) const
{
    mfem::BlockVector block_coarse(coarse.GetData(), Offsets(level));
    fine = hierarchy_.Interpolate(level, block_coarse);
}

void EllipticNLMG::Project(
    int level, const mfem::Vector& fine, mfem::Vector& coarse) const
{
    mfem::BlockVector block_fine(fine.GetData(), Offsets(level));
    coarse = hierarchy_.Project(level, block_fine);
}

void EllipticNLMG::Smoothing(int level, const mfem::Vector& in, mfem::Vector& out)
{
    solvers_[level].Solve(in, out);
}

const mfem::Array<int>& EllipticNLMG::Offsets(int level) const
{
    return hierarchy_.GetMatrix(level).BlockOffsets();
}

mfem::Vector EllipticNLMG::AssembleTrueVector(const mfem::Vector& vec) const
{
    return hierarchy_.GetMatrix(0).AssembleTrueVector(vec);
}

int EllipticNLMG::LevelSize(int level) const
{
    return hierarchy_.GetMatrix(level).NumTotalDofs();
}

// Kappa(p) = exp(\alpha p)
void Kappa(const mfem::Vector& p, mfem::Vector& kp)
{
    assert(kp.Size() == p.Size());
    for (int i = 0; i < p.Size(); i++)
    {
        kp[i] = std::exp(5e-1 * (p[i]));
        assert(kp[i] > 0.0);
    }
}

//// Kappa(p) = K\alpha / (\alpha + |p(x, y, z) - z|^\beta)
//void Kappa(const mfem::Vector& p, std::vector<double>& kp,
//           const mfem::Vector& Z_vector, bool help)
//{
//    // Loam
//    double alpha = 124.6;
//    double beta = 1.57;
//    double K_s = 1.067;//* 0.01; // cm/day

//    // Sand
////    double alpha = 1.175e6;
////    double beta = 4.74;
////    double K_s = 816.0;// * 0.01; // cm/day

//    double alpha_K_s = K_s * alpha;

//    Vector p_copy(p);
//    p_copy -= Z_vector;
////    std::cout<<"max = "<<linalgcpp::AbsMax(p_copy)<<"\n";

//    assert(kp.size() == p.size());
//    for (int i = 0; i < p.size(); i++)
//    {
//        kp[i] = alpha_K_s / (alpha + std::pow(std::fabs(p[i] - Z_vector[i]), beta));
//        if (help)
//        {
//            kp[i] = std::max(kp[i], 1e-4);
//        }
//        assert(kp[i] > 0.0);
//    }
//}
