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
#include "well.hpp"

#include "../src/picojson.h"
#include "../src/smoothG.hpp"

using namespace smoothg;

enum SteppingScheme { IMPES = 1, SequentiallyImplicit, FullyImplcit };

struct EvolveParamenters
{
    double total_time = 10.0;    // Total time
    double dt = 1.0;   // Time step size
    int vis_step = 0;
    SteppingScheme scheme = IMPES;
};

void SetOptions(FASParameters& param, bool use_vcycle, int num_backtrack, double diff_tol);

mfem::Vector TotalMobility(const mfem::Vector& S);
mfem::Vector dTMinv_dS(const mfem::Vector& S);
mfem::Vector FractionalFlow(const mfem::Vector& S);
mfem::Vector dFdS(const mfem::Vector& S);

/**
   This computes dS/dt that solves W dS/dt + Adv F(S) = b, which is the
   semi-discrete form of dS/dt + div(vF(S)) = b, where W and Adv are the mass
   and advection matrices, F is a nonlinear function, b is the influx source.
 */
class TwoPhaseSolver
{
    const int level_;
    const EvolveParamenters& evolve_param_;
    const FASParameters& solver_param_;
    const TwoPhase& problem_;
    Hierarchy& hierarchy_;
    std::vector<mfem::BlockVector> blk_helper_;

    mfem::Array<int> blk_offsets_;
    unique_ptr<mfem::BlockVector> source_;
    unique_ptr<mfem::HypreParMatrix> D_te_e_;
    int nonlinear_iter_;
    bool step_converged_;

    std::vector<mfem::Vector> micro_upwind_flux_;

    // TODO: these should be defined in / extracted from the problem, not here
    const double density_ = 1e3;
    const double porosity_ = 0.3;
    const double weight_;
public:
    TwoPhaseSolver(const TwoPhase& problem, Hierarchy& hierarchy,
                   const int level, const EvolveParamenters& evolve_param,
                   const FASParameters& solver_param);

    void TimeStepping(const double dt, mfem::BlockVector& x);
    mfem::BlockVector Solve(const mfem::BlockVector& init_val);
};

class CoupledSolver : public NonlinearSolver
{
    const MixedMatrix& darcy_system_;
    mfem::GMRESSolver gmres_;
    unique_ptr<mfem::HypreParMatrix> D_;
    unique_ptr<mfem::HypreParMatrix> DT_;
    std::vector<mfem::DenseMatrix> local_dMdS_;
    mfem::SparseMatrix Ms_;

    mfem::Array<int> blk_offsets_;
    mfem::Array<int> true_blk_offsets_;
    const mfem::Array<int>& ess_dofs_;
    const mfem::Array<int>& vert_starts_;
    mfem::Array<int> true_edof_starts_;
//    const std::vector<mfem::DenseMatrix>& traces_;

    const double dt_;
    const double weight_;
    const double density_;

    mfem::Vector normalizer_;
    bool is_first_resid_eval_;

    void Step(const mfem::Vector& rhs, mfem::Vector& x, mfem::Vector& dx) override;
    void Build_dMdS(const mfem::Vector& flux, const mfem::Vector& S);
    mfem::SparseMatrix Assemble_dMdS(const mfem::Vector& flux, const mfem::Vector& S);
    mfem::Vector AssembleTrueVector(const mfem::Vector& vec) const;
public:
    CoupledSolver(const MixedMatrix& darcy_system,
//                  const std::vector<mfem::DenseMatrix>& edge_traces,
                  const double dt,
                  const double weight,
                  const double density,
                  NLSolverParameters param);

    mfem::Vector Residual(const mfem::Vector& x, const mfem::Vector& y) override;
    double ResidualNorm(const mfem::Vector& x, const mfem::Vector& y) override;
    double Norm(const mfem::Vector& vec);
    const mfem::Array<int>& BlockOffsets() const { return blk_offsets_; }

    void BackTracking(const mfem::Vector& rhs,  double prev_resid_norm,
                      mfem::Vector& x, mfem::Vector& dx) override;

};

class CoupledFAS : public FAS
{
    const Hierarchy& hierarchy_;

    double Norm(int level, const mfem::Vector& vec) const override;
    void Restrict(int level, const mfem::Vector& fine, mfem::Vector& coarse) const override;
    void Interpolate(int level, const mfem::Vector& coarse, mfem::Vector& fine) const override;
    void Project(int level, const mfem::Vector& fine, mfem::Vector& coarse) const override;
    mfem::Vector ProjectS(int level, const mfem::Vector& S) const;
public:
    CoupledFAS(const Hierarchy& hierarchy,
               const double dt,
               const double weight,
               const double density,
               FASParameters param);
};

class TransportSolver : public NonlinearSolver
{
    const MixedMatrix& darcy_system_;
    const mfem::Array<int>& starts_;
    mfem::Array<int> ess_dofs_;
    mfem::GMRESSolver gmres_;
    const mfem::HypreParMatrix& Adv_;
    mfem::SparseMatrix Ms_;

    virtual void Step(const mfem::Vector& rhs, mfem::Vector& x, mfem::Vector& dx) override;
public:
    TransportSolver(const mfem::HypreParMatrix& Adv_,
                    const MixedMatrix& darcy_system,
                    const double vol_dt_inv,
                    NLSolverParameters param)
        : NonlinearSolver(Adv_.GetComm(), param), darcy_system_(darcy_system),
          starts_(darcy_system.GetGraph().VertexStarts()), gmres_(comm_),
          Adv_(Adv_), Ms_(SparseIdentity(Adv_.NumCols()) *= vol_dt_inv)
    {
        gmres_.SetMaxIter(200);
        gmres_.SetRelTol(1e-9);
//        gmres_.SetPrintLevel(1);
    }

    mfem::Vector Residual(const mfem::Vector& x, const mfem::Vector& y) override;
    double ResidualNorm(const mfem::Vector& x, const mfem::Vector& y) override
    {
        return mfem::ParNormlp(Residual(x, y), 2, comm_);
    }
};

int main(int argc, char* argv[])
{
    int num_procs, myid;

    // 1. Initialize MPI
    mpi_session session(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // program options from command line
    EvolveParamenters evolve_param;
    mfem::OptionsParser args(argc, argv);
    const char* perm_file = "spe_perm.dat";
    args.AddOption(&perm_file, "-p", "--perm", "SPE10 permeability file data.");
    int dim = 3;
    args.AddOption(&dim, "-d", "--dim", "Dimension of the physical space.");
    int slice = 0;
    args.AddOption(&slice, "-s", "--slice", "Slice of SPE10 data for 2D run.");
    int well_height = 1;
    args.AddOption(&well_height, "-wh", "--well-height", "Well Height.");
    double inject_rate = 0.02;
    args.AddOption(&inject_rate, "-ir", "--inject-rate", "Injector rate.");
    double bhp = 1.0e5;
    args.AddOption(&bhp, "-bhp", "--bottom-hole-pressure", "Bottom Hole Pressure.");
    args.AddOption(&evolve_param.dt, "-dt", "--delta-t", "Time step.");
    args.AddOption(&evolve_param.total_time, "-time", "--total-time",
                   "Total time to step.");
    args.AddOption(&evolve_param.vis_step, "-vs", "--vis-step",
                   "Step size for visualization.");
    int scheme = 3;
    args.AddOption(&scheme, "-scheme", "--stepping-scheme",
                   "Time stepping: 1. IMPES, 2. sequentially implicit, 3. fully implicit. ");
    bool use_vcycle = true;
    args.AddOption(&use_vcycle, "-VCycle", "--use-VCycle", "-FMG",
                   "--use-FMG", "Use V-cycle or FMG-cycle.");
    int num_backtrack = 0;
    args.AddOption(&num_backtrack, "--num-backtrack", "--num-backtrack",
                   "Maximum number of backtracking steps.");
    double diff_tol = -1.0;
    args.AddOption(&diff_tol, "--diff-tol", "--diff-tol",
                   "Tolerance for coefficient change.");
    UpscaleParameters upscale_param;
    upscale_param.spect_tol = 1.0;
    upscale_param.max_evects = 1;
    upscale_param.max_traces = 1;
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

    evolve_param.scheme = static_cast<SteppingScheme>(scheme);

    evolve_param.dt = evolve_param.total_time;

    const int max_iter = 100;

    FASParameters fas_param;
    fas_param.fine.max_num_iter = fas_param.mid.max_num_iter = use_vcycle ? 1 : max_iter;
    fas_param.coarse.max_num_iter = use_vcycle ? 20 : max_iter;
    fas_param.coarse.print_level = use_vcycle ? -1 : 1;
    fas_param.fine.print_level = use_vcycle ? -1 : 1;
    fas_param.mid.print_level = use_vcycle ? -1 : 1;
//    fas_param.coarse.rtol = 1e-10;
//    fas_param.coarse.atol = 1e-12;
    fas_param.nl_solve.print_level = 1;
    fas_param.nl_solve.max_num_iter = use_vcycle ? max_iter : 1;
    fas_param.nl_solve.atol = 1e-10;
    fas_param.nl_solve.rtol = 1e-8;
    SetOptions(fas_param, use_vcycle, num_backtrack, diff_tol);

    mfem::Array<int> ess_attr(dim == 3 ? 6 : 4);
    ess_attr = 1;

    // Setting up finite volume discretization problem
    bool use_metis = true;
    TwoPhase problem(perm_file, dim, 5, slice, use_metis, ess_attr,
                     well_height, inject_rate, bhp);

    mfem::Array<int> part;
    mfem::Array<int> coarsening_factors(dim);
    coarsening_factors = 1;
    coarsening_factors[0] = upscale_param.coarse_factor;
    problem.Partition(use_metis, coarsening_factors, part);
    upscale_param.num_iso_verts = problem.NumIsoVerts();

    Hierarchy hierarchy(problem.GetFVGraph(true), upscale_param,
                        &part, &problem.EssentialAttribute());
    hierarchy.PrintInfo();

//    if (upscale_param.hybridization)
//    {
//        hierarchy.SetAbsTol(1e-18);
//        hierarchy.SetRelTol(1e-15);
//    }

    // Fine scale transport based on fine flux
    std::vector<mfem::Vector> Ss(upscale_param.max_levels);

    //    int l = 0;
    for (int l = upscale_param.max_levels-1; l < upscale_param.max_levels; ++l)
    {
        fas_param.num_levels = l + 1;
        TwoPhaseSolver solver(problem, hierarchy, 0, evolve_param, fas_param);

        mfem::BlockVector initial_value(problem.BlockOffsets());
        initial_value = 0.0;

        mfem::StopWatch chrono;
        chrono.Start();
        mfem::BlockVector sol = solver.Solve(initial_value);

        if (myid == 0)
        {
            std::cout << "Level " << l << ":\n    Time stepping done in "
                      << chrono.RealTime() << "s.\n";
        }

        Ss[l] = sol.GetBlock(2);

        double norm = mfem::ParNormlp(Ss[l], 1, comm);
        if (myid == 0) { std::cout << "    || S ||_1 = " << norm << "\n"; }

//        if (l) { Ss[l] -= Ss[0]; }
//        double diff = mfem::ParNormlp(Ss[l], 2, comm);
//        norm = mfem::ParNormlp(Ss[0], 2, comm);
//        if (myid == 0) { std::cout << "    rel err = " << diff / norm << "\n"; }

//        mfem::socketstream sout;
//        if (l && evolve_param.vis_step)
//        {
//            problem.VisSetup(sout, Ss[l], 0.0, 0.0, "Solution difference");
//        }
    }
    return EXIT_SUCCESS;
}

void SetOptions(FASParameters& param, bool use_vcycle, int num_backtrack, double diff_tol)
{
    param.cycle = use_vcycle ? V_CYCLE : FMG;
    param.nl_solve.linearization = Newton;
    param.coarse_correct_tol = 1e-6;
    param.fine.check_converge = use_vcycle ? false : true;
    param.fine.linearization = param.nl_solve.linearization;
    param.mid.linearization = param.nl_solve.linearization;
    param.coarse.linearization = param.nl_solve.linearization;
    param.fine.num_backtrack = num_backtrack;
    param.mid.num_backtrack = num_backtrack;
    param.coarse.num_backtrack = num_backtrack;
    param.fine.diff_tol = diff_tol;
    param.mid.diff_tol = diff_tol;
    param.coarse.diff_tol = diff_tol;
    param.nl_solve.diff_tol = diff_tol;
}

//mfem::Vector ComputeFaceFlux(const MixedMatrix& darcy_system,
//                             const mfem::Vector& flux)
//{
//    mfem::Vector out(darcy_system.GetTraceFluxes());
//    RescaleVector(flux, out);
//    return out;
//}

mfem::SparseMatrix BuildUpwindPattern(const GraphSpace& graph_space,
                                      const mfem::Vector& flux)
{
    const Graph& graph = graph_space.GetGraph();
    const mfem::SparseMatrix& edge_vert = graph.EdgeToVertex();
    const mfem::SparseMatrix e_te_diag = GetDiag(graph.EdgeToTrueEdge());

    mfem::SparseMatrix upwind_pattern(graph.NumEdges(), graph.NumVertices());

    for (int i = 0; i < graph.NumEdges(); ++i)
    {
        if (edge_vert.RowSize(i) == 2) // edge is interior
        {
            const int upwind_vert = flux[i] > 0.0 ? 0 : 1;
            upwind_pattern.Set(i, edge_vert.GetRowColumns(i)[upwind_vert], 1.0);
        }
        else
        {
            assert(edge_vert.RowSize(i) == 1);
            const bool edge_is_owned = e_te_diag.RowSize(i);

            if ((flux[i] > 0.0 && edge_is_owned) || (flux[i] <= 0.0 && !edge_is_owned))
            {
                upwind_pattern.Set(i, edge_vert.GetRowColumns(i)[0], 1.0);
            }
        }
    }
    upwind_pattern.Finalize(); // TODO: use sparsity pattern of DT and update the values

    return upwind_pattern;
}

TwoPhaseSolver::TwoPhaseSolver(const TwoPhase& problem, Hierarchy& hierarchy,
                               const int level, const EvolveParamenters& evolve_param,
                               const FASParameters& solver_param)
    : level_(level), evolve_param_(evolve_param), solver_param_(solver_param),
      problem_(problem), hierarchy_(hierarchy), blk_offsets_(4), nonlinear_iter_(0),
      step_converged_(true), weight_(problem.CellVolume() * porosity_ * density_)
{
    blk_helper_.reserve(level + 1);
    blk_helper_.emplace_back(hierarchy.BlockOffsets(0));
    blk_helper_[0].GetBlock(0) = problem_.GetEdgeRHS();
    blk_helper_[0].GetBlock(1) = problem_.GetVertexRHS();

    for (int l = 0; l < level_; ++l)
    {
        blk_helper_.push_back(hierarchy.Restrict(l, blk_helper_[l]));
    }

    blk_offsets_[0] = 0;
    blk_offsets_[1] = hierarchy.BlockOffsets(level)[1];
    blk_offsets_[2] = hierarchy.BlockOffsets(level)[2];
    blk_offsets_[3] = 2 * blk_offsets_[2] - blk_offsets_[1];

    source_.reset(new mfem::BlockVector(blk_offsets_));
    source_->GetBlock(0) = blk_helper_[level].GetBlock(0);
    source_->GetBlock(1) = blk_helper_[level].GetBlock(1);
    source_->GetBlock(2) = blk_helper_[level].GetBlock(1);

    auto& e_te_e = hierarchy.GetMatrix(level).GetGraph().EdgeToTrueEdgeToEdge();
    auto& starts = hierarchy.GetMatrix(level).GetGraph().VertexStarts();
    D_te_e_ = ParMult(hierarchy.GetMatrix(level).GetD(), e_te_e, starts);
}

mfem::BlockVector TwoPhaseSolver::Solve(const mfem::BlockVector& init_val)
{
    int myid;
    MPI_Comm_rank(hierarchy_.GetComm(), &myid);

    mfem::BlockVector x(blk_offsets_);

    blk_helper_[0].GetBlock(0) = init_val.GetBlock(0);
    blk_helper_[0].GetBlock(1) = init_val.GetBlock(1);
    mfem::Vector x_blk2 = init_val.GetBlock(2);

    mfem::socketstream sout;
    if (evolve_param_.vis_step) { problem_.VisSetup(sout, x_blk2, -0.02, 1.0, "Fine scale"); }

    for (int l = 0; l < level_; ++l)
    {
        hierarchy_.Project(l, blk_helper_[l], blk_helper_[l + 1]);
        x_blk2 = hierarchy_.Project(l, x_blk2);
    }

    x.GetBlock(0) = blk_helper_[level_].GetBlock(0);
    x.GetBlock(1) = blk_helper_[level_].GetBlock(1);
    x.GetBlock(2) = x_blk2;

    double time = 0.0;
    double dt_real = std::min(evolve_param_.dt, evolve_param_.total_time - time) / 2.0;

    bool done = false;
    int step;
    for (step = 1; !done; step++)
    {
        mfem::BlockVector previous_x(x);
//        dt_real = std::min(std::min(dt_real * 2.0, evolve_param_.total_time - time), 900.);
        dt_real = std::min(dt_real * 2.0, evolve_param_.total_time - time);
        step_converged_ = false;

        TimeStepping(dt_real, x);
        while (!step_converged_)
        {
            x = previous_x;
            dt_real /= 2.0;
            TimeStepping(dt_real, x);
        }

        time += dt_real;
        done = (time >= evolve_param_.total_time);

        if (myid == 0)
        {
            std::cout << "Time step " << step << ": step size = " << dt_real
                      << ", time = " << time << "\n";
        }
        if (evolve_param_.vis_step && (done || step % evolve_param_.vis_step == 0))
        {
            x_blk2 = x.GetBlock(2);
            for (int l = level_; l > 0; --l)
            {
                x_blk2 = hierarchy_.Interpolate(l, x_blk2);
            }

            problem_.VisUpdate(sout, x_blk2);
        }
    }

    if (myid == 0)
    {
        std::cout << "Total nonlinear iterations: " << nonlinear_iter_ << "\n";
        std::cout << "Average nonlinear iterations: "
                  << double(nonlinear_iter_) / double(step-1) << "\n";
    }

    blk_helper_[level_].GetBlock(0) = x.GetBlock(0);
    blk_helper_[level_].GetBlock(1) = x.GetBlock(1);
    x_blk2 = x.GetBlock(2);

    for (int l = level_; l > 0; --l)
    {
        hierarchy_.Interpolate(l, blk_helper_[l], blk_helper_[l - 1]);
        x_blk2 = hierarchy_.Interpolate(l, x_blk2);
    }

    mfem::BlockVector out(problem_.BlockOffsets());
    out.GetBlock(0) = blk_helper_[0].GetBlock(0);
    out.GetBlock(1) = blk_helper_[0].GetBlock(1);
    out.GetBlock(2) = x_blk2;

    return out;
}

void TwoPhaseSolver::TimeStepping(const double dt, mfem::BlockVector& x)
{
    const MixedMatrix& system = hierarchy_.GetMatrix(level_);
    std::vector<mfem::DenseMatrix> traces;

    if (evolve_param_.scheme == FullyImplcit) // coupled: solve all unknowns together
    {
//        CoupledSolver solver(system, dt, weight_, density_, solver_param_.nl_solve);
        CoupledFAS solver(hierarchy_, dt, weight_, density_, solver_param_);

        mfem::BlockVector rhs(*source_);
        rhs.GetBlock(0) *= (1. / dt / density_);
        rhs.GetBlock(1) *= (dt * density_);
        add(dt * density_, rhs.GetBlock(2), weight_, x.GetBlock(2), rhs.GetBlock(2));
//        x = 0.0;
        solver.Solve(rhs, x);
        step_converged_ = solver.IsConverged();
        nonlinear_iter_ += solver.GetNumIterations();
    }
    else // sequential: solve for flux and pressure first, and then saturation
    {
        const mfem::Vector S = system.PWConstProject(x.GetBlock(2));
        hierarchy_.RescaleCoefficient(level_, TotalMobility(S));
        mfem::BlockVector flow_rhs(*source_, hierarchy_.BlockOffsets(level_));
        mfem::BlockVector flow_sol(x, hierarchy_.BlockOffsets(level_));
        hierarchy_.Solve(level_, flow_rhs, flow_sol);

        auto upwind = BuildUpwindPattern(system.GetGraphSpace(), x.GetBlock(0));
        assert(mfem::ParNormlp(x.GetBlock(0), 2, D_te_e_->GetComm()) < mfem::infinity());
        upwind.ScaleRows(x.GetBlock(0));

        if (evolve_param_.scheme == IMPES) // explcict: new_S = S + dt W^{-1} (b - Adv F(S))
        {
            mfem::Vector dSdt(source_->GetBlock(2));
            D_te_e_->Mult(-1.0, Mult(upwind, FractionalFlow(S)), 1.0, dSdt);
            x.GetBlock(2).Add(dt * density_ / weight_, dSdt);
            step_converged_ = true;
        }
        else // implicit: new_S solves new_S = S + dt W^{-1} (b - Adv F(new_S))
        {
            auto Adv = ParMult(*D_te_e_, upwind, system.GetGraph().VertexStarts());
            const double scaling = weight_ / density_ / dt;
            TransportSolver solver(*Adv, system, scaling, solver_param_.nl_solve);

            mfem::Vector rhs(source_->GetBlock(2));
            rhs.Add(weight_ / density_ / dt, x.GetBlock(2));
            solver.Solve(rhs, x.GetBlock(2));
            step_converged_ = solver.IsConverged();
            nonlinear_iter_ += solver.GetNumIterations();
        }
    }
}

CoupledSolver::CoupledSolver(const MixedMatrix& darcy_system,
//                             const std::vector<mfem::DenseMatrix>& edge_traces,
                             const double dt,
                             const double weight,
                             const double density,
                             NLSolverParameters param)
    : NonlinearSolver(darcy_system.GetComm(), param), darcy_system_(darcy_system),
      gmres_(comm_), local_dMdS_(darcy_system.GetGraph().NumVertices()),
      Ms_(SparseIdentity(darcy_system.GetGraph().NumVertices()) *= weight),
      blk_offsets_(4), true_blk_offsets_(4), ess_dofs_(darcy_system.GetEssDofs()),
      vert_starts_(darcy_system.GetGraph().VertexStarts()),
//      traces_(edge_traces),
      dt_(dt), weight_(weight), density_(density), is_first_resid_eval_(true)
{
    mfem::SparseMatrix D_proc(darcy_system_.GetD());
    if (ess_dofs_.Size()) { D_proc.EliminateCols(ess_dofs_); }
    D_.reset(darcy_system_.MakeParallelD(D_proc));
    DT_.reset(D_->Transpose());
    *D_ *= (dt_ * density_);
    *DT_ *= (1. / dt_ / density_);

    GenerateOffsets(comm_, D_->NumCols(), true_edof_starts_);

    blk_offsets_[0] = 0;
    blk_offsets_[1] = darcy_system.NumEDofs();
    blk_offsets_[2] = blk_offsets_[1] + darcy_system.NumVDofs();
    blk_offsets_[3] = blk_offsets_[2] + Ms_.NumCols();

    true_blk_offsets_[0] = 0;
    true_blk_offsets_[1] = D_->NumCols();
    true_blk_offsets_[2] = true_blk_offsets_[1] + darcy_system.NumVDofs();
    true_blk_offsets_[3] = true_blk_offsets_[2] + Ms_.NumCols();

    gmres_.SetMaxIter(10000);
    gmres_.SetRelTol(1e-12);
    gmres_.SetPrintLevel(0);
    gmres_.SetKDim(100);

    normalizer_.SetSize(Ms_.NumCols());
    normalizer_ = 800. * (weight_ / density_);
}

mfem::Vector CoupledSolver::AssembleTrueVector(const mfem::Vector& vec) const
{
    mfem::Vector true_v(true_blk_offsets_.Last());
    mfem::BlockVector blk_v(vec.GetData(), blk_offsets_);
    mfem::BlockVector blk_true_v(true_v.GetData(), true_blk_offsets_);
    blk_true_v = 0.0;

    auto& truedof_dof = darcy_system_.GetGraphSpace().TrueEDofToEDof();
    truedof_dof.Mult(blk_v.GetBlock(0), blk_true_v.GetBlock(0));
    blk_true_v.GetBlock(1) = blk_v.GetBlock(1);
    blk_true_v.GetBlock(2) = blk_v.GetBlock(2);

    return true_v;
}

mfem::Vector CoupledSolver::Residual(const mfem::Vector& x, const mfem::Vector& y)
{
    mfem::BlockVector blk_x(x.GetData(), blk_offsets_);
    mfem::BlockVector out(blk_offsets_);
    out = 0.0;

    mfem::BlockVector darcy_x(x.GetData(), darcy_system_.BlockOffsets());
    mfem::BlockVector darcy_Rx(out.GetData(), darcy_system_.BlockOffsets());

    const mfem::Vector S = darcy_system_.PWConstProject(blk_x.GetBlock(2));
    darcy_system_.Mult(TotalMobility(S), darcy_x, darcy_Rx);

    darcy_Rx.GetBlock(0) *= (1. / dt_ / density_);
    darcy_Rx.GetBlock(1) *= (dt_ * density_);

    out.GetBlock(2) = blk_x.GetBlock(2);

    const GraphSpace& space = darcy_system_.GetGraphSpace();
    auto upwind = BuildUpwindPattern(space, blk_x.GetBlock(0));
    auto upw_FS = Mult(upwind, FractionalFlow(S));
    RescaleVector(blk_x.GetBlock(0), upw_FS);
    auto U_FS = Mult(space.TrueEDofToEDof(), upw_FS);
    D_->Mult(1.0, U_FS, Ms_(0, 0), out.GetBlock(2)); //TODO: Ms_

    out -= y;
    SetZeroAtMarker(ess_dofs_, out.GetBlock(0));

    if (is_first_resid_eval_)
    {
        normalizer_ = S;
        normalizer_ -= 1.0;
        normalizer_ *= -800.0;
        normalizer_.Add(1000.0, S);
        normalizer_ *= (weight_ / density_);
        is_first_resid_eval_ = false;
    }

    return out;
}

double CoupledSolver::ResidualNorm(const mfem::Vector& x, const mfem::Vector& y)
{
//    std::cout<<"Resid norm: num of dofs = " << true_blk_offsets_.Last()<<"\n";
    return Norm(Residual(x, y));
}

double CoupledSolver::Norm(const mfem::Vector& vec)
{
    auto true_resid = AssembleTrueVector(vec);
    mfem::BlockVector blk_resid(true_resid.GetData(), true_blk_offsets_);

    InvRescaleVector(normalizer_, blk_resid.GetBlock(1));
    InvRescaleVector(normalizer_, blk_resid.GetBlock(2));

    return mfem::ParNormlp(blk_resid, mfem::infinity(), comm_);
}

void CoupledSolver::Build_dMdS(const mfem::Vector& flux, const mfem::Vector& S)
{
    // TODO: saturation is only 1 dof per cell
    auto& vert_edof = darcy_system_.GetGraphSpace().VertexToEDof();
    auto& vert_vdof = darcy_system_.GetGraphSpace().VertexToVDof();

    auto& MB = dynamic_cast<const ElementMBuilder&>(darcy_system_.GetMBuilder());
    auto& M_el = MB.GetElementMatrices();
    auto& proj_pwc = const_cast<mfem::SparseMatrix&>(darcy_system_.GetPWConstProj());

    mfem::Array<int> local_edofs, local_vdofs, vert(1);
    mfem::Vector sigma_loc, Msigma_vec;
    mfem::DenseMatrix proj_pwc_loc;

    const mfem::Vector dTMinv_dS_vec = dTMinv_dS(S);

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
        proj_pwc_loc *= dTMinv_dS_vec[i];

        local_dMdS_[i].SetSize(local_edofs.Size(), local_vdofs.Size());
        mfem::Mult(Msigma_loc, proj_pwc_loc, local_dMdS_[i]);
    }
}

mfem::SparseMatrix CoupledSolver::Assemble_dMdS(const mfem::Vector& flux, const mfem::Vector& S)
{
    Build_dMdS(flux, S); // local_dMdS_ is constructed here

    auto& vert_edof = darcy_system_.GetGraphSpace().VertexToEDof();
    mfem::Array<int> local_edofs, local_vert(1);
    mfem::SparseMatrix out(vert_edof.NumCols(), vert_edof.NumRows());
    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        local_vert[0] = i;
        out.AddSubMatrix(local_edofs, local_vert, local_dMdS_[i]);
    }
    out.Finalize();
    return out;
}

void CoupledSolver::Step(const mfem::Vector& rhs, mfem::Vector& x, mfem::Vector& dx)
{
    mfem::BlockVector blk_x(x.GetData(), blk_offsets_);
    const GraphSpace& space = darcy_system_.GetGraphSpace();

    for (int ii = 0; ii < blk_x.BlockSize(2); ++ii)
    {
        if (blk_x.GetBlock(2)[ii] < 0.0)
        {
            blk_x.GetBlock(2)[ii]  = 0.0;
        }
    }


    mfem::Vector true_resid = AssembleTrueVector(Residual(x, rhs));
    mfem::BlockVector true_blk_dx(true_blk_offsets_);
    true_blk_dx = 0.0;


    const mfem::Vector S = darcy_system_.PWConstProject(blk_x.GetBlock(2));
    auto M_proc = darcy_system_.GetMBuilder().BuildAssembledM(TotalMobility(S));
    auto dMdS_proc = Assemble_dMdS(blk_x.GetBlock(0), S);

//    std::cout.precision(24);
//    std::cout<< "|| x || = " << blk_x.GetBlock(0).Norml2() << " " << S.Norml2() << "\n";
//    std::cout<< " before: min(S) max(S) = "<< S.Min() << " " << S.Max() <<"\n";

    for (int mm = 0; mm < ess_dofs_.Size(); ++mm)
    {
        if (ess_dofs_[mm])
        {
            M_proc.EliminateRowCol(mm); // assume essential data = 0
            dMdS_proc.EliminateRow(mm);
        }
    }

    unique_ptr<mfem::HypreParMatrix> M(darcy_system_.MakeParallelM(M_proc));
    auto dMdS = ParMult(space.TrueEDofToEDof(), dMdS_proc, vert_starts_);

    *M *= (1. / dt_ / density_);
    *dMdS *= (1. / dt_ / density_);

    auto upwind = BuildUpwindPattern(space, blk_x.GetBlock(0));

    auto U_FS = Mult(space.TrueEDofToEDof(), Mult(upwind, FractionalFlow(S)));
    auto dTdsigma = ParMult(*D_, SparseDiag(std::move(U_FS)), true_edof_starts_);
//    unique_ptr<mfem::HypreParMatrix> temp(D_->Transpose());
//    temp->ScaleRows(U_FS);
//    unique_ptr<mfem::HypreParMatrix> dTdsigma(temp->Transpose());


    upwind.ScaleRows(blk_x.GetBlock(0));
    upwind.ScaleColumns(dFdS(S));

    auto U = ParMult(space.TrueEDofToEDof(), upwind, vert_starts_);
    auto U_pwc = ParMult(*U, darcy_system_.GetPWConstProj(), vert_starts_);
    unique_ptr<mfem::HypreParMatrix> dTdS(mfem::ParMult(D_.get(), U_pwc.get()));
    GetDiag(*dTdS) += Ms_;

    mfem::BlockOperator op(true_blk_offsets_);
    op.SetBlock(0, 0, M.get());
    op.SetBlock(0, 1, DT_.get());
    op.SetBlock(1, 0, D_.get());
    op.SetBlock(0, 2, dMdS.get());
    op.SetBlock(2, 0, dTdsigma.get());
    op.SetBlock(2, 2, dTdS.get());

    // preconditioner

    mfem::Vector Md;
    M->GetDiag(Md);
    Md *= -1.0;
    DT_->InvScaleRows(Md);
    unique_ptr<mfem::HypreParMatrix> schur(mfem::ParMult(D_.get(), DT_.get()));
    DT_->ScaleRows(Md);

    auto M_inv = make_unique<mfem::HypreDiagScale>(*M);
    unique_ptr<mfem::HypreBoomerAMG> schur_inv(BoomerAMG(*schur));

//    unique_ptr<mfem::HypreBoomerAMG> dTdS_inv(BoomerAMG(*dTdS));
    auto dTdS_inv = make_unique<mfem::HypreDiagScale>(*dTdS);

    mfem::BlockLowerTriangularPreconditioner prec(true_blk_offsets_);
    prec.SetDiagonalBlock(0, M_inv.get());
    prec.SetDiagonalBlock(1, schur_inv.get());
    prec.SetDiagonalBlock(2, dTdS_inv.get());
    prec.SetBlock(1, 0, D_.get());
    prec.SetBlock(2, 0, dTdsigma.get());

    gmres_.SetOperator(op);
    gmres_.SetPreconditioner(prec);
//    gmres_.iterative_mode = true;
//gmres_.SetPrintLevel(1);

////    std::cout<< "|| M || = " << FroNorm(M_proc) << "\n";
////        std::cout<< "|| D || = " << FroNorm(GetDiag(*D_)) << "\n";
////    std::cout<< "|| dMdS || = " << FroNorm(dMdS_proc) << "\n";
////    std::cout<< "|| dTdsigma || = " << FroNorm(GetDiag(*dTdsigma)) << "\n";
////    std::cout<< "|| U_FS || = " << U_FS.Norml2() << "\n";
////    std::cout<< "|| dTdS || = " << FroNorm(GetDiag(*dTdS)) << "\n";
////    std::cout << "|| rhs || " << mfem::ParNormlp(true_resid, 2, comm_) << "\n";

    gmres_.Mult(true_resid *= -1.0, true_blk_dx);

//    std::cout << "|| sol || " << mfem::ParNormlp(true_blk_dx, 2, comm_) << "\n";


//    mfem::SparseMatrix M_diag = GetDiag(*M);
//    mfem::SparseMatrix DT_diag = GetDiag(*DT_);
//    mfem::SparseMatrix D_diag = GetDiag(*D_);
//    mfem::SparseMatrix dMdS_diag = GetDiag(*dMdS);
//    mfem::SparseMatrix dTdsigma_diag = GetDiag(*dTdsigma);
//    mfem::SparseMatrix dTdS_diag = GetDiag(*dTdS);

////    dTdsigma_diag *= -1.;
////    dTdS_diag *= -1.;
////    mfem::BlockVector blk_true_resid(true_resid.GetData(), blk_offsets_);
////    blk_true_resid.GetBlock(2) *= -1.;

//    mfem::BlockMatrix mat(true_blk_offsets_);
//    mat.SetBlock(0, 0, &M_diag);
//    mat.SetBlock(0, 1, &DT_diag);
//    mat.SetBlock(1, 0, &D_diag);
//    mat.SetBlock(0, 2, &dMdS_diag);
//    mat.SetBlock(2, 0, &dTdsigma_diag);
//    mat.SetBlock(2, 2, &dTdS_diag);

//    unique_ptr<mfem::SparseMatrix> mono_mat(mat.CreateMonolithic());

//    std::cout<< "|| A || = " << FroNorm(*mono_mat) << "\n";
//    std::cout << "|| rhs || " << mfem::ParNormlp(true_resid, 2, comm_) << "\n";


//    if (mono_mat->NumCols() < 10000) mono_mat->Print();
//    std::cout<<"|| mat || = " << mono_mat->MaxNorm()<<"\n";

//    (*mono_mat) *= (dt_);
//    true_resid *= (dt_);

//    mfem::UMFPackSolver direct_solve(*mono_mat, true);
//    direct_solve.Mult(true_resid *= -1.0, true_blk_dx);

//    std::cout << "|| sol || " << mfem::ParNormlp(true_blk_dx, 2, comm_) << "\n";

//    if (!myid_ && !gmres_.GetConverged())
//    {
//        std::cout << "this level has " << dTdS->N() << " dofs\n";
//    }

//    if (!myid_) std::cout << "GMRES took " << gmres_.GetNumIterations()
//                          << " iterations, residual = " << gmres_.GetFinalNorm() << "\n";

    mfem::BlockVector blk_dx(dx.GetData(), blk_offsets_);
    blk_dx = 0.0;
    auto& dof_truedof = darcy_system_.GetGraphSpace().EDofToTrueEDof();
    dof_truedof.Mult(true_blk_dx.GetBlock(0), blk_dx.GetBlock(0));
    blk_dx.GetBlock(1) = true_blk_dx.GetBlock(1);
    blk_dx.GetBlock(2) = true_blk_dx.GetBlock(2);

    const mfem::Vector dS = darcy_system_.PWConstProject(blk_dx.GetBlock(2));
    blk_dx *= std::min(1.0, param_.diff_tol / mfem::ParNormlp(dS, mfem::infinity(), comm_));

//    std::cout << "|| S ||_inf " << mfem::ParNormlp(dS, mfem::infinity(), comm_) << "\n";

    x += blk_dx;


    for (int ii = 0; ii < blk_x.BlockSize(2); ++ii)
    {
//        blk_x.GetBlock(2)[ii] = std::fabs(blk_x.GetBlock(2)[ii]);
        if (blk_x.GetBlock(2)[ii] < 0.0)
        {
            blk_x.GetBlock(2)[ii]  = 0.0;
        }
    }

//    const mfem::Vector S2 = darcy_system_.PWConstProject(blk_x.GetBlock(2));
//    std::cout<< " after: min(S) max(S) = "<< S2.Min() << " " << S2.Max() <<"\n";

}

void CoupledSolver::BackTracking(const mfem::Vector& rhs,  double prev_resid_norm,
                                 mfem::Vector& x, mfem::Vector& dx)
{
    if (param_.num_backtrack == 0) return;

    x -= dx;
    mfem::BlockVector blk_x(x, true_blk_offsets_);
    mfem::BlockVector blk_dx(dx, true_blk_offsets_);

    const mfem::Vector S = darcy_system_.PWConstProject(blk_x.GetBlock(2));
    const mfem::Vector dS = darcy_system_.PWConstProject(blk_dx.GetBlock(2));

    auto violate = (mfem::Vector(1) = mfem::infinity());
    for (int i = 0; i < S.Size(); ++i)
    {
//        if (dS[i] < S[i]) violate = std::min(violate, S[i]);
        if (dS[i] > 1.0 - S[i])
            violate[0] = std::min(violate[0], (1.0 - S[i]) / dS[i] *.9);
    }

    violate = Min(violate = std::min(violate[0], 1.0), comm_);

    if (!myid_ && violate[0] < 1.0)
        std::cout<< "backtracking = " << violate[0] << "\n";

    blk_dx *= violate[0];

    x += blk_dx;

    resid_norm_ = violate[0] < 1e-2 ? 0.0 : ResidualNorm(x, rhs);
}

CoupledFAS::CoupledFAS(const Hierarchy& hierarchy,
                       const double dt,
                       const double weight,
                       const double density,
                       FASParameters param)
    : FAS(hierarchy.GetComm(), param), hierarchy_(hierarchy)
{
    for (int l = 0; l < param_.num_levels; ++l)
    {
        auto& system_l = hierarchy.GetMatrix(l);
        auto& param_l = l ? (l < param.num_levels - 1 ? param.mid : param.coarse) : param.fine;
        solvers_[l].reset(new CoupledSolver(system_l, dt, weight, density, param_l));
//        solvers_[l]->SetPrintLevel(param_.cycle == V_CYCLE ? -1 : 1);

        if (l > 0)
        {
            rhs_[l].SetSize(system_l.NumTotalDofs() + system_l.NumVDofs());
            sol_[l].SetSize(system_l.NumTotalDofs() + system_l.NumVDofs());
            rhs_[l] = 0.0;
            sol_[l] = 0.0;
        }
        help_[l].SetSize(system_l.NumTotalDofs() + system_l.NumVDofs());
        help_[l] = 0.0;
    }
}

double CoupledFAS::Norm(int level, const mfem::Vector& vec) const
{
    return static_cast<CoupledSolver&>(*solvers_[level]).Norm(vec);
}

void CoupledFAS::Restrict(int level, const mfem::Vector& fine, mfem::Vector& coarse) const
{
    auto& solver_f = static_cast<CoupledSolver&>(*solvers_[level]);
    auto& solver_c = static_cast<CoupledSolver&>(*solvers_[level + 1]);
    mfem::BlockVector blk_fine(fine.GetData(), solver_f.BlockOffsets());
    mfem::BlockVector blk_coarse(coarse.GetData(), solver_c.BlockOffsets());
    hierarchy_.Restrict(level, blk_fine, blk_coarse);
    hierarchy_.Restrict(level, blk_fine.GetBlock(2), blk_coarse.GetBlock(2));
}

void CoupledFAS::Interpolate(int level, const mfem::Vector& coarse, mfem::Vector& fine) const
{
    auto& solver_f = static_cast<CoupledSolver&>(*solvers_[level - 1]);
    auto& solver_c = static_cast<CoupledSolver&>(*solvers_[level]);
    mfem::BlockVector blk_fine(fine.GetData(), solver_f.BlockOffsets());
    mfem::BlockVector blk_coarse(coarse.GetData(), solver_c.BlockOffsets());
    hierarchy_.Interpolate(level, blk_coarse, blk_fine);
    hierarchy_.Interpolate(level, blk_coarse.GetBlock(2), blk_fine.GetBlock(2));
}

mfem::Vector CoupledFAS::ProjectS(int level, const mfem::Vector& x) const
{
    const auto& darcy_system = hierarchy_.GetMatrix(level);
    const auto& agg_vert = hierarchy_.GetAggVert(level);
    const mfem::Vector S = darcy_system.PWConstProject(x);

    mfem::Vector S_loc, S_coarse(agg_vert.NumRows());
    mfem::Array<int> verts;
    for (int i = 0; i < agg_vert.NumRows(); ++i)
    {
        GetTableRow(agg_vert, i, verts);
        S.GetSubVector(verts, S_loc);
        S_coarse[i] = S_loc.Max();
    }

    return hierarchy_.GetMatrix(level + 1).PWConstInterpolate(S_coarse);
}

void CoupledFAS::Project(int level, const mfem::Vector& fine, mfem::Vector& coarse) const
{
    auto& solver_f = static_cast<CoupledSolver&>(*solvers_[level]);
    auto& solver_c = static_cast<CoupledSolver&>(*solvers_[level + 1]);
    mfem::BlockVector blk_fine(fine.GetData(), solver_f.BlockOffsets());
    mfem::BlockVector blk_coarse(coarse.GetData(), solver_c.BlockOffsets());
    hierarchy_.Project(level, blk_fine, blk_coarse);
    hierarchy_.Project(level, blk_fine.GetBlock(2), blk_coarse.GetBlock(2));
//    blk_coarse.GetBlock(2) = ProjectS(level, blk_fine.GetBlock(2));
}

mfem::Vector TransportSolver::Residual(const mfem::Vector& x, const mfem::Vector& y)
{
    mfem::Vector out(x);
    auto FS = FractionalFlow(darcy_system_.PWConstProject(x));
    Adv_.Mult(1.0, FS, Ms_(0, 0), out);
    out -= y;
    return out;
}

void TransportSolver::Step(const mfem::Vector& rhs, mfem::Vector& x, mfem::Vector& dx)
{
    mfem::SparseMatrix df_ds = darcy_system_.GetPWConstProj();
    df_ds.ScaleRows(dFdS(darcy_system_.PWConstProject(x)));

    auto A = ParMult(Adv_, df_ds, starts_);
    GetDiag(*A) += Ms_;

    unique_ptr<mfem::HypreBoomerAMG> solver(BoomerAMG(*A));
    gmres_.SetPreconditioner(*solver);
    gmres_.SetOperator(*A);

    dx = 0.0;
    auto resid = Residual(x, rhs);
    gmres_.Mult(resid, dx);
//    if (!myid_) std::cout << "GMRES took " << gmres_.GetNumIterations() << " iterations\n";

    const mfem::Vector dS = darcy_system_.PWConstProject(dx);
    dx *= std::min(1.0, param_.diff_tol / mfem::ParNormlp(dS, mfem::infinity(), comm_));
    x -= dx;
}

//mfem::Vector TotalMobility(const mfem::Vector& S)
//{
//    mfem::Vector LamS(S.Size());
//    LamS = 1000.;
//    return LamS;
//}

//mfem::Vector dTMinv_dS(const mfem::Vector& S)
//{
//    mfem::Vector out(S.Size());
//    out = 0.0;
//    return out;
//}

//mfem::Vector FractionalFlow(const mfem::Vector& S)
//{
//    mfem::Vector FS(S);
//    return FS;
//}

//mfem::Vector dFdS(const mfem::Vector& S)
//{
//    mfem::Vector out(S.Size());
//    out = 1.0;
//    return out;
//}

//mfem::Vector TotalMobility(const mfem::Vector& S)
//{
//    mfem::Vector LamS(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        LamS(i)  = S_w * S_w + S_o * S_o / 5.0;
//    }
//    return LamS;
//}

//mfem::Vector dTMinv_dS(const mfem::Vector& S)
//{
//    mfem::Vector out(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        out(i)  = 2.0 * (S_w - S_o / 5.0);
//        double Lam_S  = S_w * S_w + S_o * S_o / 5.0;
//        out(i) = -1.0 * out(i) / (Lam_S * Lam_S);
//    }
//    return out;
//}

//mfem::Vector FractionalFlow(const mfem::Vector& S)
//{
//    mfem::Vector FS(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        double Lam_S  = S_w * S_w + S_o * S_o / 5.0;
//        FS(i) = S_w * S_w / Lam_S;
//    }
//    return FS;
//}

//mfem::Vector dFdS(const mfem::Vector& S)
//{
//    mfem::Vector out(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        double Lam_S  = S_w * S_w + S_o * S_o / 5.0;
//        out(i) = 0.4 * (S_w - S_w * S_w) / (Lam_S * Lam_S);
//    }
//    return out;
//}

// case 1
//mfem::Vector TotalMobility(const mfem::Vector& S)
//{
//    mfem::Vector LamS(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        LamS(i)  = S_w * S_w / 1e-3 + std::pow(S_o, 1.5) / 1e-4;
//    }
//    return LamS;
//}

//mfem::Vector dTMinv_dS(const mfem::Vector& S)
//{
//    mfem::Vector out(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        out(i)  = 2. * S_w / 1e-3 - 1.5 * std::pow(S_o, 0.5) / 1e-4;
//        double Lam_S  = S_w * S_w / 1e-3 + std::pow(S_o, 1.5) / 1e-4;
//        out(i) = -1.0 * out(i) / (Lam_S * Lam_S);
//    }
//    return out;
//}

//mfem::Vector FractionalFlow(const mfem::Vector& S)
//{
//    mfem::Vector FS(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        double Lam_S  = S_w * S_w / 1e-3 + std::pow(S_o, 1.5) / 1e-4;
//        FS(i) = S_w * S_w / 1e-3 / Lam_S;
//    }
//    return FS;
//}

//mfem::Vector dFdS(const mfem::Vector& S)
//{
//    mfem::Vector out(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        double dLw_dS = 2. * S_w / 1e-3;
//        double Lam_S  = S_w * S_w / 1e-3 + std::pow(S_o, 1.5) / 1e-4;
//        double dLam_dS = 2. * S_w / 1e-3 - 1.5 * std::pow(S_o, 0.5) / 1e-4;
//        out(i) = (dLw_dS * Lam_S - dLam_dS * S_w * S_w / 1e-3) / (Lam_S * Lam_S);
//    }
//    return out;
//}

// case 2
mfem::Vector TotalMobility(const mfem::Vector& S)
{
    mfem::Vector LamS(S.Size());
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);
        double S_o = 1.0 - S_w;
        LamS(i)  = S_w * S_w / 1e-3 + std::pow(S_o, 3.0) / 1e-2;
    }
    return LamS;
}

mfem::Vector dTMinv_dS(const mfem::Vector& S)
{
    mfem::Vector out(S.Size());
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);
        double S_o = 1.0 - S_w;
        out(i)  = 2. * S_w / 1e-3 - 3.0 * std::pow(S_o, 2.0) / 1e-2;
        double Lam_S  = S_w * S_w / 1e-3 + std::pow(S_o, 3.0) / 1e-2;
        out(i) = -1.0 * out(i) / (Lam_S * Lam_S);
    }
    return out;
}

mfem::Vector FractionalFlow(const mfem::Vector& S)
{
    mfem::Vector FS(S.Size());
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);
        double S_o = 1.0 - S_w;
        double Lam_S  = S_w * S_w / 1e-3 + std::pow(S_o, 3.0) / 1e-2;
        FS(i) = S_w * S_w / 1e-3 / Lam_S;
    }
    return FS;
}

mfem::Vector dFdS(const mfem::Vector& S)
{
    mfem::Vector out(S.Size());
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);
        double S_o = 1.0 - S_w;
        double dLw_dS = 2. * S_w / 1e-3;
        double Lam_S  = S_w * S_w / 1e-3 + std::pow(S_o, 3.0) / 1e-2;
        double dLam_dS = 2. * S_w / 1e-3 - 3.0 * std::pow(S_o, 2.0) / 1e-2;
        out(i) = (dLw_dS * Lam_S - dLam_dS * S_w * S_w / 1e-3) / (Lam_S * Lam_S);
    }
    return out;
}
