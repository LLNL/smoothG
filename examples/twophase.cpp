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
    const NLSolverParameters& solver_param_;
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

    std::vector<mfem::Vector> ComputeMicroUpwindFlux();
public:
    TwoPhaseSolver(const TwoPhase& problem, Hierarchy& hierarchy,
                   const int level, const EvolveParamenters& evolve_param,
                   const NLSolverParameters& solver_param);

    void TimeStepping(const double dt, mfem::BlockVector& x);
    mfem::BlockVector Solve(const mfem::BlockVector& init_val);
};

class CoupledSolver : public NonlinearSolver
{
    const MixedMatrix& darcy_system_;
    mfem::GMRESSolver gmres_;
    mfem::SparseMatrix Ms_;
    unique_ptr<mfem::HypreParMatrix> D_;
    unique_ptr<mfem::HypreParMatrix> DT_;

    mfem::Array<int> blk_offsets_;
    mfem::Array<int> true_blk_offsets_;
    const mfem::Array<int>& ess_dofs_;
    const mfem::Array<int>& vert_starts_;
    mfem::Array<int> true_edof_starts_;
    const std::vector<mfem::DenseMatrix>& traces_;

    const double dt_;
    const double weight_;
    const double density_;

    mfem::Vector normalizer_;

    void Step(const mfem::Vector& rhs, mfem::Vector& x, mfem::Vector& dx) override;
    mfem::Vector AssembleTrueVector(const mfem::Vector& vec) const;
    std::vector<mfem::DenseMatrix> Build_dMdS(const mfem::BlockVector& x);
    mfem::SparseMatrix Assemble_dMdS(const mfem::BlockVector& blk_x);
public:
    CoupledSolver(const MixedMatrix& darcy_system,
                  const std::vector<mfem::DenseMatrix>& edge_traces,
                  const double dt,
                  const double weight,
                  const double density,
                  NLSolverParameters param);

    mfem::Vector Residual(const mfem::Vector& x, const mfem::Vector& y) override;
    double ResidualNorm(const mfem::Vector& x, const mfem::Vector& y) override;
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
    NLSolverParameters solver_param;
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
    int scheme = 1;
    args.AddOption(&scheme, "-scheme", "--stepping-scheme",
                   "Time stepping: 1. IMPES, 2. sequentially implicit, 3. fully implicit. ");
    int num_backtrack = 0;
    args.AddOption(&num_backtrack, "--num-backtrack", "--num-backtrack",
                   "Maximum number of backtracking steps.");
    double diff_tol = -1.0;
    args.AddOption(&diff_tol, "--diff-tol", "--diff-tol",
                   "Tolerance for coefficient change.");UpscaleParameters upscale_param;
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
    solver_param.num_backtrack = num_backtrack;
    solver_param.diff_tol = diff_tol;
    solver_param.print_level = -1;
    solver_param.max_num_iter = 25;
    solver_param.atol = 1e-10;
    solver_param.rtol = 0.0;

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

    // Fine scale transport based on fine flux
    for (int l = 0; l < upscale_param.max_levels; ++l)
    {
        TwoPhaseSolver solver(problem, hierarchy, l, evolve_param, solver_param);

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

        mfem::Vector fine_S = l ? hierarchy.Interpolate(1, sol.GetBlock(2)) : sol.GetBlock(2);

        double norm = mfem::ParNormlp(fine_S, 1, comm);
        if (myid == 0) { std::cout << "    || S ||_1 = " << norm << "\n"; }
    }
    return EXIT_SUCCESS;
}

mfem::SparseMatrix BuildUpwindPattern(const GraphSpace& graph_space,
                                      const mfem::SparseMatrix& micro_upwind_fluxes,
                                      const mfem::Vector& flux)
{
    const Graph& graph = graph_space.GetGraph();
    const mfem::SparseMatrix& edge_vert = graph.EdgeToVertex();
    const mfem::SparseMatrix e_te_diag = GetDiag(graph.EdgeToTrueEdge());

    mfem::SparseMatrix upwind_pattern(graph.NumEdges(), graph.NumVertices());

    for (int i = 0; i < graph.NumEdges(); ++i)
    {
        const double weight = micro_upwind_fluxes.GetRowEntries(i)[0];
        if (edge_vert.RowSize(i) == 2) // edge is interior
        {
//            const int upwind_vert = flux[i] > 0.0 ? 0 : 1;
//            upwind_pattern.Set(i, edge_vert.GetRowColumns(i)[upwind_vert], weight);

            int upwind_vert = micro_upwind_fluxes.GetRowColumns(i)[0];
            if (flux[i] < 0.0)
            {
                if (edge_vert.GetRowColumns(i)[0] == upwind_vert)
                    upwind_vert = edge_vert.GetRowColumns(i)[1];
                else
                    upwind_vert = edge_vert.GetRowColumns(i)[0];
            }

            upwind_pattern.Set(i, upwind_vert, weight);

            assert(micro_upwind_fluxes.RowSize(i) == 1);
//            if (micro_upwind_fluxes.RowSize(i) == 1) { continue; }

//            const int downwind_vert = flux[i] > 0.0 ? 0 : 1;
//            upwind_pattern.Set(i, edge_vert.GetRowColumns(i)[downwind_vert], 1.0);
        }
        else
        {
            assert(edge_vert.RowSize(i) == 1);
            const bool edge_is_owned = e_te_diag.RowSize(i);

            if ((flux[i] > 0.0 && edge_is_owned) || (flux[i] <= 0.0 && !edge_is_owned))
            {
                upwind_pattern.Set(i, edge_vert.GetRowColumns(i)[0], weight);
            }
        }
    }
    upwind_pattern.Finalize(); // TODO: use sparsity pattern of DT and update the values

    return upwind_pattern;
}

mfem::Vector ComputeFaceFlux(const MixedMatrix& darcy_system,
                             const mfem::Vector& flux)
{
    mfem::Vector out(darcy_system.GetTraceFluxes());
    RescaleVector(flux, out);
    return out;
}

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
                               const NLSolverParameters& solver_param)
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
    if (evolve_param_.vis_step) { problem_.VisSetup(sout, x_blk2, 0.0, 1.0, "Fine scale"); }

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
    for (int step = 1; !done; step++)
    {
        mfem::BlockVector previous_x(x);
        dt_real = std::min(std::min(dt_real * 2.0, evolve_param_.total_time - time), evolve_param_.dt);
        //        dt_real = std::min(param_.dt, param_.total_time - time);
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

    return x;
}
int yo = 1;
void TwoPhaseSolver::TimeStepping(const double dt, mfem::BlockVector& x)
{
    const MixedMatrix& system = hierarchy_.GetMatrix(level_);
    std::vector<mfem::DenseMatrix> traces;

    if (evolve_param_.scheme == FullyImplcit) // coupled: solve all unknowns together
    {
        CoupledSolver solver(system, traces, dt, weight_, density_, solver_param_);

        mfem::BlockVector rhs(*source_);
        rhs.GetBlock(0) *= (1. / dt / density_);
        rhs.GetBlock(1) *= (dt * density_);
        add(dt * density_, rhs.GetBlock(2), weight_, x.GetBlock(2), rhs.GetBlock(2));
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

        auto upwind = !level_ ? BuildUpwindPattern(system.GetGraphSpace(), x.GetBlock(0))
                              : BuildUpwindPattern(system.GetGraphSpace(),
                                                   hierarchy_.GetUpwindFlux(level_ - 1),
                                                   x.GetBlock(0));
        upwind.ScaleRows(x.GetBlock(0));
//        if (level_&& yo == 1){upwind.Print();yo++;}

        unique_ptr<mfem::HypreParMatrix> Adv;
        if (level_<0)
        {
            auto& system0 = hierarchy_.GetMatrix(0);
            auto& e_te_e = system0.GetGraph().EdgeToTrueEdgeToEdge();
            auto& starts = system0.GetGraph().VertexStarts();
            auto D_te_e = ParMult(system0.GetD(), e_te_e, starts);

            auto flux0 = Mult(hierarchy_.GetPsigma(0), x.GetBlock(0));
            auto upwind0 = BuildUpwindPattern(system0.GetGraphSpace(), flux0);
            upwind0.ScaleRows(flux0);

            auto Adv0 = ParMult(*D_te_e, upwind0, starts);

            auto vert_agg = smoothg::Transpose(hierarchy_.GetAggVert(0));
            auto tmp = ParMult(*Adv0, vert_agg, system.GetGraph().VertexStarts());

//            auto tmp = ParMult(*Adv0, hierarchy_.GetPu(0), system.GetGraph().VertexStarts());

            auto PuT = smoothg::Transpose(hierarchy_.GetPu(0));
            Adv = ParMult(PuT, *tmp, system.GetGraph().VertexStarts());



//            auto upwind2 = smoothg::Mult(QU, vert_agg);
//            if (level_&& yo == 1){upwind2.Print();yo++;}

//            upwind2.Add(-1.0, upwind);
//            assert(upwind2.MaxNorm() < 1e-12);
//            std::cout<<"upwind diff = " << upwind2.MaxNorm()<<"\n";

//            auto Dc = RAP(hierarchy_.GetPu(0), hierarchy_.GetMatrix(0).GetD(), hierarchy_.GetPsigma(0));
//            if (level_&& yo ==1){hierarchy_.GetMatrix(1).GetD().Print();;yo++;}
        }
        else
        {
            Adv = ParMult(*D_te_e_, upwind, system.GetGraph().VertexStarts());
        }
//        if (level_&& yo ==1){GetDiag(*Adv).Print();yo++;}

        if (evolve_param_.scheme == IMPES) // explcict: new_S = S + dt W^{-1} (b - Adv F(S))
        {
            mfem::Vector dSdt(source_->GetBlock(2));

            auto FS = FractionalFlow(S);
            Adv->Mult(-1.0, system.PWConstInterpolate(FS), 1.0, dSdt);


//            D_te_e_->Mult(-1.0, Mult(upwind, FractionalFlow(S)), 1.0, dSdt);
            x.GetBlock(2).Add(dt * density_ / weight_, dSdt);
            step_converged_ = true;
        }
        else // implicit: new_S solves new_S = S + dt W^{-1} (b - Adv F(new_S))
        {

            TransportSolver solver(*Adv, system, weight_ / density_ / dt, solver_param_);

            mfem::Vector rhs(source_->GetBlock(2));
            rhs.Add(weight_ / density_ / dt, x.GetBlock(2));
            solver.Solve(rhs, x.GetBlock(2));
            step_converged_ = solver.IsConverged();
            nonlinear_iter_ += solver.GetNumIterations();
        }
    }
}

CoupledSolver::CoupledSolver(const MixedMatrix& darcy_system,
                             const std::vector<mfem::DenseMatrix>& edge_traces,
                             const double dt,
                             const double weight,
                             const double density,
                             NLSolverParameters param)
    : NonlinearSolver(darcy_system.GetComm(), param),
      darcy_system_(darcy_system), gmres_(comm_),
      Ms_(SparseIdentity(darcy_system.GetGraph().NumVertices()) *= weight),
      blk_offsets_(4), true_blk_offsets_(4), ess_dofs_(darcy_system.GetEssDofs()),
      vert_starts_(darcy_system.GetGraph().VertexStarts()),
      traces_(edge_traces), dt_(dt), weight_(weight), density_(density)
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

    gmres_.SetMaxIter(1000);
    gmres_.SetRelTol(1e-8);
    gmres_.SetPrintLevel(-1);
    gmres_.SetKDim(100);
}

mfem::Vector CoupledSolver::AssembleTrueVector(const mfem::Vector& vec) const
{
    mfem::Vector true_v(true_blk_offsets_.Last());
    mfem::BlockVector blk_v(vec.GetData(), blk_offsets_);
    mfem::BlockVector blk_true_v(true_v.GetData(), true_blk_offsets_);

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

    mfem::BlockVector darcy_x(x.GetData(), darcy_system_.BlockOffsets());
    mfem::BlockVector darcy_Rx(out.GetData(), darcy_system_.BlockOffsets());

    const mfem::Vector S = darcy_system_.PWConstProject(blk_x.GetBlock(2));
    darcy_system_.Mult(TotalMobility(S), darcy_x, darcy_Rx);

    darcy_Rx.GetBlock(0) *= (1. / dt_ / density_);
    darcy_Rx.GetBlock(1) *= (dt_ * density_);

    out.GetBlock(2) = blk_x.GetBlock(2);

    const GraphSpace& space = darcy_system_.GetGraphSpace();
    auto face_flux = ComputeFaceFlux(darcy_system_, blk_x.GetBlock(0));
    auto upwind = BuildUpwindPattern(space, face_flux);
    auto upw_FS = Mult(upwind, FractionalFlow(S));
    RescaleVector(blk_x.GetBlock(0), upw_FS);
    auto U_FS = Mult(space.TrueEDofToEDof(), upw_FS);
    D_->Mult(1.0, U_FS, Ms_(0, 0), out.GetBlock(2)); //TODO: Ms_

    out -= y;
    SetZeroAtMarker(ess_dofs_, out.GetBlock(0));

    if (iter_ == 0)
    {
        normalizer_ = S;
        normalizer_ -= 1.0;
        normalizer_ *= -800.0;
        normalizer_.Add(1000.0, S);
        normalizer_ *= (weight_ / density_);
    }

    return out;
}

double CoupledSolver::ResidualNorm(const mfem::Vector& x, const mfem::Vector& y)
{
    auto true_resid = AssembleTrueVector(Residual(x, y));
    mfem::BlockVector blk_resid(true_resid.GetData(), true_blk_offsets_);

    InvRescaleVector(normalizer_, blk_resid.GetBlock(1));
    InvRescaleVector(normalizer_, blk_resid.GetBlock(2));

    return mfem::ParNormlp(blk_resid, mfem::infinity(), comm_);
}

mfem::SparseMatrix CoupledSolver::Assemble_dMdS(const mfem::BlockVector& blk_x)
{
    std::vector<mfem::DenseMatrix> local_dMdS = Build_dMdS(blk_x);

    auto& vert_edof = darcy_system_.GetGraphSpace().VertexToEDof();
    mfem::Array<int> local_edofs, local_vert(1);
    mfem::SparseMatrix out(vert_edof.NumCols(), vert_edof.NumRows());
    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        local_vert[0] = i;
        out.AddSubMatrix(local_edofs, local_vert, local_dMdS[i]);
    }
    out.Finalize();
    return out;
}

void CoupledSolver::Step(const mfem::Vector& rhs, mfem::Vector& x, mfem::Vector& dx)
{
    mfem::BlockVector blk_x(x.GetData(), blk_offsets_);

    const GraphSpace& space = darcy_system_.GetGraphSpace();

    const mfem::Vector S = darcy_system_.PWConstProject(blk_x.GetBlock(2));
    auto M_proc = darcy_system_.GetMBuilder().BuildAssembledM(TotalMobility(S));
    auto dMdS_proc = Assemble_dMdS(blk_x);

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

    auto face_flux = ComputeFaceFlux(darcy_system_, blk_x.GetBlock(0));
    auto upwind = BuildUpwindPattern(space, face_flux);

    auto U_FS = Mult(space.TrueEDofToEDof(), Mult(upwind, FractionalFlow(S)));
    auto dTdsigma = ParMult(*D_, SparseDiag(U_FS), true_edof_starts_);

    upwind.ScaleRows(blk_x.GetBlock(0));
    upwind.ScaleColumns(dFdS(S));
    auto U = ParMult(space.TrueEDofToEDof(), upwind, vert_starts_);
    unique_ptr<mfem::HypreParMatrix> dTdS(mfem::ParMult(D_.get(), U.get()));
    GetDiag(*dTdS) += Ms_;

    mfem::BlockOperator op(true_blk_offsets_);
    op.SetBlock(0, 0, M.get());
    op.SetBlock(0, 1, DT_.get());
    op.SetBlock(1, 0, D_.get());
    op.SetBlock(0, 2, dMdS.get());
    op.SetBlock(2, 0, dTdsigma.get());
    op.SetBlock(2, 2, dTdS.get());

    mfem::Vector Md;
    M->GetDiag(Md);
    Md *= -1.0;
    DT_->InvScaleRows(Md);
    unique_ptr<mfem::HypreParMatrix> schur(mfem::ParMult(D_.get(), DT_.get()));
    DT_->ScaleRows(Md);

    mfem::BlockDiagonalPreconditioner prec(true_blk_offsets_);
    prec.SetDiagonalBlock(0, new mfem::HypreDiagScale(*M));
    prec.SetDiagonalBlock(1, BoomerAMG(*schur));
    prec.SetDiagonalBlock(2, BoomerAMG(*dTdS));
    prec.owns_blocks = true;

    gmres_.SetOperator(op);
    gmres_.SetPreconditioner(prec);

    mfem::Vector true_resid = AssembleTrueVector(Residual(x, rhs));
    mfem::BlockVector true_blk_dx(true_blk_offsets_);
    true_blk_dx = 0.0;
    gmres_.Mult(true_resid *= -1.0, true_blk_dx);

//    if (!myid_) std::cout << "GMRES took " << gmres_.GetNumIterations() << " iterations\n";

    mfem::BlockVector blk_dx(dx.GetData(), blk_offsets_);
    auto& dof_truedof = darcy_system_.GetGraphSpace().EDofToTrueEDof();
    dof_truedof.Mult(true_blk_dx.GetBlock(0), blk_dx.GetBlock(0));
    blk_dx.GetBlock(1) = true_blk_dx.GetBlock(1);
    blk_dx.GetBlock(2) = true_blk_dx.GetBlock(2);

    const mfem::Vector dS = darcy_system_.PWConstProject(blk_dx.GetBlock(2));
    blk_dx *= std::min(1.0, 0.2 / mfem::ParNormlp(dS, mfem::infinity(), comm_));

    x += blk_dx;
}

std::vector<mfem::DenseMatrix> CoupledSolver::Build_dMdS(const mfem::BlockVector& x)
{
    // TODO: saturation is only 1 dof per cell
    auto& vert_edof = darcy_system_.GetGraphSpace().VertexToEDof();
    auto& vert_vdof = darcy_system_.GetGraphSpace().VertexToVDof();

    auto& MB = dynamic_cast<const ElementMBuilder&>(darcy_system_.GetMBuilder());
    auto& M_el = MB.GetElementMatrices();

    auto& proj_pwc = const_cast<mfem::SparseMatrix&>(darcy_system_.GetPWConstProj());

    std::vector<mfem::DenseMatrix> out(M_el.size());
    mfem::Array<int> local_edofs, local_vdofs, vert(1);
    mfem::Vector sigma_loc, Msigma_vec;
    mfem::DenseMatrix proj_pwc_loc;

    mfem::Vector dTMinv_dS_vec = dTMinv_dS(x.GetBlock(2)); // TODO: use S

    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        GetTableRow(vert_vdof, i, local_vdofs);
        vert[0] = i;

        x.GetSubVector(local_edofs, sigma_loc);
        Msigma_vec.SetSize(local_edofs.Size());
        M_el[i].Mult(sigma_loc, Msigma_vec);
        mfem::DenseMatrix Msigma_loc(Msigma_vec.GetData(), M_el[i].Size(), 1);

        proj_pwc_loc.SetSize(1, local_vdofs.Size());
        proj_pwc_loc = 0.0;
        proj_pwc.GetSubMatrix(vert, local_vdofs, proj_pwc_loc);
        proj_pwc_loc *= dTMinv_dS_vec[i];

        out[i].SetSize(local_edofs.Size(), local_vdofs.Size());
        mfem::Mult(Msigma_loc, proj_pwc_loc, out[i]);
    }

    return out;
}

mfem::Vector TransportSolver::Residual(const mfem::Vector& x, const mfem::Vector& y)
{
    mfem::Vector out(x);
    auto FS = FractionalFlow(darcy_system_.PWConstProject(x));
    Adv_.Mult(1.0, darcy_system_.PWConstInterpolate(FS), Ms_(0, 0), out);
    out -= y;
    return out;
}

void TransportSolver::Step(const mfem::Vector& rhs, mfem::Vector& x, mfem::Vector& dx)
{
    mfem::SparseMatrix df_ds = darcy_system_.GetPWConstProj();

//    auto tmp = dFdS(darcy_system_.PWConstProject(x));
//    df_ds.ScaleRows(darcy_system_.PWConstInterpolate(tmp));

    df_ds.ScaleRows(dFdS(darcy_system_.PWConstProject(x)));

    auto A = ParMult(Adv_, df_ds, starts_);
    GetDiag(*A) += Ms_;

    unique_ptr<mfem::HypreBoomerAMG> solver(BoomerAMG(*A));
    gmres_.SetPreconditioner(*solver);
    gmres_.SetOperator(*A);

    dx = 0.0;
    auto resid = Residual(x, rhs);
    gmres_.Mult(resid, dx);

    const mfem::Vector dS = darcy_system_.PWConstProject(dx);
    dx *= std::min(1.0, 0.2 / mfem::ParNormlp(dS, mfem::infinity(), comm_));
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
mfem::Vector TotalMobility(const mfem::Vector& S)
{
    mfem::Vector LamS(S.Size());
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);
        double S_o = 1.0 - S_w;
        LamS(i)  = S_w * S_w / 1e-3 + std::pow(S_o, 1.5) / 1e-4;
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
        out(i)  = 2. * S_w / 1e-3 - 1.5 * std::pow(S_o, 0.5) / 1e-4;
        double Lam_S  = S_w * S_w / 1e-3 + std::pow(S_o, 1.5) / 1e-4;
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
        double Lam_S  = S_w * S_w / 1e-3 + std::pow(S_o, 1.5) / 1e-4;
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
        double Lam_S  = S_w * S_w / 1e-3 + std::pow(S_o, 1.5) / 1e-4;
        double dLam_dS = 2. * S_w / 1e-3 - 1.5 * std::pow(S_o, 0.5) / 1e-4;
        out(i) = (dLw_dS * Lam_S - dLam_dS * S_w * S_w / 1e-3) / (Lam_S * Lam_S);
    }
    return out;
}

// case 2
//mfem::Vector TotalMobility(const mfem::Vector& S)
//{
//    mfem::Vector LamS(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        LamS(i)  = S_w * S_w / 1e-3 + std::pow(S_o, 3.0) / 1e-2;
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
//        out(i)  = 2. * S_w / 1e-3 - 3.0 * std::pow(S_o, 2.0) / 1e-2;
//        double Lam_S  = S_w * S_w / 1e-3 + std::pow(S_o, 3.0) / 1e-2;
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
//        double Lam_S  = S_w * S_w / 1e-3 + std::pow(S_o, 3.0) / 1e-2;
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
//        double Lam_S  = S_w * S_w / 1e-3 + std::pow(S_o, 3.0) / 1e-2;
//        double dLam_dS = 2. * S_w / 1e-3 - 3.0 * std::pow(S_o, 2.0) / 1e-2;
//        out(i) = (dLw_dS * Lam_S - dLam_dS * S_w * S_w / 1e-3) / (Lam_S * Lam_S);
//    }
//    return out;
//}
