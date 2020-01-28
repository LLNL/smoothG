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
    unique_ptr<mfem::HypreParMatrix> D_te_e;
    int nonlinear_iter_;
    bool step_converged_;

    // TODO: these should be defined in / extracted from the problem, not here
    const double density_ = 1e3;
    const double porosity_ = 0.3;
    const double weight_;
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

    const mfem::Array<int>& starts_;
    const std::vector<mfem::DenseMatrix>& traces_;
    mfem::Array<int> block_offsets_;
    mfem::Array<int> true_block_offsets_;
    double dt_;
    double weight_;
    double density_;

    mfem::Vector normalizer_;

    void Step(const mfem::Vector& rhs, mfem::Vector& x, mfem::Vector& dx) override;
    mfem::Vector AssembleTrueVector(const mfem::Vector& vec) const;
    std::vector<mfem::DenseMatrix> Build_dMdS(const mfem::BlockVector& x);
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
    solver_param.print_level = 0;
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
        mfem::BlockVector S = solver.Solve(initial_value);

        if (myid == 0)
        {
            std::cout << "Level " << l << ":\n    Time stepping done in "
                      << chrono.RealTime() << "s.\n";
        }

        double norm = mfem::ParNormlp(S.GetBlock(2), 2, comm);
        if (myid == 0) { std::cout << "    || S || = " << norm << "\n"; }
    }
    return EXIT_SUCCESS;
}

mfem::Vector ComputeFaceFlux(const GraphSpace& graph_space,
                             const std::vector<mfem::DenseMatrix>& edge_traces_,
                             const mfem::Vector& flux)
{
    const mfem::SparseMatrix& edge_edof = graph_space.EdgeToEDof();
    mfem::Array<int> local_edofs;
    mfem::Vector local_flux;
    mfem::Vector out(edge_edof.NumRows());

    for (int i = 0; i < edge_edof.NumRows(); ++i)
    {
        GetTableRow(edge_edof, i, local_edofs);
        flux.GetSubVector(local_edofs, local_flux);

        mfem::Vector local_finer_flux(edge_traces_[i].NumRows());
        edge_traces_[i].Mult(local_flux, local_finer_flux);
        out[i] = local_finer_flux.Sum();
    }

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
    D_te_e = ParMult(hierarchy.GetMatrix(level).GetD(), e_te_e, starts);
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

void TwoPhaseSolver::TimeStepping(const double dt, mfem::BlockVector& x)
{
    const MixedMatrix& system = hierarchy_.GetMatrix(level_);
    const std::vector<mfem::DenseMatrix>& traces = hierarchy_.GetTraces(level_);

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

        auto flux = ComputeFaceFlux(system.GetGraphSpace(), traces, x.GetBlock(0));
        auto upwind = BuildUpwindPattern(system.GetGraphSpace(), flux);
        upwind.ScaleRows(flux);

        if (evolve_param_.scheme == IMPES) // explcict: new_S = S + dt W^{-1} (b - Adv F(S))
        {
            mfem::Vector dSdt(source_->GetBlock(2));
            D_te_e->Mult(-1.0, Mult(upwind, FractionalFlow(S)), 1.0, dSdt);
            x.GetBlock(2).Add(dt * density_ / weight_, dSdt);
            step_converged_ = true;
        }
        else // implicit: new_S solves new_S = S + dt W^{-1} (b - Adv F(new_S))
        {
            auto Adv = ParMult(*D_te_e, upwind, system.GetGraph().VertexStarts());
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
      starts_(darcy_system.GetGraph().VertexStarts()), traces_(edge_traces),
      block_offsets_(4), true_block_offsets_(4),
      dt_(dt), weight_(weight), density_(density)
{
    block_offsets_[0] = 0;
    block_offsets_[1] = darcy_system.NumEDofs();
    block_offsets_[2] = block_offsets_[1] + darcy_system.NumVDofs();
    block_offsets_[3] = block_offsets_[2] + darcy_system.GetGraph().NumVertices();

    true_block_offsets_[0] = 0;
    true_block_offsets_[1] = darcy_system.GetGraphSpace().EDofToTrueEDof().NumCols();
    true_block_offsets_[2] = true_block_offsets_[1] + darcy_system.NumVDofs();
    true_block_offsets_[3] = true_block_offsets_[2] + darcy_system.GetGraph().NumVertices();

    gmres_.SetMaxIter(1000);
    gmres_.SetRelTol(1e-8);
}

mfem::Vector CoupledSolver::AssembleTrueVector(const mfem::Vector& vec) const
{
    mfem::Vector true_v(true_block_offsets_.Last());
    mfem::BlockVector blk_v(vec.GetData(), block_offsets_);
    mfem::BlockVector blk_true_v(true_v.GetData(), true_block_offsets_);

    auto& truedof_dof = darcy_system_.GetGraphSpace().TrueEDofToEDof();
    truedof_dof.Mult(blk_v.GetBlock(0), blk_true_v.GetBlock(0));
    blk_true_v.GetBlock(1) = blk_v.GetBlock(1);
    blk_true_v.GetBlock(2) = blk_v.GetBlock(2);

    return true_v;
}

mfem::Vector CoupledSolver::Residual(const mfem::Vector& x, const mfem::Vector& y)
{
    mfem::BlockVector blk_x(x.GetData(), block_offsets_);
    mfem::BlockVector out(block_offsets_);

    mfem::BlockVector darcy_x(x.GetData(), darcy_system_.BlockOffsets());
    mfem::BlockVector darcy_Rx(out.GetData(), darcy_system_.BlockOffsets());

    const mfem::Vector S = darcy_system_.PWConstProject(blk_x.GetBlock(2));
    darcy_system_.Mult(TotalMobility(S), darcy_x, darcy_Rx);

    darcy_Rx.GetBlock(0) *= (1. / dt_ / density_);
    darcy_Rx.GetBlock(1) *= (dt_ * density_);

    out.GetBlock(2) = blk_x.GetBlock(2);

    auto face_flux = ComputeFaceFlux(darcy_system_.GetGraphSpace(),
                                     traces_, blk_x.GetBlock(0));
    auto upwind = BuildUpwindPattern(darcy_system_.GetGraphSpace(), face_flux);
    upwind.ScaleRows(blk_x.GetBlock(0));

    unique_ptr<mfem::HypreParMatrix> D(darcy_system_.MakeParallelD(darcy_system_.GetD()));

    auto U = ParMult(darcy_system_.GetGraphSpace().TrueEDofToEDof(), upwind, starts_);
    auto Adv = mfem::ParMult(D.get(), U.get());
    Adv->Mult(dt_ * density_, FractionalFlow(S), Ms_(0, 0), out.GetBlock(2)); //TODO: Ms_

    out -= y;
    SetZeroAtMarker(darcy_system_.GetEssDofs(), out.GetBlock(0));

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
    mfem::BlockVector blk_resid(true_resid.GetData(), true_block_offsets_);

    InvRescaleVector(normalizer_, blk_resid.GetBlock(1));
    InvRescaleVector(normalizer_, blk_resid.GetBlock(2));

    return mfem::ParNormlp(blk_resid, mfem::infinity(), comm_);
}

void CoupledSolver::Step(const mfem::Vector& rhs, mfem::Vector& x, mfem::Vector& dx)
{
    mfem::BlockVector blk_x(x.GetData(), block_offsets_);

    const GraphSpace& space = darcy_system_.GetGraphSpace();
    auto& vert_edof = space.VertexToEDof();

    const mfem::Vector S = darcy_system_.PWConstProject(blk_x.GetBlock(2));
    auto M_proc = darcy_system_.GetMBuilder().BuildAssembledM(TotalMobility(S));
    mfem::SparseMatrix D_proc(darcy_system_.GetD());

    std::vector<mfem::DenseMatrix> local_dMdS = Build_dMdS(blk_x);
    mfem::Array<int> local_edofs, local_vert(1);
    mfem::SparseMatrix dMdS_proc(vert_edof.NumCols(), vert_edof.NumRows());
    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        local_vert[0] = i;
        dMdS_proc.AddSubMatrix(local_edofs, local_vert, local_dMdS[i]);
    }
    dMdS_proc.Finalize();

    auto face_flux = ComputeFaceFlux(space, traces_, blk_x.GetBlock(0));
    auto upwind_pattern = BuildUpwindPattern(space, face_flux);

    mfem::Vector pattern_FS = Mult(upwind_pattern, FractionalFlow(S));
    upwind_pattern.ScaleRows(blk_x.GetBlock(0));
    upwind_pattern.ScaleColumns(dFdS(S));

    const auto& ess_dofs = darcy_system_.GetEssDofs();
    for (int mm = 0; mm < ess_dofs.Size(); ++mm)
    {
        if (ess_dofs[mm])
        {
            M_proc.EliminateRowCol(mm); // assume essential data = 0
            dMdS_proc.EliminateRow(mm);
            pattern_FS[mm] = 0.0;
        }
    }
    if (ess_dofs.Size()) { D_proc.EliminateCols(ess_dofs); }

    unique_ptr<mfem::HypreParMatrix> M(darcy_system_.MakeParallelM(M_proc));
    unique_ptr<mfem::HypreParMatrix> D(darcy_system_.MakeParallelD(D_proc));
    unique_ptr<mfem::HypreParMatrix> DT(D->Transpose());

    auto dMdS = ParMult(space.TrueEDofToEDof(), dMdS_proc, starts_);

    auto U_FS = Copy(space.EDofToTrueEDof());
    U_FS->ScaleRows(pattern_FS);
    auto dTdsigma = ParMult(D_proc, *U_FS, starts_);

    auto U = ParMult(space.TrueEDofToEDof(), upwind_pattern, starts_);
    unique_ptr<mfem::HypreParMatrix> dTdS(mfem::ParMult(D.get(), U.get()));

    *D *= (dt_ * density_);
    *DT *= (1. / dt_ / density_);
    *M *= (1. / dt_ / density_);
    *dMdS *= (1. / dt_ / density_);

    *dTdsigma *= (dt_ * density_);
    *dTdS *= (dt_ * density_);

    GetDiag(*dTdS) += Ms_;

    mfem::BlockOperator op(true_block_offsets_);
    op.SetBlock(0, 0, M.get());
    op.SetBlock(0, 1, DT.get());
    op.SetBlock(1, 0, D.get());
    op.SetBlock(0, 2, dMdS.get());
    op.SetBlock(2, 0, dTdsigma.get());
    op.SetBlock(2, 2, dTdS.get());

    mfem::Vector Md;
    M->GetDiag(Md);
    DT->InvScaleRows(Md);
    unique_ptr<mfem::HypreParMatrix> schur(mfem::ParMult(D.get(), DT.get()));
    (*schur) *= -1.0;
    DT->ScaleRows(Md);

    mfem::BlockDiagonalPreconditioner prec(true_block_offsets_);
    prec.SetDiagonalBlock(0, new mfem::HypreDiagScale(*M));
    prec.SetDiagonalBlock(1, BoomerAMG(*schur));
    prec.SetDiagonalBlock(2, BoomerAMG(*dTdS));
    prec.owns_blocks = true;

    gmres_.SetPrintLevel(-1);
    gmres_.SetKDim(100);
    gmres_.SetOperator(op);
    gmres_.SetPreconditioner(prec);

    mfem::Vector true_resid = AssembleTrueVector(Residual(x, rhs));
    mfem::BlockVector true_blk_dx(true_block_offsets_);
    true_blk_dx = 0.0;
    gmres_.Mult(true_resid *= -1.0, true_blk_dx);

//    if (!myid_) std::cout << "GMRES took " << gmres_.GetNumIterations() << " iterations\n";

    mfem::BlockVector blk_dx(dx.GetData(), block_offsets_);
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

    mfem::Vector dTMinv_dS_vec = dTMinv_dS(x.GetBlock(2));

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
    Adv_.Mult(1.0, FractionalFlow(darcy_system_.PWConstProject(x)), Ms_(0, 0), out);
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
