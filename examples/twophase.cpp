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

struct EvolveParamenters
{
    double total_time = 1.0;    // Total time
    double dt = 1.0;   // Time step size
    int vis_step = 0;
    bool is_explicit = true;
};

mfem::Vector TotalMobility(const mfem::Vector& S);
mfem::Vector FractionalFlow(const mfem::Vector& S);
mfem::Vector dFdS(const mfem::Vector& S);

/**
   This computes dS/dt that solves W dS/dt + Adv F(S) = b, which is the
   semi-discrete form of dS/dt + div(vF(S)) = b, where W and Adv are the mass
   and advection matrices, F is a nonlinear function, b is the influx source.
 */
class TwoPhaseSolver
{
    const EvolveParamenters& param_;
    const TwoPhase& problem_;
    Hierarchy& hierarchy_;

    mfem::BlockVector source_;
    unique_ptr<mfem::HypreParMatrix> Winv_D_;

    mfem::SparseMatrix BuildUpwindFlux(const mfem::Vector& flux) const;

    // k = W^{-1} (b - Adv F(x))
    mfem::Vector ExplicitSolve(const mfem::BlockVector& x) const;

    // k solves k = W^{-1} (b - Adv F(x + dt * k)) at t_{n+1} = t_n + dt
    mfem::Vector ImplicitSolve(const double dt, const mfem::BlockVector& x);

public:
    TwoPhaseSolver(const TwoPhase& problem, Hierarchy& hierarchy,
                   const EvolveParamenters& param);

    void Step(mfem::BlockVector& x, double& t, double& dt);
    mfem::BlockVector Solve(const mfem::BlockVector& initial_value);
};

class ImplicitTimeStepSolver : public NonlinearSolver
{
    mfem::Array<int> ess_dofs_;
    double dt_;
    unique_ptr<mfem::HypreParMatrix> Winv_Adv_;
    const mfem::Array<int>& starts_;
    mfem::GMRESSolver gmres_;

    virtual void Mult(const mfem::Vector& x, mfem::Vector& Rx) override;
    virtual void IterationStep(const mfem::Vector& rhs, mfem::Vector& sol) override;
    mfem::Vector AssembleTrueVector(const mfem::Vector& v) const override { return v; }
    const mfem::Array<int>& GetEssDofs() const override { return ess_dofs_; }

public:
    ImplicitTimeStepSolver(const mfem::HypreParMatrix& Winv_Adv,
                           const mfem::SparseMatrix& U,
                           const mfem::Array<int>& starts,
                           const double dt);
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
    bool use_metis = false;
    args.AddOption(&use_metis, "-ma", "--metis", "-nm", "--no-metis",
                   "Use Metis for partitioning (instead of geometric).");
    int well_height = 1;
    args.AddOption(&well_height, "-wh", "--well-height", "Well Height.");
    double inject_rate = 0.3;
    args.AddOption(&inject_rate, "-ir", "--inject-rate", "Injector rate.");
    double bhp = 175.0;
    args.AddOption(&bhp, "-bhp", "--bottom-hole-pressure", "Bottom Hole Pressure.");
    args.AddOption(&evolve_param.dt, "-dt", "--delta-t", "Time step.");
    args.AddOption(&evolve_param.total_time, "-time", "--total-time",
                   "Total time to step.");
    args.AddOption(&evolve_param.vis_step, "-vs", "--vis-step",
                   "Step size for visualization.");
    args.AddOption(&evolve_param.is_explicit, "-explicit", "--explicit",
                   "-implicit", "--implicit", "Use forward or backward Euler.");

    UpscaleParameters upscale_param;
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

    mfem::Array<int> ess_attr(dim == 3 ? 6 : 4);
    ess_attr = 1;

    // Setting up finite volume discretization problem
    TwoPhase problem(perm_file, dim, 5, slice, use_metis, ess_attr,
                     well_height, inject_rate, bhp);

    Hierarchy hierarchy(problem.GetFVGraph(true), upscale_param,
                        nullptr, &problem.EssentialAttribute());
    hierarchy.PrintInfo();

    // Fine scale transport based on fine flux
    TwoPhaseSolver solver(problem, hierarchy, evolve_param);

    mfem::BlockVector initial_value(problem.BlockOffsets());
    initial_value = 0.0;

    mfem::StopWatch chrono;
    chrono.Start();
    mfem::BlockVector S_fine = solver.Solve(initial_value);

    if (myid == 0)
    {
        std::cout << "Time stepping done in " << chrono.RealTime() << "s.\n";
    }

    double norm = mfem::ParNormlp(S_fine.GetBlock(2), 2, comm);
    if (myid == 0) { std::cout<<"|| S || = "<< norm <<"\n"; }

    return EXIT_SUCCESS;
}

TwoPhaseSolver::TwoPhaseSolver(const TwoPhase& problem, Hierarchy& hierarchy,
                               const EvolveParamenters& param)
    : param_(param), problem_(problem), hierarchy_(hierarchy), source_(problem.BlockOffsets())
{
    auto Winv = SparseIdentity(source_.BlockSize(2));
    Winv *= 1. / problem.CellVolume(); // assume W is diagonal

    const MixedMatrix& system = hierarchy_.GetMatrix(0);
    const GraphSpace& space = system.GetGraphSpace();
    unique_ptr<mfem::HypreParMatrix> D(system.MakeParallelD(system.GetD()));
    auto tmp(ParMult(Winv, *D, space.VDofStarts()));
    Winv_D_.reset(mfem::ParMult(tmp.get(), &space.TrueEDofToEDof()));

    source_.GetBlock(0) = problem_.GetEdgeRHS();
    source_.GetBlock(1) = problem_.GetVertexRHS();
    Winv.Mult(problem_.GetVertexRHS(), source_.GetBlock(2));
}

mfem::SparseMatrix TwoPhaseSolver::BuildUpwindFlux(const mfem::Vector& flux) const
{
    const GraphSpace& space = hierarchy_.GetMatrix(0).GetGraphSpace();
    const Graph& graph = space.GetGraph();
    const mfem::SparseMatrix& e_v = graph.EdgeToVertex();
    const mfem::SparseMatrix e_te_diag = GetDiag(graph.EdgeToTrueEdge());

    mfem::SparseMatrix upwind(graph.NumEdges(), graph.NumVertices());

    for (int i = 0; i < graph.NumEdges(); ++i)
    {
        if (e_v.RowSize(i) == 2) // edge is interior
        {
            const int upwind_vert = flux(i) > 0.0 ? 0 : 1;
            upwind.Set(i, e_v.GetRowColumns(i)[upwind_vert], flux(i));
        }
        else
        {
            assert(e_v.RowSize(i) == 1);
            const bool edge_is_owned = e_te_diag.RowSize(i);

            if ((flux(i) > 0.0 && edge_is_owned) || (flux(i) <= 0.0 && !edge_is_owned))
            {
                upwind.Set(i, e_v.GetRowColumns(i)[0], flux(i));
            }
        }
    }
    upwind.Finalize(); // TODO: use sparsity pattern of DT and update the values

    return upwind;
}

mfem::Vector TwoPhaseSolver::ExplicitSolve(const mfem::BlockVector& x) const
{
    hierarchy_.RescaleCoefficient(0, TotalMobility(x.GetBlock(2)));
    mfem::BlockVector flow_rhs(source_.GetData(), hierarchy_.BlockOffsets(0));
    mfem::BlockVector flow_sol = hierarchy_.Solve(0, flow_rhs);

    mfem::Vector dxdt(source_.GetBlock(2));
    mfem::SparseMatrix U = BuildUpwindFlux(flow_sol.GetBlock(0));
    mfem::Vector upwind_flux(x.GetBlock(0).Size());
    upwind_flux = 0.0;
    U.Mult(FractionalFlow(x.GetBlock(2)), upwind_flux);
    Winv_D_->Mult(-1.0, upwind_flux, 1.0, dxdt);
    return dxdt;
}

mfem::Vector TwoPhaseSolver::ImplicitSolve(const double dt, const mfem::BlockVector& x)
{
    hierarchy_.RescaleCoefficient(0, TotalMobility(x.GetBlock(2)));
    mfem::BlockVector flow_rhs(source_.GetData(), hierarchy_.BlockOffsets(0));
    mfem::BlockVector flow_sol = hierarchy_.Solve(0, flow_rhs);


    mfem::SparseMatrix U = BuildUpwindFlux(flow_sol.GetBlock(0));
    auto& starts = hierarchy_.GetGraph(0).VertexStarts();
    ImplicitTimeStepSolver implicit_step_solver(*Winv_D_, U, starts, dt);
    implicit_step_solver.SetPrintLevel(-1);

    mfem::Vector dxdt(x.GetBlock(2));
    mfem::Vector rhs(source_.GetBlock(2));
    rhs.Add(1. / dt, x.GetBlock(2));
    implicit_step_solver.Solve(rhs, dxdt);

    dxdt -= x.GetBlock(2);
    dxdt /= dt;

    return dxdt;
}

void TwoPhaseSolver::Step(mfem::BlockVector& x, double& t, double& dt)
{
    x.GetBlock(2).Add(dt, param_.is_explicit ? ExplicitSolve(x) : ImplicitSolve(dt, x));
    t += dt;
}

mfem::BlockVector TwoPhaseSolver::Solve(const mfem::BlockVector& initial_value)
{
    int myid;
    MPI_Comm_rank(hierarchy_.GetComm(), &myid);

    mfem::BlockVector x = initial_value;

    mfem::socketstream sout;
    if (param_.vis_step) { problem_.VisSetup(sout, x.GetBlock(2), 0.0, 1.0, "Fine scale"); }

    double time = 0.0;
    bool done = false;
    for (int step = 1; !done; step++)
    {

        double dt_real = std::min(param_.dt, param_.total_time - time);
        Step(x, time, dt_real);

        done = (time >= param_.total_time - 1e-8 * param_.dt);

        if (myid == 0)
        {
            std::cout << "time step: " << step << ", time: " << time << "\r";
        }
        if (param_.vis_step && (done || step % param_.vis_step == 0))
        {
            problem_.VisUpdate(sout, x.GetBlock(2));
        }
    }

    return x;
}

ImplicitTimeStepSolver::ImplicitTimeStepSolver(const mfem::HypreParMatrix& Winv_D,
                                               const mfem::SparseMatrix& U,
                                               const mfem::Array<int>& starts,
                                               const double dt)
    : NonlinearSolver(Winv_D.GetComm(), U.NumCols(), Newton, "", 1e-6), dt_(dt),
      Winv_Adv_(ParMult(Winv_D, U, starts)), starts_(starts), gmres_(Winv_D.GetComm())
{
    gmres_.SetMaxIter(500);
    gmres_.SetRelTol(1e-9);
}

void ImplicitTimeStepSolver::IterationStep(const mfem::Vector& rhs, mfem::Vector& sol)
{
    auto A = ParMult(*Winv_Adv_, SparseDiag(dFdS(sol)), starts_);
    GetDiag(*A).Add(1.0 / dt_, SparseIdentity(A->NumRows()));

    mfem::HypreBoomerAMG solver(*A);
    solver.SetPrintLevel(-1);
    gmres_.SetOperator(*A);
    gmres_.SetPreconditioner(solver);

    mfem::Vector delta_sol(rhs.Size());
    delta_sol = 0.0;
    gmres_.Mult(residual_, delta_sol);
    sol -= delta_sol;
}

void ImplicitTimeStepSolver::Mult(const mfem::Vector& x, mfem::Vector& Rx)
{
    Rx = x;
    Winv_Adv_->Mult(1.0, FractionalFlow(x), 1.0 / dt_, Rx);
}

mfem::Vector TotalMobility(const mfem::Vector& S)
{
    mfem::Vector LamS(S.Size());
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);
        double S_o = 1.0 - S_w;
        LamS(i)  = S_w * S_w + S_o * S_o / 5.0;
    }
    return LamS;
}

mfem::Vector FractionalFlow(const mfem::Vector& S)
{
    mfem::Vector FS(S.Size());
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);
        double S_o = 1.0 - S_w;
        double Lam_S  = S_w * S_w + S_o * S_o / 5.0;
        FS(i) = S_w * S_w / Lam_S;
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
        double Lam_S  = S_w * S_w + S_o * S_o / 5.0;
        out(i) = 0.4 * (S_w - S_w * S_w) / (Lam_S * Lam_S);
    }
    return out;
}

