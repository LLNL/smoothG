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
    double T = 1.0;    // Total time
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
class TwoPhaseSolver : public mfem::ODESolver, NonlinearSolver
{
    const EvolveParamenters& param_;
    const TwoPhase& problem_;
    Hierarchy& hierarchy_;

    mfem::Vector b_;
    unique_ptr<mfem::HypreParMatrix> Adv_;
    mfem::Vector W_diag_;
    unique_ptr<mfem::HypreParMatrix> e_te_v_;

    double dt_;
    mfem::Array<int> ess_dofs_;
    mfem::Array<int> starts_;
    mfem::GMRESSolver gmres_;

    mfem::Vector dxdt_;

    void UpdateAdvection(const mfem::Vector& flux);

    // k = W^{-1} (b - Adv F(x))
    void ExplicitSolve(const mfem::Vector& x, mfem::Vector& k) const;

    // k solves k = W^{-1} (b - Adv F(x + dt * k))
    void ImplicitSolve(const double dt, const mfem::Vector& x, mfem::Vector& k);

    // For nonlinear solve
    virtual void IterationStep(const mfem::Vector& rhs, mfem::Vector& sol) override;
    mfem::Vector AssembleTrueVector(const mfem::Vector& v) const override { return v; }
    const mfem::Array<int>& GetEssDofs() const override { return ess_dofs_; }

public:
    TwoPhaseSolver(const TwoPhase& problem, Hierarchy& hierarchy,
                   const EvolveParamenters& param);
    virtual void Mult(const mfem::Vector& x, mfem::Vector& Rx) override;
    virtual void Step(mfem::Vector& x, double& t, double& dt) override;
    mfem::Vector Solve(const mfem::Vector& initial_guess);
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
    args.AddOption(&evolve_param.T, "-time", "--total-time", "Total time to step.");
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

    mfem::Vector initial_value(problem.GetVertexRHS().Size());
    initial_value = 0.0;

    mfem::StopWatch chrono;
    chrono.Start();
    mfem::Vector S_fine = solver.Solve(initial_value);

    if (myid == 0)
    {
        std::cout << "Time stepping done in " << chrono.RealTime() << "s.\n";
    }

    double norm = mfem::ParNormlp(S_fine, 2, comm);
    if (myid == 0) { std::cout<<"|| S || = "<< norm <<"\n"; }

    return EXIT_SUCCESS;
}

TwoPhaseSolver::TwoPhaseSolver(const TwoPhase& problem, Hierarchy& hierarchy,
                               const EvolveParamenters& param)
    : NonlinearSolver(hierarchy.GetComm(), hierarchy.GetGraph(0).NumVertices(), Newton, "", 1e-6),
      param_(param), problem_(problem), hierarchy_(hierarchy),
      b_(problem.GetVertexRHS()), W_diag_(b_.Size()), starts_(3),
      gmres_(hierarchy_.GetComm()), dxdt_(b_.Size())
{
    const Graph& graph = hierarchy_.GetGraph(0);
    const mfem::HypreParMatrix& e_te_e = graph.EdgeToTrueEdgeToEdge();
    e_te_v_ = ParMult(e_te_e, graph.EdgeToVertex(), graph.VertexStarts());
    W_diag_ = problem.CellVolume(); // assume W is diagonal

    gmres_.SetMaxIter(500);
    gmres_.SetRelTol(1e-9);
}

void TwoPhaseSolver::UpdateAdvection(const mfem::Vector& flux)
{
    const MixedMatrix& mixed_matrix = hierarchy_.GetMatrix(0);
    const Graph& graph = hierarchy_.GetGraph(0);
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
    upwind.Finalize(0);

    // TODO: put this in setup
    const GraphSpace& space = mixed_matrix.GetGraphSpace();
    unique_ptr<mfem::HypreParMatrix> D(mixed_matrix.MakeParallelD(mixed_matrix.GetD()));
    auto U = ParMult(space.TrueEDofToEDof(), upwind, space.VDofStarts());

    Adv_.reset(mfem::ParMult(D.get(), U.get()));
}

void TwoPhaseSolver::ExplicitSolve(const mfem::Vector& x, mfem::Vector& k) const
{
    k = b_;
    Adv_->Mult(-1.0, FractionalFlow(x), 1.0, k);
    InvRescaleVector(W_diag_, k);
}

void TwoPhaseSolver::ImplicitSolve(const double dt, const mfem::Vector& x, mfem::Vector& k)
{
    dt_ = dt;
    NonlinearSolver::SetPrintLevel(-1);
    mfem::Vector rhs(x);
    rhs /= dt;
    NonlinearSolver::Solve(rhs, k);
    k /= dt;
    k -= rhs;
}

void TwoPhaseSolver::IterationStep(const mfem::Vector& rhs, mfem::Vector& sol)
{
    const Graph& graph = hierarchy_.GetGraph(0);
    auto A = ParMult(*Adv_, SparseDiag(dFdS(sol)), graph.VertexStarts());
    A->InvScaleRows(W_diag_);
    GetDiag(*A).Add(1. / dt_, SparseIdentity(rhs.Size()));

    mfem::HypreBoomerAMG solver(*A);
    solver.SetPrintLevel(-1);
    gmres_.SetOperator(*A);
    gmres_.SetPreconditioner(solver);

    mfem::Vector delta_sol(sol.Size());
    delta_sol = 0.0;
    gmres_.Mult(residual_, delta_sol);
    sol -= delta_sol;
}

void TwoPhaseSolver::Mult(const mfem::Vector& x, mfem::Vector& Rx)
{
    mfem::Vector y;
    ExplicitSolve(x, y);
    Rx = x;
    Rx /= dt_;
    Rx -= y;
}

void TwoPhaseSolver::Step(mfem::Vector& x, double& t, double& dt)
{
    if (param_.is_explicit)
    {
        ExplicitSolve(x, dxdt_);
    }
    else
    {
        ImplicitSolve(dt, x, dxdt_); // solve for k: k = f(x + dt*k, t + dt)
    }
    x.Add(dt, dxdt_);
    t += dt;
}

mfem::Vector TwoPhaseSolver::Solve(const mfem::Vector& initial_value)
{
    int myid;
    MPI_Comm_rank(hierarchy_.GetComm(), &myid);

    mfem::BlockVector p_rhs(hierarchy_.BlockOffsets(0));
    p_rhs.GetBlock(0) = problem_.GetEdgeRHS();
    p_rhs.GetBlock(1) = problem_.GetVertexRHS();

    mfem::Vector S = initial_value;

    mfem::socketstream sout;
    if (param_.vis_step) { problem_.VisSetup(sout, S, 0.0, 1.0, "Fine scale"); }

    double time = 0.0;
    bool done = false;
    for (int step = 1; !done; step++)
    {
        hierarchy_.RescaleCoefficient(0, TotalMobility(S));
        mfem::BlockVector flow_sol = hierarchy_.Solve(0, p_rhs);
        UpdateAdvection(flow_sol.GetBlock(0));

        double dt_real = std::min(param_.dt, param_.T - time);
        Step(S, time, dt_real);

        done = (time >= param_.T - 1e-8 * param_.dt);

        if (myid == 0)
        {
            std::cout << "time step: " << step << ", time: " << time << "\r";
        }
        if (param_.vis_step && (done || step % param_.vis_step == 0))
        {
            problem_.VisUpdate(sout, S);
        }
    }

    return S;
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

