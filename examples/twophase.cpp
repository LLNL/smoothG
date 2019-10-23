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

/**
   This computes dS/dt that solves W dS/dt + K F(S) = b, which is the
   semi-discrete form of dS/dt + div(vF(S)) = b, where W and K are the mass
   and advection matrices F is a nonlinear function b is the influx source.
 */
class Transport : public mfem::TimeDependentOperator
{
    const Graph& graph_;
    unique_ptr<mfem::HypreParMatrix> e_te_v_;
    unique_ptr<mfem::HypreParMatrix> K_;
    mfem::Vector W_diag_;
    mfem::Vector b_;
public:
    Transport(const Graph& graph, const mfem::SparseMatrix& W, const mfem::Vector& b);
    void UpdateK(const mfem::Vector& flux);
    // y = W^{-1} (b - K F(x))
    virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const;
};

class TwoPhaseSolver
{
    const EvolveParamenters& param_;
    const TwoPhase& problem_;
    Hierarchy& hierarchy_;
    unique_ptr<Transport> transport_;
    unique_ptr<mfem::ODESolver> ode_solver_;
public:
    TwoPhaseSolver(const TwoPhase& problem, Hierarchy& hierarchy,
                   const EvolveParamenters& param);
    mfem::Vector Solve(const mfem::Vector& initial_guess);
};

mfem::Vector TotalMobility(const mfem::Vector& S);
mfem::Vector FractionalFlow(const mfem::Vector& S);

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
    bool use_metis = true;
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
    TwoPhase problem(perm_file, dim, 5, slice, false, ess_attr,
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

Transport::Transport(const Graph& graph, const mfem::SparseMatrix& W, const mfem::Vector& b)
    : mfem::TimeDependentOperator(b.Size()), graph_(graph), b_(b)
{
    const mfem::HypreParMatrix& e_te_e = graph_.EdgeToTrueEdgeToEdge();
    e_te_v_ = ParMult(e_te_e, graph_.EdgeToVertex(), graph_.VertexStarts());
    W.GetDiag(W_diag_); // assume W is diagonal
}

void Transport::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    y = b_;
    K_->Mult(-1.0, FractionalFlow(x), 1.0, y);
    InvRescaleVector(W_diag_, y);
}

void Transport::UpdateK(const mfem::Vector& flux)
{
    const mfem::SparseMatrix& e_v = graph_.EdgeToVertex();
    const mfem::SparseMatrix e_te_v_offd = GetOffd(*e_te_v_);
    const mfem::SparseMatrix e_te_diag = GetDiag(graph_.EdgeToTrueEdge());

    mfem::SparseMatrix diag(graph_.NumVertices(), graph_.NumVertices());
    mfem::SparseMatrix offd(graph_.NumVertices(), e_te_v_offd.NumCols());

    for (int i = 0; i < graph_.NumEdges(); ++i)
    {
        const double flux_0 = (fabs(flux(i)) - flux(i)) / 2;
        const double flux_1 = (fabs(flux(i)) + flux(i)) / 2;

        if (e_v.RowSize(i) == 2) // edge is interior
        {
            const int* verts = e_v.GetRowColumns(i);
            diag.Set(verts[0], verts[1], -flux_1);
            diag.Add(verts[1], verts[1], flux_1);
            diag.Set(verts[1], verts[0], -flux_0);
            diag.Add(verts[0], verts[0], flux_0);
        }
        else
        {
            assert(e_v.RowSize(i) == 1);
            const int diag_v = e_v.GetRowColumns(i)[0];
            const bool edge_is_owned = e_te_diag.RowSize(i);
            const bool edge_is_shared = e_te_v_offd.RowSize(i);
            assert(edge_is_owned || edge_is_shared);

            diag.Add(diag_v, diag_v, edge_is_owned ? flux_0 : flux_1);

            if (edge_is_shared) // edge is shared
            {
                assert(e_te_v_offd.RowSize(i) == 1);
                const int offd_v = e_te_v_offd.GetRowColumns(i)[0];
                offd.Set(diag_v, offd_v, edge_is_owned ? -flux_1 : -flux_0);
            }
        }
    }

    diag.Finalize(0);
    offd.Finalize(0);

    mfem::Array<int> starts;
    GenerateOffsets(graph_.GetComm(), graph_.NumVertices(), starts);

    HYPRE_Int* col_map = new HYPRE_Int[e_te_v_offd.NumCols()];
    std::copy_n(GetColMap(*e_te_v_), e_te_v_offd.NumCols(), col_map);

    K_.reset(new mfem::HypreParMatrix(graph_.GetComm(), e_te_v_->N(), e_te_v_->N(),
                                      starts, starts, &diag, &offd, col_map));

    // Adjust ownership and copy starts arrays
    K_->CopyRowStarts();
    K_->CopyColStarts();
    K_->SetOwnerFlags(3, 3, 1);
    diag.LoseData();
    offd.LoseData();
}

TwoPhaseSolver::TwoPhaseSolver(const TwoPhase& problem, Hierarchy& hierarchy,
                               const EvolveParamenters& param)
    : param_(param), problem_(problem), hierarchy_(hierarchy)
{
    if (param_.is_explicit)
    {
        ode_solver_.reset(new mfem::ForwardEulerSolver());
    }
    else
    {
        ode_solver_.reset(new mfem::BackwardEulerSolver());
    }

    mfem::Vector influx = problem.GetVertexRHS();
    influx *= -1.0;
    mfem::SparseMatrix W = SparseIdentity(influx.Size()) *= problem.CellVolume();
    transport_.reset(new Transport(hierarchy.GetGraph(0), W, influx));
    ode_solver_->Init(*transport_);
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
        transport_->UpdateK(flow_sol.GetBlock(0));

        double dt_real = std::min(param_.dt, param_.T - time);
        ode_solver_->Step(S, time, dt_real);

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
