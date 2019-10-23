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

/** A time-dependent operator for the right-hand side of the ODE. The semi-discrete
    equation of dS/dt + div(vF(S)) = b is M dS/dt + K F(S) = b, where M and K are
    the mass and advection matrices, F is a nonlinear function, and b describes the
    influx source. This can be written as a general ODE, dS/dt = M^{-1} (b - K F(S)),
    and this class is used to evaluate the right-hand side. */
class FV_Evolution : public mfem::TimeDependentOperator
{
private:
    mfem::HypreParMatrix K_ref_;
    mfem::Vector W_diag_;
    const mfem::Vector& b_;
    mutable mfem::Vector FS_;
public:
    FV_Evolution(const mfem::SparseMatrix& W, const mfem::Vector& b);
    void UpdateK(const mfem::HypreParMatrix& K) { K_ref_.MakeRef(K); }
    virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const;
};

void TotalMobility(const mfem::Vector& S, mfem::Vector& LamS);
void FractionalFlow(const mfem::Vector& S, mfem::Vector& FS);

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
    const char* perm_file = "spe_perm.dat";
    args.AddOption(&perm_file, "-p", "--perm",
                   "SPE10 permeability file data.");
    int dim = 3;
    args.AddOption(&dim, "-d", "--dim",
                   "Dimension of the physical space.");
    int slice = 0;
    args.AddOption(&slice, "-s", "--slice",
                   "Slice of SPE10 data to take for 2D run.");
    bool use_metis = true;
    args.AddOption(&use_metis, "-ma", "--metis-agglomeration",
                   "-nm", "--no-metis-agglomeration",
                   "Use Metis as the partitioner (instead of geometric).");
    int spe10_scale = 5;
    args.AddOption(&spe10_scale, "-sc", "--spe10-scale",
                   "Scale of problem, 1=small, 5=full SPE10.");
    double delta_t = 1.0;
    args.AddOption(&delta_t, "-dt", "--delta-t", "Time step.");
    double total_time = 1000.0;
    args.AddOption(&total_time, "-time", "--total-time", "Total time to step.");
    int vis_step = 0;
    args.AddOption(&vis_step, "-vs", "--vis-step", "Step size for visualization.");
    int well_height = 1;
    args.AddOption(&well_height, "-wh", "--well-height", "Well Height.");
    double inject_rate = 0.3;
    args.AddOption(&inject_rate, "-ir", "--inject-rate", "Injector rate.");
    double bottom_hole_pressure = 175.0;
    args.AddOption(&bottom_hole_pressure, "-bhp", "--bottom-hole-pressure",
                   "Bottom Hole Pressure.");
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
    TwoPhase fv_problem(perm_file, dim, spe10_scale, slice, false, ess_attr,
                          well_height, inject_rate, bottom_hole_pressure);

    Graph graph = fv_problem.GetFVGraph(true);
    auto& combined_ess_attr = fv_problem.EssentialAttribute();

    mfem::Array<int> geo_coarsening_factor(dim);
    geo_coarsening_factor[0] = 5;
    geo_coarsening_factor[1] = 5;
    if (dim == 3) { geo_coarsening_factor[2] = 2; }
//    spe10problem.CartPart(partition, nz, geo_coarsening_factor, well_vertices);

    Hierarchy hierarchy(graph, upscale_param, nullptr, &combined_ess_attr);
    hierarchy.PrintInfo();

    // Fine scale transport based on fine flux
    auto S_fine = fv_problem.Solve(hierarchy, delta_t, total_time, vis_step);

    double norm = mfem::ParNormlp(S_fine, 2, comm);
    if (myid == 0) { std::cout<<"|| S || = "<< norm <<"\n"; }

    return EXIT_SUCCESS;
}

FV_Evolution::FV_Evolution(const mfem::SparseMatrix& W, const mfem::Vector& b)
    : mfem::TimeDependentOperator(W.NumRows()), b_(b), FS_(W.NumRows())
{
    W.GetDiag(W_diag_); // assume W is diagonal
}

void FV_Evolution::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    // y = W^{-1} (b - K F(x))
    y = b_;
    FractionalFlow(x, FS_);
    K_ref_.Mult(-1.0, FS_, 1.0, y);
    InvRescaleVector(W_diag_, y);
}

mfem::HypreParMatrix* Advection(const mfem::Vector& flux, const Graph& graph)
{
    const mfem::HypreParMatrix& e_te_e = graph.EdgeToTrueEdgeToEdge();
    const mfem::SparseMatrix& e_v = graph.EdgeToVertex();
    auto e_te_v = ParMult(e_te_e, e_v, graph.VertexStarts());
    const mfem::SparseMatrix e_te_v_offd = GetOffd(*e_te_v);
    const mfem::SparseMatrix e_te_diag = GetDiag(graph.EdgeToTrueEdge());

    mfem::SparseMatrix diag(graph.NumVertices(), graph.NumVertices());
    mfem::SparseMatrix offd(graph.NumVertices(), e_te_v_offd.NumCols());

    for (int i = 0; i < graph.NumEdges(); ++i)
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
    GenerateOffsets(graph.GetComm(), graph.NumVertices(), starts);

    HYPRE_Int* col_map = new HYPRE_Int[e_te_v_offd.NumCols()];
    std::copy_n(GetColMap(*e_te_v), e_te_v_offd.NumCols(), col_map);

    auto out = new mfem::HypreParMatrix(graph.GetComm(), e_te_v->N(), e_te_v->N(),
                                        starts, starts, &diag, &offd, col_map);

    // Adjust ownership and copy starts arrays
    out->CopyRowStarts();
    out->CopyColStarts();
    out->SetOwnerFlags(3, 3, 1);
    diag.LoseData();
    offd.LoseData();

    return out;
}

mfem::Vector TwoPhase::Solve(Hierarchy& hierarchy, double delta_t,
                             double total_time, int vis_step)
{
    mfem::StopWatch chrono;

    const Graph& graph = hierarchy.GetMatrix(0).GetGraph();

    mfem::BlockVector p_rhs(hierarchy.BlockOffsets(0));
    p_rhs.GetBlock(0) = GetEdgeRHS();
    p_rhs.GetBlock(1) = GetVertexRHS();

    mfem::Vector influx = GetVertexRHS();
    std::string full_caption = "Fine scale ";

    mfem::SparseMatrix W = SparseIdentity(graph.NumVertices());
    W *= CellVolume();

    influx *= -1.0;
    mfem::Vector S = influx;
    S = 0.0;

    mfem::Vector total_mobility = S;
    mfem::BlockVector flow_sol(p_rhs);

    chrono.Clear();
    MPI_Barrier(graph.GetComm());
    chrono.Start();

    mfem::Vector S_vis;
    S_vis.SetDataAndSize(S.GetData(), S.Size());

    mfem::socketstream sout;
    if (vis_step) { VisSetup(sout, S_vis, 0.0, 1.0, full_caption); }

    double time = 0.0;

    FV_Evolution adv(W, influx);
    adv.SetTime(time);

    mfem::ForwardEulerSolver ode_solver;
    ode_solver.Init(adv);

    unique_ptr<mfem::HypreParMatrix> Adv;

    mfem::Vector flux;

    bool done = false;
    for (int ti = 0; !done; )
    {
        TotalMobility(S, total_mobility);
        hierarchy.RescaleCoefficient(0, total_mobility);
        hierarchy.Solve(0, p_rhs, flow_sol);

        flux.SetDataAndSize(flow_sol.GetData(), flow_sol.BlockSize(0));

        Adv.reset(Advection(flux, graph));
        adv.UpdateK(*Adv);

        double dt_real = std::min(delta_t, total_time - time);
        ode_solver.Step(S, time, dt_real);
        ti++;

        done = (time >= total_time - 1e-8 * delta_t);

        if (myid_ == 0)
        {
            std::cout << "time step: " << ti << ", time: " << time << "\r";//std::endl;
        }
        if (vis_step && (done || ti % vis_step == 0))
        {
            VisUpdate(sout, S_vis);
        }
    }

    MPI_Barrier(graph.GetComm());
    if (myid_ == 0)
    {
        std::cout << "Time stepping done in " << chrono.RealTime() << "s.\n";
    }

    return S;
}

void TotalMobility(const mfem::Vector& S, mfem::Vector& LamS)
{
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);
        double S_o = 1.0 - S_w;
        LamS(i)  = S_w * S_w + S_o * S_o / 5.0;
    }
}

void FractionalFlow(const mfem::Vector& S, mfem::Vector& FS)
{
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);
        double S_o = 1.0 - S_w;
        double Lam_S  = S_w * S_w + S_o * S_o / 5.0;
        FS(i) = S_w * S_w / Lam_S;
    }
}
