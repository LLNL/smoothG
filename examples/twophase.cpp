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
    mfem::Vector Minv_;
    const mfem::Vector& b_;
    mutable mfem::Vector FS_;
public:
    FV_Evolution(const mfem::SparseMatrix& M, const mfem::Vector& b);
    void UpdateK(const mfem::HypreParMatrix& K) { K_ref_.MakeRef(K); }
    virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const;
};

std::unique_ptr<mfem::HypreParMatrix> Advection(const mfem::Vector& flux, const Graph& graph);

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

FV_Evolution::FV_Evolution(const mfem::SparseMatrix& M, const mfem::Vector& b)
    : mfem::TimeDependentOperator(M.Height()), b_(b)
{
    M.GetDiag(Minv_); // assume M is diagonal
    for (int i = 0; i < Minv_.Size(); i++)
    {
        Minv_(i) = 1.0 / Minv_(i);
    }
    FS_.SetSize(M.Height());
}

void FV_Evolution::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    // y = M^{-1} (b - K F(x))
    y = b_;
    FractionalFlow(x, FS_);
    K_ref_.Mult(-1.0, FS_, 1.0, y);
    RescaleVector(Minv_, y);
}

std::unique_ptr<mfem::HypreParMatrix> Advection(const mfem::Vector& flux, const Graph& graph)
{
    MPI_Comm comm = graph.GetComm();
    const int num_elems_diag = graph.NumVertices();
    const int num_facets = graph.NumEdges();

    const mfem::HypreParMatrix& e_te_e = graph.EdgeToTrueEdgeToEdge();
    const mfem::SparseMatrix& e_v = graph.EdgeToVertex();
    auto facet_truefacet_elem = ParMult(e_te_e, e_v, graph.VertexStarts());

    mfem::SparseMatrix f_tf_e_diag, f_tf_e_offd, f_tf_diag;
    HYPRE_Int* elem_map;
    facet_truefacet_elem->GetDiag(f_tf_e_diag);
    facet_truefacet_elem->GetOffd(f_tf_e_offd, elem_map);
    graph.EdgeToTrueEdge().GetDiag(f_tf_diag);

    HYPRE_Int* col_map = new HYPRE_Int[f_tf_e_offd.Width()];
    std::copy_n(elem_map, f_tf_e_offd.Width(), col_map);

    mfem::SparseMatrix diag(num_elems_diag, num_elems_diag);
    mfem::SparseMatrix offd(num_elems_diag, f_tf_e_offd.Width());

    mfem::Array<int> facedofs;

    for (int f = 0; f < num_facets; ++f)
    {
        const double flux_0 = (fabs(flux(f)) - flux(f)) / 2;
        const double flux_1 = (fabs(flux(f)) + flux(f)) / 2;

        if (f_tf_e_diag.RowSize(f) == 2) // facet is interior
        {
            const int* elems = f_tf_e_diag.GetRowColumns(f);

            diag.Set(elems[0], elems[1], -flux_1);
            diag.Add(elems[1], elems[1], flux_1);
            diag.Set(elems[1], elems[0], -flux_0);
            diag.Add(elems[0], elems[0], flux_0);
        }
        else
        {
            const int diag_elem = f_tf_e_diag.GetRowColumns(f)[0];

            if (f_tf_e_offd.RowSize(f) > 0) // facet is shared
            {
                assert(f_tf_e_offd.RowSize(f) == 1);
                const int offd_elem = f_tf_e_offd.GetRowColumns(f)[0];

                if (f_tf_diag.RowSize(f) > 0) // facet is owned by local proc
                {
                    offd.Set(diag_elem, offd_elem, -flux_1);
                    diag.Add(diag_elem, diag_elem, flux_0);
                }
                else // facet is owned by the neighbor proc
                {
                    diag.Add(diag_elem, diag_elem, flux_1);
                    offd.Set(diag_elem, offd_elem, -flux_0);
                }
            }
            else if (f_tf_e_diag.RowSize(f) == 1) // global boundary
            {
                diag.Add(diag_elem, diag_elem, flux_0);
            }
            else
            {
                assert(f_tf_e_diag.RowSize(f) == 0);
                assert(f_tf_e_offd.RowSize(f) == 0);
            }
        }
    }

    diag.Finalize(0);
    offd.Finalize(0);

    mfem::Array<int> elem_starts;
    GenerateOffsets(comm, num_elems_diag, elem_starts);

    int num_elems = elem_starts.Last();
    auto out = new mfem::HypreParMatrix(comm, num_elems, num_elems, elem_starts,
                                        elem_starts, &diag, &offd, col_map);

    // Adjust ownership and copy starts arrays
    out->CopyRowStarts();
    out->CopyColStarts();
    out->SetOwnerFlags(3, 3, 1);

    diag.LoseData();
    offd.LoseData();

    return unique_ptr<mfem::HypreParMatrix>(out);
}

mfem::Vector TwoPhase::Solve(Hierarchy& hierarchy, double delta_t,
                             double total_time, int vis_step)
{
    const Graph& graph = hierarchy.GetMatrix(0).GetGraph();

    mfem::BlockVector p_rhs(hierarchy.BlockOffsets(0));
    p_rhs.GetBlock(0) = GetEdgeRHS();
    p_rhs.GetBlock(1) = GetVertexRHS();

    mfem::SparseMatrix vertex_edge;
    mfem::HypreParMatrix edge_d_td;
    mfem::Vector influx;
    std::string full_caption;

    {
        vertex_edge.MakeRef(graph.VertexToEdge());
        edge_d_td.MakeRef(graph.EdgeToTrueEdge());
        influx = GetVertexRHS();
        full_caption = "Fine scale ";
    }

    int myid;
    MPI_Comm_rank(edge_d_td.GetComm(), &myid);

    mfem::StopWatch chrono;

    mfem::SparseMatrix M = SparseIdentity(graph.NumVertices());
    M *= CellVolume();

    influx *= -1.0;
    mfem::Vector S = influx;
    S = 0.0;

    mfem::Vector total_mobility = S;
    mfem::BlockVector flow_sol(p_rhs);

    chrono.Clear();
    MPI_Barrier(edge_d_td.GetComm());
    chrono.Start();

    mfem::Vector S_vis;
//    if (S_level == Fine)
    {
        S_vis.SetDataAndSize(S.GetData(), S.Size());
    }

    mfem::socketstream sout;
    if (vis_step)
    {
//        if (S_level == Coarse)
//        {
//            up.Interpolate(S, S_vis);
//        }
        VisSetup(sout, S_vis, 0.0, 1.0, full_caption);
//        setup = false;
    }

    double time = 0.0;

    FV_Evolution adv(M, influx);
    adv.SetTime(time);

    mfem::ForwardEulerSolver ode_solver;
    ode_solver.Init(adv);

//    std::vector<mfem::Vector> sats(well_vertices.Size(), mfem::Vector(total_time / delta_t + 2));
//    for (unsigned int i = 0; i < sats.size(); i++)
//    {
//        sats[i] = 0.0;
//    }

    unique_ptr<mfem::HypreParMatrix> Adv;
    mfem::BlockVector upscaled_flow_sol(hierarchy.BlockOffsets(0));
    mfem::Vector upscaled_total_mobility;

    {
        upscaled_total_mobility.SetSize(hierarchy.GetMatrix(0).NumVDofs());
    }
    mfem::Vector flux;

    bool done = false;
    for (int ti = 0; !done; )
    {
//        if (p_level == Fine)
        {
            TotalMobility(S, total_mobility);
            hierarchy.RescaleCoefficient(0, total_mobility);
            hierarchy.Solve(0, p_rhs, flow_sol);
//            up.ShowFineSolveInfo();
        }

        {
            flux.SetDataAndSize(flow_sol.GetData(), flow_sol.BlockSize(0));
        }

//        if (coarse_Adv == CoarseAdv::Upwind)
//        {
//            Adv = DiscreteAdvection(normal_flux, vertex_edge, edge_d_td);
//        }
//        else if (coarse_Adv == CoarseAdv::RAP)
//        {
//            auto Adv_f = DiscreteAdvection(normal_flux, vertex_edge, edge_d_td);
//            auto PuT = smoothg::Transpose(up.GetPu());

//            mfem::Array<int> starts;
//            GenerateOffsets(edge_d_td.GetComm(), PuT.Height(), starts);

//            unique_ptr<mfem::HypreParMatrix> PAdv_f(Adv_f->LeftDiagMult(PuT, starts));
//            unique_ptr<mfem::HypreParMatrix> Adv_fP(PAdv_f->Transpose());
//            unique_ptr<mfem::HypreParMatrix> AdvT(Adv_fP->LeftDiagMult(PuT, starts));
//            Adv.reset(AdvT->Transpose());
//        }
//        else
        {
            Adv = Advection(flux, graph);
        }
        adv.UpdateK(*Adv);

        double dt_real = std::min(delta_t, total_time - time);
        ode_solver.Step(S, time, dt_real);
        ti++;
//std::cout<<"S norm = "<<S.Norml2()<<"\n";
        //        for (unsigned int i = 0; i < sats.size(); i++)
        //        {
        //            if (level == Coarse)
        //            {
        //                up.Interpolate(S, S_vis);
        //                sats[i](ti) = S_vis(well_vertices[i]);
        //            }
        //            else
        //            {
        //                sats[i](ti) = S(well_vertices[i]);
        //            }
        //        }

        done = (time >= total_time - 1e-8 * delta_t);

        if (myid == 0)
        {
            std::cout << "time step: " << ti << ", time: " << time << "\r";//std::endl;
        }
        if (vis_step && (done || ti % vis_step == 0))
        {
//            if (S_level == Coarse)
//            {
//                up.Interpolate(S, S_vis);
//            }
            VisUpdate(sout, S_vis);
        }
    }
    MPI_Barrier(edge_d_td.GetComm());
    if (myid == 0)
    {
        std::cout << "Time stepping done in " << chrono.RealTime() << "s.\n";
//        std::cout << "Size of discrete saturation space: " << Adv->N() << "\n";
    }

//    if (S_level == Coarse)
//    {
//        up.Interpolate(S, S_vis);
//        S = S_vis;
//    }

//    for (unsigned int i = 0; i < sats.size(); i++)
//    {
//        std::ofstream ofs("sat_prod_" + std::to_string(i) + "_" + std::to_string(myid)
//                          + "_" + std::to_string(option) + ".txt");
//        sats[i].Print(ofs, 1);
//    }

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
