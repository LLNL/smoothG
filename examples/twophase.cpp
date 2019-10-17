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
    mfem::HypreParMatrix K_;
    mfem::Vector Minv_;
    const mfem::Vector& b_;
    mutable mfem::Vector FS_;
public:
    FV_Evolution(const mfem::SparseMatrix& M,
                 const mfem::Vector& b);
    void UpdateK(const mfem::HypreParMatrix& K);
    virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const;
    virtual ~FV_Evolution() { }
};

void TotalMobility(const mfem::Vector& S, mfem::Vector& LamS);
void FractionalFlow(const mfem::Vector& S, mfem::Vector& FS);

enum Level {Fine, Coarse};
enum CoarseAdv {Upwind, RAP, FastRAP};

//mfem::Vector TwoPhaseFlow(const SPE10Problem& spe10problem, FiniteVolumeMLMC &up,
//                          const mfem::BlockVector& p_rhs, double delta_t,
//                          double total_time, int vis_step, Level p_level, Level S_level,
//                          const std::string& caption, CoarseAdv coarse_Adv);

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
    int max_evects = 4;
    args.AddOption(&max_evects, "-m", "--max-evects",
                   "Maximum eigenvectors per aggregate.");
    double spect_tol = 1.0;
    args.AddOption(&spect_tol, "-t", "--spect-tol",
                   "Spectral tolerance for eigenvalue problems.");
    bool metis_agglomeration = false;
    args.AddOption(&metis_agglomeration, "-ma", "--metis-agglomeration",
                   "-nm", "--no-metis-agglomeration",
                   "Use Metis as the partitioner (instead of geometric).");
    int spe10_scale = 5;
    args.AddOption(&spe10_scale, "-sc", "--spe10-scale",
                   "Scale of problem, 1=small, 5=full SPE10.");
    bool hybridization = false;
    args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                   "--no-hybridization", "Enable hybridization.");
    bool dual_target = true;
    args.AddOption(&dual_target, "-du", "--dual-target", "-no-du",
                   "--no-dual-target", "Use dual graph Laplacian in trace generation.");
    bool scaled_dual = true;
    args.AddOption(&scaled_dual, "-sd", "--scaled-dual", "-no-sd",
                   "--no-scaled-dual", "Scale dual graph Laplacian by (inverse) edge weight.");
    bool energy_dual = true;
    args.AddOption(&energy_dual, "-ed", "--energy-dual", "-no-ed",
                   "--no-energy-dual", "Use energy matrix in trace generation.");
    double delta_t = 1.0;
    args.AddOption(&delta_t, "-dt", "--delta-t", "Time step.");
    double total_time = 1000.0;
    args.AddOption(&total_time, "-time", "--total-time", "Total time to step.");
    int vis_step = 0;
    args.AddOption(&vis_step, "-vs", "--vis-step", "Step size for visualization.");
    int write_step = 0;
    args.AddOption(&write_step, "-ws", "--write-step", "Step size for writing data to file.");
    int well_height = 1;
    args.AddOption(&well_height, "-wh", "--well-height", "Well Height.");
    double inject_rate = 0.3;
    args.AddOption(&inject_rate, "-ir", "--inject-rate", "Injector rate.");
    double bottom_hole_pressure = 175.0;
    args.AddOption(&bottom_hole_pressure, "-bhp", "--bottom-hole-pressure",
                   "Bottom Hole Pressure.");
    double well_shift = 1.0;
    args.AddOption(&well_shift, "-wsh", "--well-shift", "Shift well from corners");
    int nz = 15;
    args.AddOption(&nz, "-nz", "--num-z", "Num of slices in z direction for 3d run.");
    int coarsening_factor = 10;
    args.AddOption(&coarsening_factor, "-cf", "--coarsen-factor", "Coarsening factor");
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

    const int nbdr = 6;
    mfem::Array<int> ess_attr(nbdr);
    ess_attr = 1;

    // Setting up finite volume discretization problem
    bool use_metis = true;
    TwoPhase flow_problem(perm_file, dim, spe10_scale, slice, use_metis, ess_attr,
                          well_height, inject_rate, bottom_hole_pressure);

    Graph graph = flow_problem.GetFVGraph(true);
    auto& rhs_sigma_fine = flow_problem.GetEdgeRHS();
    auto& rhs_u_fine = flow_problem.GetVertexRHS();
    auto& well_list = flow_problem.GetWells();

//    int num_producer = 0;
//    for (auto& well : well_list)
//    {
//        if (well.type == WellType::Producer)
//            num_producer++;
//    }

//    // add one boundary attribute for edges connecting production wells to reservoir
//    int total_num_producer;
//    MPI_Allreduce(&num_producer, &total_num_producer, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
//    if (total_num_producer > 0)
//    {
//        ess_attr.Append(0);
//    }

//    mfem::Array<int> partition;
//    int nparts = std::max(vertex_edge.Height() / coarsening_factor, 1);
//    bool adaptive_part = false;
//    bool use_edge_weight = (dim == 3) && (nz > 1);
//    PartitionVerticesByMetis(vertex_edge, weight, well_vertices, nparts,
//                             partition, adaptive_part, use_edge_weight);

    mfem::Array<int> geo_coarsening_factor(3);
    geo_coarsening_factor[0] = 5;
    geo_coarsening_factor[1] = 5;
    geo_coarsening_factor[2] = dim == 3 ? 1 : nz;
//    spe10problem.CartPart(partition, nz, geo_coarsening_factor, well_vertices);

    // Create Upscaler and Solve
//    Hierarchy hierarchy(comm, vertex_edge, local_weight, partition, edge_d_td,
//                        edge_bdr_att, ess_attr, spect_tol, max_evects,
//                        dual_target, scaled_dual, energy_dual, hybridization, false);
//    hierarchy.PrintInfo();
//    hierarchy.ShowSetupTime();
//    hierarchy.MakeFineSolver();

//    mfem::BlockVector rhs_fine(fvupscale.GetFineBlockVector());
//    rhs_fine.GetBlock(0) = rhs_sigma_fine;
//    rhs_fine.GetBlock(1) = rhs_u_fine;
//    mfem::BlockVector rhs_coarse = fvupscale.Restrict(rhs_fine);

    // Fine scale transport based on fine flux
//    auto S_fine = TwoPhaseFlow(
//                flow_problem, fvupscale, rhs_fine, delta_t, total_time, vis_step,
//                Fine, Fine, "saturation based on fine scale upwind", CoarseAdv::Upwind);

//    // Fine scale transport based on upscaled flux
//    auto S_upscaled = TwoPhaseFlow(
//                spe10problem, fvupscale, rhs_coarse, delta_t, total_time, vis_step,
//                Coarse, Fine, "saturation based on coarse scale upwind", CoarseAdv::Upwind);

//    // Coarse scale transport based on upscaled flux
//    auto S_coarse = TwoPhaseFlow(
//                spe10problem, fvupscale, rhs_coarse, delta_t, total_time, vis_step,
//                Coarse, Coarse, "saturation based on coarse scale upwind", CoarseAdv::Upwind);

//    auto S_coarse2 = TwoPhaseFlow(
//                flow_problem, fvupscale, rhs_coarse, delta_t, total_time, vis_step,
//                Coarse, Coarse, "saturation based on RAP", CoarseAdv::RAP);

//    auto S_coarse3 = TwoPhaseFlow(
//                spe10problem, fvupscale, rhs_coarse, delta_t, total_time, vis_step,
//                Coarse, Coarse, "saturation based on fastRAP", CoarseAdv::FastRAP);

////    double sat_err = CompareError(comm, S_upscaled, S_fine);
////    double sat_err2 = CompareError(comm, S_coarse, S_fine);
////    double sat_err3 = CompareError(comm, S_coarse2, S_fine);
//    double sat_err4 = CompareError(comm, S_coarse2, S_fine);
//    if (myid == 0)
//    {
////        std::cout << "Flow errors:\n";
////        ShowErrors(error_info);
//        std::cout << "Saturation errors: " << sat_err4 << ", " << sat_err4//<< "\n";
//                  << " " << sat_err4 << " " << sat_err4 << "\n";
//    }

    return EXIT_SUCCESS;
}

FV_Evolution::FV_Evolution(const mfem::SparseMatrix& M,
                           const mfem::Vector& b)
    : mfem::TimeDependentOperator(M.Height()), b_(b)
{
    M.GetDiag(Minv_); // assume M is diagonal
    for (int i = 0; i < Minv_.Size(); i++)
    {
        Minv_(i) = 1.0 / Minv_(i);
    }
    FS_.SetSize(M.Height());
}

void FV_Evolution::UpdateK(const mfem::HypreParMatrix& K)
{
    K_.MakeRef(K);
}

void FV_Evolution::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    // y = M^{-1} (b - K F(x))
    y = b_;
    FractionalFlow(x, FS_);
    K_.Mult(-1.0, FS_, 1.0, y);
    RescaleVector(Minv_, y);
}

//std::unique_ptr<mfem::HypreParMatrix> DiscreteAdvection(
//    const mfem::Vector& normal_flux, const mfem::SparseMatrix& elem_facet,
//    const mfem::HypreParMatrix& facet_truefacet, const Upscale* up = nullptr)
//{
//    MPI_Comm comm = facet_truefacet.GetComm();
//    const int num_elems_diag = elem_facet.Height();
//    const int num_facets = elem_facet.Width();

//    int myid; MPI_Comm_rank(comm, &myid);

//    mfem::Array<int> elem_starts;
//    GenerateOffsets(comm, num_elems_diag, elem_starts);

//    using ParMatPtr = std::unique_ptr<mfem::HypreParMatrix>;
//    ParMatPtr elem_truefacet(facet_truefacet.LeftDiagMult(elem_facet, elem_starts));

//    ParMatPtr truefacet_elem(elem_truefacet->Transpose());
//    ParMatPtr facet_truefacet_elem(mfem::ParMult(&facet_truefacet, truefacet_elem.get()));

//    mfem::SparseMatrix f_tf_e_diag, f_tf_e_offd, f_tf_diag;
//    HYPRE_Int* elem_map;
//    facet_truefacet_elem->GetDiag(f_tf_e_diag);
//    facet_truefacet_elem->GetOffd(f_tf_e_offd, elem_map);
//    facet_truefacet.GetDiag(f_tf_diag);

//    HYPRE_Int* elem_map_copy = new HYPRE_Int[f_tf_e_offd.Width()];
//    std::copy_n(elem_map, f_tf_e_offd.Width(), elem_map_copy);

//    mfem::SparseMatrix diag(num_elems_diag, num_elems_diag);
//    mfem::SparseMatrix offd(num_elems_diag, f_tf_e_offd.Width());

//    mfem::Array<int> facedofs;
//    mfem::Vector normal_flux_loc;
//    mfem::Vector normal_flux_fine;

//    for (int ifacet = 0; ifacet < num_facets; ifacet++)
//    {
//        double normal_flux_i_0 = 0.0;
//        double normal_flux_i_1 = 0.0;
//        if (up == nullptr)
//        {
//            normal_flux_i_0 = (fabs(normal_flux(ifacet)) - normal_flux(ifacet)) / 2;
//            normal_flux_i_1 = (fabs(normal_flux(ifacet)) + normal_flux(ifacet)) / 2;
//        }
//        else
//        {
//            const auto& normal_flip = up->GetCoarseToFineNormalFlip()[ifacet];
//            GetTableRow(up->GetFaceToFaceDof(), ifacet, facedofs);
//            normal_flux.GetSubVector(facedofs, normal_flux_loc);
//            normal_flux_loc *= -1.0; // P matrix stores negative traces
//            const mfem::DenseMatrix& traces = up->GetTraces()[ifacet];
//            normal_flux_fine.SetSize(traces.Height());
//            traces.Mult(normal_flux_loc, normal_flux_fine);
//            for (int i = 0; i < normal_flux_fine.Size(); i++)
//            {
//                double fine_flux_i = normal_flux_fine(i) * normal_flip[i];
//                normal_flux_i_0 += (fabs(fine_flux_i) - fine_flux_i) / 2;
//                normal_flux_i_1 += (fabs(fine_flux_i) + fine_flux_i) / 2;
//            }
//        }

//        if (f_tf_e_diag.RowSize(ifacet) == 2) // facet is interior
//        {
//            const int* elem_pair = f_tf_e_diag.GetRowColumns(ifacet);

//            diag.Set(elem_pair[0], elem_pair[1], -normal_flux_i_1);
//            diag.Add(elem_pair[1], elem_pair[1], normal_flux_i_1);
//            diag.Set(elem_pair[1], elem_pair[0], -normal_flux_i_0);
//            diag.Add(elem_pair[0], elem_pair[0], normal_flux_i_0);
//        }
//        else
//        {
//            const int diag_elem = f_tf_e_diag.GetRowColumns(ifacet)[0];

//            if (f_tf_e_offd.RowSize(ifacet) > 0) // facet is shared
//            {
//                assert(f_tf_e_offd.RowSize(ifacet) == 1);
//                const int offd_elem = f_tf_e_offd.GetRowColumns(ifacet)[0];

//                if (f_tf_diag.RowSize(ifacet) > 0) // facet is owned by local proc
//                {
//                    offd.Set(diag_elem, offd_elem, -normal_flux_i_1);
//                    diag.Add(diag_elem, diag_elem, normal_flux_i_0);
//                }
//                else // facet is owned by the neighbor proc
//                {
//                    diag.Add(diag_elem, diag_elem, normal_flux_i_1);
//                    offd.Set(diag_elem, offd_elem, -normal_flux_i_0);
//                }
//            }
//            else if (f_tf_e_diag.RowSize(ifacet) == 1) // global boundary
//            {
//                diag.Add(diag_elem, diag_elem, normal_flux_i_0);
//            }
//            else
//            {
//                assert(f_tf_e_diag.RowSize(ifacet) == 0);
//                assert(f_tf_e_offd.RowSize(ifacet) == 0);
//            }
//        }
//    }

//    diag.Finalize(0);
//    offd.Finalize(0);

//    int num_elems = elem_starts.Last();
//    auto out = make_unique<mfem::HypreParMatrix>(comm, num_elems, num_elems, elem_starts,
//                                                 elem_starts, &diag, &offd, elem_map_copy);

//    // Adjust ownership and copy starts arrays
//    out->CopyRowStarts();
//    out->CopyColStarts();
//    out->SetOwnerFlags(3, 3, 1);

//    diag.LoseData();
//    offd.LoseData();

//    return out;
//}


//mfem::socketstream sout;
//int option = 0;
//bool setup = true;
//mfem::Vector TwoPhaseFlow(const SPE10Problem& spe10problem, FiniteVolumeMLMC& up,
//                          const mfem::BlockVector& p_rhs, double delta_t,
//                          double total_time, int vis_step, Level p_level, Level S_level,
//                          const std::string& caption, CoarseAdv coarse_Adv)
//{
//    option++;
//    mfem::SparseMatrix vertex_edge;
//    mfem::HypreParMatrix edge_d_td;
//    mfem::Vector influx;
//    std::string full_caption;
//    if (S_level == Fine)
//    {
//        vertex_edge.MakeRef(spe10problem.GetVertexEdge());
//        edge_d_td.MakeRef(spe10problem.GetEdgeToTrueEdge());
//        influx = spe10problem.GetVertexRHS();
//        full_caption = "Fine scale ";
//    }
//    else
//    {
//        if (coarse_Adv == CoarseAdv::RAP)
//        {
//            vertex_edge.MakeRef(spe10problem.GetVertexEdge());
//            edge_d_td.MakeRef(spe10problem.GetEdgeToTrueEdge());
//        }
//        else if (coarse_Adv == CoarseAdv::Upwind)
//        {
//            vertex_edge.MakeRef(up.GetCoarseMatrix().GetD());
//            edge_d_td.MakeRef(up.GetCoarseMatrix().GetEdgeDofToTrueDof());
//        }
//        else
//        {
//            vertex_edge.MakeRef(up.GetAggFace());
//            edge_d_td.MakeRef(up.GetFaceTrueFace());
//        }
//        influx = up.Restrict(spe10problem.GetVertexRHS());
//        full_caption = "Coarse scale ";
//    }
//    full_caption += caption;

//    int myid;
//    MPI_Comm_rank(edge_d_td.GetComm(), &myid);

//    mfem::StopWatch chrono;

//    mfem::SparseMatrix M = SparseIdentity(spe10problem.GetVertexRHS().Size());
//    M *= spe10problem.CellVolume();
//    if (S_level == Coarse)
//    {
//        unique_ptr<mfem::SparseMatrix> M_c(mfem::RAP(up.GetPu(), M, up.GetPu()));
//        M.Swap(*M_c);
//    }

//    influx *= -1.0;
//    mfem::Vector S = influx;
//    S = 0.0;

//    mfem::Vector total_mobility = S;
//    mfem::BlockVector flow_sol(p_rhs);

//    chrono.Clear();
//    MPI_Barrier(edge_d_td.GetComm());
//    chrono.Start();

//    mfem::Vector S_vis;
//    if (S_level == Fine)
//    {
//        S_vis.SetDataAndSize(S.GetData(), S.Size());
//    }
//    else
//    {
//        S_vis.SetSize(spe10problem.GetVertexRHS().Size());
//    }

//    if (vis_step && setup)
//    {
//        if (S_level == Coarse)
//        {
//            up.Interpolate(S, S_vis);
//        }
//        spe10problem.VisSetup(sout, S_vis, 0.0, 1.0, caption);
////        setup = false;
//    }

//    double time = 0.0;

//    FV_Evolution adv(M, influx);
//    adv.SetTime(time);

//    mfem::ForwardEulerSolver ode_solver;
//    ode_solver.Init(adv);

////    std::vector<mfem::Vector> sats(well_vertices.Size(), mfem::Vector(total_time / delta_t + 2));
////    for (unsigned int i = 0; i < sats.size(); i++)
////    {
////        sats[i] = 0.0;
////    }

//    unique_ptr<mfem::HypreParMatrix> Adv;
//    mfem::BlockVector upscaled_flow_sol(up.GetFineBlockVector());
//    mfem::Vector upscaled_total_mobility;
//    if (S_level == Coarse)
//    {
//        upscaled_total_mobility.SetSize(up.GetFineBlockVector().BlockSize(1));
//    }
//    else
//    {
//        upscaled_total_mobility.SetSize(up.GetCoarseBlockVector().BlockSize(1));
//    }
//    mfem::Vector normal_flux;

//    bool done = false;
//    for (int ti = 0; !done; )
//    {
//        if (p_level == Fine)
//        {
//            TotalMobility(S, total_mobility);
//            up.RescaleFineCoefficient(total_mobility);
//            up.SolveFine(p_rhs, flow_sol);
////            up.ShowFineSolveInfo();
//        }
//        else
//        {
//            if (S_level == Coarse)
//            {
//                up.Interpolate(S, S_vis);
//                TotalMobility(S_vis, upscaled_total_mobility);
//                up.RescaleCoarseCoefficient(upscaled_total_mobility);
//            }
//            else
//            {
//                TotalMobility(S, total_mobility);
//                up.ComputeAggAverages(total_mobility, upscaled_total_mobility);
//                /// @todo fix local coarse M matrix
//                up.RescaleCoarseCoefficient(upscaled_total_mobility);
//            }
//            up.SolveCoarse(p_rhs, flow_sol);
////            up.ShowCoarseSolveInfo();
//        }

//        if ( (p_level == Coarse && S_level == Fine) || coarse_Adv == CoarseAdv::RAP)
//        {
//            up.Interpolate(flow_sol, upscaled_flow_sol);
//            normal_flux.SetDataAndSize(upscaled_flow_sol.GetData(), upscaled_flow_sol.BlockSize(0));
//        }
//        else
//        {
//            normal_flux.SetDataAndSize(flow_sol.GetData(), flow_sol.BlockSize(0));
//        }

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
//        {
//            Adv = DiscreteAdvection(normal_flux, vertex_edge, edge_d_td, &up);
//        }
//        adv.UpdateK(*Adv);

//        double dt_real = std::min(delta_t, total_time - time);
//        ode_solver.Step(S, time, dt_real);
//        ti++;
////std::cout<<"S norm = "<<S.Norml2()<<"\n";
//        //        for (unsigned int i = 0; i < sats.size(); i++)
//        //        {
//        //            if (level == Coarse)
//        //            {
//        //                up.Interpolate(S, S_vis);
//        //                sats[i](ti) = S_vis(well_vertices[i]);
//        //            }
//        //            else
//        //            {
//        //                sats[i](ti) = S(well_vertices[i]);
//        //            }
//        //        }

//        done = (time >= total_time - 1e-8 * delta_t);

//        if (myid == 0)
//        {
//            std::cout << "time step: " << ti << ", time: " << time << "\r";//std::endl;
//        }
//        if (vis_step && (done || ti % vis_step == 0))
//        {
//            if (S_level == Coarse)
//            {
//                up.Interpolate(S, S_vis);
//            }
//            spe10problem.VisUpdate(sout, S_vis);
//        }
//    }
//    MPI_Barrier(edge_d_td.GetComm());
//    if (myid == 0)
//    {
//        std::cout << "Time stepping done in " << chrono.RealTime() << "s.\n";
//        std::cout << "Size of discrete saturation space: " << Adv->N() << "\n";
//    }

//    if (S_level == Coarse)
//    {
//        up.Interpolate(S, S_vis);
//        S = S_vis;
//    }

////    for (unsigned int i = 0; i < sats.size(); i++)
////    {
////        std::ofstream ofs("sat_prod_" + std::to_string(i) + "_" + std::to_string(myid)
////                          + "_" + std::to_string(option) + ".txt");
////        sats[i].Print(ofs, 1);
////    }

//    return S;
//}

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
