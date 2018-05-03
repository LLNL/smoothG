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
   @file timestep.cpp
   @brief Visualized pressure over time of a simple reservior model.
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "mfem.hpp"
#include "spe10.hpp"

#include "../src/picojson.h"
#include "../src/smoothG.hpp"

using std::unique_ptr;

using namespace smoothg;

void InitialCondition(mfem::ParFiniteElementSpace& ufespace, mfem::BlockVector& fine_u,
                      double initial_val);

void VisSetup(MPI_Comm comm, mfem::socketstream& vis_v, mfem::ParGridFunction& field,
              mfem::ParMesh& pmesh,
              double range = 1.0, const std::string& caption = "");

void VisUpdate(MPI_Comm comm, mfem::socketstream& vis_v, mfem::ParGridFunction& field,
               mfem::ParMesh& pmesh);

int main(int argc, char* argv[])
{
    int num_procs, myid;
    picojson::object serialize;

    // 1. Initialize MPI
    mpi_session session(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // program options from command line
    mfem::OptionsParser args(argc, argv);
    const char* permFile = "spe_perm.dat";
    args.AddOption(&permFile, "-p", "--perm",
                   "SPE10 permeability file data.");
    int nDimensions = 2;
    args.AddOption(&nDimensions, "-d", "--dim",
                   "Dimension of the physical space.");
    int slice = 0;
    args.AddOption(&slice, "-s", "--slice",
                   "Slice of SPE10 data to take for 2D run.");
    int max_evects = 4;
    args.AddOption(&max_evects, "-m", "--max-evects",
                   "Maximum eigenvectors per aggregate.");
    double spect_tol = 1.e-3;
    args.AddOption(&spect_tol, "-t", "--spect-tol",
                   "Spectral tolerance for eigenvalue problems.");
    bool metis_agglomeration = false;
    args.AddOption(&metis_agglomeration, "-ma", "--metis-agglomeration",
                   "-nm", "--no-metis-agglomeration",
                   "Use Metis as the partitioner (instead of geometric).");
    double proc_part_ubal = 2.0;
    args.AddOption(&proc_part_ubal, "-pub", "--part-unbalance",
                   "Processor partition unbalance factor.");
    int spe10_scale = 5;
    args.AddOption(&spe10_scale, "-sc", "--spe10-scale",
                   "Scale of problem, 1=small, 5=full SPE10.");
    bool hybridization = false;
    args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                   "--no-hybridization", "Enable hybridization.");
    int num_refine = 0;
    args.AddOption(&num_refine, "-nr", "--num-refine",
                   "Number of time to refine mesh.");
    double delta_t = 10.0;
    args.AddOption(&delta_t, "-dt", "--delta-t",
                   "Time step.");
    double total_time = 10000.0;
    args.AddOption(&total_time, "-time", "--total-time",
                   "Total time to step.");
    double initial_val = 1.0;
    args.AddOption(&initial_val, "-iv", "--initial-value",
                   "Initial pressure difference.");
    int vis_step = 0;
    args.AddOption(&vis_step, "-vs", "--vis_step",
                   "Step size for visualization.");
    int k = 1;
    args.AddOption(&k, "-k", "--level",
                   "Level. Fine = 0, Coarse = 1");
    bool dual_target = false;
    args.AddOption(&dual_target, "-dt", "--dual-target", "-no-dt",
                   "--no-dual-target", "Use dual graph Laplacian in trace generation.");
    bool scaled_dual = false;
    args.AddOption(&scaled_dual, "-sd", "--scaled-dual", "-no-sd",
                   "--no-scaled-dual", "Scale dual graph Laplacian by (inverse) edge weight.");
    bool energy_dual = false;
    args.AddOption(&energy_dual, "-ed", "--energy-dual", "-no-ed",
                   "--no-energy-dual", "Use energy matrix in trace generation.");
    const char* caption = "";
    args.AddOption(&caption, "-cap", "--caption",
                   "Caption for visualization");

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

    assert(k == 0 || k == 1);

    mfem::Array<int> coarseningFactor(nDimensions);
    coarseningFactor[0] = 10;
    coarseningFactor[1] = 10;
    if (nDimensions == 3)
        coarseningFactor[2] = 5;

    int nbdr;
    if (nDimensions == 3)
        nbdr = 6;
    else
        nbdr = 4;
    mfem::Array<int> ess_zeros(nbdr);
    mfem::Array<int> nat_one(nbdr);
    mfem::Array<int> nat_zeros(nbdr);
    ess_zeros = 1;
    nat_one = 0;
    nat_zeros = 0;

    mfem::Array<int> ess_attr;
    mfem::Vector weight;
    mfem::Vector rhs_sigma_fine;
    mfem::Vector rhs_u_fine;

    // Setting up finite volume discretization problem
    SPE10Problem spe10problem(permFile, nDimensions, spe10_scale, slice,
                              metis_agglomeration, proc_part_ubal, coarseningFactor);
    mfem::ParMesh* pmesh = spe10problem.GetParMesh();

    for (int i = 0; i < num_refine; ++i)
    {
        pmesh->UniformRefinement();
    }

    ess_attr.SetSize(nbdr);
    for (int i(0); i < nbdr; ++i)
        ess_attr[i] = ess_zeros[i];

    // Construct "finite volume mass" matrix using mfem instead of parelag
    mfem::RT_FECollection sigmafec(0, nDimensions);
    mfem::ParFiniteElementSpace sigmafespace(pmesh, &sigmafec);

    mfem::ParBilinearForm a(&sigmafespace);
    a.AddDomainIntegrator(
        new FiniteVolumeMassIntegrator(*spe10problem.GetKInv()) );
    a.Assemble();
    a.Finalize();
    a.SpMat().GetDiag(weight);

    for (int i = 0; i < weight.Size(); ++i)
    {
        weight[i] = 1.0 / weight[i];
    }

    mfem::L2_FECollection ufec(0, nDimensions);
    mfem::ParFiniteElementSpace ufespace(pmesh, &ufec);

    mfem::ConstantCoefficient pinflow_coeff(0.);
    mfem::LinearForm b(&sigmafespace);
    b.AddBoundaryIntegrator(
        new mfem::VectorFEBoundaryFluxLFIntegrator(pinflow_coeff));
    b.Assemble();
    rhs_sigma_fine = b;

    rhs_u_fine = 0.0;

    // Construct vertex_edge table in mfem::SparseMatrix format
    auto& vertex_edge_table = nDimensions == 2 ? pmesh->ElementToEdgeTable()
                              : pmesh->ElementToFaceTable();
    mfem::SparseMatrix vertex_edge = TableToMatrix(vertex_edge_table);

    // Construct agglomerated topology based on METIS or Cartesion aggloemration
    mfem::Array<int> partitioning;
    if (metis_agglomeration)
    {
        MetisPart(partitioning, sigmafespace, ufespace, coarseningFactor);
    }
    else
    {
        auto num_procs_xyz = spe10problem.GetNumProcsXYZ();
        CartPart(partitioning, num_procs_xyz, *pmesh, coarseningFactor);
    }

    const auto& edge_d_td(sigmafespace.Dof_TrueDof_Matrix());

    auto edge_boundary_att = GenerateBoundaryAttributeTable(pmesh);

    mfem::SparseMatrix W_block = SparseIdentity(vertex_edge.Height());

    const double cell_volume = spe10problem.CellVolume(nDimensions);
    W_block *= cell_volume / delta_t;     // W_block = Mass matrix / delta_t

    //W_block *= 1.0 / delta_t;

    // Time Stepping
    {
        FiniteVolumeUpscale fvupscale(comm, vertex_edge, weight, W_block,
                                      partitioning, *edge_d_td, edge_boundary_att,
                                      ess_attr, spect_tol, max_evects, dual_target,
                                      scaled_dual, energy_dual, hybridization);

        fvupscale.PrintInfo();

        // Input Vectors
        std::vector<mfem::Array<int>> offsets(2);
        fvupscale.FineBlockOffsets(offsets[0]);
        fvupscale.CoarseBlockOffsets(offsets[1]);

        mfem::BlockVector fine_rhs(offsets[0]);
        fine_rhs.GetBlock(0) = 0.0;
        fine_rhs.GetBlock(1) = rhs_u_fine;

        // Set some pressure initial condition
        mfem::BlockVector fine_u(offsets[0]);
        InitialCondition(ufespace, fine_u, initial_val);

        // Create Workspace
        mfem::BlockVector tmp(offsets[k]);
        tmp = 0.0;

        mfem::BlockVector work_rhs(offsets[k]);
        mfem::BlockVector work_u(offsets[k]);

        if (k == 0)
        {
            fvupscale.MakeFineSolver();

            work_rhs = fine_rhs;
            work_u = fine_u;
        }
        else
        {
            fvupscale.Restrict(fine_u, work_u);
            fvupscale.Restrict(fine_rhs, work_rhs);
        }

        const mfem::SparseMatrix* W = fvupscale.GetMatrix(k).GetW();
        assert(W);

        // Setup visualization
        mfem::socketstream vis_v, vis_w;
        mfem::ParGridFunction field(&ufespace);

        if (vis_step > 0)
        {
            field = fine_u.GetBlock(1);
            VisSetup(comm, vis_v, field, *pmesh, initial_val, caption);
        }

        fvupscale.ShowSetupTime();

        double time = 0.0;
        int count = 0;

        mfem::StopWatch chrono;
        chrono.Start();

        while (time < total_time)
        {
            W->Mult(work_u.GetBlock(1), tmp.GetBlock(1));

            //tmp += work_rhs; // RHS is zero for now
            tmp *= -1.0;

            if (k == 0)
            {
                fvupscale.SolveFine(tmp, work_u);
            }
            else
            {
                fvupscale.SolveCoarse(tmp, work_u);
            }

            if (myid == 0)
            {
                std::cout << std::fixed << std::setw(8) << count << "\t" << time << "\n";
            }

            time += delta_t;
            count++;

            if (vis_step > 0 && count % vis_step == 0)
            {
                if (k == 0)
                {
                    fine_u.GetBlock(1) = work_u.GetBlock(1);
                }
                else
                {
                    fvupscale.Interpolate(work_u.GetBlock(1), fine_u.GetBlock(1));
                }

                field = fine_u.GetBlock(1);
                VisUpdate(comm, vis_v, field, *pmesh);
            }
        }

        chrono.Stop();

        fvupscale.ShowCoarseSolveInfo();

        if (myid == 0)
        {
            std::cout << "Total Time: " << chrono.RealTime() << "\n";
        }
    }

    return 0;
}

void InitialCondition(mfem::ParFiniteElementSpace& ufespace, mfem::BlockVector& fine_u,
                      double initial_val)
{
    HalfCoeffecient half(initial_val);

    mfem::GridFunction init(&ufespace);
    init.ProjectCoefficient(half);

    fine_u.GetBlock(1) = init;
}

void VisSetup(MPI_Comm comm, mfem::socketstream& vis_v, mfem::ParGridFunction& field,
              mfem::ParMesh& pmesh,
              double range, const std::string& caption)
{
    const char vishost[] = "localhost";
    const int  visport   = 19916;
    vis_v.open(vishost, visport);
    vis_v.precision(8);

    vis_v << "parallel " << pmesh.GetNRanks() << " " << pmesh.GetMyRank() << "\n";
    vis_v << "solution\n" << pmesh << field;
    vis_v << "window_size 500 800\n";
    vis_v << "window_title 'pressure'\n";
    vis_v << "autoscale off\n"; // update value-range; keep mesh-extents fixed
    vis_v << "valuerange " << -std::fabs(range) << " " << std::fabs(range) <<
          "\n"; // update value-range; keep mesh-extents fixed

    if (pmesh.SpaceDimension() == 2)
    {
        vis_v << "view 0 0\n"; // view from top
        vis_v << "keys jl\n";  // turn off perspective and light
        vis_v << "keys ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]\n";  // increase size
    }
    else
    {
        vis_v << "keys ]]]]]]]]]]]]]\n";  // increase size
        //vis_v << "keys IYYYYY\n";  // cut and rotate
    }

    vis_v << "keys c\n";         // show colorbar and mesh
    //vis_v << "pause\n"; // Press space to play!

    if (!caption.empty())
    {
        vis_v << "plot_caption '" << caption << "'\n";
    }

    MPI_Barrier(comm);

    vis_v << "keys S\n";         //Screenshot

    MPI_Barrier(comm);
}

void VisUpdate(MPI_Comm comm, mfem::socketstream& vis_v, mfem::ParGridFunction& field,
               mfem::ParMesh& pmesh)
{
    vis_v << "parallel " << pmesh.GetNRanks() << " " << pmesh.GetMyRank() << "\n";
    vis_v << "solution\n" << pmesh << field;

    MPI_Barrier(comm);

    vis_v << "keys S\n";         //Screenshot

    MPI_Barrier(comm);
}
