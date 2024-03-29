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

#include "pde.hpp"

#include "../src/picojson.h"
#include "../src/smoothG.hpp"

using std::unique_ptr;

using namespace smoothg;

int main(int argc, char* argv[])
{
    int num_procs, myid;

    // 1. Initialize MPI
    mpi_session session(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // program options from command line
    UpscaleParameters upscale_param;
    mfem::OptionsParser args(argc, argv);
    const char* perm_file = "spe_perm.dat";
    args.AddOption(&perm_file, "-p", "--perm",
                   "SPE10 permeability file data.");
    int nDimensions = 2;
    args.AddOption(&nDimensions, "-d", "--dim",
                   "Dimension of the physical space.");
    int slice = 0;
    args.AddOption(&slice, "-s", "--slice",
                   "Slice of SPE10 data to take for 2D run.");
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
    const char* caption = "";
    args.AddOption(&caption, "-cap", "--caption",
                   "Caption for visualization");
    // Read upscaling options from command line into upscale_param object
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

    assert(k == 0 || k == 1);

    mfem::Array<int> coarsening_factors(nDimensions);
    coarsening_factors = 10;
    coarsening_factors.Last() = nDimensions == 3 ? 5 : 10;

    mfem::Array<int> ess_attr(nDimensions == 3 ? 6 : 4);
    ess_attr = 1;

    // Setting up finite volume discretization problem
    SPE10Problem spe10problem(perm_file, nDimensions, spe10_scale, slice,
                              metis_agglomeration, ess_attr);
    Graph graph = spe10problem.GetFVGraph();

    // Construct agglomerated topology based on METIS or Cartesion aggloemration
    mfem::Array<int> partitioning;
    spe10problem.Partition(metis_agglomeration, coarsening_factors, partitioning);

    mfem::SparseMatrix W_block = SparseIdentity(graph.NumVertices());
    W_block *= spe10problem.CellVolume() / delta_t;     // Mass matrix / delta_t

    // Time Stepping
    {
        Hierarchy hierarchy(graph, upscale_param, &partitioning, &ess_attr, W_block);
        hierarchy.PrintInfo();

        // Set some pressure initial condition
        mfem::BlockVector fine_u(hierarchy.BlockOffsets(0));
        fine_u.GetBlock(0) = 0.0;
        fine_u.GetBlock(1) = spe10problem.InitialCondition(initial_val);

        // Create Workspace
        mfem::BlockVector work_rhs(hierarchy.BlockOffsets(k));
        work_rhs = 0.0;

        mfem::BlockVector work_u = k == 0 ? fine_u : hierarchy.Restrict(0, fine_u);

        assert(hierarchy.GetMatrix(k).CheckW());
        const mfem::SparseMatrix& W = hierarchy.GetMatrix(k).GetW();

        // Setup visualization
        mfem::socketstream vis_v;

        if (vis_step > 0)
        {
            spe10problem.VisSetup(vis_v, fine_u.GetBlock(1), -std::fabs(initial_val),
                                  std::fabs(initial_val), caption);
        }

        double time = 0.0;
        int count = 0;

        mfem::StopWatch chrono;
        chrono.Start();

        while (time < total_time)
        {
            W.Mult(work_u.GetBlock(1), work_rhs.GetBlock(1));

            //tmp += work_rhs; // RHS is zero for now
            work_rhs *= -1.0;

            hierarchy.Solve(k, work_rhs, work_u);

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
                    hierarchy.Interpolate(1, work_u.GetBlock(1), fine_u.GetBlock(1));
                }

                spe10problem.VisUpdate(vis_v, fine_u.GetBlock(1));
            }
        }

        chrono.Stop();

        if (myid == 0)
        {
            std::cout << "Total Time: " << chrono.RealTime() << "\n";
        }
    }

    return 0;
}
