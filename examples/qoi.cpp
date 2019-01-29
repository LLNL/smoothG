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
   @file qoi.cpp

   An example of multilevel Monte Carlo on a simple finite volume
   problem.

   The default configuration has a source on the left side and a sink on
   the right side of a square 2D domain, and the quantity of interest is
   the average pressure on the bottom boundary. Coefficient samples are
   drawn using a PDE sampler from a lognormal distribution.

   To see the effect of multilevel Monte Carlo, compare the following runs:

   One-level:

      ./qoi --coarse-factor 16 \
          --max-levels 1 --fine-samples 100 \
          --coarse-samples 0 --shared-samples 0 --choose-samples 0

   Multi (two)-level

      ./qoi --coarse-factor 16 \
          --max-levels 2 --fine-samples 0 \
          --coarse-samples 5 --shared-samples 5 --choose-samples 90

   Each gives essentially the same estimate for the quantity of interest, but
   the two-level estimator is much more efficient because most samples are
   taken on the coarse grid. Also note in the two-level run that the
   variance for the correction on the fine grid is much smaller than the
   variance on the coarse grid, which is what justifies taking few samples
   on the fine grid.

   See Osborn, Vassilevski, and Villa, A multilevel, hierarchical sampling
   technique for spatially correlated random fields, SISC 39 (2017) pp. S543-S562
   and its references for more information on these techniques.
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "../src/picojson.h"
#include "../src/smoothG.hpp"
#include "pde.hpp"

using namespace smoothg;

int main(int argc, char* argv[])
{
    int num_procs, myid;
    picojson::object serialize;

    mpi_session session(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // program options from command line
    UpscaleParameters upscale_param;
    mfem::OptionsParser args(argc, argv);
    double kappa = 0.001;
    args.AddOption(&kappa, "--kappa", "--kappa",
                   "(Inverse) correlation length for Gaussian samples.");
    int dump_number = 3;
    args.AddOption(&dump_number, "--dump-number", "--dump-number",
                   "Number of samples for which to dump solutions for eg. visualization.");
    int shared_samples = 3;
    args.AddOption(&shared_samples, "--shared-samples", "--shared-samples",
                   "(minimum) Number of samples to draw and simulate on all levels, "
                   "estimating difference.");
    int fine_samples = 0;
    args.AddOption(&fine_samples, "--fine-samples", "--fine-samples",
                   "Number of samples to draw and simulate on the fine level alone,"
                   " normally only for debugging/comparison.");
    int coarse_samples = 3;
    args.AddOption(&coarse_samples, "--coarse-samples", "--coarse-samples",
                   "(minimum) Number of samples to draw and simulate on the coarse level alone.");
    int choose_samples = 10;
    args.AddOption(&choose_samples, "--choose-samples", "--choose-samples",
                   "Number of samples where MLMC will choose to optimize.");
    int argseed = 1;
    args.AddOption(&argseed, "--seed", "--seed", "Seed for random number generator.");
    int dimension = 2;
    args.AddOption(&dimension, "--dimension", "--dimension",
                   "Spatial dimension of simulation.");
    double cell_volume = 10000.0;
    args.AddOption(&cell_volume, "--volume", "--volume",
                   "Volume of typical cell (for PDE sampler).");
    int scale = 50;
    args.AddOption(&scale, "--scale", "--scale",
                   "Scale of problems (number of volumes in x,y directions)");

    // Read upscaling options from command line into upscale_param object
    upscale_param.RegisterInOptionsParser(args);
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(std::cout);
        }
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(std::cout);
    }
    MFEM_ASSERT(fine_samples == 0 ||
                (coarse_samples == 0 && choose_samples == 0 && shared_samples == 0),
                "Either do standard (fine) MC, or MLMC, no combination!");
    serialize["fine-samples"] = picojson::value((double) fine_samples);
    serialize["coarse-samples"] = picojson::value((double) coarse_samples);
    serialize["choose-samples"] = picojson::value((double) choose_samples);
    serialize["shared-samples"] = picojson::value((double) shared_samples);
    serialize["kappa"] = picojson::value(kappa);
    // serialize["coarsening-factor"] = picojson::value((double) coarsening_factor);

    // Setting up a mesh
    std::unique_ptr<mfem::ParMesh> pmesh;
    if (dimension == 3)
    {
        mfem::Mesh mesh(scale, scale, 10, mfem::Element::HEXAHEDRON, 1,
                        scale * 100.0, scale * 100.0, 100.0);
        pmesh = make_unique<mfem::ParMesh>(comm, mesh);
    }
    else
    {
        mfem::Mesh mesh(scale, scale, mfem::Element::QUADRILATERAL, 1,
                        scale * 100.0, scale * 100.0);
        pmesh = make_unique<mfem::ParMesh>(comm, mesh);
    }

    // Construct a graph from a finite volume problem defined on the mesh
    mfem::Array<int> ess_attr(dimension == 3 ? 6 : 4);
    DarcyProblem fvproblem(*pmesh, ess_attr);
    Graph graph = fvproblem.GetFVGraph();
    Upscale upscale(graph, upscale_param);

    upscale.PrintInfo();
    upscale.ShowSetupTime();
    upscale.MakeSolver(0);

    mfem::BlockVector rhs_fine(upscale.GetBlockVector(0));
    rhs_fine.GetBlock(0) = 0.0;
    rhs_fine.GetBlock(1) = 0.0;
    // it may make sense to have more general, not hard-coded, right-hand side
    int index = 0.5 * scale * scale + 0.1 * scale;
    rhs_fine.GetBlock(1)(index) = 1.0;
    index = 0.6 * scale * scale + 0.75 * scale;
    rhs_fine.GetBlock(1)(index) = -1.0;

    std::unique_ptr<MultilevelSampler> sampler;
    const int seed = argseed + myid;
    sampler = make_unique<PDESampler>(
                  dimension, cell_volume, kappa, seed,
                  graph, upscale_param);

    mfem::Vector functional(rhs_fine.GetBlock(1));
    functional = 0.0;
    double dscale = 1.0 / scale;
    for (int i = 0; i < scale; ++i)
    {
        functional(i) = dscale;
    }
    mfem::Vector dummy;
    PressureFunctionalQoI qoi(upscale, functional);

    MLMCManager mlmc(*sampler, qoi, upscale, rhs_fine, dump_number);

    const bool verbose = (myid == 0);

    if (upscale_param.max_levels == 1)
    {
        for (int sample = 0; sample < fine_samples; ++sample)
        {
            if (myid == 0)
                std::cout << "---\nFine sample " << sample << "\n---" << std::endl;

            mlmc.FineSample(verbose);
        }
    }
    else if (upscale_param.max_levels == 2)
    {
        for (int sample = 0; sample < coarse_samples; ++sample)
        {
            if (myid == 0)
                std::cout << "---\nCoarsest " << upscale_param.max_levels - 1 << " sample "
                          << sample << "\n---" << std::endl;
            mlmc.CoarseSample(verbose);
        }
        for (int sample = 0; sample < shared_samples; ++sample)
        {
            if (myid == 0)
                std::cout << "---\nShared sample " << sample << "\n---" << std::endl;

            mlmc.CorrectionSample(0, verbose);
        }
    }
    else
    {
        // at least two samples on each level to ensure a meaningful variance (and cost)
        // three is better; sometimes you get stuck if there are very few samples on a level
        for (int i = 0; i < coarse_samples; ++i)
        {
            std::cout << "---\nCoarsest " << upscale_param.max_levels - 1 << " sample "
                      << i << "\n---" << std::endl;
            mlmc.CoarseSample(verbose);
        }
        for (int level = 0; level < upscale_param.max_levels - 1; ++level)
        {
            for (int i = 0; i < shared_samples; ++i)
            {
                std::cout << "---\nInitial level " << level << " correction " << i
                          << "\n---" << std::endl;
                mlmc.CorrectionSample(level, verbose);
            }
        }
    }

    for (int sample = 0; sample < choose_samples; ++sample)
    {
        if (myid == 0)
            std::cout << "---\nChoose sample " << sample << "\n---" << std::endl;

        mlmc.BestSample(verbose);
    }

    if (myid == 0)
    {
        mlmc.DisplayStatus(serialize);
    }

    return EXIT_SUCCESS;
}
