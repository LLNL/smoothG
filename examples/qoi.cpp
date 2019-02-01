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
          --max-levels 1 --fine-samples 200 \
          --coarse-samples 0 --shared-samples 0 --choose-samples 0

   Multi (two)-level

      ./qoi --coarse-factor 16 \
          --max-levels 2 --fine-samples 0 \
          --coarse-samples 10 --shared-samples 10 --choose-samples 180

   Each gives essentially the same estimate for the quantity of interest, but
   the two-level estimator is much more efficient because most samples are
   taken on the coarse grid. Also note in the two-level run that the
   variance for the correction on the fine grid is much smaller than the
   variance on the coarse grid, which is what justifies taking few samples
   on the fine grid.

   See Osborn, Vassilevski, and Villa, A multilevel, hierarchical sampling
   technique for spatially correlated random fields, SISC 39 (2017) pp. S543-S562
   and its references for more information on these techniques.

   Output from this example can be visualized (in 2D) with examples/qoivis.py
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
    int problem_size = 50;
    args.AddOption(&problem_size, "--problem-size", "--problem-size",
                   "Scale of problem (number of volumes in x,y directions)");

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
        mfem::Mesh mesh(problem_size, problem_size, 10, mfem::Element::HEXAHEDRON, 1,
                        problem_size * 100.0, problem_size * 100.0, 100.0);
        pmesh = make_unique<mfem::ParMesh>(comm, mesh);
    }
    else
    {
        mfem::Mesh mesh(problem_size, problem_size, mfem::Element::QUADRILATERAL, 1,
                        problem_size * 100.0, problem_size * 100.0);
        pmesh = make_unique<mfem::ParMesh>(comm, mesh);
    }

    // Construct a graph from a finite volume problem defined on the mesh
    mfem::Array<int> ess_attr(dimension == 3 ? 6 : 4);
    ess_attr = 0;
    ess_attr[0] = 1;
    DarcyProblem fvproblem(*pmesh, ess_attr);
    double cell_volume = fvproblem.CellVolume();
    Graph graph = fvproblem.GetFVGraph();
    Upscale upscale(graph, upscale_param, nullptr, &ess_attr);

    upscale.PrintInfo();
    upscale.ShowSetupTime();
    upscale.MakeSolver(0);

    mfem::BlockVector rhs_fine(upscale.GetBlockVector(0));
    rhs_fine.GetBlock(0) = 0.0;
    rhs_fine.GetBlock(1) = 0.0;
    // it may make sense to have a more general, not hard-coded, right-hand side
    int index = 0.5 * problem_size * problem_size + 0.1 * problem_size;
    rhs_fine.GetBlock(1)(index) = 1.0;
    index = 0.6 * problem_size * problem_size + 0.75 * problem_size;
    rhs_fine.GetBlock(1)(index) = -1.0;

    const int seed = argseed + myid;
    PDESampler sampler(dimension, cell_volume, kappa, seed,
                       graph, upscale_param, nullptr, &ess_attr);

    mfem::Vector functional(rhs_fine.GetBlock(1));
    functional = 0.0;
    double dscale = 1.0 / problem_size;
    for (int i = 0; i < problem_size; ++i)
    {
        functional(i) = dscale;
    }
    mfem::Vector dummy;
    PressureFunctionalQoI qoi(upscale, functional);

    MLMCManager mlmc(sampler, qoi, upscale, rhs_fine, dump_number);
    const bool verbose = (myid == 0);

    const int num_levels = upscale_param.max_levels;
    if (num_levels == 1)
    {
        mlmc.SetInitialSamplesLevel(0, fine_samples);
    }
    else
    {
        mlmc.SetInitialSamples(shared_samples);
        mlmc.SetInitialSamplesLevel(num_levels - 1, coarse_samples);
    }
    mlmc.SetNumChooseSamples(choose_samples);
    mlmc.Simulate(verbose);

    if (myid == 0)
    {
        mlmc.DisplayStatus(serialize);
    }

    return EXIT_SUCCESS;
}
