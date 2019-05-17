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
   @file spe10_mlmc.cpp
   @brief This is an example of multilevel Monte Carlo for a graph
   Laplacian coming from a finite
   volume discretization of a simple reservior model in parallel.

   If lateral_pressure == true, Dirichlet pressure boundary condition will be
   imposed on left (p = -1) and right (p = 0) side of the domain boundary.
   No flow boundary condition (v.n = 0) is imposed on the rest of the boundary.
   In this case, the quantity of interest (QoI) is the total out flux
   \f$ \int v \cdot n dS \f$ on the left boundary.

   Coefficient samples are
   drawn using a PDE sampler from a lognormal distribution.

   To see the effect of multilevel Monte Carlo, compare the following runs:

   One-level:

      ./spe10_mlmc --coarse-factor 16 --spe10-scale 1 \
          --max-levels 1 --fine-samples 200 \
          --coarse-samples 0 --shared-samples 0 --choose-samples 0

   Multi (two)-level

      ./spe10_mlmc --coarse-factor 16 --spe10-scale 1 \
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

#include "mfem.hpp"
#include "pde.hpp"

#include "../src/picojson.h"
#include "../src/smoothG.hpp"

using namespace smoothg;

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
    UpscaleParameters upscale_param;
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
    bool metis_agglomeration = false;
    args.AddOption(&metis_agglomeration, "-ma", "--metis-agglomeration",
                   "-nm", "--no-metis-agglomeration",
                   "Use Metis as the partitioner (instead of geometric).");
    int spe10_scale = 5;
    args.AddOption(&spe10_scale, "-sc", "--spe10-scale",
                   "Scale of problem, 1=small, 5=full SPE10.");
    bool visualization = false;
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization", "Enable visualization.");
    bool lateral_pressure = true;
    args.AddOption(&lateral_pressure, "-lat-pres", "--lateral-pressure",
                   "-no-lat-pres", "--no-lateral-pressure",
                   "Impose Dirichlet pressure condition on lateral sides.");

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

    MFEM_ASSERT(fine_samples == 0 ||
                (coarse_samples == 0 && choose_samples == 0 && shared_samples == 0),
                "Either do standard (fine) MC, or MLMC, no combination!");
    serialize["fine-samples"] = picojson::value((double) fine_samples);
    serialize["coarse-samples"] = picojson::value((double) coarse_samples);
    serialize["choose-samples"] = picojson::value((double) choose_samples);
    serialize["shared-samples"] = picojson::value((double) shared_samples);
    serialize["kappa"] = picojson::value(kappa);

    mfem::Array<int> coarsening_factors(nDimensions);
    coarsening_factors = 10;
    coarsening_factors.Last() = nDimensions == 3 ? 5 : 10;

    mfem::Array<int> ess_attr(nDimensions == 3 ? 6 : 4);
    ess_attr = 1;
    if (lateral_pressure)
    {
        ess_attr[0] = ess_attr[2] = 0;
    }

    // Setting up finite volume discretization problem
    SPE10Problem spe10problem(permFile, nDimensions, spe10_scale, slice,
                              metis_agglomeration, ess_attr);
    double cell_volume = spe10problem.CellVolume();
    Graph graph = spe10problem.GetFVGraph();

    // Construct agglomerated topology based on METIS or Cartesian agglomeration
    mfem::Array<int> partitioning;
    spe10problem.Partition(metis_agglomeration, coarsening_factors, partitioning);

    Hierarchy hierarchy(graph, upscale_param, &partitioning, &ess_attr);
    hierarchy.PrintInfo();

    mfem::BlockVector rhs_fine(hierarchy.BlockOffsets(0));
    rhs_fine.GetBlock(0) = spe10problem.GetEdgeRHS();
    rhs_fine.GetBlock(1) = spe10problem.GetVertexRHS();

    const int seed = argseed + myid;
    PDESampler sampler(nDimensions, cell_volume, kappa, seed,
                       graph, upscale_param, &partitioning, &ess_attr);

    FunctionalQoI qoi_evaluator(hierarchy, rhs_fine);

    MLMCManager mlmc(sampler, qoi_evaluator, hierarchy, rhs_fine, dump_number);
    const bool verbose = false;

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

    if (lateral_pressure && myid == 0)
        std::cout << picojson::value(serialize).serialize(true) << std::endl;

    if (myid == 0)
    {
        mlmc.DisplayStatus(serialize);
    }

    return EXIT_SUCCESS;
}
