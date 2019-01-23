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
   @file mlmc.cpp

   @brief This is an example for upscaling a graph Laplacian coming from a finite
   volume discretization of a simple reservior model, where we change coefficients
   in the model without re-coarsening.

   A simple way to run the example:

   ./mlmc --perm spe_perm.dat
*/

// best multilevel command line so far (appears to create a reasonable result):
// ./mlmc --sampler-type pde --num-samples 2 --max-levels 3 --hybridization --no-coarse-components --max-evects 1 --coarse-factor 8

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "pde.hpp"

#include "../src/picojson.h"
#include "../src/smoothG.hpp"

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
    const char* permFile = "";
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
    bool elem_mass = false;
    args.AddOption(&elem_mass, "-el-mass", "--element-mass", "-no-el-mass",
                   "--no-element-mass", "Store fine M in element matrices format.");
    const char* sampler_type = "simple";
    args.AddOption(&sampler_type, "--sampler-type", "--sampler-type",
                   "Which sampler to use for coefficient: simple, pde");
    double kappa = 0.001;
    args.AddOption(&kappa, "--kappa", "--kappa",
                   "Correlation length for Gaussian samples.");
    int num_samples = 3;
    args.AddOption(&num_samples, "--num-samples", "--num-samples",
                   "Number of samples to draw and simulate.");
    int argseed = 1;
    args.AddOption(&argseed, "--seed", "--seed", "Seed for random number generator.");
    upscale_param.coarse_components = true;
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

    mfem::Array<int> coarsening_factors(nDimensions);
    coarsening_factors = 10;
    coarsening_factors.Last() = nDimensions == 3 ? 5 : 10;

    mfem::Array<int> ess_attr(nDimensions == 3 ? 6 : 4);
    ess_attr = 1;

    // Setting up finite volume discretization problem
    SPE10Problem spe10problem(permFile, nDimensions, spe10_scale, slice,
                              metis_agglomeration, ess_attr);
    Graph graph = spe10problem.GetFVGraph(elem_mass);

    // Construct agglomerated topology based on METIS or Cartesion aggloemration
    mfem::Array<int> partitioning;
    spe10problem.Partition(metis_agglomeration, coarsening_factors, partitioning);

    // Create Upscaler and Solve
    Upscale upscale(graph, upscale_param, &partitioning, &ess_attr);

    upscale.PrintInfo();

    mfem::BlockVector rhs_fine(upscale.GetBlockVector(0));
    rhs_fine.GetBlock(0) = 0.0;
    rhs_fine.GetBlock(1) = spe10problem.GetVertexRHS();

    const int num_levels = upscale_param.max_levels;
    unique_ptr<MultilevelSampler> sampler;
    if (std::string(sampler_type) == "simple")
    {
        auto vertex_sizes = upscale.GetHierarchy().GetVertexSizes();
        sampler = make_unique<SimpleSampler>(vertex_sizes);
    }
    else if (std::string(sampler_type) == "pde")
    {
        const int seed = argseed + myid;
        sampler = make_unique<PDESampler>(
                      nDimensions, spe10problem.CellVolume(), kappa, seed,
                      graph, partitioning, ess_attr, upscale_param);
    }
    else
    {
        if (myid == 0)
            std::cerr << "Unrecognized sampler: " << sampler_type << "!" << std::endl;
        return 1;
    }

    for (int sample = 0; sample < num_samples; ++sample)
    {
        if (myid == 0)
            std::cout << "---\nSample " << sample << "\n---" << std::endl;

        sampler->NewSample();
        std::vector<mfem::Vector> coefficient(num_levels);
        std::vector<mfem::BlockVector> sol;

        for (int level = 0; level < num_levels; ++level)
        {
            coefficient[level] = sampler->GetCoefficient(level);
            upscale.RescaleCoefficient(level, coefficient[level]);
            sol.push_back(upscale.Solve(level, rhs_fine));
            upscale.ShowSolveInfo(level);

            if (level > 0)
            {
                upscale.ShowErrors(sol[level], sol[0], level);
            }

            std::stringstream filename;
            filename << "pressure_s" << sample << "_l" << level;
            spe10problem.SaveFigure(sol[level].GetBlock(1), filename.str());
            if (visualization)
            {
                mfem::socketstream vis_v;
                std::stringstream caption;
                caption << "pressure sample " << sample << " level " << level;
                spe10problem.VisSetup(vis_v, sol[level].GetBlock(1), 0.0, 0.0, caption.str());
            }
        }

        // for more informative visualization
        for (int i = 0; i < coefficient[0].Size(); ++i)
        {
            coefficient[0](i) = std::log(coefficient[0](i));
        }
        std::stringstream coeffname;
        coeffname << "coefficient" << sample;
        spe10problem.SaveFigure(coefficient[0], coeffname.str());

        if (visualization)
        {
            mfem::socketstream vis_v;
            std::stringstream caption;
            caption << "coefficient" << sample;
            spe10problem.VisSetup(vis_v, coefficient[0], 0.0, 0.0, caption.str());
        }
    }

    return EXIT_SUCCESS;
}
