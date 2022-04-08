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
   @file sampler.cpp

   @brief Try to do scalable hierarchical sampling with finite volumes

   See Osborn, Vassilevski, and Villa, A multilevel, hierarchical sampling technique for
   spatially correlated random fields, SISC 39 (2017) pp. S543-S562.

   A simple way to run the example:

   mpirun -n 4 ./sampler

   I like the following runs:

   examples/sampler --visualization --kappa 0.001 --cartesian-factor 3
   examples/sampler --visualization --kappa 0.001 --cartesian-factor 2
*/

// previously used command line here for multilevel (vary kappa to see nice pictures)
// ./sampler --visualization --kappa 0.001 --max-levels 3

#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <mpi.h>

#include "pde.hpp"

#include "../src/picojson.h"
#include "../src/smoothG.hpp"

using namespace smoothg;

mfem::Vector InterpolateToFine(const PDESampler& pdesampler, int level, const mfem::Vector& in)
{
    mfem::Vector vec1, vec2;
    vec1 = in;
    for (int k = level; k > 0; k--)
    {
        vec2 = pdesampler.GetHierarchy().Interpolate(k, vec1);
        vec2.Swap(vec1);
    }
    return vec1;
}

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
    double kappa = 0.001;
    args.AddOption(&kappa, "--kappa", "--kappa",
                   "Correlation length for Gaussian samples.");
    int coarsening_factor = 2;
    args.AddOption(&coarsening_factor, "--cartesian-factor", "--cartesian-factor",
                   "Coarsening factor for Cartesian agglomeration");
    int seed = 0;
    args.AddOption(&seed, "--seed", "--seed",
                   "Seed for random number generator.");
    int num_samples = 1;
    args.AddOption(&num_samples, "--num-samples", "--num-samples",
                   "Number of samples to take.");
    bool save_samples = false;
    args.AddOption(&save_samples, "--save-samples", "--save-samples",
                   "--no-save-samples", "--no-save-samples",
                   "Save images of each sample.");
    bool save_statistics = true;
    args.AddOption(&save_statistics, "--save-statistics", "--save-statistics",
                   "--no-save-statistics", "--no-save-statistics",
                   "Save images of mean and variance.");
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
    coarsening_factors = coarsening_factor * 2;
    coarsening_factors.Last() = nDimensions == 3 ? coarsening_factor : coarsening_factor * 2;

    mfem::Array<int> ess_attr(nDimensions == 3 ? 6 : 4);
    ess_attr = 0;

    // Setting up finite volume discretization problem
    bool unit_edge_weight = true;
    SPE10Problem spe10problem("", nDimensions, spe10_scale, slice,
                              metis_agglomeration, ess_attr, unit_edge_weight);
    Graph graph = spe10problem.GetFVGraph();

    // Construct agglomerated topology based on METIS or Cartesian agglomeration
    mfem::Array<int> partitioning;
    spe10problem.Partition(metis_agglomeration, coarsening_factors, partitioning);

    mfem::SparseMatrix W_block = SparseIdentity(graph.NumVertices());
    const double cell_volume = spe10problem.CellVolume();
    W_block *= cell_volume * kappa * kappa;

    const int num_levels = upscale_param.max_levels;
    std::vector<mfem::Vector> mean(num_levels);
    std::vector<mfem::Vector> m2(num_levels);
    std::vector<int> total_iterations(num_levels);
    std::vector<double> total_time(num_levels);
    std::vector<double> p_error(num_levels);
    for (int level = 0; level < num_levels; ++level)
    {
        mean[level].SetSize(graph.NumVertices());
        mean[level] = 0.0;
        m2[level].SetSize(graph.NumVertices());
        m2[level] = 0.0;
        total_iterations[level] = 0;
        total_time[level] = 0.0;
    }

    // Create Hierarchy
    upscale_param.coarse_factor = 4;
    Hierarchy hierarchy(graph, upscale_param, &partitioning, &ess_attr, W_block);
    hierarchy.PrintInfo();

    PDESampler pdesampler(nDimensions, kappa, seed + myid, std::move(hierarchy));

    double max_p_error = 0.0;
    for (int sample = 0; sample < num_samples; ++sample)
    {
        if (myid == 0)
        {
            std::cout << "  Sample " << sample << ":" << std::endl;
        }
        double count = static_cast<double>(sample) + 1.0;
        pdesampler.NewSample();

        auto sol_fine = pdesampler.GetCoefficient(0);
        for (int i = 0; i < sol_fine.Size(); ++i)
            sol_fine(i) = std::log(sol_fine(i));
        par_orthogonalize_from_constant(sol_fine, graph.VertexStarts().Last());
        int iterations = pdesampler.GetHierarchy().GetSolveIters(0);
        total_iterations[0] += iterations;
        double time = pdesampler.GetHierarchy().GetSolveTime(0);
        total_time[0] += time;
        for (int i = 0; i < mean[0].Size(); ++i)
        {
            const double delta = (sol_fine(i) - mean[0](i));
            mean[0](i) += delta / count;
            const double delta2 = (sol_fine(i) - mean[0](i));
            m2[0](i) += delta * delta2;
        }
        p_error[0] = 0.0;

        for (int level = 1; level < num_levels; ++level)
        {
            auto sol_coarse = pdesampler.GetCoefficientForVisualization(level);
            auto sol_upscaled = InterpolateToFine(pdesampler, level, sol_coarse);
            for (int i = 0; i < sol_upscaled.Size(); ++i)
                sol_upscaled(i) = std::log(sol_upscaled(i));
            par_orthogonalize_from_constant(sol_upscaled, graph.VertexStarts().Last());
            iterations = pdesampler.GetHierarchy().GetSolveIters(level);
            total_iterations[level] += iterations;
            time = pdesampler.GetHierarchy().GetSolveTime(level);
            total_time[level] += time;
            for (int i = 0; i < mean[level].Size(); ++i)
            {
                const double delta = (sol_upscaled(i) - mean[level](i));
                mean[level](i) += delta / count;
                const double delta2 = (sol_upscaled(i) - mean[level](i));
                m2[level](i) += delta * delta2;
            }
            p_error[level] = CompareError(comm, sol_upscaled, sol_fine);
            if (myid == 0)
            {
                std::cout << "    p_error_level_" << level << ": " << p_error[level] << std::endl;
            }

            if (level == 1)
            {
                max_p_error = (max_p_error > p_error[level]) ? max_p_error : p_error[level];
            }

            if (save_samples)
            {
                std::stringstream name;
                name << "sample_l" << level << "_s" << sample;
                if (level == 0)
                {
                    spe10problem.SaveFigure(sol_fine, name.str());
                }
                else
                {
                    spe10problem.SaveFigure(sol_upscaled, name.str());
                }
            }
        }
    }

    double count = static_cast<double>(num_samples);
    if (count > 1.1)
    {
        for (int level = 0; level < num_levels; ++level)
            m2[level] *= (1.0 / (count - 1.0));
    }

    serialize["total-coarse-iterations"] = picojson::value((double) total_iterations[1]);
    serialize["total-fine-iterations"] = picojson::value((double) total_iterations[0]);
    serialize["fine-mean-typical"] = picojson::value(
                                         mean[0][mean[0].Size() / 2]);
    serialize["fine-mean-l1"] = picojson::value(
                                    mean[0].Norml1() / static_cast<double>(mean[0].Size()));
    serialize["coarse-mean-l1"] = picojson::value(
                                      mean[1].Norml1() / static_cast<double>(mean[1].Size()));
    serialize["coarse-mean-typical"] = picojson::value(
                                           mean[1][mean[1].Size() / 2]);
    serialize["fine-variance-mean"] = picojson::value(
                                          m2[0].Sum() / static_cast<double>(m2[0].Size()));
    serialize["coarse-variance-mean"] = picojson::value(
                                            m2[1].Sum() / static_cast<double>(m2[1].Size()));
    serialize["max-p-error"] = picojson::value(max_p_error);
    for (int i = 0; i < num_levels; ++i)
    {
        std::stringstream s;
        s << "p-error-level-" << i;
        serialize[s.str()] = picojson::value(p_error[i]);
    }

    if (visualization)
    {
        for (int level = 0; level < num_levels; ++level)
        {
            mfem::socketstream vis_v;
            std::stringstream filename;
            filename << "pressure " << level;
            spe10problem.VisSetup(vis_v, mean[level], 0.0, 0.0, filename.str());
            if (count > 1.1)
            {
                mfem::socketstream vis_v;
                std::stringstream filename;
                filename << "pressure " << 10 + level;
                spe10problem.VisSetup(vis_v, m2[level], 0.0, 0.0, filename.str());
            }
        }
    }
    if (save_statistics)
    {
        for (int level = 0; level < num_levels; ++level)
        {
            std::stringstream filename;
            filename << "level_" << level << "_mean";
            spe10problem.SaveFigure(mean[level], filename.str());
            filename.str("");
            filename << "level_" << level << "_variance";
            spe10problem.SaveFigure(m2[level], filename.str());
        }
    }

    if (myid == 0)
        std::cout << picojson::value(serialize).serialize(true) << std::endl;

    return EXIT_SUCCESS;
}
