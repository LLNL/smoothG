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

#include "mfem.hpp"
#include "spe10.hpp"

#include "../src/picojson.h"
#include "../src/smoothG.hpp"

using namespace smoothg;

void SaveFigure(const mfem::Vector& sol,
                mfem::ParFiniteElementSpace& fespace,
                const std::string& name)
{
    mfem::ParGridFunction field(&fespace);
    mfem::ParMesh* pmesh = fespace.GetParMesh();
    field = sol;
    {
        std::stringstream filename;
        filename << name << ".mesh";
        std::ofstream out(filename.str().c_str());
        pmesh->Print(out);
    }
    {
        std::stringstream filename;
        filename << name << ".gridfunction";
        std::ofstream out(filename.str().c_str());
        field.Save(out);
    }
}

void Visualize(const mfem::Vector& sol,
               mfem::ParFiniteElementSpace& fespace,
               int tag)
{
    char vishost[] = "localhost";
    int  visport   = 19916;

    mfem::socketstream vis_v;
    vis_v.open(vishost, visport);
    vis_v.precision(8);

    mfem::ParGridFunction field(&fespace);
    mfem::ParMesh* pmesh = fespace.GetParMesh();
    field = sol;

    vis_v << "parallel " << pmesh->GetNRanks() << " " << pmesh->GetMyRank() << "\n";
    vis_v << "solution\n" << *pmesh << field;
    vis_v << "window_size 500 800\n";
    vis_v << "window_title 'pressure" << tag << "'\n";
    vis_v << "autoscale values\n";

    if (pmesh->Dimension() == 2)
    {
        vis_v << "view 0 0\n"; // view from top
        vis_v << "keys ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]\n";  // increase size
    }

    vis_v << "keys cjl\n";

    MPI_Barrier(pmesh->GetComm());
};

mfem::Vector InterpolateToFine(const Upscale& upscale, int level, const mfem::Vector& in)
{
    mfem::Vector vec1, vec2;
    vec1 = in;
    for (int k = level; k > 0; k--)
    {
        vec2 = upscale.Interpolate(k, vec1);
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

    mfem::Array<int> coarseningFactor(nDimensions);
    coarseningFactor[0] = coarsening_factor * 2;
    coarseningFactor[1] = coarsening_factor * 2;
    if (nDimensions == 3)
        coarseningFactor[2] = coarsening_factor;

    mfem::Vector weight;

    // Setting up finite volume discretization problem
    SPE10Problem spe10problem("", nDimensions, spe10_scale, 0,
                              metis_agglomeration, 2.0, coarseningFactor);

    mfem::ParMesh* pmesh = spe10problem.GetParMesh();

    if (myid == 0)
    {
        std::cout << pmesh->GetNEdges() << " fine edges, " <<
                  pmesh->GetNFaces() << " fine faces, " <<
                  pmesh->GetNE() << " fine elements\n";
    }

    mfem::Array<int> ess_attr;
    int nbdr;
    if (nDimensions == 3)
        nbdr = 6;
    else
        nbdr = 4;
    ess_attr.SetSize(nbdr);
    ess_attr = 0;

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
        // weight[i] = 1.0 / weight[i];
        weight[i] = 1.0;
    }

    mfem::L2_FECollection ufec(0, nDimensions);
    mfem::ParFiniteElementSpace ufespace(pmesh, &ufec);

    // Construct vertex_edge table in mfem::SparseMatrix format
    auto& vertex_edge_table = nDimensions == 2 ? pmesh->ElementToEdgeTable()
                              : pmesh->ElementToFaceTable();
    mfem::SparseMatrix vertex_edge = TableToMatrix(vertex_edge_table);

    // Construct agglomerated topology based on METIS or Cartesian agglomeration
    mfem::Array<int> partitioning;
    if (metis_agglomeration)
    {
        FESpaceMetisPartition(partitioning, sigmafespace, ufespace, coarseningFactor);
    }
    else
    {
        auto num_procs_xyz = spe10problem.GetNumProcsXYZ();
        FVMeshCartesianPartition(partitioning, num_procs_xyz, *pmesh, coarseningFactor);
    }

    const auto& edge_d_td(sigmafespace.Dof_TrueDof_Matrix());

    auto edge_boundary_att = GenerateBoundaryAttributeTable(pmesh);

    mfem::SparseMatrix W_block = SparseIdentity(vertex_edge.Height());

    const double cell_volume = spe10problem.CellVolume(nDimensions);
    W_block *= cell_volume * kappa * kappa;

    const int num_levels = upscale_param.max_levels;
    std::vector<mfem::Vector> mean(num_levels);
    std::vector<mfem::Vector> m2(num_levels);
    std::vector<int> total_iterations(num_levels);
    std::vector<double> total_time(num_levels);
    std::vector<double> p_error(num_levels);
    for (int level = 0; level < num_levels; ++level)
    {
        mean[level].SetSize(ufespace.GetVSize());
        mean[level] = 0.0;
        m2[level].SetSize(ufespace.GetVSize());
        m2[level] = 0.0;
        total_iterations[level] = 0;
        total_time[level] = 0.0;
    }

    // Create Upscaler
    upscale_param.coarse_factor = 4;
    Graph graph(vertex_edge, *edge_d_td, weight);
    auto upscale = std::make_shared<Upscale>(
                       graph, W_block, partitioning, &edge_boundary_att, &ess_attr, upscale_param);

    upscale->MakeFineSolver();
    upscale->PrintInfo();
    upscale->ShowSetupTime();

    PDESampler pdesampler(upscale, nDimensions, cell_volume, kappa, seed + myid);
    // PDESampler pdesampler(upscale, ufespace.GetVSize(), num_aggs, nDimensions,
    //                   cell_volume, kappa, seed + myid);

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
        int iterations = upscale->GetSolveIters(0);
        total_iterations[0] += iterations;
        double time = upscale->GetSolveTime(0);
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
            auto sol_upscaled = InterpolateToFine(*upscale, level, sol_coarse);
            for (int i = 0; i < sol_upscaled.Size(); ++i)
                sol_upscaled(i) = std::log(sol_upscaled(i));
            upscale->Orthogonalize(0, sol_upscaled);
            iterations = upscale->GetSolveIters(level);
            total_iterations[level] += iterations;
            time = upscale->GetSolveTime(level);
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

                if (save_samples)
                {
                    std::stringstream coarsename;
                    coarsename << "coarse_" << sample;
                    SaveFigure(sol_upscaled, ufespace, coarsename.str());
                    std::stringstream finename;
                    finename << "fine_" << sample;
                    SaveFigure(sol_fine, ufespace, finename.str());
                }

                {
                    //std::cout << "    fine: iterations: " << fine_iterations
                    // << ", time: " << fine_time << std::endl;
                    // std::cout << "    coarse: iterations: " << coarse_iterations
                    // << ", time: " << coarse_time << std::endl;
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
            Visualize(mean[level], ufespace, level);
            if (count > 1.1)
            {
                Visualize(m2[level], ufespace, 10 + level);
            }
        }
    }
    if (save_statistics)
    {
        SaveFigure(mean[1], ufespace, "coarse_mean");
        SaveFigure(mean[0], ufespace, "fine_mean");
        SaveFigure(m2[1], ufespace, "coarse_variance");
        SaveFigure(m2[0], ufespace, "fine_variance");
    }

    if (myid == 0)
        std::cout << picojson::value(serialize).serialize(true) << std::endl;

    return EXIT_SUCCESS;
}
