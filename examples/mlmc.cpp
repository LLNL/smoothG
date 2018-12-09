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

void Visualize(const mfem::Vector& sol, mfem::ParGridFunction& field,
               const mfem::ParMesh& pmesh, const std::string& title)
{
    char vishost[] = "localhost";
    int  visport   = 19916;

    mfem::socketstream vis_v;
    vis_v.open(vishost, visport);
    vis_v.precision(8);

    field = sol;

    vis_v << "parallel " << pmesh.GetNRanks() << " " << pmesh.GetMyRank() << "\n";
    vis_v << "solution\n" << pmesh << field;
    vis_v << "window_size 500 800\n";
    vis_v << "window_title '" << title << "'\n";
    vis_v << "autoscale values\n";

    if (pmesh.Dimension() == 2)
    {
        vis_v << "view 0 0\n"; // view from top
        vis_v << "keys ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]\n";  // increase size
    }

    vis_v << "keys cjl\n";

    MPI_Barrier(pmesh.GetComm());
};

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
    std::vector<mfem::Vector> local_weight;
    mfem::Vector rhs_u_fine;

    // Setting up finite volume discretization problem
    const double proc_part_ubal = 2.0;
    SPE10Problem spe10problem(
        permFile, nDimensions, spe10_scale, slice, metis_agglomeration,
        proc_part_ubal, coarseningFactor);
    mfem::ParMesh* pmesh = spe10problem.GetParMesh();

    if (myid == 0)
    {
        std::cout << pmesh->GetNEdges() << " fine FE edges, " <<
                  pmesh->GetNFaces() << " fine FE faces, " <<
                  pmesh->GetNE() << " fine FE elements\n";
    }

    ess_attr.SetSize(nbdr);
    for (int i(0); i < nbdr; ++i)
        ess_attr[i] = ess_zeros[i];

    // Construct "finite volume mass" matrix using mfem
    mfem::RT_FECollection sigmafec(0, nDimensions);
    mfem::ParFiniteElementSpace sigmafespace(pmesh, &sigmafec);
    {
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
        if (elem_mass)
        {
            local_weight.resize(pmesh->GetNE());
            mfem::DenseMatrix M_el_i;
            for (int i = 0; i < pmesh->GetNE(); i++)
            {
                a.ComputeElementMatrix(i, M_el_i);
                mfem::Vector& local_weight_i = local_weight[i];
                local_weight_i.SetSize(M_el_i.Height());
                for (int j = 0; j < local_weight_i.Size(); j++)
                {
                    local_weight_i[j] = 1.0 / M_el_i(j, j);
                }
            }
        }
    }

    mfem::L2_FECollection ufec(0, nDimensions);
    mfem::ParFiniteElementSpace ufespace(pmesh, &ufec);

    mfem::LinearForm q(&ufespace);
    q.AddDomainIntegrator(
        new mfem::DomainLFIntegrator(*spe10problem.GetForceCoeff()) );
    q.Assemble();
    rhs_u_fine = q;

    // Construct vertex_edge table in mfem::SparseMatrix format
    auto& vertex_edge_table = nDimensions == 2 ? pmesh->ElementToEdgeTable()
                              : pmesh->ElementToFaceTable();
    mfem::SparseMatrix vertex_edge = TableToMatrix(vertex_edge_table);

    // Construct agglomerated topology based on METIS or Cartesion aggloemration
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

    unique_ptr<Graph> graph;
    if (elem_mass == false)
    {
        graph = make_unique<Graph>(vertex_edge, *edge_d_td, weight, &edge_boundary_att);
    }
    else
    {
        graph = make_unique<Graph>(vertex_edge, *edge_d_td, local_weight, &edge_boundary_att);
    }

    // Create Upscaler and Solve
    Upscale upscale(*graph, upscale_param, &partitioning, &ess_attr);

    upscale.PrintInfo();
    upscale.ShowSetupTime();

    mfem::BlockVector rhs_fine(upscale.GetBlockVector(0));
    rhs_fine.GetBlock(0) = 0.0;
    rhs_fine.GetBlock(1) = rhs_u_fine;

    const int num_levels = upscale_param.max_levels;
    unique_ptr<MultilevelSampler> sampler;
    if (std::string(sampler_type) == "simple")
    {
        std::vector<int> vertex_sizes = upscale.GetVertexSizes();
        sampler = make_unique<SimpleSampler>(vertex_sizes);
    }
    else if (std::string(sampler_type) == "pde")
    {
        const int seed = argseed + myid;
        sampler = make_unique<PDESampler>(
                      comm, nDimensions, spe10problem.CellVolume(nDimensions), kappa, seed,
                      vertex_edge, weight, partitioning, *edge_d_td, edge_boundary_att, ess_attr,
                      upscale_param);
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
                auto error_info = upscale.ComputeErrors(sol[level], sol[0]);
                if (myid == 0)
                    ShowErrors(error_info);
            }

            std::stringstream filename;
            filename << "pressure_s" << sample << "_l" << level;
            SaveFigure(sol[level].GetBlock(1), ufespace, filename.str());
            if (visualization)
            {
                mfem::ParGridFunction field(&ufespace);
                std::stringstream caption;
                caption << "pressure sample " << sample << " level " << level;
                Visualize(sol[level].GetBlock(1), field, *pmesh, caption.str());
            }
        }

        // for more informative visualization
        for (int i = 0; i < coefficient[0].Size(); ++i)
        {
            coefficient[0](i) = std::log(coefficient[0](i));
        }
        std::stringstream coeffname;
        coeffname << "coefficient" << sample;
        SaveFigure(coefficient[0], ufespace, coeffname.str());

        if (visualization)
        {
            mfem::ParGridFunction field(&ufespace);
            std::stringstream caption;
            caption << "coefficient" << sample;
            Visualize(coefficient[0], field, *pmesh, caption.str());
        }
    }

    return EXIT_SUCCESS;
}
