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
   @file coarse_assembling.cpp

   @brief Test if the coarse M and D constructed in GraphCoarsen coincide
   with the RAP approach.
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "../examples/pde.hpp"

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
    int spe10_scale = 1;
    args.AddOption(&spe10_scale, "-sc", "--spe10-scale",
                   "Scale of problem, 1=small, 5=full SPE10");
    int slice = 0;
    args.AddOption(&slice, "-s", "--slice",
                   "Slice of SPE10 data to take for 2D run.");

    upscale_param.max_levels = 3;
    upscale_param.coarse_factor = 10;
    upscale_param.hybridization = true;

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

    mfem::Array<int> ess_attr(nDimensions == 3 ? 6 : 4);
    ess_attr = 1;

    const bool metis_agglomeration = true;
    const double proc_part_ubal = 1.001;
    mfem::Array<int> junk(3);

    // Setting up finite volume discretization problem
    SPE10Problem spe10problem(permFile, nDimensions, spe10_scale, slice,
                              metis_agglomeration, proc_part_ubal, junk);
    mfem::ParMesh* pmesh = spe10problem.GetParMesh();

    // Construct "finite volume mass" matrix using mfem instead of parelag
    mfem::Vector weight;

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

    // Construct vertex_edge table in mfem::SparseMatrix format
    auto& vertex_edge_table = nDimensions == 2 ? pmesh->ElementToEdgeTable()
                              : pmesh->ElementToFaceTable();
    const mfem::SparseMatrix vertex_edge = TableToMatrix(vertex_edge_table);
    const auto& edge_d_td(sigmafespace.Dof_TrueDof_Matrix());
    auto edge_boundary_att = GenerateBoundaryAttributeTable(pmesh);

    Graph graph(vertex_edge, *edge_d_td, weight, &edge_boundary_att);

    // Create Upscaler
    Upscale upscale(graph, upscale_param, nullptr, &ess_attr);

    upscale.PrintInfo();
    upscale.ShowSetupTime();

    auto CoarseNotEqualRAP = [&](const char* name)
    {
        bool out = EXIT_SUCCESS;
        const bool is_M = !(std::strcmp(name, "M"));
        for (int level = 1; level < upscale_param.max_levels; ++level)
        {
            auto& fine_system = upscale.GetMatrix(level - 1);
            auto& coarse_system = upscale.GetMatrix(level);
            auto& fine_mat = is_M ? fine_system.GetM() : fine_system.GetD();

            auto& P = upscale.GetPsigma(level - 1);
            auto& R = is_M ? P : upscale.GetPu(level - 1);
            unique_ptr<mfem::SparseMatrix> coarse_mat_RAP(mfem::RAP(R, fine_mat, P));

            coarse_system.BuildM();
            auto& coarse_mat_smoothG = is_M ? coarse_system.GetM() : coarse_system.GetD();

            coarse_mat_RAP->Add(-1.0, coarse_mat_smoothG);
            double diff_maxnorm_loc = coarse_mat_RAP->MaxNorm();
            double diff_maxnorm;
            MPI_Allreduce(&diff_maxnorm_loc, &diff_maxnorm, 1, MPI_DOUBLE, MPI_MAX, comm);

            if (diff_maxnorm > 1e-8)
            {
                if (myid == 0)
                {
                    std::cout << "Level " << level << ": || " << name << "_smoothG - "
                              << name << "_RAP ||_inf = " << diff_maxnorm << "\n";
                }
                out = EXIT_FAILURE;
            }
        }
        return out;
    };

    return CoarseNotEqualRAP("M") + CoarseNotEqualRAP("D");
}
