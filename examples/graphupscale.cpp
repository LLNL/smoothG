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

/** @file graphupscale.cpp
    @brief Example usage of the Upscale operator.

    Provides sample use cases of the Upscale operator.
    This includes:
    * several simple upscale constructors,
    * performing work on the coarse level,
    * comparing upscaled solution to the fine level solution,
    * comparing solver types.
*/

#include <mpi.h>

#include "mfem.hpp"
#include "../src/picojson.h"
#include "../src/smoothG.hpp"

using namespace smoothg;
using namespace mfem;

int main(int argc, char* argv[])
{
    // Initialize MPI
    int myid;
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &myid);

    constexpr auto ve_filename = "../../graphdata/vertex_edge_sample.txt";
    constexpr auto rhs_filename = "../../graphdata/fiedler_sample.txt";

    UpscaleParameters param;
    param.coarsen_param.coarse_factor = 100;
    param.coarsen_param.max_evects = 4;
    param.coarsen_param.spect_tol = 1.e-3;
    param.coarsen_param.dual_target = false;
    param.coarsen_param.scaled_dual = false;
    param.coarsen_param.energy_dual = false;
    param.hybridization = false;

    // vertex_edge and partition
    const Graph graph(comm, ReadVertexEdge(ve_filename));
    {
        mfem::Array<int> partitioning(graph.NumVertices());
        PartitionAAT(graph.VertexToEdge(), partitioning, param.coarsen_param.coarse_factor);

        const auto upscale = Upscale(graph, param, &partitioning);

        const auto rhs_u_fine = graph.ReadVertexVector(rhs_filename);
        const auto sol = upscale.Solve(1, rhs_u_fine);

        graph.WriteVertexVector(sol, "sol1.out");
    }

    // vertex_edge and coarse factor
    {
        const auto upscale = Upscale(graph, param);

        const auto rhs_u_fine = graph.ReadVertexVector(rhs_filename);
        const auto sol = upscale.Solve(1, rhs_u_fine);

        graph.WriteVertexVector(sol, "sol2.out");
    }

    // Using coarse space
    {
        const auto hierarchy = Hierarchy(graph, param.coarsen_param);

        // Start at Fine Level
        const auto rhs_u_fine = graph.ReadVertexVector(rhs_filename);

        // Do work at Coarse Level
        auto rhs_u_coarse = hierarchy.Restrict(0, rhs_u_fine);
        auto sol_u_coarse = hierarchy.Solve(1, rhs_u_coarse);

        // If multiple iterations, reuse vector
        for (int i = 0; i < 5; ++i)
        {
            hierarchy.Solve(1, rhs_u_coarse, sol_u_coarse);
        }

        // Interpolate back to Fine Level
        auto sol_u_fine = hierarchy.Interpolate(1, sol_u_coarse);

        graph.WriteVertexVector(sol_u_fine, "sol3.out");
    }

    // Comparing Error; essentially generalgraph.cpp
    {
        const auto upscale = Upscale(graph, param);

        mfem::BlockVector fine_rhs(upscale.BlockOffsets(0));
        fine_rhs.GetBlock(0) = 0.0;
        fine_rhs.GetBlock(1) = graph.ReadVertexVector(rhs_filename);

        mfem::BlockVector fine_sol = upscale.Solve(0, fine_rhs);
        mfem::BlockVector upscaled_sol = upscale.Solve(1, fine_rhs);

        upscale.PrintInfo();

        if (myid == 0)
        {
            std::cout << "Upscale:\n";
            std::cout << "---------------------\n";
        }
        upscale.ShowErrors(upscaled_sol, fine_sol, 1);
    }

    // Compare Minres vs hybridization solvers
    {
        param.hybridization = false;
        const auto minres_upscale = Upscale(graph, param);

        param.hybridization = true;
        const auto hb_upscale = Upscale(graph, param);

        const auto rhs_u_fine = graph.ReadVertexVector(rhs_filename);

        const auto minres_sol = minres_upscale.Solve(1, rhs_u_fine);
        const auto hb_sol = hb_upscale.Solve(1, rhs_u_fine);

        const auto error = CompareError(comm, hb_sol, minres_sol);

        if (myid == 0)
        {
            std::cout.precision(3);
            std::cout << "---------------------\n";
            std::cout << "HB (BoomerAMG) vs Minres Error: " <<  error << "\n";
            std::cout.precision(3);
        }

#if SMOOTHG_USE_SAAMGE
        SAAMGeParam saamge_param;
        param.coarse_components = false;
        param.saamge_param = &saamge_param;
        const auto hbsa_upscale = Upscale(graph, param);

        const auto hbsa_sol = hbsa_upscale.Solve(1, rhs_u_fine);
        const auto error_sa_mr = CompareError(comm, hbsa_sol, minres_sol);
        const auto error_sa_ba = CompareError(comm, hbsa_sol, hb_sol);
        if (myid == 0)
        {
            std::cout.precision(3);
            std::cout << "HB (SAAMGe) vs Minres Error: " <<  error_sa_mr << "\n";
            std::cout << "HB (SAAMGe) vs HB (BoomerAMG): " <<  error_sa_ba << "\n";
            std::cout.precision(3);
        }
#endif
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
