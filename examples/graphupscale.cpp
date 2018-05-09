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

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "smoothG.hpp"

using namespace smoothg;

using linalgcpp::ReadText;
using linalgcpp::WriteText;
using linalgcpp::ReadCSR;

int main(int argc, char* argv[])
{
    // Initialize MPI
    MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm_;
    int myid = mpi_info.myid_;
    int num_procs = mpi_info.num_procs_;

    std::string ve_filename = "../../graphdata/vertex_edge_sample.txt";
    std::string rhs_filename = "../../graphdata/fiedler_sample.txt";

    int coarse_factor = 100;
    int num_partitions = 10;
    int max_evects = 4;
    double spect_tol = 1.e-3;
    bool dual_target = false;
    bool scaled_dual = false;
    bool energy_dual = false;
    bool hybridization = false;

    SparseMatrix vertex_edge = ReadCSR(ve_filename);

    // vertex_edge and partition
    {
        std::vector<int> part = PartitionAAT(vertex_edge, coarse_factor);
        Graph graph(comm, vertex_edge, part);

        GraphUpscale upscale(graph, spect_tol, max_evects, hybridization);

        Vector rhs_u_fine = upscale.ReadVertexVector(rhs_filename);
        Vector sol = upscale.Solve(rhs_u_fine);

        upscale.WriteVertexVector(sol, "sol1.out");
    }

    // Mimic distributed data
    {
        std::vector<int> part = PartitionAAT(vertex_edge, coarse_factor);
        Graph graph_global(comm, vertex_edge, part);

        // Pretend these came from some outside distributed source
        const auto& vertex_edge_local = graph_global.vertex_edge_local_;
        const auto& edge_true_edge = graph_global.edge_true_edge_;
        const auto& part_local = graph_global.part_local_;
        const auto& weight_local = graph_global.weight_local_;

        // Use distrubted constructor
        Graph graph_local(vertex_edge_local, edge_true_edge, part_local, weight_local);
        GraphUpscale upscale(graph_local, spect_tol, max_evects, hybridization);

        // This right hand side may not be permuted the same as in the upscaler,
        // since only local vertex information was given and the vertex map was generated
        Vector rhs_u_fine = upscale.ReadVertexVector(rhs_filename);
        Vector sol = upscale.Solve(rhs_u_fine);

        upscale.WriteVertexVector(sol, "sol2.out");
    }

    // Using coarse space
    {
        std::vector<int> part = PartitionAAT(vertex_edge, coarse_factor);
        Graph graph(comm, vertex_edge, part);

        GraphUpscale upscale(graph, spect_tol, max_evects, hybridization);

        // Start at Fine Level
        Vector rhs_u_fine = upscale.ReadVertexVector(rhs_filename);

        // Do work at Coarse Level
        Vector rhs_u_coarse = upscale.Restrict(rhs_u_fine);
        Vector sol_u_coarse = upscale.SolveCoarse(rhs_u_coarse);

        // If multiple iterations, reuse vector
        for (int i = 0; i < 5; ++i)
        {
            upscale.SolveCoarse(rhs_u_coarse, sol_u_coarse);
        }

        // Interpolate back to Fine Level
        Vector sol_u_fine = upscale.Interpolate(sol_u_coarse);
        upscale.Orthogonalize(sol_u_fine);

        upscale.WriteVertexVector(sol_u_fine, "sol3.out");
    }

    // Comparing Error; essentially generalgraph.cpp
    {
        std::vector<int> part = PartitionAAT(vertex_edge, coarse_factor);
        Graph graph(comm, vertex_edge, part);

        GraphUpscale upscale(graph, spect_tol, max_evects, hybridization);

        BlockVector fine_rhs = upscale.ReadVertexBlockVector(rhs_filename);

        BlockVector fine_sol = upscale.SolveFine(fine_rhs);
        BlockVector upscaled_sol = upscale.Solve(fine_rhs);

        upscale.PrintInfo();

        auto error_info = upscale.ComputeErrors(upscaled_sol, fine_sol);

        if (myid == 0)
        {
            std::cout << "Upscale:\n";
            std::cout << "---------------------\n";

            ShowErrors(error_info);
        }
    }

    // Compare hybridization vs Minres solvers
    {
        std::vector<int> part = PartitionAAT(vertex_edge, coarse_factor);
        Graph graph(comm, vertex_edge, part);

        bool use_hybridization = true;

        GraphUpscale hb_upscale(graph, spect_tol, max_evects, use_hybridization);
        GraphUpscale minres_upscale(graph, spect_tol, max_evects, !use_hybridization);

        Vector rhs_u_fine = minres_upscale.ReadVertexVector(rhs_filename);

        Vector minres_sol = minres_upscale.Solve(rhs_u_fine);
        Vector hb_sol = hb_upscale.Solve(rhs_u_fine);

        auto error = CompareError(comm, minres_sol, hb_sol);

        if (myid == 0)
        {
            std::cout.precision(3);
            std::cout << "---------------------\n";
            std::cout << "HB vs Minres Error: " <<  error << "\n";
            std::cout.precision(3);
        }
    }

    return EXIT_SUCCESS;
}
