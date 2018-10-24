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

/** @file poweriter.cpp
    @brief Example use of upscale operator.  Performs inverse power iterations
    to find an approximate Fiedler vector.
*/

#include <mpi.h>

#include "mfem.hpp"
#include "../src/GraphUpscale.hpp"
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

    picojson::object serialize;

    // Setup Parameters
    UpscaleParameters param;
    param.coarse_factor = 80;
    param.max_evects = 4;
    param.spect_tol = 1.0;
    param.dual_target = false;
    param.scaled_dual = false;
    param.energy_dual = false;
    param.hybridization = false;

    // Solve Parameters
    constexpr auto max_iter = 800;
    constexpr auto solve_tol = 1e-12;
    constexpr auto verbose = false;
    constexpr auto seed = 1;

    // Global Input Information
    constexpr auto ve_filename = "../../graphdata/vertex_edge_sample.txt";
    constexpr auto rhs_filename = "../../graphdata/fiedler_sample.txt";
    const auto vertex_edge = ReadVertexEdge(ve_filename);

    // Power Iteration With Upscale Operators
    {
        // Upscaler
        const GraphUpscale upscale(comm, vertex_edge, param);

        // Wrapper for solving on the fine level, no upscaling
        const UpscaleFineSolve fine_solver(upscale);

        upscale.PrintInfo();

        // Read and normalize true Fiedler vector
        Vector true_sol = upscale.ReadVertexVector(rhs_filename);
        true_sol /= ParNormlp(true_sol, 2, comm);

        // Power Iteration for each Operator
        std::map<const Operator*, std::string> op_to_name;
        op_to_name[&upscale] = "coarse";
        op_to_name[&fine_solver] = "fine";

        for (const auto& op_pair : op_to_name)
        {
            auto& op = op_pair.first;
            auto& name = op_pair.second;

            // Power Iteration
            Vector result(op->Height());
            result.Randomize(seed);

            double eval = PowerIterate(comm, *op, result, max_iter, solve_tol, verbose);

            // Normalize
            result /= ParNormlp(result, 2, comm);
            upscale.Orthogonalize(result);

            // Match Signs
            double true_sign = true_sol[0] / std::fabs(true_sol[0]);
            double result_sign = result[0] / std::fabs(result[0]);
            if (std::fabs(true_sign - result_sign) > 1e-8)
            {
                result *= -1.0;
            }

            // Compute Error
            double error = CompareError(comm, result, true_sol);
            serialize[name + "-eval"] = picojson::value(eval);
            serialize[name + "-error"] = picojson::value(error);
        }
    }

    if (myid == 0)
    {
        std::cout << "\nResults:\n";
        std::cout << "---------------------\n";
        std::cout << picojson::value(serialize).serialize(true) << std::endl;
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
