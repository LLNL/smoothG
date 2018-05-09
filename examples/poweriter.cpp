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

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "smoothG.hpp"

using namespace smoothg;

using linalgcpp::Operator;
using linalgcpp::ReadCSR;

using parlinalgcpp::ParL2Norm;

int main(int argc, char* argv[])
{
    // Initialize MPI
    MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm_;
    int myid = mpi_info.myid_;
    int num_procs = mpi_info.num_procs_;

    // Setup Parameters
    double coarse_factor = 80;
    int max_evects = 4;
    double spect_tol = 1.0;
    bool hybridization = false;

    // Solve Parameters
    int max_iter = 800;
    double solve_tol = 1e-8;
    bool verbose = false;
    int seed = 1;

    // Global Input Information
    std::string ve_filename = "../../graphdata/vertex_edge_sample.txt";
    std::string rhs_filename = "../../graphdata/fiedler_sample.txt";

    SparseMatrix vertex_edge = ReadCSR(ve_filename);

    // Power Iteration With Upscale Operators
    // Upscaler
    std::vector<int> part = PartitionAAT(vertex_edge, coarse_factor);
    Graph graph(comm, vertex_edge, part);

    GraphUpscale upscale(graph, spect_tol, max_evects, hybridization);

    // Wrapper for solving on the fine level, no upscaling
    UpscaleFineSolve fine_solver(upscale);

    upscale.PrintInfo();

    // Read and normalize true Fiedler vector
    Vector true_sol = upscale.ReadVertexVector(rhs_filename);
    true_sol /= ParL2Norm(comm, true_sol);

    // Power Iteration for each Operator
    std::map<const Operator*, std::string> op_to_name;
    op_to_name[&upscale] = "coarse";
    op_to_name[&fine_solver] = "fine";

    std::map<std::string, double> error_info;

    for (const auto& op_pair : op_to_name)
    {
        auto& op = op_pair.first;
        auto& name = op_pair.second;

        // Power Iteration
        Vector result(op->Rows());
        result.Randomize(seed);

        double eval = PowerIterate(comm, *op, result, max_iter, solve_tol, verbose);

        // Normalize
        result /= ParL2Norm(comm, result);
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
        error_info[name + "-eval"] = eval;
        error_info[name + "-error"] = error;
    }

    if (myid == 0)
    {
        std::cout << "\nResults:\n";
        std::cout << "---------------------\n";
        PrintJSON(error_info);
    }

    return EXIT_SUCCESS;
}
