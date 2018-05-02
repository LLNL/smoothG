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
   Test code for the Watts-Strogatz random graph generator
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "smoothG.hpp"

using namespace smoothg;

int main(int argc, char* argv[])
{
    // initialize MPI
    MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm_;
    int myid = mpi_info.myid_;
    int num_procs = mpi_info.num_procs_;

    // program options from command line
    linalgcpp::ArgParser arg_parser(argc, argv);

    int nvertices = 100;
    int mean_degree = 20;
    double beta = 0.15;
    unsigned int seed = 5;

    arg_parser.Parse(nvertices, "--nv", "Number of vertices of the graph to be generated.");
    arg_parser.Parse(mean_degree, "--md", "Average vertex degree of the graph to be generated.");
    arg_parser.Parse(beta, "--b", "Probability of rewiring in the Watts-Strogatz model.");

    if (!arg_parser.IsGood())
    {
        ParPrint(myid, arg_parser.ShowHelp());
        ParPrint(myid, arg_parser.ShowErrors());

        return EXIT_FAILURE;
    }

    ParPrint(myid, arg_parser.ShowOptions());

    Timer chrono(Timer::Start::True);

    SparseMatrix vertex_edge = GenerateGraph(comm, nvertices, mean_degree, beta, seed);

    chrono.Click();

    std::map<int, double> graph_stats;

    for (int i = 0; i < vertex_edge.Rows(); i++)
    {
        int tmp = vertex_edge.RowSize(i);
        if (graph_stats[tmp])
        {
            graph_stats[tmp] += 1;
        }
        else
        {
            graph_stats[tmp] = 1;
        }
    }

    if (myid == 0)
    {
        std::cout << "Vertex degree statistics:\n";
        std::cout << "  Degree   Percentage\n";

        for (auto&& stat : graph_stats)
        {
            std::cout << "    " << stat.first
                      << "       " << std::setprecision(4)
                      << stat.second / nvertices * 100 << "% \n";
        }
    }

    if (myid == 0)
    {
        std::cout << "A random graph is generated in "
                  << chrono.TotalTime() << " seconds \n";
    }

    int failures = 0;

    if (vertex_edge.Rows() != nvertices)
    {
        failures += 1;

        std::cout << "The generated graph does not have "
                  << "the expected number of vertices\n";
        std::cout << "Expect: " << nvertices << "\n";
        std::cout << "Actual: " << vertex_edge.Rows() << "\n";
    }

    if (vertex_edge.Cols() != nvertices * mean_degree / 2)
    {
        failures += 1;

        std::cout << "The generated graph does not have "
                  << "the expected number of edges\n";
        std::cout << "Expect: " << nvertices * mean_degree / 2 << "\n";
        std::cout << "Actual: " << vertex_edge.Cols() << "\n";
    }

    return failures;
}


