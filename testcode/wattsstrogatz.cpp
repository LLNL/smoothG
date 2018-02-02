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

#include "mfem.hpp"
#include "../src/smoothG.hpp"

using namespace smoothg;

int main(int argc, char* argv[])
{
    // initialize MPI
    mpi_session session(argc, argv);

    int myid;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &myid);

    // program options from command line
    mfem::OptionsParser args(argc, argv);

    int nvertices = 100;
    args.AddOption(&nvertices, "-nv", "--num-vert",
                   "Number of vertices of the graph to be generated.");
    int mean_degree = 20;
    args.AddOption(&mean_degree, "-md", "--mean-degree",
                   "Average vertex degree of the graph to be generated.");
    double beta = 0.15;
    args.AddOption(&beta, "-b", "--beta",
                   "Probability of rewiring in the Watts-Strogatz model.");
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

    mfem::StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    GraphGenerator graph_gen(nvertices, mean_degree, beta);
    mfem::SparseMatrix vertex_edge = graph_gen.Generate();

    std::map<int, double> graph_stats;
    for (int i = 0; i < vertex_edge.Height(); i++)
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
        for (auto it = graph_stats.begin(); it != graph_stats.end(); it++)
            std::cout << "    " << it->first
                      << "       " << std::setprecision(4)
                      << it->second / nvertices * 100 << "% \n";
    }

    chrono.Stop();
    if (myid == 0)
        std::cout << "A random graph is generated in "
                  << chrono.RealTime() << " seconds \n";

    bool success = true;
    if (vertex_edge.Height() != nvertices)
    {
        success &= false;
        std::cout << "The generated graph does not have "
                  << "the expected number of vertices\n";
        std::cout << "Expect: " << nvertices << "\n";
        std::cout << "Actual: " << vertex_edge.Height() << "\n";
    }
    if (vertex_edge.Width() != nvertices * mean_degree / 2)
    {
        success &= false;
        std::cout << "The generated graph does not have "
                  << "the expected number of edges\n";
        std::cout << "Expect: " << nvertices* mean_degree / 2 << "\n";
        std::cout << "Actual: " << vertex_edge.Width() << "\n";
    }

    if (success)
        return 0;
    else
        return 1;
}


