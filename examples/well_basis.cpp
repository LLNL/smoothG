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
   @file singlephase.cpp
   @brief This is an example for upscaling a graph Laplacian coming from a finite
   volume discretization of a single phase flow and transport model in parallel.

   A simple way to run the example:

   mpirun -n 4 ./singlephase
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "mfem.hpp"
#include "well.hpp"

#include "../src/picojson.h"
#include "../src/smoothG.hpp"

using namespace smoothg;

int main(int argc, char* argv[])
{
    int num_procs, myid;

    // 1. Initialize MPI
    mpi_session session(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // program options from command line
    mfem::OptionsParser args(argc, argv);
    int dim = 3;
    args.AddOption(&dim, "-d", "--dim", "Dimension of the physical space.");
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

    // Setting up finite volume discretization problem
    LocalProblem problem(comm, dim, std::vector<int>(dim, 29));
    Upscale upscale(problem.GetFVGraph(true));
    upscale.PrintInfo();

    mfem::BlockVector rhs(upscale.BlockOffsets(0));
    rhs.GetBlock(0) = 0.0;
    rhs.GetBlock(1) = 1.0/(rhs.BlockSize(1)-1);
    rhs[rhs.Size()-1] = -1.0;

    mfem::BlockVector sol = upscale.Solve(0, rhs);

    std::cout<<"P max: " << sol.GetBlock(1).Max() <<", P min: " << sol.GetBlock(1).Min() <<"\n";
    std::cout<<"P mean: " << sol.GetBlock(1).Sum() / sol.BlockSize(1) <<"\n";

    mfem::socketstream vis_v;
    problem.VisSetup(vis_v, sol.GetBlock(1), 0.0, 0.0, "Pressure");

    return EXIT_SUCCESS;
}
