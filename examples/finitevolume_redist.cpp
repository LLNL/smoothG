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
   @file finitevolume.cpp
   @brief This is an example for upscaling a graph Laplacian coming from a finite
   volume discretization of a simple reservior model in parallel.

   If lateral_pressure == true, Dirichlet pressure boundary condition will be
   imposed on left (p = -1) and right (p = 0) side of the domain boundary.
   No flow boundary condition (v.n = 0) is imposed on the rest of the boundary.
   In this case, the quantity of interest (QoI) is the total out flux
   \f$ \int v \cdot n dS \f$ on the left boundary.

   A simple way to run the example:

   mpirun -n 4 ./finitevolume
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "mfem.hpp"
#include "pde.hpp"

#include "../src/picojson.h"
#include "../src/smoothG.hpp"

using namespace smoothg;

void RedistSolve(const Hierarchy& hierarchy, Redistributor& redistributor,
                 const LinearSolverParameters& lin_solve_param, int level,
                 const mfem::BlockVector& x, mfem::BlockVector& y)
{
    auto& mixed_system = hierarchy.GetMatrix(level);
    MixedMatrix redist_system = redistributor.Redistribute(mixed_system);

    if (!lin_solve_param.hybridization) { redist_system.BuildM(); }
    auto solver = hierarchy.MakeSolver(redist_system, lin_solve_param, true);

    std::vector<mfem::BlockVector> rhs, sol;
    rhs.reserve(level + 1);
    sol.reserve(level + 1);

    rhs.push_back(x);
    for (int i = 0; i < level; ++i)
    {
        rhs.push_back(hierarchy.Restrict(i, rhs[i]));
        sol.emplace_back(hierarchy.BlockOffsets(i));
    }
    sol.emplace_back(hierarchy.BlockOffsets(level));

    auto& redTVD_TVD = redistributor.TrueDofRedistribution(0);
    auto& redTED_TED = redistributor.TrueDofRedistribution(1);

    mfem::Array<int> red_offsets(3);
    red_offsets[0] = 0;
    red_offsets[1] = redTED_TED.NumRows();
    red_offsets[2] = red_offsets[1] + redTVD_TVD.NumRows();

    auto assembled_rhs = mixed_system.Assemble(rhs[level]);
    auto& true_offsets = mixed_system.BlockTrueOffsets();
    mfem::BlockVector true_rhs(assembled_rhs.GetData(), true_offsets);

    mfem::BlockVector redist_rhs(red_offsets), redist_sol(red_offsets);
    redTED_TED.Mult(true_rhs.GetBlock(0), redist_rhs.GetBlock(0));
    redTVD_TVD.Mult(true_rhs.GetBlock(1), redist_rhs.GetBlock(1));

    redist_sol = 0.0;
    solver->Mult(redist_rhs, redist_sol);

    mfem::BlockVector true_sol(true_offsets);
    redTED_TED.MultTranspose(redist_sol.GetBlock(0), true_sol.GetBlock(0));
    redTVD_TVD.MultTranspose(redist_sol.GetBlock(1), true_sol.GetBlock(1));

    // ((mfem::Vector&)sol[level]) = mixed_system.Distribute(true_sol);
    auto dist_sol = mixed_system.Distribute(true_sol);
    std::copy_n(dist_sol.GetData(), dist_sol.Size(), sol[level].GetData());

    // interpolate solution
    for (int i = level; i > 0; --i)
    {
        hierarchy.Interpolate(i, sol[i], sol[i - 1]);
    }
    y = sol[0];
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
    const char* perm_file = "spe_perm.dat";
    args.AddOption(&perm_file, "-p", "--perm",
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
    bool lateral_pressure = false;
    args.AddOption(&lateral_pressure, "-lat-pres", "--lateral-pressure",
                   "-no-lat-pres", "--no-lateral-pressure",
                   "Impose Dirichlet pressure condition on lateral sides.");
    // Read upscaling options from command line into upscale_param object
    upscale_param.coarsen_param.spect_tol = 1.0;
    upscale_param.coarsen_param.max_evects = 2;
    upscale_param.coarsen_param.max_traces = 1;
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

    mfem::Array<int> coarsening_factors(nDimensions);
    if (metis_agglomeration)
    {
        coarsening_factors = 1;
        coarsening_factors.Last() = upscale_param.coarsen_param.coarse_factor;
    }
    else
    {
        coarsening_factors = 10;
        coarsening_factors.Last() = nDimensions == 3 ? 2 : 10;
    }

    mfem::Array<int> ess_attr(nDimensions == 3 ? 6 : 4);
    ess_attr = 1;
    if (lateral_pressure)
    {
        ess_attr[nDimensions - 2] = ess_attr[nDimensions] = 0;
    }

    // Setting up finite volume discretization problem
    SPE10Problem problem(perm_file, nDimensions, spe10_scale, slice,
                         metis_agglomeration, ess_attr);
    Graph graph = problem.GetFVGraph();

    // Construct agglomerated topology based on METIS or Cartesian agglomeration
    mfem::Array<int> partitioning;
    problem.Partition(metis_agglomeration, coarsening_factors, partitioning);

    // Create Upscaler and Solve
    Upscale upscale(std::move(graph), upscale_param, &partitioning, &ess_attr);
    upscale.PrintInfo();

    mfem::BlockVector rhs_fine(upscale.BlockOffsets(0));
    rhs_fine.GetBlock(0) = problem.GetEdgeRHS();
    rhs_fine.GetBlock(1) = problem.GetVertexRHS();

    /// [Solve]
    const int num_levels = upscale_param.coarsen_param.max_levels;
    std::vector<mfem::BlockVector> sol(num_levels, rhs_fine);
    std::vector<mfem::BlockVector> redist_sol(num_levels, rhs_fine);
    std::vector<double> QoI(num_levels);
    FunctionalQoI qoi_evaluator(upscale.GetHierarchy(), rhs_fine);
    for (int level = 0; level < num_levels; ++level)
    {
        upscale.Solve(level, rhs_fine, sol[level]);
        upscale.ShowSolveInfo(level);

        auto& hierarchy = upscale.GetHierarchy();
        auto& mgL = hierarchy.GetMatrix(level);
        int num_procs_redist = num_procs / 2;
        Redistributor redistributor(mgL.GetGraph(), num_procs_redist);

        RedistSolve(hierarchy, redistributor, upscale_param.lin_solve_param,
                    level, rhs_fine, redist_sol[level]);

        auto errors = upscale.ComputeErrors(sol[level], redist_sol[level], 0);
        if (myid == 0)
        {
            serialize["relative-vertex-error"] = picojson::value(errors[0]);
            serialize["relative-edge-error"] = picojson::value(errors[1]);
            serialize["relative-D-edge-error"] = picojson::value(errors[2]);
            std::cout << picojson::value(serialize).serialize(true) << "\n";
        }
    }
    /// [Solve]

    return EXIT_SUCCESS;
}
