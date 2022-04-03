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

double mu_o = 3e-3; //0.005; //0.0002; //
double mu_w = 3e-4; // 0.3*centi*poise //1e-3;
int relperm_order = 2;
//mfem::Array<int> well_cells;
mfem::Array<int> well_perforations;
mfem::Vector well_cell_gf;


enum SteppingScheme { IMPES = 1, SequentiallyImplicit, FullyImplcit };

struct EvolveParamenters
{
    double total_time = 10.0;    // Total time
    double dt = 1.0;   // Time step size
    int vis_step = 0;
    SteppingScheme scheme = IMPES;
};

void SetOptions(FASParameters& param, bool use_vcycle, int num_backtrack, double diff_tol);

mfem::Vector TotalMobility(const mfem::Vector& S);
mfem::Vector dTMinv_dS(const mfem::Vector& S);
mfem::Vector FractionalFlow(const mfem::Vector& S);
mfem::Vector dFdS(const mfem::Vector& S);

DarcyProblem* problem_ptr;
std::vector<mfem::socketstream> sout_resid_(50); // this should not be needed (only for debug)


int main(int argc, char* argv[])
{
    int num_procs, myid;

    // 1. Initialize MPI
    mpi_session session(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // program options from command line
    EvolveParamenters evolve_param;
    mfem::OptionsParser args(argc, argv);
    std::string base_dir = "/Users/lee1029/Downloads/";
    const char* problem_dir = "";
    args.AddOption(&problem_dir, "-pd", "--problem-directory",
                   "Directory where data files are located");
    const char* perm_file = "spe_perm.dat";
    args.AddOption(&perm_file, "-p", "--perm", "SPE10 permeability file data.");
    int dim = 3;
    args.AddOption(&dim, "-d", "--dim", "Dimension of the physical space.");
    int slice = 0;
    args.AddOption(&slice, "-s", "--slice", "Slice of SPE10 data for 2D run.");
    int well_height = 1;
    args.AddOption(&well_height, "-wh", "--well-height", "Well Height.");
    double inject_rate = 5.6544e-04;//0.00005;// * 0.6096;
    args.AddOption(&inject_rate, "-ir", "--inject-rate", "Injector rate.");
    double bhp = -2.7579e07;//-1.0e6;
    args.AddOption(&bhp, "-bhp", "--bottom-hole-pressure", "Bottom Hole Pressure.");
    args.AddOption(&evolve_param.dt, "-dt", "--delta-t", "Time step.");
    args.AddOption(&evolve_param.total_time, "-time", "--total-time",
                   "Total time to step.");
    args.AddOption(&evolve_param.vis_step, "-vs", "--vis-step",
                   "Step size for visualization.");
    int scheme = 3;
    args.AddOption(&scheme, "-scheme", "--stepping-scheme",
                   "Time stepping: 1. IMPES, 2. sequentially implicit, 3. fully implicit. ");
    bool use_vcycle = true;
    args.AddOption(&use_vcycle, "-VCycle", "--use-VCycle", "-FMG",
                   "--use-FMG", "Use V-cycle or FMG-cycle.");
    int num_backtrack = 0;
    args.AddOption(&num_backtrack, "--num-backtrack", "--num-backtrack",
                   "Maximum number of backtracking steps.");
    double diff_tol = -1.0;
    args.AddOption(&diff_tol, "--diff-tol", "--diff-tol",
                   "Tolerance for coefficient change.");
    int print_level = -1;
    args.AddOption(&print_level, "-print-level", "--print-level",
                   "Solver print level (-1 = no to print, 0 = final error, 1 = all errors.");
    bool smeared_front = true;
    args.AddOption(&smeared_front, "-smeared-front", "--smeared-front", "-sharp-front",
                   "--sharp-front", "Control density to produce smeared or sharp saturation front.");
    args.AddOption(&relperm_order, "-ro", "--relperm-order",
                   "Exponent of relperm function.");
    UpscaleParameters upscale_param;
    upscale_param.spect_tol = 1.0;
    upscale_param.max_evects = 1;
    upscale_param.max_traces = 1;
    upscale_param.max_levels = 1;
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

    evolve_param.scheme = static_cast<SteppingScheme>(scheme);

    // Setting up finite volume discretization problem
    unique_ptr<DarcyProblem> problem;
    problem.reset(new LocalProblem(comm, dim, std::vector<int>(dim, 29)));

    unique_ptr<mfem::Array<int>> partition(nullptr);
    Upscale upscale(problem->GetFVGraph(true), upscale_param, partition.get(),
                    &problem->EssentialAttribute());
    upscale.PrintInfo();

    mfem::BlockVector rhs(upscale.BlockOffsets(0));
    rhs.GetBlock(0) = 0.0;
    rhs.GetBlock(1) = 1.0/(rhs.BlockSize(1)-1);
    rhs[rhs.Size()-1] = -1.0;

    mfem::BlockVector sol = upscale.Solve(0, rhs);

//    mfem::Vector p_wc(19);
//    for (int i = 0; i < 19; ++i) { p_wc[i] = sol.GetBlock(1)[sol.BlockSize(1)-19+i]; }
//    p_wc.Print();
    std::cout<<"P max: " << sol.GetBlock(1).Max() <<", P min: " << sol.GetBlock(1).Min() <<"\n";
    std::cout<<"P mean: " << sol.GetBlock(1).Sum() / sol.BlockSize(1) <<"\n";

    mfem::socketstream vis_v;
    problem->VisSetup(vis_v, sol.GetBlock(1), 0.0, 0.0, "Pressure");

    return EXIT_SUCCESS;
}
