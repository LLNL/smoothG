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
   @example
   @file generalgraph.cpp
   @brief Compares a graph upscaled solution to the fine solution.
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "mfem.hpp"

#include "../src/picojson.h"
#include "../src/smoothG.hpp"

using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;
using std::flush;
using std::cout;
using std::endl;
using std::cerr;

using namespace smoothg;
using namespace mfem;

void MetisPart(const mfem::SparseMatrix& vertex_edge, mfem::Array<int>& part, int num_parts,
               int isolate);
mfem::Vector ComputeFiedlerVector(const MixedMatrix& mixed_laplacian);

void Split(const mfem::SparseMatrix& A, mfem::SparseMatrix& vertex_edge, mfem::Vector& weight);

int main(int argc, char* argv[])
{
    // 1. Initialize MPI
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // program options from command line
    mfem::OptionsParser args(argc, argv);
    int coarse_factor = 100;
    args.AddOption(&coarse_factor, "-cf", "--coarse-factor",
            "Coarsening factor");
    const char* graphFileName = "../../graphdata/vertex_edge_sample.txt";
    args.AddOption(&graphFileName, "-g", "--graph",
            "File to load for graph connection data.");
    const char* FiedlerFileName = "../../graphdata/fiedler_sample.txt";
    args.AddOption(&FiedlerFileName, "-f", "--fiedler",
            "File to load for the Fiedler vector.");
    const char* partition_filename = "../../graphdata/partition_sample.txt";
    args.AddOption(&partition_filename, "-p", "--partition",
            "Partition file to load (instead of using metis).");
    const char* weight_filename = "";
    args.AddOption(&weight_filename, "-w", "--weight",
            "File to load for graph edge weights.");
    const char* w_block_filename = "";
    args.AddOption(&w_block_filename, "-wb", "--w_block",
            "File to load for w block.");
    bool metis_agglomeration = false;
    args.AddOption(&metis_agglomeration, "-ma", "--metis-agglomeration",
            "-nm", "--no-metis-agglomeration",
            "Use Metis as the partitioner (instead of loading partition).");
    int max_evects = 4;
    args.AddOption(&max_evects, "-m", "--max-evects",
            "Maximum eigenvectors per aggregate.");
    double spect_tol = 1.e-3;
    args.AddOption(&spect_tol, "-t", "--spect-tol",
            "Spectral tolerance for eigenvalue problems.");
    bool hybridization = true;
    args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
            "--no-hybridization", "Enable hybridization.");
    bool generate_graph = false;
    args.AddOption(&generate_graph, "-gg", "--generate-graph", "-no-gg",
            "--no-generate-graph", "Generate a graph at runtime.");
    bool generate_fiedler = false;
    args.AddOption(&generate_fiedler, "-gf", "--generate-fiedler", "-no-gf",
            "--no-generate-fiedler", "Generate a fiedler vector at runtime.");
    bool save_fiedler = false;
    args.AddOption(&save_fiedler, "-sf", "--save-fiedler", "-no-sf",
            "--no-save-fiedler", "Save a generate a fiedler vector at runtime.");
    int gen_vertices = 1000;
    args.AddOption(&gen_vertices, "-nv", "--num-vert",
            "Number of vertices of the graph to be generated.");
    int mean_degree = 40;
    args.AddOption(&mean_degree, "-md", "--mean-degree",
            "Average vertex degree of the graph to be generated.");
    double beta = 0.15;
    args.AddOption(&beta, "-b", "--beta",
            "Probability of rewiring in the Watts-Strogatz model.");
    int seed = 0;
    args.AddOption(&seed, "-s", "--seed",
            "Seed (unsigned integer) for the random number generator.");
    int isolate = -1;
    args.AddOption(&isolate, "--isolate", "--isolate",
            "Isolate a single vertex (for debugging so far).");
    bool dual_target = false;
    args.AddOption(&dual_target, "-dt", "--dual-target", "-no-dt",
            "--no-dual-target", "Use dual graph Laplacian in trace generation.");
    bool scaled_dual = false;
    args.AddOption(&scaled_dual, "-sd", "--scaled-dual", "-no-sd",
            "--no-scaled-dual", "Scale dual graph Laplacian by (inverse) edge weight.");
    bool energy_dual = false;
    args.AddOption(&energy_dual, "-ed", "--energy-dual", "-no-ed",
            "--no-energy-dual", "Use energy matrix in trace generation.");
    bool verbose = false;
    args.AddOption(&verbose, "-v", "--verbose", "-no-v",
            "--no-verbose", "Verbose solver output.");
    int max_iter = 10000;
    args.AddOption(&max_iter, "-mi", "--max-iter",
            "Max number of solver iterations.");


    // MFEM Options
    const char* mesh_file = "../data/star.mesh";
    args.AddOption(&mesh_file, "-mesh", "--mesh",
            "Mesh file to use.");
    int order = 1;
    args.AddOption(&order, "-o", "--order",
            "Finite element order (polynomial degree) or -1 for"
            " isoparametric space.");
    int ref_levels = 1;
    args.AddOption(&ref_levels, "-nr", "--num-refine",
            "Number of times to refine mesh.");
    bool static_cond = false;
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
            "--no-static-condensation", "Enable static condensation.");
    bool visualization = 1;
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
            "--no-visualization",
            "Enable or disable GLVis visualization.");
   bool amg_elast = 0;
   args.AddOption(&amg_elast, "-elast", "--amg-for-elasticity", "-sys",
                  "--amg-for-systems",
                  "Use the special AMG elasticity solver (GM/LN approaches), "
                  "or standard AMG for systems (unknown approach).");

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

    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    if (mesh->attributes.Max() < 2 || mesh->bdr_attributes.Max() < 2)
    {
        if (myid == 0)
            cerr << "\nInput mesh should have at least two materials and "
                << "two boundary attributes! (See schematic in ex2.cpp)\n"
                << endl;
        MPI_Finalize();
        return 3;
    }

    // 4. Select the order of the finite element discretization space. For NURBS
    //    meshes, we increase the order by degree elevation.
    if (mesh->NURBSext && order > mesh->NURBSext->GetOrder())
    {
        mesh->DegreeElevate(order - mesh->NURBSext->GetOrder());
    }

    // 5. Refine the serial mesh on all processors to increase the resolution. In
    //    this example we do 'ref_levels' of uniform refinement. We choose
    //    'ref_levels' to be the largest number that gives a final mesh with no
    //    more than 1,000 elements.
    for (int l = 0; l < ref_levels; l++)
    {
        mesh->UniformRefinement();
    }

    // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh further in parallel to increase the resolution. Once the
    //    parallel mesh is defined, the serial mesh can be deleted.
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;

    // 7. Define a parallel finite element space on the parallel mesh. Here we
    //    use vector finite elements, i.e. dim copies of a scalar finite element
    //    space. We use the ordering by vector dimension (the last argument of
    //    the FiniteElementSpace constructor) which is expected in the systems
    //    version of BoomerAMG preconditioner. For NURBS meshes, we use the
    //    (degree elevated) NURBS space associated with the mesh nodes.
    FiniteElementCollection *fec;
    ParFiniteElementSpace *fespace;
    const bool use_nodal_fespace = pmesh->NURBSext && !amg_elast;
    if (use_nodal_fespace)
    {
        fec = NULL;
        fespace = (ParFiniteElementSpace *)pmesh->GetNodes()->FESpace();
    }
    else
    {
        fec = new H1_FECollection(order, dim);
        fespace = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byVDIM);
    }
    HYPRE_Int size = fespace->GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "Number of finite element unknowns: " << size << endl
            << "Assembling: " << flush;
    }

    // 8. Determine the list of true (i.e. parallel conforming) essential
    //    boundary dofs. In this example, the boundary conditions are defined by
    //    marking only boundary attribute 1 from the mesh as essential and
    //    converting it to a list of true dofs.
    Array<int> ess_tdof_list, ess_bdr(pmesh->bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 0;
    fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    // 9. Set up the parallel linear form b(.) which corresponds to the
    //    right-hand side of the FEM linear system. In this case, b_i equals the
    //    boundary integral of f*phi_i where f represents a "pull down" force on
    //    the Neumann part of the boundary and phi_i are the basis functions in
    //    the finite element fespace. The force is defined by the object f, which
    //    is a vector of Coefficient objects. The fact that f is non-zero on
    //    boundary attribute 2 is indicated by the use of piece-wise constants
    //    coefficient for its last component.
    VectorArrayCoefficient f(dim);
    for (int i = 0; i < dim-1; i++)
    {
        f.Set(i, new ConstantCoefficient(0.0));
    }
    {
        Vector pull_force(pmesh->bdr_attributes.Max());
        pull_force = 0.0;
        pull_force(1) = -1.0e-2;
        f.Set(dim-1, new PWConstCoefficient(pull_force));
    }

    ParLinearForm *b = new ParLinearForm(fespace);
    b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
    if (myid == 0)
    {
        cout << "r.h.s. ... " << flush;
    }
    b->Assemble();

    // 10. Define the solution vector x as a parallel finite element grid
    //     function corresponding to fespace. Initialize x with initial guess of
    //     zero, which satisfies the boundary conditions.
    ParGridFunction x(fespace);
    x = 0.0;

    // 11. Set up the parallel bilinear form a(.,.) on the finite element space
    //     corresponding to the linear elasticity integrator with piece-wise
    //     constants coefficient lambda and mu.
    Vector lambda(pmesh->attributes.Max());
    lambda = 1.0;
    lambda(0) = lambda(1)*50;
    PWConstCoefficient lambda_func(lambda);
    Vector mu(pmesh->attributes.Max());
    mu = 1.0;
    mu(0) = mu(1)*50;
    PWConstCoefficient mu_func(mu);

    ParBilinearForm *a = new ParBilinearForm(fespace);
    a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));

    // 12. Assemble the parallel bilinear form and the corresponding linear
    //     system, applying any necessary transformations such as: parallel
    //     assembly, eliminating boundary conditions, applying conforming
    //     constraints for non-conforming AMR, static condensation, etc.
    if (myid == 0) { cout << "matrix ... " << flush; }
    if (static_cond) { a->EnableStaticCondensation(); }
    a->Assemble();

    HypreParMatrix A;
    Vector B, X;
    a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
    if (myid == 0)
    {
        cout << "done." << endl;
        cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
    }

    if (myid == 0)
    {
        cout << "done." << endl;
        cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
    }

    // 13. Define and apply a parallel PCG solver for A X = B with the BoomerAMG
    //     preconditioner from hypre.
    HypreBoomerAMG *amg = new HypreBoomerAMG(A);
    if (amg_elast && !a->StaticCondensationIsEnabled())
    {
        amg->SetElasticityOptions(fespace);
    }
    else
    {
        amg->SetSystemsOptions(dim);
    }

    //HyprePCG *solver = new HyprePCG(A);
    HypreGMRES *solver = new HypreGMRES(A);
    solver->SetTol(1e-8);
    solver->SetMaxIter(500);
    solver->SetPrintLevel(2);
    solver->SetPreconditioner(*amg);
    solver->Mult(B, X);

    // 14. Recover the parallel grid function corresponding to X. This is the
    //     local finite element solution on each processor.
    a->RecoverFEMSolution(X, *b, x);

    // 15. For non-NURBS meshes, make the mesh curved based on the finite element
    //     space. This means that we define the mesh elements through a fespace
    //     based transformation of the reference element.  This allows us to save
    //     the displaced mesh as a curved mesh when using high-order finite
    //     element displacement field. We assume that the initial mesh (read from
    //     the file) is not higher order curved mesh compared to the chosen FE
    //     space.
    if (!use_nodal_fespace)
    {
        pmesh->SetNodalFESpace(fespace);
    }

    // 18. Free the used memory.
    delete solver;
    delete amg;
    delete a;
    delete b;
    if (fec)
    {
        delete fespace;
        delete fec;
    }
    delete pmesh;

    MPI_Finalize();

    return 0;
}
