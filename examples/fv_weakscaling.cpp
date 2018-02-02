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
   This is an example for upscaling a graph Laplacian coming from a finite
   volume discretization of a simple reservior model in parallel.

   A simple way to run the example:

   mpirun -n 4 ./parfinitevolume
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

using namespace smoothg;

/**
   A forcing function that is supposed to very roughly represent some wells
   that are resolved on the *coarse* level.

   The forcing function is 1 on the top-left coarse cell, and -1 on the
   bottom-right coarse cell, and 0 elsewhere.

   @param Lx length of entire domain in x direction
   @param Hx size in x direction of a coarse cell.
*/
class GCoefficient : public mfem::Coefficient
{
public:
    GCoefficient(double Lx, double Ly, double Lz,
                 double Hx, double Hy, double Hz);
    double Eval(mfem::ElementTransformation& T,
                const mfem::IntegrationPoint& ip);
private:
    double Lx_, Ly_, Lz_;
    double Hx_, Hy_, Hz_;
};

GCoefficient::GCoefficient(double Lx, double Ly, double Lz,
                           double Hx, double Hy, double Hz)
    :
    Lx_(Lx),
    Ly_(Ly),
    Lz_(Lz),
    Hx_(Hx),
    Hy_(Hy),
    Hz_(Hz)
{
}

double GCoefficient::Eval(mfem::ElementTransformation& T,
                          const mfem::IntegrationPoint& ip)
{
    double dx[3];
    mfem::Vector transip(dx, 3);

    T.Transform(ip, transip);

    if ((transip(0) < Hx_) && (transip(1) > (Ly_ - Hy_)))
        return 1.0;
    else if ((transip(0) > (Lx_ - Hx_)) && (transip(1) < Hy_))
        return -1.0;
    return 0.0;
}

class DarcyFlowProblem
{
public:
    DarcyFlowProblem(MPI_Comm comm, int nDimensions,
                     const mfem::Array<int>& coarsening_factor);
    ~DarcyFlowProblem();
    mfem::ParMesh* GetParMesh()
    {
        return pmesh_;
    }
    GCoefficient* GetForceCoeff()
    {
        return source_coeff_;
    }
private:
    double Lx, Ly, Lz, Hx, Hy, Hz;
    mfem::ParMesh* pmesh_;
    GCoefficient* source_coeff_;
};

DarcyFlowProblem::DarcyFlowProblem(MPI_Comm comm, int nDimensions,
                                   const mfem::Array<int>& coarsening_factor)
{
    int num_procs;
    MPI_Comm_size(comm, &num_procs);

    int np_per_dim;
    if (nDimensions == 3)
        np_per_dim = static_cast<int>(std::cbrt(num_procs) + 0.5);
    else
        np_per_dim = static_cast<int>(std::sqrt(num_procs) + 0.5);

    mfem::Array<int> N(3);
    N[0] = coarsening_factor[0] * np_per_dim * 4;
    N[1] = coarsening_factor[0] * np_per_dim * 4;
    N[2] = coarsening_factor[0] * np_per_dim;

    // SPE10 grid cell dimensions
    mfem::Vector h(3);
    h(0) = 20.0;
    h(1) = 10.0;
    h(2) = 2.0;
    unique_ptr<mfem::Mesh> mesh;

    int Lx = N[0] * h(0);
    int Ly = N[1] * h(1);
    int Lz = N[2] * h(2);

    // Create a serial mesh.
    if (nDimensions == 3)
        mesh = make_unique<mfem::Mesh>(
                   N[0], N[1], N[2], mfem::Element::HEXAHEDRON, 1, Lx, Ly, Lz);
    else
        mesh = make_unique<mfem::Mesh>(
                   N[0], N[1], mfem::Element::QUADRILATERAL, 1, Lx, Ly);

    int* nxyz = new int[nDimensions];
    if (nDimensions == 3)
    {
        nxyz[0] = np_per_dim;
        nxyz[1] = np_per_dim;
        nxyz[2] = np_per_dim;
    }
    else
    {
        nxyz[0] = np_per_dim;
        nxyz[1] = np_per_dim;
    }

    int nparts = 1;
    for (int d = 0; d < nDimensions; d++)
        nparts *= nxyz[d];
    assert(nparts == num_procs);

    int* cart_part = mesh->CartesianPartitioning(nxyz);
    pmesh_  = new mfem::ParMesh(comm, *mesh, cart_part);

    delete [] cart_part;
    delete [] nxyz;

    // Free the serial mesh
    mesh.reset();

    if (nDimensions == 3)
        pmesh_->ReorientTetMesh();

    Hx = coarsening_factor[0] * h(0);
    Hy = coarsening_factor[1] * h(1);
    Hz = 1.0;
    if (nDimensions == 3)
        Hz = coarsening_factor[2] * h(2);
    source_coeff_ = new GCoefficient(Lx, Ly, Lz, Hx, Hy, Hz);
}

DarcyFlowProblem::~DarcyFlowProblem()
{
    delete source_coeff_;
    delete pmesh_;
}

enum
{
    TOPOLOGY = 0, SEQUENCE, SOLVER, NSTAGES
};

int main(int argc, char* argv[])
{
    int num_procs, myid;
    picojson::object serialize;

    // 1. Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // program options from command line
    mfem::OptionsParser args(argc, argv);
    const char* permFile = "spe_perm.dat";
    args.AddOption(&permFile, "-p", "--perm",
                   "SPE10 permeability file data.");
    int nDimensions = 2;
    args.AddOption(&nDimensions, "-d", "--dim",
                   "Dimension of the physical space.");
    int slice = 0;
    args.AddOption(&slice, "-s", "--slice",
                   "Slice of SPE10 data to take for 2D run.");
    int max_evects = 4;
    args.AddOption(&max_evects, "-m", "--max-evects",
                   "Maximum eigenvectors per aggregate.");
    double spect_tol = 1.e-3;
    args.AddOption(&spect_tol, "-t", "--spect-tol",
                   "Spectral tolerance for eigenvalue problems.");
    bool metis_agglomeration = false;
    args.AddOption(&metis_agglomeration, "-ma", "--metis-agglomeration",
                   "-nm", "--no-metis-agglomeration",
                   "Use Metis as the partitioner (instead of geometric).");
    bool hybridization = false;
    args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                   "--no-hybridization", "Enable hybridization.");
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

    mfem::Array<int> coarsening_factor(nDimensions);
    if (nDimensions == 3)
    {
        coarsening_factor[0] = 10;
        coarsening_factor[1] = 10;
        coarsening_factor[2] = 10;
    }
    else
    {
        coarsening_factor[0] = 30;
        coarsening_factor[1] = 30;
    }

    int nbdr;
    if (nDimensions == 3)
        nbdr = 6;
    else
        nbdr = 4;
    mfem::Array<int> ess_zeros(nbdr);
    mfem::Array<int> nat_one(nbdr);
    mfem::Array<int> nat_zeros(nbdr);
    ess_zeros = 1;
    nat_one = 0;
    nat_zeros = 0;

    mfem::StopWatch chrono;

    const int nLevels = 2;
    UpscalingStatistics stats(nLevels);

    mfem::Array<int> ess_attr;
    mfem::FiniteElementCollection* sigmafec;
    mfem::FiniteElementCollection* ufec;
    mfem::ParFiniteElementSpace* sigmafespace;
    mfem::ParFiniteElementSpace* ufespace;
    mfem::Vector weight_inv;
    mfem::Vector rhs_sigma_fine;
    mfem::Vector rhs_u_fine;
    unique_ptr<mfem::HypreParMatrix> pvertex_edge;

    // Setting up finite volume discretization problem
    DarcyFlowProblem darcyproblem(comm, nDimensions, coarsening_factor);
    mfem::ParMesh* pmesh = darcyproblem.GetParMesh();
    if (myid == 0)
    {
        std::cout << pmesh->GetNEdges() << " fine edges, " <<
                  pmesh->GetNFaces() << " fine faces, " <<
                  pmesh->GetNE() << " fine elements\n";
    }

    ess_attr.SetSize(nbdr);
    for (int i(0); i < nbdr; ++i)
        ess_attr[i] = ess_zeros[i];

    // Construct "finite volume mass" matrix
    sigmafec = new mfem::RT_FECollection(0, nDimensions);
    sigmafespace = new mfem::ParFiniteElementSpace(pmesh, sigmafec);
    mfem::ParBilinearForm a(sigmafespace);
    a.AddDomainIntegrator(new FiniteVolumeMassIntegrator());
    a.Assemble();
    a.Finalize();
    a.SpMat().GetDiag(weight_inv);

    ufec = new mfem::L2_FECollection(0, nDimensions);
    ufespace = new mfem::ParFiniteElementSpace(pmesh, ufec);

    mfem::ConstantCoefficient pinflow_coeff(0.);
    mfem::LinearForm b(sigmafespace);
    b.AddBoundaryIntegrator(
        new mfem::VectorFEBoundaryFluxLFIntegrator(pinflow_coeff));
    b.Assemble();
    rhs_sigma_fine = b;

    mfem::LinearForm q(ufespace);
    q.AddDomainIntegrator(
        new mfem::DomainLFIntegrator(*darcyproblem.GetForceCoeff()) );
    q.Assemble();
    rhs_u_fine = q;

    // Construct vertex_edge table in mfem::SparseMatrix format
    const mfem::Table* vertex_edge_table;
    if (nDimensions == 2)
        vertex_edge_table = &(pmesh->ElementToEdgeTable());
    else
        vertex_edge_table = &(pmesh->ElementToFaceTable());
    int nvertices = vertex_edge_table->Size();
    int nedges = vertex_edge_table->Width();
    int vertex_edge_nnz = vertex_edge_table->Size_of_connections();
    int* vertex_edge_i = new int[nvertices + 1];
    int* vertex_edge_j = new int[vertex_edge_nnz];
    double* vertex_edge_data = new double [vertex_edge_nnz];
    std::copy_n(vertex_edge_table->GetI(), nvertices + 1, vertex_edge_i);
    std::copy_n(vertex_edge_table->GetJ(), vertex_edge_nnz, vertex_edge_j);
    std::fill_n(vertex_edge_data, vertex_edge_nnz, 1.);
    auto vertex_edge =
        make_shared<mfem::SparseMatrix>(vertex_edge_i, vertex_edge_j,
                                        vertex_edge_data, nvertices, nedges);

    chrono.Clear();
    MPI_Barrier(comm);
    chrono.Start();

    // Prepare storage for partitioning
    mfem::Array<int> partitioning;
    if (metis_agglomeration) // Construct agglomerated topology based on METIS
    {
        MetisGraphPartitioner partitioner;
        partitioner.setUnbalanceTol(2);
        int metis_coarsening_factor = 1;

        mfem::DiscreteLinearOperator DivOp(sigmafespace, ufespace);
        DivOp.AddDomainInterpolator(new mfem::DivergenceInterpolator);
        DivOp.Assemble();
        DivOp.Finalize();
        mfem::SparseMatrix& DivMat = DivOp.SpMat();
        mfem::SparseMatrix* DivMatT = Transpose(DivMat);
        mfem::SparseMatrix* vertex_vertex = Mult(DivMat, *DivMatT);
        delete DivMatT;

        for (int d = 0; d < nDimensions; d++)
            metis_coarsening_factor *= coarsening_factor[d];

        int num_partitions = nvertices / metis_coarsening_factor;
        if (num_partitions == 0) num_partitions = 1;

        partitioner.doPartition(*vertex_vertex, num_partitions,
                                partitioning );
        delete vertex_vertex;
    }
    else // Or use cartesian agglomeration to build topology
    {
        // Use cartesian agglomeration to build topology
        int* nxyz = new int[nDimensions];
        if (nDimensions == 3)
        {
            nxyz[0] = 1;
            nxyz[1] = 1;
            nxyz[2] = 1;
        }
        else
        {
            nxyz[0] = 4;
            nxyz[1] = 4;
        }
        mfem::Array<int> cart_part(pmesh->CartesianPartitioning(nxyz),
                                   pmesh->GetNE());
        partitioning.Append(cart_part);

        cart_part.MakeDataOwner();
        cart_part.DeleteAll();
        delete [] nxyz;
    }
    MPI_Barrier(comm);
    if (myid == 0)
        std::cout << "Timing ELEM_AGG: 'vertices' partition done in "
                  << chrono.RealTime() << " seconds \n";

    auto edge_d_td = make_shared<mfem::HypreParMatrix>();
    edge_d_td->MakeRef(*(sigmafespace->Dof_TrueDof_Matrix()));

    std::vector<unique_ptr<mfem::SparseMatrix> > edge_edgedof(nLevels);
    std::vector<shared_ptr<mfem::SparseMatrix> > edge_boundaryattribute(nLevels);

    std::vector<MixedMatrix> mixed_laplacians;
    std::vector<std::unique_ptr<Mixed_GL_Coarsener>> mixed_gl_coarseners;

    // build fine topology, data structures, and try to coarsen
    chrono.Clear();
    chrono.Start();
    {
        int i = 0;

        mixed_laplacians.emplace_back(*vertex_edge, weight_inv, edge_d_td);

        stats.BeginTiming();

        edge_boundaryattribute[0] = GenerateBoundaryAttributeTable(pmesh);
        unique_ptr<GraphTopology> graph_topology = make_unique<GraphTopology>(
                                                       vertex_edge, edge_d_td, partitioning, edge_boundaryattribute[0]);
        stats.EndTiming(0, TOPOLOGY);

        if (myid == 0)
            std::cout << "Start coarsening level " << i << " ...\n";
        stats.BeginTiming();

        mixed_gl_coarseners.push_back(
            make_unique<SpectralAMG_MGL_Coarsener>(
                mixed_laplacians[i], std::move(graph_topology),
                spect_tol, max_evects, hybridization));
        mixed_gl_coarseners[0]->construct_coarse_subspace();

        mixed_laplacians.emplace_back(mixed_gl_coarseners[0]->GetCoarseM(),
                                      mixed_gl_coarseners[0]->GetCoarseD(),
                                      mixed_gl_coarseners[0]->get_face_dof_truedof_table());

        edge_boundaryattribute[1] =
            mixed_gl_coarseners[0]->get_GraphTopology_ref().face_bdratt_;

        edge_edgedof[1] = make_unique<mfem::SparseMatrix>(
                              mixed_gl_coarseners[0]->construct_face_facedof_table() );

        mixed_laplacians[0].set_Drow_start(
            mixed_gl_coarseners[0]->get_GraphTopology_ref().GetVertexStart());

        mixed_laplacians[1].set_Drow_start(
            mixed_gl_coarseners[0]->get_GraphCoarsen_ref().GetVertexCoarseDofStart());

        stats.EndTiming(0, SEQUENCE);
    }
    chrono.Stop();
    if (myid == 0)
        std::cout << "Timing all levels: Coarsening done in "
                  << chrono.RealTime() << " seconds \n";

    std::vector<std::unique_ptr<mfem::BlockVector>> rhs(nLevels);
    rhs[0] = mixed_laplacians[0].subvecs_to_blockvector(rhs_sigma_fine, rhs_u_fine);
    for (int i = 1; i < nLevels; i++)
    {
        rhs[i] = mixed_gl_coarseners[i - 1]->coarsen_rhs(*rhs[i - 1]);
    }

    std::vector<std::unique_ptr<mfem::BlockVector>> sol(nLevels);
    for (int k(0); k < nLevels; ++k)
    {
        sol[k] = make_unique<mfem::BlockVector>(
                     mixed_laplacians[k].get_blockoffsets());
    }

    std::vector<std::unique_ptr<mfem::Vector>> ess_data(nLevels);
    for (int k = 0; k < nLevels; ++k)
    {
        ess_data[k] = make_unique<mfem::Vector>(
                          mixed_laplacians[k].get_num_edge_dofs());
        (*ess_data[k]) = 0.;
    }

    for (int k(0); k < nLevels; ++k)
    {
        if (myid == 0)
            std::cout << "Begin solve loop level " << k << std::endl;
        stats.BeginTiming();

        mfem::SparseMatrix& Mref = mixed_laplacians[k].getWeight();
        mfem::SparseMatrix& Dref = mixed_laplacians[k].getD();

        const mfem::HypreParMatrix& edgedof_d_td(
            mixed_laplacians[k].get_edge_d_td() );

        // ndofs[k] = edgedof_d_td.GetGlobalNumCols()
        //      + mixed_laplacians[k].get_Drow_start().Last();

        // deal with boundary conditions
        auto marker = make_shared<mfem::Array<int> >(Dref.Width());
        *marker = 0;
        if (k == 0)
        {
            // on fine level, just use the DofHandler
            sigmafespace->GetEssentialVDofs(ess_attr, *marker);
        }
        else
        {
            MarkDofsOnBoundary(*edge_boundaryattribute[k], *edge_edgedof[k],
                               ess_attr, *marker);
        }

        if (hybridization) // Hybridization solver
        {
            std::unique_ptr<HybridSolver> hb;
            if (k == 0)
                hb = make_unique<HybridSolver>(comm, mixed_laplacians[k],
                                               edge_boundaryattribute[k], marker);
            else
                hb = make_unique<HybridSolver>(
                         comm, mixed_laplacians[k], *mixed_gl_coarseners[k - 1],
                         edge_boundaryattribute[k], marker);

            hb->Mult(*rhs[k], *sol[k]);
            stats.RegisterSolve(*hb, k);
        }
        else // L2-H1 block diagonal preconditioner
        {

            for (int mm = 0; mm < Dref.Width(); ++mm)
            {
                if ((*marker)[mm])
                    Mref.EliminateRowCol(mm, (*ess_data[k])(mm), *(rhs[k]));
            }
            Dref.EliminateCols(*marker);
            if (myid == 0)
            {
                Dref.EliminateRow(0);
                rhs[k]->GetBlock(1)(0) = 0.;
            }

            mfem::BlockVector true_rhs(mixed_laplacians[k].get_blockTrueOffsets());
            edgedof_d_td.MultTranspose(rhs[k]->GetBlock(0), true_rhs.GetBlock(0));
            true_rhs.GetBlock(1) = rhs[k]->GetBlock(1);

            mfem::BlockVector true_sol(mixed_laplacians[k].get_blockTrueOffsets());
            MinresBlockSolver mgp(mixed_laplacians[k], comm);
            mgp.Mult(true_rhs, true_sol);
            stats.RegisterSolve(mgp, k);

            edgedof_d_td.Mult(true_sol.GetBlock(0), sol[k]->GetBlock(0));
            sol[k]->GetBlock(1) = true_sol.GetBlock(1);
        }
        if (k == 0)
            par_orthogonalize_from_constant(sol[k]->GetBlock(1),
                                            mixed_laplacians[k].get_Drow_start().Last());
        stats.EndTiming(k, SOLVER);

        // error norms
        stats.ComputeErrorSquare(k, mixed_laplacians, *mixed_gl_coarseners[0], sol);
    }

    stats.PrintStatistics(comm, serialize);

    delete sigmafespace;
    delete ufespace;
    delete sigmafec;
    delete ufec;

    if (myid == 0)
        std::cout << picojson::value(serialize).serialize() << std::endl;

    MPI_Finalize();
    return 0;
}

