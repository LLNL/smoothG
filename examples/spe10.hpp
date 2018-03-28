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
   @file spe10.hpp
   @brief Implementation of spe10 problem.

   Reads data from file and creates the appropriate finite element structures.
*/

#include "../src/smoothG.hpp"

using std::unique_ptr;

/**
   @brief A forcing function that is supposed to very roughly represent some wells
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

/**
   @brief A function that marks half the resevior w/ value and the other -value.

   @param spe10_scale scale for length
*/
class HalfCoeffecient : public mfem::Coefficient
{
public:
    HalfCoeffecient(double value, int spe10_scale = 5)
        : value_(value), spe10_scale_(spe10_scale) {}
    double Eval(mfem::ElementTransformation& T,
                const mfem::IntegrationPoint& ip);
private:
    double value_;
    int spe10_scale_;
};

double HalfCoeffecient::Eval(mfem::ElementTransformation& T,
                             const mfem::IntegrationPoint& ip)
{
    double dx[3];
    mfem::Vector transip(dx, 3);

    T.Transform(ip, transip);

    return transip(1) < (spe10_scale_ * 44 * 5) ? -value_ : value_;
}


/**
   @brief Manages data from the SPE10 dataset.
*/
class SPE10Problem
{
public:
    SPE10Problem(MPI_Comm comm, const char* permFile, int nDimensions, int spe10_scale,
                 int slice, bool metis_partition,
                 const mfem::Array<int>& coarsening_factor);
    ~SPE10Problem();
    mfem::ParMesh* GetParMesh()
    {
        return pmesh_;
    }
    mfem::VectorFunctionCoefficient* GetKInv()
    {
        return kinv_;
    }
    GCoefficient* GetForceCoeff()
    {
        return source_coeff_;
    }
    const std::vector<int>& GetNumProcsXYZ()
    {
        return num_procs_xyz_;
    }
    static double CellVolume(int nDimensions)
    {
        return (nDimensions == 2 ) ? (20.0 * 10.0) : (20.0 * 10.0 * 2.0);
    }
private:
    double Lx, Ly, Lz, Hx, Hy, Hz;
    mfem::ParMesh* pmesh_;
    mfem::VectorFunctionCoefficient* kinv_;
    GCoefficient* source_coeff_;
    std::vector<int> num_procs_xyz_;
};

SPE10Problem::SPE10Problem(MPI_Comm comm, const char* permFile, int nDimensions,
                           int spe10_scale, int slice,  bool metis_partition,
                           const mfem::Array<int>& coarsening_factor)
{
    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    mfem::Array<int> N(3);
    N[0] = 12 * spe10_scale; // 60
    N[1] = 44 * spe10_scale; // 220
    N[2] = 17 * spe10_scale; // 85

    // SPE10 grid cell dimensions
    mfem::Vector h(3);
    h(0) = 20.0;
    h(1) = 10.0;
    h(2) = 2.0;
    unique_ptr<mfem::Mesh> mesh;

    using IPF = smoothg::InversePermeabilityFunction;

    IPF::SetNumberCells(N[0], N[1], N[2]);
    IPF::SetMeshSizes(h(0), h(1), h(2));
    IPF::ReadPermeabilityFile(permFile, MPI_COMM_WORLD);

    if (nDimensions == 2)
        IPF::Set2DSlice(IPF::XY, slice);

    kinv_ = new mfem::VectorFunctionCoefficient(
        nDimensions, IPF::InversePermeability);

    const bool use_egg_model = false;
    if (use_egg_model)
    {
        std::string meshfile = "Egg_model.mesh";
        std::ifstream imesh(meshfile.c_str());
        if (!imesh)
        {
            if (myid == 0)
                std::cerr << "\nCan not open mesh file: " << meshfile
                          << std::endl;
            throw 2;
        }
        mesh = make_unique<mfem::Mesh>(imesh, 1, 1);
        imesh.close();
    }
    else if (nDimensions == 3)
    {
        mesh = make_unique<mfem::Mesh>(
                   N[0], N[1], N[2], mfem::Element::HEXAHEDRON, 1,
                   h(0) * N[0], h(1) * N[1], h(2) * N[2]);
    }
    else
    {
        mesh = make_unique<mfem::Mesh>(
                   N[0], N[1], mfem::Element::QUADRILATERAL, 1,
                   h(0) * N[0], h(1) * N[1]);
    }

    if (metis_partition)
    {
        pmesh_  = new mfem::ParMesh(comm, *mesh);
    }
    else
    {
        int num_procs_x = static_cast<int>(std::sqrt(num_procs) + 0.5);
        while (num_procs % num_procs_x)
            num_procs_x -= 1;

        num_procs_xyz_.resize(nDimensions);
        num_procs_xyz_[0] = num_procs_x;
        num_procs_xyz_[1] = num_procs / num_procs_x;
        if (nDimensions == 3)
            num_procs_xyz_[2] = 1;

        int nparts = 1;
        for (int d = 0; d < nDimensions; d++)
            nparts *= num_procs_xyz_[d];
        assert(nparts == num_procs);

        int* cart_part = mesh->CartesianPartitioning(num_procs_xyz_.data());
        pmesh_  = new mfem::ParMesh(comm, *mesh, cart_part);
        delete [] cart_part;
    }

    // Free the serial mesh
    mesh.reset();

    if (nDimensions == 3)
        pmesh_->ReorientTetMesh();

    // this should probably be in a different method
    Lx = N[0] * h(0);
    Ly = N[1] * h(1);
    Lz = N[2] * h(2);
    Hx = coarsening_factor[0] * h(0);
    Hy = coarsening_factor[1] * h(1);
    Hz = 1.0;
    if (nDimensions == 3)
        Hz = coarsening_factor[2] * h(2);
    source_coeff_ = new GCoefficient(Lx, Ly, Lz, Hx, Hy, Hz);
}

SPE10Problem::~SPE10Problem()
{
    smoothg::InversePermeabilityFunction::ClearMemory();
    delete source_coeff_;
    delete kinv_;
    delete pmesh_;
}

