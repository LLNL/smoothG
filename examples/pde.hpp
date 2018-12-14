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

namespace smoothg
{

/**
   @brief Construct edge to boundary attribute table (orientation is not considered)

   Copied from parelag::AgglomeratedTopology::generateFacetBdrAttributeTable

   Given a mesh this computes a table with a row for every face and a column for
   every boundary attribute, with a 1 if the face has that boundary attribute.

   This only works for the fine level, because of the mfem::Mesh. To get
   this table on a coarser mesh, premultiply by AEntity_entity.
*/
mfem::SparseMatrix GenerateBoundaryAttributeTable(const mfem::Mesh* mesh)
{
    int nedges = mesh->Dimension() == 2 ? mesh->GetNEdges() : mesh->GetNFaces();
    int nbdr = mesh->bdr_attributes.Max();
    int nbdr_edges = mesh->GetNBE();

    int* edge_bdrattr_i = new int[nedges + 1]();
    int* edge_bdrattr_j = new int[nbdr_edges];

    // in the loop below, edge_bdrattr_i is used as a temporary array
    for (int j = 0; j < nbdr_edges; j++)
    {
        int edge = mesh->GetBdrElementEdgeIndex(j);
        edge_bdrattr_i[edge + 1] = mesh->GetBdrAttribute(j);
    }
    edge_bdrattr_i[0] = 0;

    int count = 0;
    for (int j = 1; j <= nedges; j++)
    {
        if (edge_bdrattr_i[j])
        {
            edge_bdrattr_j[count++] = edge_bdrattr_i[j] - 1;
            edge_bdrattr_i[j] = edge_bdrattr_i[j - 1] + 1; // single nonzero in this row
        }
        else
        {
            edge_bdrattr_i[j] = edge_bdrattr_i[j - 1]; // no nonzeros in this row
        }
    }

    double* edge_bdrattr_data = new double[nbdr_edges];
    std::fill_n(edge_bdrattr_data, nbdr_edges, 1.0);

    return mfem::SparseMatrix(edge_bdrattr_i, edge_bdrattr_j, edge_bdrattr_data,
                              nedges, nbdr);
}

/**
   @brief A utility class for working with the SPE10 data set.

   The SPE10 data set can be found at: http://www.spe.org/web/csp/datasets/set02.htm
*/
class InversePermeabilityFunction
{
public:
    enum SliceOrientation {NONE, XY, XZ, YZ};
    static void SetNumberCells(int Nx_, int Ny_, int Nz_);
    static void SetReadRange(int max_Nx_, int max_Ny_, int max_Nz_);
    static void SetMeshSizes(double hx, double hy, double hz);
    static void Set2DSlice(SliceOrientation o, int npos );
    static void ReadPermeabilityFile(const std::string& fileName);
    static void ReadPermeabilityFile(const std::string& fileName, MPI_Comm comm);
    static void BlankPermeability();
    static void InversePermeability(const mfem::Vector& x, mfem::Vector& val);
    static double InvNorm2(const mfem::Vector& x);
    static void ClearMemory();
private:
    static int Nx;
    static int Ny;
    static int Nz;
    static int max_Nx;
    static int max_Ny;
    static int max_Nz;
    static double hx;
    static double hy;
    static double hz;
    static double* inversePermeability;
    static SliceOrientation orientation;
    static int npos;
};

void InversePermeabilityFunction::SetNumberCells(int Nx_, int Ny_, int Nz_)
{
    Nx = Nx_;
    Ny = Ny_;
    Nz = Nz_;
}

void InversePermeabilityFunction::SetReadRange(int max_Nx_, int max_Ny_, int max_Nz_)
{
    max_Nx = max_Nx_;
    max_Ny = max_Ny_;
    max_Nz = max_Nz_;
}

void InversePermeabilityFunction::SetMeshSizes(double hx_, double hy_, double hz_)
{
    hx = hx_;
    hy = hy_;
    hz = hz_;
}

void InversePermeabilityFunction::Set2DSlice(SliceOrientation o, int npos_ )
{
    orientation = o;
    npos = npos_;
}

void InversePermeabilityFunction::ReadPermeabilityFile(const std::string& fileName)
{
    std::ifstream permfile(fileName.c_str());

    if (!permfile.is_open())
    {
        std::cerr << "Error in opening file " << fileName << std::endl;
        mfem::mfem_error("File does not exist");
    }

    inversePermeability = new double [3 * Nx * Ny * Nz];
    double* ip = inversePermeability;
    double tmp;
    for (int l = 0; l < 3; l++)
    {
        for (int k = 0; k < Nz; k++)
        {
            for (int j = 0; j < Ny; j++)
            {
                for (int i = 0; i < Nx; i++)
                {
                    permfile >> *ip;
                    *ip = 1. / (*ip);
                    ip++;
                }
                for (int i = 0; i < max_Nx - Nx; i++)
                    permfile >> tmp; // skip unneeded part
            }
            for (int j = 0; j < max_Ny - Ny; j++)
                for (int i = 0; i < max_Nx; i++)
                    permfile >> tmp;  // skip unneeded part
        }

        if (l < 2) // if not processing Kz, skip unneeded part
            for (int k = 0; k < max_Nz - Nz; k++)
                for (int j = 0; j < max_Ny; j++)
                    for (int i = 0; i < max_Nx; i++)
                        permfile >> tmp;
    }
}

void InversePermeabilityFunction::ReadPermeabilityFile(const std::string& fileName,
                                                       MPI_Comm comm)
{
    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    mfem::StopWatch chrono;

    chrono.Start();
    if (myid == 0)
        ReadPermeabilityFile(fileName);
    else
        inversePermeability = new double [3 * Nx * Ny * Nz];
    chrono.Stop();

    if (myid == 0)
        std::cout << "Permeability file read in " << chrono.RealTime() << ".s \n";

    chrono.Clear();

    chrono.Start();
    MPI_Bcast(inversePermeability, 3 * Nx * Ny * Nz, MPI_DOUBLE, 0, comm);
    chrono.Stop();

    if (myid == 0)
        std::cout << "Permeability field distributed in " << chrono.RealTime() << ".s \n";

}

void InversePermeabilityFunction::BlankPermeability()
{
    inversePermeability = new double[3 * Nx * Ny * Nz];
    std::fill_n(inversePermeability, 3 * Nx * Ny * Nz, 1.0);
}

void InversePermeabilityFunction::InversePermeability(const mfem::Vector& x,
                                                      mfem::Vector& val)
{
    val.SetSize(x.Size());

    unsigned int i = 0, j = 0, k = 0;

    switch (orientation)
    {
        case NONE:
            i = Nx - 1 - (int)floor(x[0] / hx / (1. + 3e-16));
            j = (int)floor(x[1] / hy / (1. + 3e-16));
            k = Nz - 1 - (int)floor(x[2] / hz / (1. + 3e-16));
            break;
        case XY:
            i = Nx - 1 - (int)floor(x[0] / hx / (1. + 3e-16));
            j = (int)floor(x[1] / hy / (1. + 3e-16));
            k = npos;
            break;
        case XZ:
            i = Nx - 1 - (int)floor(x[0] / hx / (1. + 3e-16));
            j = npos;
            k = Nz - 1 - (int)floor(x[2] / hz / (1. + 3e-16));
            break;
        case YZ:
            i = npos;
            j = (int)floor(x[1] / hy / (1. + 3e-16));
            k = Nz - 1 - (int)floor(x[2] / hz / (1. + 3e-16));
            break;
        default:
            mfem::mfem_error("InversePermeabilityFunction::InversePermeability");
    }

    val[0] = inversePermeability[Ny * Nx * k + Nx * j + i];
    val[1] = inversePermeability[Ny * Nx * k + Nx * j + i + Nx * Ny * Nz];

    if (orientation == NONE)
        val[2] = inversePermeability[Ny * Nx * k + Nx * j + i + 2 * Nx * Ny * Nz];

}

double InversePermeabilityFunction::InvNorm2(const mfem::Vector& x)
{
    mfem::Vector val(3);
    InversePermeability(x, val);
    return 1.0 / val.Norml2();
}

void InversePermeabilityFunction::ClearMemory()
{
    delete[] inversePermeability;
}

int InversePermeabilityFunction::Nx(60);
int InversePermeabilityFunction::Ny(220);
int InversePermeabilityFunction::Nz(85);
int InversePermeabilityFunction::max_Nx(60);
int InversePermeabilityFunction::max_Ny(220);
int InversePermeabilityFunction::max_Nz(85);
double InversePermeabilityFunction::hx(20);
double InversePermeabilityFunction::hy(10);
double InversePermeabilityFunction::hz(2);
double* InversePermeabilityFunction::inversePermeability(NULL);
InversePermeabilityFunction::SliceOrientation InversePermeabilityFunction::orientation(
    InversePermeabilityFunction::NONE );
int InversePermeabilityFunction::npos(-1);

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
    /// constructor for the usual SPE10 dataset permeabilities
    SPE10Problem(const char* permFile, int nDimensions, int spe10_scale,
                 int slice, bool metis_partition, double proc_part_ubal,
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
    void Init(
        const char* permFile, int nDimensions, int spe10_scale, int slice,
        bool metis_partition, double proc_part_ubal,
        const mfem::Array<int>& coarsening_factor);

    double Lx, Ly, Lz, Hx, Hy, Hz;
    mfem::ParMesh* pmesh_;
    mfem::VectorFunctionCoefficient* kinv_;
    GCoefficient* source_coeff_;
    std::vector<int> num_procs_xyz_;
};

SPE10Problem::SPE10Problem(const char* permFile, int nDimensions,
                           int spe10_scale, int slice, bool metis_partition, double proc_part_ubal,
                           const mfem::Array<int>& coarsening_factor)
{
    Init(permFile, nDimensions, spe10_scale, slice, metis_partition, proc_part_ubal,
         coarsening_factor);
}

void SPE10Problem::Init(
    const char* permFile, int nDimensions, int spe10_scale, int slice,
    bool metis_partition, double proc_part_ubal, const mfem::Array<int>& coarsening_factor)
{
    int num_procs, myid;
    MPI_Comm comm = MPI_COMM_WORLD;
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
    if (permFile != NULL && (std::strcmp(permFile, "") == 0))
    {
        IPF::BlankPermeability();
    }
    else
    {
        IPF::ReadPermeabilityFile(permFile, MPI_COMM_WORLD);
    }

    if (nDimensions == 2)
        IPF::Set2DSlice(IPF::XY, slice);

    kinv_ = new mfem::VectorFunctionCoefficient(
        nDimensions, IPF::InversePermeability);

    if (nDimensions == 3)
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
        auto elem_elem = TableToMatrix(mesh->ElementToElementTable());

        mfem::Array<int> partition;
        MetisGraphPartitioner partitioner;
        partitioner.setUnbalanceTol(proc_part_ubal);
        partitioner.doPartition(elem_elem, num_procs, partition);

        pmesh_  = new mfem::ParMesh(comm, *mesh, partition);

        assert(partition.Max() + 1 == num_procs);
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

} // namespace smoothg

