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
   @brief A utility class for working with the SPE10 or Egg model data set.

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
   @brief Darcy's flow problem discretized in finite volume (TPFA)
*/
class DarcyProblem
{
public:
    DarcyProblem(MPI_Comm comm, const mfem::Array<int>& ess_v_attr);
    DarcyProblem(const mfem::ParMesh& pmesh, const mfem::Array<int> &ess_v_attr);

    Graph GetFVGraph(bool use_local_weight = false);

    const mfem::Vector& GetVertexRHS() const
    {
        return rhs_u_;
    }
    const std::vector<mfem::Vector>& GetLocalWeight() const
    {
        return local_weight_;
    }
    double CellVolume() const
    {
        assert(pmesh_);
        return pmesh_->GetElementVolume(0);
    }
    const mfem::Array<int>& GetEssentialVertDofs() const
    {
        return ess_vdofs_;
    }
    void PrintMeshWithPartitioning(mfem::Array<int>& partition);
    void VisSetup(mfem::socketstream& vis_v, mfem::Vector& vec, double range_min = 0,
                  double range_max = 0, const std::string& caption = "", int coef = 0) const;
    void VisUpdate(mfem::socketstream& vis_v, mfem::Vector& vec) const;
    void CartPart(const mfem::Array<int>& coarsening_factor, mfem::Array<int>& partitioning) const;
    void MetisPart(const mfem::Array<int>& coarsening_factor, mfem::Array<int>& partitioning) const;
protected:
    void BuildReservoirGraph();
    void InitGraph();
    void ComputeGraphWeight();

    unique_ptr<mfem::ParMesh> pmesh_;

    std::vector<int> num_procs_xyz_;
    unique_ptr<mfem::RT_FECollection> sigma_fec_;
    unique_ptr<mfem::L2_FECollection> u_fec_;
    unique_ptr<mfem::ParFiniteElementSpace> sigma_fes_;
    unique_ptr<mfem::ParFiniteElementSpace> u_fes_;

    unique_ptr<mfem::GridFunction> coeff_gf_;

    mfem::SparseMatrix vertex_edge_;
    mfem::SparseMatrix edge_bdratt_;

    mfem::Vector weight_;
    std::vector<mfem::Vector> local_weight_;

    unique_ptr<mfem::VectorCoefficient> kinv_vector_;
    unique_ptr<mfem::Coefficient> kinv_scalar_;

    mfem::Vector rhs_sigma_;
    mfem::Vector rhs_u_;

    mfem::Array<int> ess_v_attr_;
    mfem::Array<int> ess_vdofs_;
    int num_ess_vdof_;

    mutable mfem::ParGridFunction u_fes_gf_;

    MPI_Comm comm_;
    int myid_;
    int num_procs_;
};

DarcyProblem::DarcyProblem(MPI_Comm comm, const mfem::Array<int> &ess_v_attr)
    : num_ess_vdof_(0), comm_(comm)
{
    MPI_Comm_size(comm_, &num_procs_);
    MPI_Comm_rank(comm_, &myid_);

    ess_v_attr.Copy(ess_v_attr_);
}

DarcyProblem::DarcyProblem(const mfem::ParMesh& pmesh, const mfem::Array<int> &ess_v_attr)
    : DarcyProblem(pmesh.GetComm(), ess_v_attr)
{
    pmesh_ = make_unique<mfem::ParMesh>(pmesh, false);
    InitGraph();
    kinv_scalar_ = make_unique<mfem::ConstantCoefficient>(1.0);
    ComputeGraphWeight();
}

Graph DarcyProblem::GetFVGraph(bool use_local_weight)
{
    if (use_local_weight)
    {
        std::cout << "use_local_weight is currently not supported! \n";
    }
    return Graph(vertex_edge_, *sigma_fes_->Dof_TrueDof_Matrix(), weight_, &edge_bdratt_);
}

// Keep only boundary faces associated with essential pressure condition
// For these faces, add the associated attribute as a (ghost) element
void DarcyProblem::BuildReservoirGraph()
{
    mfem::SparseMatrix edge_bdratt = GenerateBoundaryAttributeTable(pmesh_.get());
    edge_bdratt_.Swap(edge_bdratt);
    assert(edge_bdratt_.NumCols() == ess_v_attr_.Size());

    const mfem::Table& v_e_table = pmesh_->Dimension() == 2 ?
                pmesh_->ElementToEdgeTable() : pmesh_->ElementToFaceTable();
    mfem::SparseMatrix v_e = TableToMatrix(v_e_table);
    vertex_edge_.Swap(v_e);
}

void DarcyProblem::InitGraph()
{
    sigma_fec_ = make_unique<mfem::RT_FECollection>(0, pmesh_->SpaceDimension());
    sigma_fes_ = make_unique<mfem::ParFiniteElementSpace>(pmesh_.get(), sigma_fec_.get());

    u_fec_ = make_unique<mfem::L2_FECollection>(0, pmesh_->SpaceDimension());
    u_fes_ = make_unique<mfem::ParFiniteElementSpace>(pmesh_.get(), u_fec_.get());
    coeff_gf_ = make_unique<mfem::GridFunction>(u_fes_.get());

    BuildReservoirGraph();

    rhs_sigma_.SetSize(vertex_edge_.NumCols());
    rhs_u_.SetSize(vertex_edge_.NumRows());
    rhs_sigma_ = 0.0;
    rhs_u_ = 0.0;
}

void DarcyProblem::ComputeGraphWeight()
{
    // Construct "finite volume mass" matrix
    mfem::ParBilinearForm a(sigma_fes_.get());
    if (kinv_vector_)
    {
        assert(kinv_scalar_ == nullptr);
        a.AddDomainIntegrator(new FiniteVolumeMassIntegrator(*kinv_vector_));
    }
    else
    {
        assert(kinv_scalar_);
        a.AddDomainIntegrator(new FiniteVolumeMassIntegrator(*kinv_scalar_));
    }

    // Compute element mass matrices, assemble mass matrix and edge weight
    a.ComputeElementMatrices();
    a.Assemble();
    a.Finalize();
    a.SpMat().GetDiag(weight_);

    for (int i = 0; i < weight_.Size(); ++i)
    {
        assert(mfem::IsFinite(weight_[i]) && weight_[i] != 0.0);
        weight_[i] = 1.0 / weight_[i];
    }

    // Store element mass matrices and local edge weights
    local_weight_.resize(vertex_edge_.NumRows());
    mfem::DenseMatrix M_el_i;
    for (int i = 0; i < pmesh_->GetNE(); i++)
    {
        a.ComputeElementMatrix(i, M_el_i);

        mfem::Vector& local_weight_i = local_weight_[i];
        local_weight_i.SetSize(M_el_i.Height());

        for (int j = 0; j < M_el_i.Height(); j++)
        {
            local_weight_i[j] = 1.0 / M_el_i(j, j);
        }
    }
}

void DarcyProblem::PrintMeshWithPartitioning(mfem::Array<int>& partition)
{
    std::stringstream fname;
    fname << "mesh0.mesh." << std::setfill('0') << std::setw(6) << myid_;
    std::ofstream ofid(fname.str().c_str());
    ofid.precision(8);
    pmesh_->PrintWithPartitioning(partition.GetData(), ofid, 1);
}

void DarcyProblem::VisSetup(mfem::socketstream& vis_v, mfem::Vector& vec, double range_min,
                            double range_max, const std::string& caption, int coef) const
{
    u_fes_gf_.MakeRef(u_fes_.get(), vec.GetData());

    const char vishost[] = "localhost";
    const int  visport   = 19916;
    vis_v.open(vishost, visport);
    vis_v.precision(8);

    vis_v << "parallel " << num_procs_ << " " << myid_ << "\n";
    vis_v << "solution\n" << *pmesh_ << u_fes_gf_;
    vis_v << "window_size 500 800\n";
    vis_v << "window_title 'vertex space unknown'\n";
    vis_v << "autoscale off\n"; // update value-range; keep mesh-extents fixed
    if (range_max > range_min)
    {
        vis_v << "valuerange " << range_min << " " << range_max <<
                 "\n"; // update value-range; keep mesh-extents fixed
    }

    if (pmesh_->SpaceDimension() == 2)
    {
        vis_v << "view 0 0\n"; // view from top
        vis_v << "keys ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]\n";  // increase size
    }

    vis_v << "keys cjl\n"; // colorbar, perspective, and light

    if (coef)
    {
        vis_v << "keys fL\n";  // smoothing and logarithmic scale
    }

    if (!caption.empty())
    {
        vis_v << "plot_caption '" << caption << "'\n";
    }

    MPI_Barrier(comm_);

    vis_v << "keys S\n";         //Screenshot

    MPI_Barrier(comm_);
}

void DarcyProblem::VisUpdate(mfem::socketstream& vis_v, mfem::Vector& vec) const
{
    u_fes_gf_.MakeRef(u_fes_.get(), vec.GetData());

    vis_v << "parallel " << pmesh_->GetNRanks() << " " << myid_ << "\n";
    vis_v << "solution\n" << *pmesh_ << u_fes_gf_;

    MPI_Barrier(pmesh_->GetComm());

    vis_v << "keys S\n";         //Screenshot

    MPI_Barrier(pmesh_->GetComm());
}

void DarcyProblem::CartPart(const mfem::Array<int>& coarsening_factor,
                            mfem::Array<int>& partitioning) const
{
    const int SPE10_num_x_volumes = 60;
    const int SPE10_num_y_volumes = 220;
    const int SPE10_num_z_volumes = 85;

    const int nDimensions = num_procs_xyz_.size();

    mfem::Array<int> nxyz(nDimensions);
    nxyz[0] = SPE10_num_x_volumes / num_procs_xyz_[0] / coarsening_factor[0];
    nxyz[1] = SPE10_num_y_volumes / num_procs_xyz_[1] / coarsening_factor[1];
    if (nDimensions == 3)
        nxyz[2] = SPE10_num_z_volumes / num_procs_xyz_[2] / coarsening_factor[2];

    for (int& i : nxyz)
    {
        i = std::max(1, i);
    }

    mfem::Array<int> cart_part(pmesh_->CartesianPartitioning(nxyz), pmesh_->GetNE());
    cart_part.MakeDataOwner();
    partitioning.Append(cart_part);
}

void DarcyProblem::MetisPart(const mfem::Array<int>& coarsening_factor,
                             mfem::Array<int>& partitioning) const
{
    mfem::DiscreteLinearOperator DivOp(sigma_fes_.get(), u_fes_.get());
    DivOp.AddDomainInterpolator(new mfem::DivergenceInterpolator);
    DivOp.Assemble();
    DivOp.Finalize();

    int metis_coarsening_factor = 1;
    for (const auto factor : coarsening_factor)
        metis_coarsening_factor *= factor;

    PartitionAAT(DivOp.SpMat(), partitioning, metis_coarsening_factor);
}

class SPE10Problem : public DarcyProblem
{
public:
    SPE10Problem(const char* permFile, int nDimensions, int spe10_scale,
                 int slice, bool metis_parition, const mfem::Array<int>& ess_attr);

    ~SPE10Problem()
    {
        InversePermeabilityFunction::ClearMemory();
    }

private:
    void SetupMeshAndCoeff(const char* permFile, int nDimensions,
                           int spe10_scale, bool metis_partition, int slice);
    void MakeRHS();

    unique_ptr<GCoefficient> source_coeff_;
};

SPE10Problem::SPE10Problem(const char* permFile, int nDimensions, int spe10_scale,
                           int slice, bool metis_parition, const mfem::Array<int>& ess_attr)
    : DarcyProblem(MPI_COMM_WORLD, ess_attr)
{
    SetupMeshAndCoeff(permFile, nDimensions, spe10_scale, metis_parition, slice);
    InitGraph();
    ComputeGraphWeight();
    MakeRHS();
}

void SPE10Problem::SetupMeshAndCoeff(const char* permFile, int nDimensions,
                                     int spe10_scale, bool metis_partition, int slice)
{
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
    if (nDimensions == 2)
    {
        mesh = make_unique<mfem::Mesh>(N[0], N[1], mfem::Element::QUADRILATERAL,
                                       1, h(0) * N[0], h(1) * N[1]);
    }
    else
    {
        mesh = make_unique<mfem::Mesh>(N[0], N[1], N[2], mfem::Element::HEXAHEDRON,
                                       1, h(0) * N[0], h(1) * N[1], h(2) * N[2]);
    }

    mfem::Array<int> partition;
    if (metis_partition)
    {
        auto elem_elem = TableToMatrix(mesh->ElementToElementTable());
        Partition(elem_elem, partition, num_procs_);
        assert(partition.Max() + 1 == num_procs_);
    }
    else
    {
        int num_procs_x = static_cast<int>(std::sqrt(num_procs_) + 0.5);
        while (num_procs_ % num_procs_x)
            num_procs_x -= 1;

        num_procs_xyz_.resize(nDimensions, 1);
        num_procs_xyz_[0] = num_procs_x;
        num_procs_xyz_[1] = num_procs_ / num_procs_x;

        int nparts = 1;
        for (int d = 0; d < nDimensions; d++)
            nparts *= num_procs_xyz_[d];
        assert(nparts == num_procs_);

        partition.MakeRef(mesh->CartesianPartitioning(num_procs_xyz_.data()), mesh->GetNE());
        partition.MakeDataOwner();
    }
    pmesh_ = make_unique<mfem::ParMesh>(comm_, partition);

    if (myid_ == 0)
    {
        std::cout << pmesh_->GetNEdges() << " fine edges, "
                  << pmesh_->GetNFaces() << " fine faces, "
                  << pmesh_->GetNE() << " fine elements\n";
    }

    using IPF = InversePermeabilityFunction;
    IPF::SetNumberCells(N[0], N[1], N[2]);
    IPF::SetMeshSizes(h(0), h(1), h(2));
    if (nDimensions == 2)
    {
        IPF::Set2DSlice(IPF::XY, slice);
    }
    IPF::ReadPermeabilityFile(permFile, comm_);
    kinv_vector_ = make_unique<mfem::VectorFunctionCoefficient>(nDimensions, IPF::InversePermeability);

    mfem::Array<int> coarsening_factor(nDimensions);
    coarsening_factor = 10;
    coarsening_factor.Last() = nDimensions == 3 ? 5 : 10;

    int Lx = N[0] * h(0);
    int Ly = N[1] * h(1);
    int Lz = N[2] * h(2);
    int Hx = coarsening_factor[0] * h(0);
    int Hy = coarsening_factor[1] * h(1);
    int Hz = 1.0;
    if (nDimensions == 3)
        Hz = coarsening_factor[2] * h(2);
    source_coeff_ = make_unique<GCoefficient>(Lx, Ly, Lz, Hx, Hy, Hz);
}

void SPE10Problem::MakeRHS()
{
    mfem::LinearForm q(u_fes_.get());
    q.AddDomainIntegrator(new mfem::DomainLFIntegrator(*source_coeff_) );
    q.Assemble();
    rhs_u_ = q;
}

} // namespace smoothg

