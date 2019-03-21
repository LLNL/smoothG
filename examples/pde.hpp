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
   @file pde.hpp
   @brief Implementation of some partial differential equation problems.

   Reads data from file and creates the appropriate finite element/volume structures.
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
*/
mfem::SparseMatrix GenerateBoundaryAttributeTable(const mfem::Mesh* mesh)
{
    int nedges = mesh->Dimension() == 2 ? mesh->GetNEdges() : mesh->GetNFaces();
    int nbdr_edges = mesh->GetNBE();

    int* I = new int[nedges + 1]();
    int* J = new int[nbdr_edges];

    for (int j = 0; j < nbdr_edges; ++j)
    {
        I[mesh->GetBdrElementEdgeIndex(j) + 1]++ ;
    }

    std::partial_sum(I, I + nedges + 1, I);

    for (int j = 0; j < nbdr_edges; ++j)
    {
        J[I[mesh->GetBdrElementEdgeIndex(j)]] = mesh->GetBdrAttribute(j) - 1;
    }

    double* Data = new double[nbdr_edges];
    std::fill_n(Data, nbdr_edges, 1.0);

    return mfem::SparseMatrix(I, J, Data, nedges, mesh->bdr_attributes.Max());
}

/**
   @brief A utility class for working with the SPE10 or Egg model data set.

   The SPE10 data set can be found at: http://www.spe.org/web/csp/datasets/set02.htm
*/
class InversePermeabilityCoefficient : public mfem::VectorCoefficient
{
public:
    enum SliceOrientation {NONE, XY, XZ, YZ};

    /**
       @brief MFEM Coefficient constructed from permeability data set
       @param comm MPI communicator
       @param fileName file name of permeability data set
       @param N number of data in each direction intended to store from data set
       @param max_N data set size in each direction.
              For SPE10, it should be {60, 220, 85}
              For egg model, it should be {60, 60, 7}
       @param h element size in each direction
       @param orientation if NONE (default), full data set will be read (3D);
              otherwise, it tells which 2D plane {XY, XZ, or YZ} to read
       @param slice which slice of the selected 2D plane to read
    */
    InversePermeabilityCoefficient(MPI_Comm comm,
                                   const std::string& fileName,
                                   const mfem::Array<int>& N,
                                   const mfem::Array<int>& max_N,
                                   const mfem::Vector& h,
                                   SliceOrientation orientation = NONE,
                                   int slice = -1);

    virtual void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
                      const mfem::IntegrationPoint& ip)
    {
        mfem::Vector transip(vdim);
        T.Transform(ip, transip);
        InversePermeability(transip, V);
    }

    /// Inverse of Frobenius norm of the inverse permeability
    double InvNorm2(const mfem::Vector& x);
private:
    void ReadPermeabilityFile(const std::string& fileName,
                              const mfem::Array<int>& max_N);
    void ReadPermeabilityFile(MPI_Comm comm, const std::string& fileName,
                              const mfem::Array<int>& max_N);
    void BlankPermeability();
    void InversePermeability(const mfem::Vector& x, mfem::Vector& val);

    mfem::Array<int> N_;
    mfem::Vector h_;
    int slice_;
    SliceOrientation orientation_;

    int N_slice_;
    int N_all_;
    std::vector<double> inverse_permeability_;
};

InversePermeabilityCoefficient::InversePermeabilityCoefficient(
    MPI_Comm comm, const std::string& file_name, const mfem::Array<int>& N,
    const mfem::Array<int>& max_N, const mfem::Vector& h,
    SliceOrientation orientation, int slice)
    : mfem::VectorCoefficient(orientation == NONE ? 3 : 2), h_(h), slice_(slice),
      orientation_(orientation), N_slice_(N[0] * N[1]), N_all_(N_slice_ * N[2]),
      inverse_permeability_(3 * N_all_)
{
    assert(N.Size() == max_N.Size());
    for (int i = 0; i < N.Size(); ++i)
    {
        assert(N[i] <= max_N[i]);
    }

    N.Copy(N_);

    if (file_name == "")
    {
        BlankPermeability();
        return;
    }
    ReadPermeabilityFile(comm, file_name, max_N);
}

void InversePermeabilityCoefficient::ReadPermeabilityFile(const std::string& fileName,
                                                          const mfem::Array<int>& max_N)
{
    std::ifstream permfile(fileName.c_str());

    if (!permfile.is_open())
    {
        std::cerr << "Error in opening file " << fileName << std::endl;
        mfem::mfem_error("File does not exist");
    }

    double* ip = inverse_permeability_.data();
    double tmp;
    for (int l = 0; l < 3; l++)
    {
        for (int k = 0; k < N_[2]; k++)
        {
            for (int j = 0; j < N_[1]; j++)
            {
                for (int i = 0; i < N_[0]; i++)
                {
                    permfile >> *ip;
                    *ip = 1. / (*ip);
                    ip++;
                }
                for (int i = 0; i < max_N[0] - N_[0]; i++)
                    permfile >> tmp; // skip unneeded part
            }
            for (int j = 0; j < max_N[1] - N_[1]; j++)
                for (int i = 0; i < max_N[0]; i++)
                    permfile >> tmp;  // skip unneeded part
        }

        if (l < 2) // if not processing Kz, skip unneeded part
            for (int k = 0; k < max_N[2] - N_[2]; k++)
                for (int j = 0; j < max_N[1]; j++)
                    for (int i = 0; i < max_N[0]; i++)
                        permfile >> tmp;
    }
}

void InversePermeabilityCoefficient::ReadPermeabilityFile(MPI_Comm comm,
                                                          const std::string& fileName,
                                                          const mfem::Array<int>& max_N)
{
    int myid;
    MPI_Comm_rank(comm, &myid);

    mfem::StopWatch chrono;

    chrono.Start();
    if (myid == 0)
        ReadPermeabilityFile(fileName, max_N);
    chrono.Stop();

    if (myid == 0)
        std::cout << "Permeability file read in " << chrono.RealTime() << ".s \n";

    chrono.Clear();

    chrono.Start();
    MPI_Bcast(inverse_permeability_.data(), 3 * N_all_, MPI_DOUBLE, 0, comm);
    chrono.Stop();

    if (myid == 0)
        std::cout << "Permeability field distributed in " << chrono.RealTime() << ".s \n";

}

void InversePermeabilityCoefficient::BlankPermeability()
{
    std::fill(inverse_permeability_.begin(), inverse_permeability_.end(), 1.0);
}

void InversePermeabilityCoefficient::InversePermeability(const mfem::Vector& x,
                                                         mfem::Vector& val)
{
    val.SetSize(x.Size());

    unsigned int i = 0, j = 0, k = 0;

    switch (orientation_)
    {
        case NONE:
            i = N_[0] - 1 - (int)floor(x[0] / h_[0] / (1. + 3e-16));
            j = (int)floor(x[1] / h_[1] / (1. + 3e-16));
            k = N_[2] - 1 - (int)floor(x[2] / h_[2] / (1. + 3e-16));
            break;
        case XY:
            i = N_[0] - 1 - (int)floor(x[0] / h_[0] / (1. + 3e-16));
            j = (int)floor(x[1] / h_[1] / (1. + 3e-16));
            k = slice_;
            break;
        case XZ:
            i = N_[0] - 1 - (int)floor(x[0] / h_[0] / (1. + 3e-16));
            j = slice_;
            k = N_[2] - 1 - (int)floor(x[2] / h_[2] / (1. + 3e-16));
            break;
        case YZ:
            i = slice_;
            j = (int)floor(x[1] / h_[1] / (1. + 3e-16));
            k = N_[2] - 1 - (int)floor(x[2] / h_[2] / (1. + 3e-16));
            break;
        default:
            mfem::mfem_error("InversePermeabilityCoefficient::InversePermeability");
    }

    const int offset = N_slice_ * k + N_[0] * j + i;
    for (int l = 0; l < vdim; ++l)
    {
        val[l] = inverse_permeability_[offset + N_all_ * l];
    }
}

double InversePermeabilityCoefficient::InvNorm2(const mfem::Vector& x)
{
    mfem::Vector val(3);
    InversePermeability(x, val);
    return 1.0 / val.Norml2();
}

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
 @brief compute transmissibility based on two-point flux approximation

 The implementation is based on (Sec. 4.4.1 of) the preliminary version of

 K.-A. Lie. An introduction to reservoir simulation using MATLAB/GNU Octave
 **/
class LocalTPFA
{
    mfem::Coefficient* Q_;
    mfem::VectorCoefficient* VQ_;
    const mfem::ParMesh& mesh_;
    const mfem::SparseMatrix& vert_edge_;

    mfem::DenseMatrix EvalKappa(int i);
    mfem::Vector ComputeShapeCenter(const mfem::Element& el);
    mfem::Vector ComputeLocalWeight(int i);
public:
    LocalTPFA(const mfem::ParMesh& mesh, const mfem::SparseMatrix& vert_edge)
        : Q_(NULL), VQ_(NULL), mesh_(mesh), vert_edge_(vert_edge) { }

    /// @param q inverse of permeability \f$ kappa^{-1} \f$ in Darcy's law
    LocalTPFA(const mfem::ParMesh& mesh, const mfem::SparseMatrix& vert_edge,
              mfem::Coefficient& q)
        : Q_(&q), VQ_(NULL), mesh_(mesh), vert_edge_(vert_edge) { }

    LocalTPFA(const mfem::ParMesh& mesh, const mfem::SparseMatrix& vert_edge,
              mfem::VectorCoefficient& q)
        : Q_(NULL), VQ_(&q), mesh_(mesh), vert_edge_(vert_edge) { }

    /// Compute local edge weights for the corresponding graph Laplacian
    std::vector<mfem::Vector> ComputeLocalWeights()
    {
        std::vector<mfem::Vector> local_weights(mesh_.GetNE());
        for (int i = 0; i < mesh_.GetNE(); ++i)
        {
            local_weights[i] = ComputeLocalWeight(i);
        }
        return local_weights;
    }
};

mfem::DenseMatrix LocalTPFA::EvalKappa(int i)
{
    const int dim = mesh_.Dimension();
    const mfem::Element& el = *(mesh_.GetElement(i));
    auto trans = const_cast<mfem::ParMesh&>(mesh_).GetElementTransformation(i);
    const auto& ip = mfem::IntRules.Get(el.GetType(), 1).IntPoint(0);

    mfem::DenseMatrix kappa(dim);

    // Note that Q is kappa^{-1}
    if (VQ_)
    {
        mfem::Vector vq(dim);
        VQ_->Eval(vq, *trans, ip);
        for (int d = 0; d < dim; ++d)
            kappa(d, d) = 1.0 / vq(d);
    }
    else if (Q_)
    {
        double sq = Q_->Eval(*trans, ip);
        for (int d = 0; d < dim; ++d)
            kappa(d, d) = 1.0 / sq;
    }
    else
    {
        for (int d = 0; d < dim; ++d)
            kappa(d, d) = 1.0;
    }

    return kappa;
}

mfem::Vector LocalTPFA::ComputeShapeCenter(const mfem::Element& el)
{
    const int dim = mesh_.Dimension();
    const int num_verts = el.GetNVertices();

    mfem::Vector center(dim);
    center = 0.0;

    for (int v = 0; v < num_verts; ++v)
    {
        const double* vert_coord = mesh_.GetVertex(el.GetVertices()[v]);
        for (int d = 0; d < dim; ++d)
        {
            center[d] += vert_coord[d];
        }
    }
    return center /= num_verts;
}

mfem::Vector LocalTPFA::ComputeLocalWeight(int i)
{
    const int dim = mesh_.Dimension();
    const int num_facets = vert_edge_.RowSize(i);
    const mfem::DenseMatrix kappa = EvalKappa(i);
    const mfem::Vector cell_center = ComputeShapeCenter(*(mesh_.GetElement(i)));

    mfem::Array<int> edges;
    GetTableRow(vert_edge_, i, edges);

    mfem::ParMesh& non_const_mesh = const_cast<mfem::ParMesh&>(mesh_);
    mfem::Vector normal(dim), c_vector(dim);
    mfem::Vector local_weight(num_facets);

    for (int e = 0; e < num_facets; ++e)
    {
        auto trans = non_const_mesh.GetFaceTransformation(edges[e]);
        auto& ip = mfem::IntRules.Get(trans->GetGeometryType(), 1).IntPoint(0);
        trans->SetIntPoint(&ip);

        double facet_measure = ip.weight * trans->Weight();
        assert(facet_measure > 0);

        mfem::CalcOrtho(trans->Jacobian(), normal);
        normal /= normal.Norml2();   // make it unit normal

        auto facet_ceter = ComputeShapeCenter(*(mesh_.GetFace(edges[e])));

        for (int d = 0; d < dim; ++d)
        {
            c_vector[d] = facet_ceter[d] - cell_center[d];
        }

        // Note that edge weight is inverse of M
        double delta_x = c_vector.Norml2();
        double nkc = std::fabs(kappa.InnerProduct(normal, c_vector));
        local_weight(e) = (facet_measure * nkc) / (delta_x * delta_x);
    }

    return local_weight;
}

/**
   @brief Darcy's flow problem discretized in finite volume (TPFA)

   Abstract class serves as interface between graph and finite volume problem.
   It produces the weighted graph, partition, right hand side of an FV problem,
   and it can visualize coefficient vectors in vertex space.
*/
class DarcyProblem
{
public:
    /**
       @brief Abstract constructor, actual construction is done in derived class
       @param comm MPI communicator
       @param num_dims number of dimensions of underlying physical space
       @param ess_attr marker for boundary attributes where essential edge
              condition is imposed
    */
    DarcyProblem(MPI_Comm comm, int num_dims, const mfem::Array<int>& ess_attr);

    /**
       @brief Construct an FV problem assuming constant 1 permeability
       @param pmesh mesh where the finite volume problem is defined on
       @param ess_attr marker for boundary attributes where essential edge
              condition is imposed
    */
    DarcyProblem(const mfem::ParMesh& pmesh, const mfem::Array<int>& ess_attr);

    /**
       @param use_local_weight whether to store "element" weight
       @return weighted graph associated with the finite volume problem
    */
    Graph GetFVGraph(bool use_local_weight = false);

    /// Getter for vertex-block right hand side
    const mfem::Vector& GetVertexRHS() const
    {
        return rhs_u_;
    }

    /// Getter for edge-block right hand side
    const mfem::Vector& GetEdgeRHS() const
    {
        return rhs_sigma_;
    }

    /// Volume of a cell in the mesh (assuming all cells have the same volume)
    double CellVolume() const
    {
        assert(pmesh_);
        return pmesh_->GetElementVolume(0); // assumed uniform mesh
    }

    /// Save mesh with partitioning information (GLVis can separate partitions)
    void PrintMeshWithPartitioning(mfem::Array<int>& partition);

    /// Setup visualization of vertex space vector
    void VisSetup(mfem::socketstream& vis_v, mfem::Vector& vec, double range_min = 0.0,
                  double range_max = 0.0, const std::string& caption = "", int coef = 0) const;

    /// Update visualization of vertex space vector, VisSetup needs to be called first
    void VisUpdate(mfem::socketstream& vis_v, mfem::Vector& vec) const;

    /// Save plot of sol (in vertex space)
    void SaveFigure(const mfem::Vector& sol, const std::string& name) const;

    /// Construct partitioning array for vertices
    void Partition(bool metis_parition, const mfem::Array<int>& coarsening_factors,
                   mfem::Array<int>& partitioning) const;
protected:
    void BuildReservoirGraph();
    void InitGraph();
    void ComputeGraphWeight(bool unit_weight = false);
    void CartPart(const mfem::Array<int>& coarsening_factor, mfem::Array<int>& partitioning) const;
    void MetisPart(const mfem::Array<int>& coarsening_factor, mfem::Array<int>& partitioning) const;

    unique_ptr<mfem::ParMesh> pmesh_;

    mfem::Array<int> num_procs_xyz_;
    mfem::RT_FECollection sigma_fec_;
    mfem::L2_FECollection u_fec_;
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

    mfem::Array<int> ess_attr_;

    mutable mfem::ParGridFunction u_fes_gf_;

    MPI_Comm comm_;
    int myid_;
    int num_procs_;
};

DarcyProblem::DarcyProblem(MPI_Comm comm, int num_dims, const mfem::Array<int>& ess_attr)
    : sigma_fec_(0, num_dims), u_fec_(0, num_dims), comm_(comm)
{
    MPI_Comm_size(comm_, &num_procs_);
    MPI_Comm_rank(comm_, &myid_);

    ess_attr.Copy(ess_attr_);
}

DarcyProblem::DarcyProblem(const mfem::ParMesh& pmesh, const mfem::Array<int>& ess_attr)
    : DarcyProblem(pmesh.GetComm(), pmesh.Dimension(), ess_attr)
{
    pmesh_ = make_unique<mfem::ParMesh>(pmesh, false);
    InitGraph();
    kinv_scalar_ = make_unique<mfem::ConstantCoefficient>(1.0);
    ComputeGraphWeight();
}

Graph DarcyProblem::GetFVGraph(bool use_local_weight)
{
    const mfem::HypreParMatrix& edge_trueedge = *sigma_fes_->Dof_TrueDof_Matrix();
    if (use_local_weight && local_weight_.size() > 0)
    {
        return Graph(vertex_edge_, edge_trueedge, local_weight_, &edge_bdratt_);
    }
    return Graph(vertex_edge_, edge_trueedge, weight_, &edge_bdratt_);
}

void DarcyProblem::BuildReservoirGraph()
{
    mfem::SparseMatrix edge_bdratt = GenerateBoundaryAttributeTable(pmesh_.get());
    edge_bdratt_.Swap(edge_bdratt);
    assert(edge_bdratt_.NumCols() == ess_attr_.Size());

    const mfem::Table& v_e_table = pmesh_->Dimension() == 2 ? pmesh_->ElementToEdgeTable()
                                   : pmesh_->ElementToFaceTable();
    mfem::SparseMatrix v_e = TableToMatrix(v_e_table);
    vertex_edge_.Swap(v_e);
}

void DarcyProblem::InitGraph()
{
    sigma_fes_ = make_unique<mfem::ParFiniteElementSpace>(pmesh_.get(), &sigma_fec_);
    u_fes_ = make_unique<mfem::ParFiniteElementSpace>(pmesh_.get(), &u_fec_);
    coeff_gf_ = make_unique<mfem::GridFunction>(u_fes_.get());

    BuildReservoirGraph();

    rhs_sigma_.SetSize(vertex_edge_.NumCols());
    rhs_u_.SetSize(vertex_edge_.NumRows());
    rhs_sigma_ = 0.0;
    rhs_u_ = 0.0;
}

void DarcyProblem::ComputeGraphWeight(bool unit_weight)
{
    if (unit_weight)
    {
        weight_.SetSize(vertex_edge_.NumCols());
        weight_ = 1.0;
        return;
    }

    // Compute local edge weights
    LocalTPFA local_TPFA(*pmesh_, vertex_edge_, *kinv_vector_);
    local_weight_ = local_TPFA.ComputeLocalWeights();

    // Assemble edge weights w_ij = (w_{ij,i}^{-1} + w_{ij,j}^{-1})^{-1}
    mfem::Array<int> edges;

    weight_.SetSize(vertex_edge_.NumCols());
    weight_ = 0.0;

    for (int i = 0; i < vertex_edge_.NumRows(); ++i)
    {
        GetTableRow(vertex_edge_, i, edges);
        for (int j = 0; j < edges.Size(); ++j)
        {
            weight_[edges[j]] += 1.0 / local_weight_[i][j];
        }
    }

    for (int i = 0; i < weight_.Size(); ++i)
    {
        assert(mfem::IsFinite(weight_[i]) && weight_[i] != 0.0);
        weight_[i] = 1.0 / weight_[i];
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
        vis_v << "valuerange " << range_min << " " << range_max << "\n";
    }

    if (pmesh_->SpaceDimension() == 2)
    {
        vis_v << "view 0 0\n"; // view from top
        vis_v << "keys jl\n";  // turn off perspective and light
        vis_v << "keys ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]\n";  // increase size
    }
    else
    {
        vis_v << "keys ]]]]]]]]]]]]]\n";  // increase size
    }

    vis_v << "keys c\n"; // colorbar

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

    vis_v << "parallel " << num_procs_ << " " << myid_ << "\n";
    vis_v << "solution\n" << *pmesh_ << u_fes_gf_;

    MPI_Barrier(comm_);

    vis_v << "keys S\n";         //Screenshot

    MPI_Barrier(comm_);
}

void DarcyProblem::SaveFigure(const mfem::Vector& sol, const std::string& name) const
{
    u_fes_gf_.MakeRef(u_fes_.get(), sol.GetData());
    {
        std::stringstream filename;
        filename << name << ".mesh";
        std::ofstream out(filename.str().c_str());
        pmesh_->Print(out);
    }
    {
        std::stringstream filename;
        filename << name << ".gridfunction";
        std::ofstream out(filename.str().c_str());
        u_fes_gf_.Save(out);
    }
}

void DarcyProblem::CartPart(const mfem::Array<int>& coarsening_factor,
                            mfem::Array<int>& partitioning) const
{
    const int SPE10_num_x_volumes = 60;
    const int SPE10_num_y_volumes = 220;
    const int SPE10_num_z_volumes = 85;

    const int nDimensions = num_procs_xyz_.Size();

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

void DarcyProblem::Partition(bool metis_parition,
                             const mfem::Array<int>& coarsening_factors,
                             mfem::Array<int>& partitioning) const
{
    if (metis_parition)
    {
        MetisPart(coarsening_factors, partitioning);
    }
    else
    {
        CartPart(coarsening_factors, partitioning);
    }
}

/**
   @brief Construct finite volume problem on the SPE10 data set
*/
class SPE10Problem : public DarcyProblem
{
public:
    /**
       @brief Constructor
       @param permFile file name
       @param nDimensions
       @param spe10_scale scale of problem size (1-5)
       @param slice
       @param metis_parition whether to call METIS/Cartesian partitioner
       @param ess_attr marker for boundary attributes where essential edge
              condition is imposed
       @param unit_weight whether set edge weight as unit weight (1.0)
    */
    SPE10Problem(const char* permFile, int nDimensions, int spe10_scale,
                 int slice, bool metis_parition,
                 const mfem::Array<int>& ess_attr, bool unit_weight = false);

    /// Setup a vector that equals initial_val in half of the domain (in y-direction)
    /// and -initial_val in the other half
    mfem::Vector InitialCondition(double initial_val) const;

private:
    void SetupMeshAndCoeff(const char* permFile, int nDimensions,
                           int spe10_scale, bool metis_partition, int slice);
    unique_ptr<mfem::ParMesh> MakeParMesh(mfem::Mesh& mesh, bool metis_partition);
    void MakeRHS();

    unique_ptr<GCoefficient> source_coeff_;
};

SPE10Problem::SPE10Problem(const char* permFile, int nDimensions, int spe10_scale,
                           int slice, bool metis_parition,
                           const mfem::Array<int>& ess_attr, bool unit_weight)
    : DarcyProblem(MPI_COMM_WORLD, nDimensions, ess_attr)
{
    SetupMeshAndCoeff(permFile, nDimensions, spe10_scale, metis_parition, slice);

    if (myid_ == 0)
    {
        std::cout << pmesh_->GetNEdges() << " fine edges, "
                  << pmesh_->GetNFaces() << " fine faces, "
                  << pmesh_->GetNE() << " fine elements\n";
    }

    InitGraph();
    ComputeGraphWeight(unit_weight);
    MakeRHS();
}

void SPE10Problem::SetupMeshAndCoeff(const char* permFile, int nDimensions,
                                     int spe10_scale, bool metis_partition, int slice)
{
    mfem::Array<int> max_N(3);
    max_N[0] = 60;
    max_N[1] = 220;
    max_N[2] = 85;

    mfem::Array<int> N(3);
    N[0] = 12 * spe10_scale; // 60
    N[1] = 44 * spe10_scale; // 220
    N[2] = 17 * spe10_scale; // 85

    // SPE10 grid cell sizes
    mfem::Vector h(3);
    h(0) = 20.0;
    h(1) = 10.0;
    h(2) = 2.0;

    const int Lx = N[0] * h(0);
    const int Ly = N[1] * h(1);
    const int Lz = N[2] * h(2);

    using IPC = InversePermeabilityCoefficient;
    IPC::SliceOrientation orient = nDimensions == 2 ? IPC::XY : IPC::NONE;
    kinv_vector_ = make_unique<IPC>(comm_, permFile, N, max_N, h, orient, slice);

    mfem::Array<int> coarsening_factor(nDimensions);
    coarsening_factor = 10;
    coarsening_factor.Last() = nDimensions == 3 ? 5 : 10;

    int Hx = coarsening_factor[0] * h(0);
    int Hy = coarsening_factor[1] * h(1);
    int Hz = 1.0;
    if (nDimensions == 3)
        Hz = coarsening_factor[2] * h(2);
    source_coeff_ = make_unique<GCoefficient>(Lx, Ly, Lz, Hx, Hy, Hz);

    if (nDimensions == 2)
    {
        mfem::Mesh mesh(N[0], N[1], mfem::Element::QUADRILATERAL, 1, Lx, Ly);
        pmesh_ = MakeParMesh(mesh, metis_partition);
        return;
    }
    mfem::Mesh mesh(N[0], N[1], N[2], mfem::Element::HEXAHEDRON, 1, Lx, Ly, Lz);
    pmesh_ = MakeParMesh(mesh, metis_partition);
}

unique_ptr<mfem::ParMesh> SPE10Problem::MakeParMesh(mfem::Mesh& mesh, bool metis_partition)
{
    mfem::Array<int> partition;
    if (metis_partition)
    {
        auto elem_elem = TableToMatrix(mesh.ElementToElementTable());
        smoothg::Partition(elem_elem, partition, num_procs_);
        assert(partition.Max() + 1 == num_procs_);
    }
    else
    {
        int num_procs_x = static_cast<int>(std::sqrt(num_procs_) + 0.5);
        while (num_procs_ % num_procs_x)
            num_procs_x -= 1;

        num_procs_xyz_.SetSize(mesh.SpaceDimension(), 1);
        num_procs_xyz_[0] = num_procs_x;
        num_procs_xyz_[1] = num_procs_ / num_procs_x;
        assert(num_procs_xyz_[0] * num_procs_xyz_[1] == num_procs_);

        partition.MakeRef(mesh.CartesianPartitioning(num_procs_xyz_), mesh.GetNE());
        partition.MakeDataOwner();
    }
    return make_unique<mfem::ParMesh>(comm_, mesh, partition);
}

void SPE10Problem::MakeRHS()
{
    bool no_flow_bc = true;
    for (auto attr : ess_attr_)
    {
        if (attr == 0)
        {
            no_flow_bc = false;
            break;
        }
    }

    if (no_flow_bc)
    {
        rhs_sigma_ = 0.0;

        mfem::LinearForm q(u_fes_.get());
        q.AddDomainIntegrator(new mfem::DomainLFIntegrator(*source_coeff_));
        q.Assemble();
        rhs_u_ = q;
    }
    else
    {
        mfem::Array<int> nat_negative_one(ess_attr_.Size());
        nat_negative_one = 0;
        nat_negative_one[0] = 1;

        mfem::ConstantCoefficient negative_one(-1.0);
        mfem::RestrictedCoefficient pinflow_coeff(negative_one, nat_negative_one);

        mfem::LinearForm g(sigma_fes_.get());
        g.AddBoundaryIntegrator(
            new mfem::VectorFEBoundaryFluxLFIntegrator(pinflow_coeff));
        g.Assemble();
        rhs_sigma_ = g;

        rhs_u_ = 0.0;
    }
}

mfem::Vector SPE10Problem::InitialCondition(double initial_val) const
{
    HalfCoeffecient half(initial_val);
    mfem::GridFunction init(u_fes_.get());
    init.ProjectCoefficient(half);

    return init;
}

} // namespace smoothg

