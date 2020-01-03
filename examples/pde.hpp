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

    /// Frobenius norm of permeability
    double FroNorm(const mfem::Vector& x);
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

void InversePermeabilityCoefficient::ReadPermeabilityFile(
    MPI_Comm comm, const std::string& fileName, const mfem::Array<int>& max_N)
{
    int myid;
    MPI_Comm_rank(comm, &myid);

    if (myid == 0)
        ReadPermeabilityFile(fileName, max_N);

    MPI_Bcast(inverse_permeability_.data(), 3 * N_all_, MPI_DOUBLE, 0, comm);
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
            i = (int)floor(x[0] / h_[0] / (1. + 3e-16));
            j = (int)floor(x[1] / h_[1] / (1. + 3e-16));
            k = (int)floor(x[2] / h_[2] / (1. + 3e-16));
            break;
        case XY:
            i = (int)floor(x[0] / h_[0] / (1. + 3e-16));
            j = (int)floor(x[1] / h_[1] / (1. + 3e-16));
            k = slice_;
            break;
        case XZ:
            i = (int)floor(x[0] / h_[0] / (1. + 3e-16));
            j = slice_;
            k = (int)floor(x[2] / h_[2] / (1. + 3e-16));
            break;
        case YZ:
            i = slice_;
            j = (int)floor(x[1] / h_[1] / (1. + 3e-16));
            k = (int)floor(x[2] / h_[2] / (1. + 3e-16));
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

double InversePermeabilityCoefficient::FroNorm(const mfem::Vector& x)
{
    mfem::Vector val(3);
    InversePermeability(x, val);

    for (int i = 0; i < val.Size(); ++i)
    {
        val[i] = 1.0 / val[i];
    }

    return val.Norml2();
}

/**
   @brief A forcing function that is supposed to very roughly represent some wells
   that are resolved on the *coarse* level.

   The forcing function is 1 on the top-right coarse cell, and -1 on the
   bottom-left coarse cell, and 0 elsewhere.

   @param Lx length of entire domain in x direction
   @param Hx size in x direction of a coarse cell.
*/
class GCoefficient : public mfem::Coefficient
{
public:
    GCoefficient(double Lx, double Ly, double Hx, double Hy);
    double Eval(mfem::ElementTransformation& T,
                const mfem::IntegrationPoint& ip);
private:
    double Lx_, Ly_;
    double Hx_, Hy_;
};

GCoefficient::GCoefficient(double Lx, double Ly, double Hx, double Hy)
    :
    Lx_(Lx),
    Ly_(Ly),
    Hx_(Hx),
    Hy_(Hy)
{
}

double GCoefficient::Eval(mfem::ElementTransformation& T,
                          const mfem::IntegrationPoint& ip)
{
    double dx[3];
    mfem::Vector transip(dx, 3);

    T.Transform(ip, transip);

    if ((transip(0) > (Lx_ - Hx_)) && (transip(1) > (Ly_ - Hy_)))
        return 1.0;
    else if ((transip(0) < Hx_) && (transip(1) < Hy_))
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

    mfem::DenseMatrix EvalKappa(int i);
    mfem::Vector ComputeShapeCenter(const mfem::Element& el);
    mfem::Vector ComputeLocalWeight(int i);
public:
    LocalTPFA(const mfem::ParMesh& mesh)
        : Q_(NULL), VQ_(NULL), mesh_(mesh) { }

    /// @param q inverse of permeability \f$ kappa^{-1} \f$ in Darcy's law
    LocalTPFA(const mfem::ParMesh& mesh, mfem::Coefficient& q)
        : Q_(&q), VQ_(NULL), mesh_(mesh) { }

    LocalTPFA(const mfem::ParMesh& mesh, mfem::VectorCoefficient& q)
        : Q_(NULL), VQ_(&q), mesh_(mesh) { }

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
    auto& elem_face = dim > 2 ? mesh_.ElementToFaceTable() : mesh_.ElementToEdgeTable();
    const int num_faces = elem_face.RowSize(i);
    const mfem::DenseMatrix kappa = EvalKappa(i);
    const mfem::Vector cell_center = ComputeShapeCenter(*(mesh_.GetElement(i)));
    const int* faces = elem_face.GetRow(i);

    mfem::ParMesh& non_const_mesh = const_cast<mfem::ParMesh&>(mesh_);
    mfem::Vector normal(dim), c_vector(dim);
    mfem::Vector local_weight(num_faces);

    for (int f = 0; f < num_faces; ++f)
    {
        auto trans = non_const_mesh.GetFaceTransformation(faces[f]);
        auto& ip = mfem::IntRules.Get(trans->GetGeometryType(), 1).IntPoint(0);
        trans->SetIntPoint(&ip);

        double face_measure = ip.weight * trans->Weight();
        assert(face_measure > 0);

        mfem::CalcOrtho(trans->Jacobian(), normal);
        normal /= normal.Norml2();   // make it unit normal

        auto facet_ceter = ComputeShapeCenter(*(mesh_.GetFace(faces[f])));

        for (int d = 0; d < dim; ++d)
        {
            c_vector[d] = facet_ceter[d] - cell_center[d];
        }

        // Note that edge weight is inverse of M
        double delta_x = c_vector.Norml2();
        double nkc = std::fabs(kappa.InnerProduct(normal, c_vector));
        local_weight(f) = (face_measure * nkc) / (delta_x * delta_x);
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

    int NumIsoVerts() const { return iso_vert_count_; }

    const mfem::ParMesh& GetMesh() const { return *pmesh_; }

    /// Save mesh with partitioning information (GLVis can separate partitions)
    void PrintMeshWithPartitioning(mfem::Array<int>& partition);

    /// Setup visualization of vertex space vector
    void VisSetup(mfem::socketstream& vis_v, mfem::Vector& vec, double range_min = 0.0,
                  double range_max = 0.0, const std::string& caption = "",
                  bool coef = false, bool vec_is_cell_based = true) const;

    /// Update visualization of vertex space vector, VisSetup needs to be called first
    void VisUpdate(mfem::socketstream& vis_v, mfem::Vector& vec,
                   bool vec_is_cell_based = true) const;

    /// Save plot of sol (in vertex space)
    void SaveFigure(const mfem::Vector& sol, const std::string& name,
                    bool vec_is_cell_based = true) const;

    /// Construct partitioning array for vertices
    void Partition(bool metis_parition, const mfem::Array<int>& coarsening_factors,
                   mfem::Array<int>& partitioning) const;

    void VisualizePermeability();

    /// @return a vector of size number of cells containing z-coordinates of cell centers
    mfem::Vector ComputeZ() const;
protected:
    void BuildReservoirGraph();
    void InitGraph();
    void ComputeGraphWeight(bool unit_weight = false);
    virtual void CartPart(const mfem::Array<int>& coarsening_factor,
                          mfem::Array<int>& partitioning) const
    {
        if (myid_ == 0)
        {
            std::cout << "Warning: CartPart is not defined, MetisPart will be called instead!\n";
        }
        MetisPart(coarsening_factor, partitioning);
    }
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

    mutable int iso_vert_count_ = 0;

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
    LocalTPFA local_TPFA(*pmesh_, *kinv_vector_);
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

void DarcyProblem::VisSetup(mfem::socketstream& vis_v, mfem::Vector& vec,
                            double range_min, double range_max, const std::string& caption,
                            bool coef, bool vec_is_cell_based) const
{
    auto& fes = vec_is_cell_based ? u_fes_ : sigma_fes_;
    mfem::ParGridFunction gf(fes.get(), vec.GetData());

    const char vishost[] = "localhost";
    const int  visport   = 19916;

    vis_v.open(vishost, visport);
    vis_v.precision(8);

    vis_v << "parallel " << num_procs_ << " " << myid_ << "\n";
    vis_v << "solution\n" << *pmesh_ << gf;
    vis_v << "window_size 500 800\n"; // Richard's example 800 250
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
        vis_v << "keys L\n";  // logarithmic scale
    }

    if (!caption.empty())
    {
        vis_v << "plot_caption '" << caption << "'\n";
    }

    MPI_Barrier(comm_);

    vis_v << "keys S\n";         //Screenshot

    MPI_Barrier(comm_);
}

void DarcyProblem::VisUpdate(mfem::socketstream& vis_v, mfem::Vector& vec,
                             bool vec_is_cell_based) const
{
    auto& fes = vec_is_cell_based ? u_fes_ : sigma_fes_;
    mfem::ParGridFunction gf(fes.get(), vec.GetData());

    vis_v << "parallel " << num_procs_ << " " << myid_ << "\n";
    vis_v << "solution\n" << *pmesh_ << gf;

    MPI_Barrier(comm_);

    vis_v << "keys S\n";         //Screenshot

    MPI_Barrier(comm_);
}

void DarcyProblem::SaveFigure(const mfem::Vector& sol, const std::string& name,
                              bool vec_is_cell_based) const
{
    auto& fes = vec_is_cell_based ? u_fes_ : sigma_fes_;
    mfem::ParGridFunction gf(fes.get(), sol.GetData());
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
        gf.Save(out);
    }
}

void DarcyProblem::MetisPart(const mfem::Array<int>& coarsening_factor,
                             mfem::Array<int>& partitioning) const
{
    mfem::DiscreteLinearOperator DivOp(sigma_fes_.get(), u_fes_.get());
    DivOp.AddDomainInterpolator(new mfem::DivergenceInterpolator);
    DivOp.Assemble();
    DivOp.Finalize();

    mfem::Vector weight_sqrt(weight_);
    for (int i = 0; i < weight_.Size(); ++i)
    {
        weight_sqrt[i] = std::sqrt(weight_[i]);
    }
    DivOp.SpMat().ScaleColumns(weight_sqrt);

    const int dim = pmesh_->Dimension();
    const int xy_cf = coarsening_factor[0] * coarsening_factor[1];
    const int metis_cf = xy_cf * (dim > 2 ? coarsening_factor[2] : 1);
    PartitionAAT(DivOp.SpMat(), partitioning, metis_cf, dim > 2);
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

mfem::Vector DarcyProblem::ComputeZ() const
{
    const int z_index = pmesh_->SpaceDimension() - 1;
    mfem::Vector Z(vertex_edge_.NumRows());
    Z = 0.0;

    mfem::Array<int> vertices;
    for (int i = 0; i < pmesh_->GetNE(); ++i)
    {
        pmesh_->GetElement(i)->GetVertices(vertices);
        for (auto& vertex : vertices)
        {
            Z[i] += pmesh_->GetVertex(vertex)[z_index];
        }
        Z[i] /= vertices.Size();
    }

    return Z;
}

void DarcyProblem::VisualizePermeability()
{
    mfem::Array<int> vertices;
    for (int i = 0; i < pmesh_->GetNE(); ++i)
    {
        pmesh_->GetElement(i)->GetVertices(vertices);
        mfem::Vector center(pmesh_->Dimension());
        center = 0.0;
        for (int index = 0; index < pmesh_->Dimension(); ++index)
        {
            for (auto& vertex : vertices)
            {
                center[index] += pmesh_->GetVertex(vertex)[index];
            }
            center[index] /= vertices.Size();
        }
        (*coeff_gf_)[i] = ((InversePermeabilityCoefficient&)(*kinv_vector_)).FroNorm(center);
    }

    mfem::socketstream soc;
    VisSetup(soc, *coeff_gf_, 0., 0., "", 1);
}

double hy_g, Ly_g;

class FrancoisCoefficient : public mfem::Coefficient
{
    double const_mult_;

    virtual double Eval(mfem::ElementTransformation& T,
                        const mfem::IntegrationPoint& ip)
    {
        double dx[3];
        mfem::Vector transip(dx, 3);
        T.Transform(ip, transip);

        double tmp = std::exp((transip[1] + hy_g / 2) / Ly_g - 0.9); // 4.0 480.
        return std::max(tmp, 1.) * const_mult_;
    }
public:
    FrancoisCoefficient(double const_mult) : const_mult_(const_mult) { }
};

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
    virtual void CartPart(const mfem::Array<int>& coarsening_factor,
                          mfem::Array<int>& partitioning) const;

    mfem::Array<int> N_;
    unique_ptr<GCoefficient> source_coeff_;
};

SPE10Problem::SPE10Problem(const char* permFile, int nDimensions, int spe10_scale,
                           int slice, bool metis_parition,
                           const mfem::Array<int>& ess_attr, bool unit_weight)
    : DarcyProblem(MPI_COMM_WORLD, nDimensions, ess_attr)
{
    SetupMeshAndCoeff(permFile, nDimensions, spe10_scale, metis_parition, slice);

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
    max_N[2] = 85;//85;

    N_.SetSize(3, 12 * spe10_scale); // 60
    N_[1] = 44 * spe10_scale; // 220
    N_[2] = max_N[2];//17 * spe10_scale; // 85

    // SPE10 grid cell sizes
    mfem::Vector h(3);
    h(0) = 20.0; // 365.76 / 60. in meters
    h(1) = 10.0; // 670.56 / 220. in meters
    h(2) = 2.0; // 51.816 / 85. in meters

    const double Lx = N_[0] * h(0);
    const double Ly = N_[1] * h(1);
    const double Lz = N_[2] * h(2);

    hy_g = h(1);
    Ly_g = Ly;

    using IPC = InversePermeabilityCoefficient;
    IPC::SliceOrientation orient = nDimensions == 2 ? IPC::XY : IPC::NONE;
    kinv_vector_ = make_unique<IPC>(comm_, permFile, N_, max_N, h, orient, slice);

    mfem::Array<int> coarsening_factor(nDimensions);
    coarsening_factor = 10;
    coarsening_factor.Last() = nDimensions == 3 ? 2 : 10;

    const double Hx = coarsening_factor[0] * h(0);
    const double Hy = coarsening_factor[1] * h(1);
    source_coeff_ = make_unique<GCoefficient>(Lx, Ly, Hx, Hy);

    if (nDimensions == 2)
    {
        mfem::Mesh mesh(N_[0], N_[1], mfem::Element::QUADRILATERAL, 1, Lx, Ly);
        pmesh_ = MakeParMesh(mesh, metis_partition);

        return;
    }
    mfem::Mesh mesh(N_[0], N_[1], N_[2], mfem::Element::HEXAHEDRON, 1, Lx, Ly, Lz);
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

void SPE10Problem::CartPart(const mfem::Array<int>& coarsening_factor,
                            mfem::Array<int>& partitioning) const
{
    mfem::Array<int> nxyz(num_procs_xyz_.Size());
    for (int i = 0; i < nxyz.Size(); ++i)
    {
        nxyz[i] = std::max(1, N_[i] / num_procs_xyz_[i] / coarsening_factor[i]);
    }

    partitioning.MakeRef(pmesh_->CartesianPartitioning(nxyz), pmesh_->GetNE());
    partitioning.MakeDataOwner();
}

void SPE10Problem::MakeRHS()
{
    if (ess_attr_.Find(0) == -1) // Neumann condition on whole boundary
    {
        rhs_sigma_ = 0.0;

        mfem::LinearForm q(u_fes_.get());
        q.AddDomainIntegrator(new mfem::DomainLFIntegrator(*source_coeff_));
        q.Assemble();
        rhs_u_ = q;
    }
    else if (ess_attr_.Size() - ess_attr_.Sum() == 2) // Dirichlet on two sides
    {
        mfem::Array<int> nat_one(ess_attr_.Size());
        nat_one = 0;
        nat_one[pmesh_->Dimension() - 2] = 1;

        mfem::ConstantCoefficient one(1.0);
        mfem::RestrictedCoefficient pinflow_coeff(one, nat_one);

        mfem::LinearForm g(sigma_fes_.get());
        g.AddBoundaryIntegrator(
            new mfem::VectorFEBoundaryFluxLFIntegrator(pinflow_coeff));

        mfem::Array<int> nat_negative_one(ess_attr_.Size());
        nat_negative_one = 0;
        nat_negative_one[pmesh_->Dimension() == 2 ? 2 : 3] = 1;

        mfem::ConstantCoefficient negative_one(1.0);
        mfem::RestrictedCoefficient poutflow_coeff(negative_one, nat_negative_one);
        g.AddBoundaryIntegrator(
            new mfem::VectorFEBoundaryFluxLFIntegrator(poutflow_coeff));
        g.Assemble();
        rhs_sigma_ = g;

        rhs_u_ = 0.0;
    }
    else
    {
        mfem::LinearForm g(sigma_fes_.get());
        g.Assemble();
        rhs_sigma_ = g;

        const double rhs_mult = -0.000025;
        FrancoisCoefficient source_coeff(rhs_mult);
        mfem::LinearForm f(u_fes_.get());
        f.AddDomainIntegrator(new mfem::DomainLFIntegrator(source_coeff));
        f.Assemble();
        rhs_u_ = f;
    }
}

mfem::Vector SPE10Problem::InitialCondition(double initial_val) const
{
    HalfCoeffecient half(initial_val);
    mfem::GridFunction init(u_fes_.get());
    init.ProjectCoefficient(half);

    return init;
}

class LognormalModel : public DarcyProblem
{
public:
    LognormalModel(int nDimensions, int num_ser_ref, int num_par_ref,
                   double correlation_length, const mfem::Array<int>& ess_attr);
private:
    void SetupMesh(int nDimensions, int num_ser_ref, int num_par_ref);
    void SetupCoeff(int nDimensions, double correlation_length);

    unique_ptr<mfem::ParMesh> pmesh_c_;
};

LognormalModel::LognormalModel(int nDimensions, int num_ser_ref,
                               int num_par_ref, double correlation_length,
                               const mfem::Array<int>& ess_attr)
    : DarcyProblem(MPI_COMM_WORLD, nDimensions, ess_attr)
{
    SetupMesh(nDimensions, num_ser_ref, num_par_ref);
    InitGraph();

    SetupCoeff(nDimensions, correlation_length);
    ComputeGraphWeight();

    rhs_u_ = -1.0 * CellVolume();
}

void LognormalModel::SetupMesh(int nDimensions, int num_ser_ref, int num_par_ref)
{
    const int N = std::pow(2, num_ser_ref);
    unique_ptr<mfem::Mesh> mesh;
    if (nDimensions == 2)
    {
        mesh = make_unique<mfem::Mesh>(N, N, mfem::Element::QUADRILATERAL, true);
    }
    else
    {
        mesh = make_unique<mfem::Mesh>(N, N, N, mfem::Element::HEXAHEDRON, true);
    }

    pmesh_ = make_unique<mfem::ParMesh>(comm_, *mesh);
    for (int i = 0; i < 0; i++)
    {
        pmesh_->UniformRefinement();
    }
    pmesh_c_ = make_unique<mfem::ParMesh>(*pmesh_);
    for (int i = 0; i < num_par_ref ; i++)
    {
        pmesh_->UniformRefinement();
    }
}

void LognormalModel::SetupCoeff(int nDimensions, double correlation_length)
{
    double nu_parameter = nDimensions == 2 ? 1.0 : 0.5;
    double kappa = std::sqrt(2.0 * nu_parameter) / correlation_length;

    double ddim = static_cast<double>(nDimensions);
    double scalar_g = std::pow(4.0 * M_PI, ddim / 4.0) * std::pow(kappa, nu_parameter) *
                      std::sqrt( std::tgamma(nu_parameter + ddim / 2.0) / tgamma(nu_parameter) );

    mfem::Array<int> ess_attr(ess_attr_.Size());
    ess_attr = 0;

    DarcyProblem darcy_problem(*pmesh_, ess_attr);
    mfem::SparseMatrix W_block = SparseIdentity(pmesh_->GetNE());
    double cell_vol = CellVolume();
    W_block = cell_vol * kappa * kappa;
    MixedMatrix mgL(darcy_problem.GetFVGraph(), W_block);
    mgL.BuildM();

    NormalDistribution normal_dist(0.0, 1.0, 22 + myid_);
    mfem::Vector rhs(mgL.GetD().NumRows());

    for (int i = 0; i < rhs.Size(); ++i)
    {
        rhs[i] = scalar_g * std::sqrt(cell_vol) * normal_dist.Sample();
    }

    MinresBlockSolverFalse solver(mgL, &ess_attr);
    mfem::Vector sol;
    sol = 0.0;
    solver.Solve(rhs, sol);

    for (int i = 0; i < coeff_gf_->Size(); ++i)
    {
        coeff_gf_->Elem(i) = std::exp(sol[i]);
    }
    kinv_scalar_ = make_unique<mfem::GridFunctionCoefficient>(coeff_gf_.get());
}

class EggModel : public DarcyProblem
{
public:
    EggModel(int num_ser_ref, int num_par_ref, const mfem::Array<int>& ess_attr);
private:
    void SetupMesh(int num_ser_ref, int num_par_ref);
    void SetupCoeff();
};

void VelocityEgg(const mfem::Vector& x, mfem::Vector& out)
{
    out.SetSize(x.Size());
    out = 0.0;
    out[0] = 1000.0;
}

EggModel::EggModel(int num_ser_ref, int num_par_ref, const mfem::Array<int>& ess_attr)
    : DarcyProblem(MPI_COMM_WORLD, 3, ess_attr)
{
    SetupMesh(num_ser_ref, num_par_ref);
    InitGraph();

    SetupCoeff();
    ComputeGraphWeight();

    {
        mfem::LinearForm g(sigma_fes_.get());
        g.Assemble();
        rhs_sigma_ = g;

        hy_g = 4.0;
        Ly_g = 480.0;

        double rhs_mult = -0.025;
        if (myid_ == 0)
        {
            std::cout << "RHS multiplier = " << rhs_mult << "\n";
        }
        FrancoisCoefficient source_coeff(rhs_mult);
        mfem::LinearForm f(u_fes_.get());
        f.AddDomainIntegrator(new mfem::DomainLFIntegrator(source_coeff));
        f.Assemble();

        rhs_u_ = f;
    }
}

void EggModel::SetupMesh(int num_ser_ref, int num_par_ref)
{
    std::ifstream imesh("egg_model.mesh");
    mfem::Mesh mesh(imesh, 1, 1);

    for (int i = 0; i < num_ser_ref; i++)
    {
        mesh.UniformRefinement();
    }

    pmesh_ = make_unique<mfem::ParMesh>(comm_, mesh);
    for (int i = 0; i < num_par_ref; i++)
    {
        pmesh_->UniformRefinement();
    }
}

void EggModel::SetupCoeff()
{
    mfem::Array<int> N(3);
    N = 60;
    N[2] = 7;

    mfem::Vector h(3);
    h = 8.0;
    h(2) = 4.0;

    using IPC = InversePermeabilityCoefficient;
    kinv_vector_ = make_unique<IPC>(comm_, "egg_perm.dat", N, N, h);
}

// domain = (0, 4000) x (0, 1000) cm
// BC 10 cm/year = (10 / 365) cm/day on (0, 2000) x {1000} cm
class Richards : public DarcyProblem
{
public:
    Richards(int num_ref, const mfem::Array<int>& ess_attr);
private:
    void SetupMeshCoeff(int num_ref);
    void SetupRHS();
};

Richards::Richards(int num_ref, const mfem::Array<int>& ess_attr)
    : DarcyProblem(MPI_COMM_WORLD, 2, ess_attr)
{
    SetupMeshCoeff(num_ref);
    InitGraph();

    kinv_scalar_ = make_unique<mfem::ConstantCoefficient>(1.0);
    ComputeGraphWeight();

    SetupRHS();
}

void Richards::SetupMeshCoeff(int num_ref)
{
    mfem::Mesh mesh(40, 10, mfem::Element::QUADRILATERAL, 1, 4000.0, 1000.0);
    for (int i = 0; i < num_ref; i++)
    {
        mesh.UniformRefinement();
    }

    pmesh_ = make_unique<mfem::ParMesh>(comm_, mesh);
}

void Velocity(const mfem::Vector& x, mfem::Vector& out)
{
    out.SetSize(x.Size());
    out[0] = 0.0;
    out[1] = x[0] <= 2000.0 ? -10. / 365.0 : 0.0;
}

void Richards::SetupRHS()
{
    mfem::ParMixedBilinearForm bVarf(sigma_fes_.get(), u_fes_.get());
    bVarf.AddDomainIntegrator(new mfem::VectorFEDivergenceIntegrator);
    bVarf.Assemble();
    bVarf.Finalize();

    mfem::ParGridFunction flux_gf(sigma_fes_.get());
    flux_gf = 0.0;

    mfem::VectorFunctionCoefficient velocity_coeff(2, Velocity);
    flux_gf.ProjectBdrCoefficientNormal(velocity_coeff, ess_attr_);

    bVarf.SpMat().AddMult(flux_gf, rhs_u_, 1.0);
}

} // namespace smoothg

