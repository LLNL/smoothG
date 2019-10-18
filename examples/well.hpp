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
   @file well.hpp
   @brief Implementation of well models.

   Build somewhat sophisticated well models, integrate them with a reservoir model
*/

#include "pde.hpp"

using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

using namespace smoothg;

enum WellType { Injector, Producer };
enum WellDirection { X = 0, Y, Z };
constexpr double ft = 0.3048;          // 1 ft = 0.3048 meter

struct Well
{
    WellType type; // injector or producer
    double value; // injector: total inject rate, producer: bottom hole pressure
    std::vector<int> cells; // cells in matrix that contain this well
    std::vector<double> well_indices; // well indices of well cells
};

class WellManager
{
public:
    WellManager(mfem::Mesh& mesh, mfem::VectorCoefficient& perm_inv_coeff)
        : mesh_(mesh), ir_(mfem::IntRules.Get(mesh_.GetElementType(0), 1)),
          ip_(ir_.IntPoint(0)), perm_inv_coeff_(perm_inv_coeff),
          dim_(mesh_.Dimension()), dir_vec_(dim_), perp_dir_vec1_(dim_),
          perp_dir_vec2_(dim_), num_well_cells_(0) { }

    void AddWell(const WellType type,
                 const double value,
                 const std::vector<int>& cells,
                 const WellDirection direction = Z,
                 const double r_w = 0.01);

    const std::vector<Well>& GetWells() const { return wells_; }

    int NumWellCells() const { return num_well_cells_; }
    int NumWellCells(WellType type) const;
    int NumWells(WellType type) const;
private:
    void SetDirectionVectors(int dir, int perp_dir1, int perp_dir2);

    mfem::Mesh& mesh_;
    const mfem::IntegrationRule& ir_;
    const mfem::IntegrationPoint& ip_;
    mfem::VectorCoefficient& perm_inv_coeff_;

    const int dim_;
    mfem::Vector dir_vec_;
    mfem::Vector perp_dir_vec1_;
    mfem::Vector perp_dir_vec2_;

    std::vector<Well> wells_;
    int num_well_cells_;
};

int WellManager::NumWellCells(WellType type) const
{
    int num_well_cells = 0;
    for (const Well& well : wells_)
    {
        num_well_cells += well.type == type ? well.cells.size() : 0;
    }
    return num_well_cells;
}

int WellManager::NumWells(WellType type) const
{
    int num_wells = 0;
    for (const Well& well : wells_) { num_wells += (well.type == type); }
    return num_wells;
}

void WellManager::SetDirectionVectors(int dir, int perp_dir1, int perp_dir2)
{
    dir_vec_ = perp_dir_vec1_ = perp_dir_vec2_ = 0.0;
    dir_vec_[dir] = perp_dir_vec1_[perp_dir1] = perp_dir_vec2_[perp_dir2] = 1.0;
}

void WellManager::AddWell(const WellType type,
                          const double value,
                          const std::vector<int>& cells,
                          const WellDirection direction,
                          const double r_w)
{
    assert (dim_ == 3 || (dim_ == 2 && direction == WellDirection::Z));

    // directions perpendicular to the direction of the well
    const int perp_dir1 = (direction + 1) % 3;
    const int perp_dir2 = (direction + 2) % 3;
    SetDirectionVectors(direction, perp_dir1, perp_dir2);

    // Simplified version of (13.12) and (13.13) in Chen, Huan, and Ma's book
    auto WellIndex = [&](int cell, double k11, double k22)
    {
        double h1 = mesh_.GetElementSize(cell, perp_dir_vec1_);
        double h2 = mesh_.GetElementSize(cell, perp_dir_vec2_);
        double numerator = 0.28 * sqrt(k22 * h1 * h1 + k11 * h2 * h2);
        double equiv_radius = numerator / (sqrt(k11) + sqrt(k22));

        double h3 = dim_ < 3 ? 2 * ft : mesh_.GetElementSize(cell, dir_vec_);
        return 2 * M_PI * h3 * sqrt(k11 * k22) / std::log(equiv_radius / r_w);
    };

    std::vector<double> well_indices;
    well_indices.reserve(cells.size());

    for (const auto& cell : cells)
    {
        auto Tr = mesh_.GetElementTransformation(cell);
        Tr->SetIntPoint(&ip_);
        mfem::Vector perm_inv;
        perm_inv_coeff_.Eval(perm_inv, *Tr, ip_);

        double k11 = 1. / perm_inv[perp_dir1];
        double k22 = 1. / perm_inv[perp_dir2];
        well_indices.push_back(WellIndex(cell, k11, k22));
    }

    wells_.push_back({type, value, cells, well_indices});
    num_well_cells_ += cells.size();
}

unique_ptr<mfem::HypreParMatrix> ConcatenateIdentity(
    const mfem::HypreParMatrix& pmat, const int id_size)
{
    MPI_Comm comm = pmat.GetComm();

    mfem::SparseMatrix diag, offd;
    HYPRE_Int* old_colmap;
    pmat.GetDiag(diag);
    pmat.GetOffd(offd, old_colmap);

    const int num_rows = diag.NumRows() + id_size;
    const int num_cols_diag = diag.NumCols() + id_size;
    const int nnz_diag = NNZ(diag) + id_size;

    mfem::Array<HYPRE_Int> row_starts, col_starts;
    mfem::Array<HYPRE_Int>* starts[2] = {&row_starts, &col_starts};
    HYPRE_Int sizes[2] = {num_rows, num_cols_diag};
    GenerateOffsets(comm, 2, sizes, starts);

    int myid_;
    int num_procs;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid_);

    int col_diff = col_starts[0] - pmat.ColPart()[0];

    int global_num_cols = pmat.GetGlobalNumCols();
    mfem::Array<int> col_change(global_num_cols);
    col_change = 0;

    for (int i = pmat.ColPart()[0]; i < pmat.ColPart()[1]; ++i)
    {
        col_change[i] = col_diff;
    }

    mfem::Array<int> col_remap(global_num_cols);
    col_remap = 0;

    MPI_Scan(col_change, col_remap, global_num_cols, HYPRE_MPI_INT, MPI_SUM, comm);
    MPI_Bcast(col_remap, global_num_cols, HYPRE_MPI_INT, num_procs - 1, comm);

    // Append identity matrix diagonally to the bottom left of diag
    int* diag_i = new int[num_rows + 1];
    std::copy_n(diag.GetI(), diag.NumRows() + 1, diag_i);
    std::iota(diag_i + diag.NumRows(), diag_i + num_rows + 1, diag_i[diag.NumRows()]);

    int* diag_j = new int[nnz_diag];
    std::copy_n(diag.GetJ(), NNZ(diag), diag_j);

    for (int i = 0; i < id_size; i++)
    {
        diag_j[NNZ(diag) + i] = diag.NumCols() + i;
    }

    double* diag_data = new double[nnz_diag];
    std::copy_n(diag.GetData(), NNZ(diag), diag_data);
    std::fill_n(diag_data + NNZ(diag), id_size, 1.0);

    // Append zero matrix to the bottom of offd
    int* offd_i = new int[num_rows + 1];
    std::copy_n(offd.GetI(), offd.NumRows() + 1, offd_i);
    std::fill_n(offd_i + offd.NumRows() + 1, id_size, offd_i[offd.NumRows()]);

    int* offd_j = new int[NNZ(offd)];
    std::copy_n(offd.GetJ(), NNZ(offd), offd_j);

    double* offd_data = new double[NNZ(offd)];
    std::copy_n(offd.GetData(), NNZ(offd), offd_data);

    HYPRE_Int* colmap = new HYPRE_Int[offd.NumCols()]();
    std::copy_n(old_colmap, offd.NumCols(), colmap);

    for (int i = 0; i < offd.NumCols(); ++i)
    {
        colmap[i] += col_remap[colmap[i]];
    }

    auto out = make_unique<mfem::HypreParMatrix>(
                   comm, row_starts.Last(), col_starts.Last(),
                   row_starts, col_starts, diag_i, diag_j, diag_data,
                   offd_i, offd_j, offd_data, offd.NumCols(), colmap);

    out->CopyRowStarts();
    out->CopyColStarts();

    return out;
}

class TwoPhase : public SPE10Problem
{
public:
    TwoPhase(const char* perm_file, int dim, int spe10_scale,
             int slice, bool metis_parition, const mfem::Array<int>& ess_attr,
             int well_height, double inject_rate, double bottom_hole_pressure);

    virtual Graph GetFVGraph(bool use_local_weight) override;

    void PrintMeshWithPartitioning(mfem::Array<int>& partition);
private:
    // Set up well model (Peaceman's five-spot pattern)
    void SetPeaceman(int well_height, double inject_rate, double bot_hole_pres);

    mfem::SparseMatrix ExtendVertexEdge(const mfem::SparseMatrix& vert_edge);
    mfem::Vector AppendWellIndex(const mfem::Vector& weight);
    std::vector<mfem::Vector> AppendWellIndex(const std::vector<mfem::Vector>& loc_weight);
    mfem::Vector AppendProducerRHS(const mfem::Vector& rhs_sigma);
    mfem::Vector AppendInjectorRHS(const mfem::Vector& rhs_u);

    // extend edge_bdr_ by adding new boundary for production wells edges
    mfem::SparseMatrix ExtendEdgeBoundary(const mfem::SparseMatrix& edge_bdr);

    WellManager well_manager_;
};

TwoPhase::TwoPhase(const char* perm_file, int dim, int spe10_scale, int slice,
                   bool metis_parition, const mfem::Array<int>& ess_attr,
                   int well_height, double inject_rate, double bottom_hole_pressure)
    : SPE10Problem(perm_file, dim, spe10_scale, slice, metis_parition, ess_attr),
      well_manager_(*pmesh_, *kinv_vector_)
{
    SetPeaceman(well_height, inject_rate, bottom_hole_pressure);

    rhs_sigma_ = 0.0;
    rhs_u_ = 0.0;
}

void TwoPhase::SetPeaceman(int well_height, double inject_rate, double bhp)
{
    const int num_wells = 5;
    std::vector<std::vector<int>> well_cells(num_wells);

    const double max_x = 365.76 - ft;
    const double max_y = 670.56 - ft;

    mfem::DenseMatrix point(pmesh_->Dimension(), num_wells);
    point = ft;
    point(0, 1) = max_x;
    point(1, 2) = max_y;
    point(0, 3) = max_x;
    point(1, 3) = max_y;
    point(0, 4) = ((max_x + ft) / 2.0) + ft;
    point(1, 4) = ((max_y + ft) / 2.0) + ft;

    for (int j = 0; j < well_height; ++j)
    {
        mfem::Array<int> ids;
        mfem::Array<mfem::IntegrationPoint> ips;
        pmesh_->FindPoints(point, ids, ips, false);

        for (int i = 0; i < num_wells; ++i)
        {
            assert(ids[i] >= 0); // TODO: parallel
            well_cells[i].push_back(ids[i]);
            if (pmesh_->Dimension() == 3) point(2, i) += 2.0 * ft; // next layer
        }
    }

    for (int i = 0; i < num_wells - 1; ++i)
    {
        well_manager_.AddWell(Producer, bhp, well_cells[i]);
    }
    well_manager_.AddWell(Injector, inject_rate, well_cells.back());
}


mfem::SparseMatrix TwoPhase::ExtendVertexEdge(const mfem::SparseMatrix& vert_edge)
{
    const int num_edges = vert_edge.NumCols() + well_manager_.NumWellCells();
    const int num_verts = vert_edge.NumRows() + well_manager_.NumWells(Injector);

    // TODO: new local numbering may not match local numbering in local weight
    mfem::SparseMatrix ext_vert_edge(num_verts, num_edges);
    for (int i = 0; i < vert_edge.NumRows(); ++i)
    {
        for (int j = 0; j < vert_edge.RowSize(i); ++j)
        {
            ext_vert_edge.Add(i, vert_edge.GetRowColumns(i)[j], 1.0);
        }
    }

    int edge = vert_edge.NumCols();         // number of reservoir faces
    int vert = vert_edge.NumRows();         // number of reservoir cells

    // Adding connection between reservoir and well to the graph
    for (const Well& well : well_manager_.GetWells())
    {
        for (auto& cell : well.cells)
        {
            if (well.type == Injector) ext_vert_edge.Add(vert, edge, 1.0);
            ext_vert_edge.Add(cell, edge++, 1.0);
        }
        vert += (well.type == Injector);
    }
    ext_vert_edge.Finalize();

    return ext_vert_edge;
}

mfem::Vector TwoPhase::AppendWellIndex(const mfem::Vector& weight)
{
    const int num_edges = weight.Size() + well_manager_.NumWellCells();

    mfem::Vector new_weight(num_edges);
    std::copy_n(weight_.GetData(), weight.Size(), new_weight.GetData());

    double* data_ptr = new_weight.GetData() + weight.Size();
    for (const Well& well : well_manager_.GetWells())
    {
        std::copy_n(well.well_indices.data(), well.cells.size(), data_ptr);
        data_ptr += well.cells.size();
    }

    return new_weight;
}

std::vector<mfem::Vector> TwoPhase::AppendWellIndex(const std::vector<mfem::Vector>& loc_weight)
{
    std::vector<mfem::Vector> new_loc_weight;
    new_loc_weight.reserve(loc_weight.size() + well_manager_.NumWells(Injector));
    for (const auto& loc_w : loc_weight) new_loc_weight.push_back(loc_w);

    for (const Well& well : well_manager_.GetWells())
    {
        const auto& cells = well.cells;
        for (unsigned int j = 0; j < cells.size(); j++)
        {
            const mfem::Vector& loc_w = loc_weight[cells[j]];
            mfem::Vector& new_loc_w = new_loc_weight[cells[j]];
            new_loc_w.SetSize(loc_w.Size() + 1);
            std::copy_n(loc_w.GetData(), loc_w.Size(), new_loc_w.GetData());
            new_loc_w[loc_w.Size()] = well.well_indices[j];
        }

        if (well.type == Injector)
        {
            mfem::Vector new_loc_w(cells.size());
            new_loc_w = 1e10;//INFINITY; // Not sure if this is ok
            new_loc_weight.push_back(new_loc_w);
        }
    }

    return new_loc_weight;
}

mfem::Vector TwoPhase::AppendProducerRHS(const mfem::Vector& rhs_sigma)
{
    mfem::Vector new_rhs_sigma(rhs_sigma.Size() + well_manager_.NumWellCells());
    std::copy_n(rhs_sigma_.GetData(), rhs_sigma.Size(), new_rhs_sigma.GetData());

    double* data_ptr = new_rhs_sigma.GetData() + rhs_sigma.Size();
    for (const Well& well : well_manager_.GetWells())
    {
        double value = well.type == Producer ? well.value : 0.0;
        std::fill_n(data_ptr, well.cells.size(), value);
        data_ptr += well.cells.size();
    }

    return new_rhs_sigma;
}

mfem::Vector TwoPhase::AppendInjectorRHS(const mfem::Vector& rhs_u)
{
    mfem::Vector new_rhs_u(rhs_u.Size() + well_manager_.NumWells(Injector));
    std::copy_n(rhs_u.GetData(), rhs_u.Size(), new_rhs_u.GetData());

    double* data_ptr = new_rhs_u.GetData() + rhs_u.Size();
    for (const Well& well : well_manager_.GetWells())
    {
        if (well.type == Injector) *(data_ptr++) = -1.0 * well.value;
    }

    return new_rhs_u;
}

mfem::SparseMatrix TwoPhase::ExtendEdgeBoundary(const mfem::SparseMatrix& edge_bdr)
{
    const int num_edges = edge_bdr.NumRows() + well_manager_.NumWellCells();
    const int nnz = NNZ(edge_bdr) + well_manager_.NumWellCells(Producer);

    int* I = new int[num_edges + 1];
    int* J = new int[nnz];
    double* Data = new double[nnz];

    std::copy_n(edge_bdr.GetI(), edge_bdr.NumRows() + 1, I);
    std::copy_n(edge_bdr.GetJ(), NNZ(edge_bdr), J);
    std::fill_n(Data, nnz, 1.0);

    int* I_ptr = I + edge_bdr.NumRows() + 1;
    int* J_ptr = J + NNZ(edge_bdr);
    int bdr = edge_bdr.NumCols();

    for (const Well& well : well_manager_.GetWells())
    {
        if (well.type == Producer)
        {
            std::iota(I_ptr, I_ptr + well.cells.size(), *(I_ptr - 1) + 1);
            std::fill_n(J_ptr, well.cells.size(), bdr++);
            J_ptr += well.cells.size();
        }
        else
        {
            std::fill_n(I_ptr, well.cells.size(), *(I_ptr - 1));
        }
        I_ptr += well.cells.size();
    }

    return mfem::SparseMatrix(I, J, Data, num_edges, bdr);
}

Graph TwoPhase::GetFVGraph(bool use_local_weight)
{
    weight_ = AppendWellIndex(weight_);
    local_weight_ = AppendWellIndex(local_weight_);
    vertex_edge_ = ExtendVertexEdge(vertex_edge_);
    rhs_sigma_ = AppendProducerRHS(rhs_sigma_);
    rhs_u_ = AppendInjectorRHS(rhs_u_);

    const int num_well_cells = well_manager_.NumWellCells();
    auto& old_edge_trueedge = *sigma_fes_->Dof_TrueDof_Matrix();
    auto edge_trueedge = ConcatenateIdentity(old_edge_trueedge, num_well_cells);

    edge_bdr_ = ExtendEdgeBoundary(edge_bdr_);

    mfem::Array<int> producer_attr(well_manager_.NumWells(Producer));
    producer_attr = 0;                // treat producer as "natural boundary"
    ess_attr_.Append(producer_attr);

    if (use_local_weight && local_weight_.size() > 0)
    {
        return Graph(vertex_edge_, *edge_trueedge, local_weight_, &edge_bdr_);
    }

    return Graph(vertex_edge_, *edge_trueedge, weight_, &edge_bdr_);
}

void TwoPhase::PrintMeshWithPartitioning(mfem::Array<int>& partition)
{
    std::stringstream fname;
    fname << "mesh.with_parts." << std::setfill('0') << std::setw(6) << myid_;
    std::ofstream ofid(fname.str().c_str());
    ofid.precision(8);
    pmesh_->PrintWithPartitioning(partition.GetData(), ofid, 1);
}
