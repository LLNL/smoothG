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
    std::vector<int> cells; // indices of cells that belong to this well
    std::vector<double> coeffs; // transmissibility from well to cells
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
                 const WellDirection direction = WellDirection::Z,
                 const double r_w = 0.01,
                 const double density = 1.0,
                 const double viscosity = 1.0);

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
                          const double r_w,
                          const double density,
                          const double viscosity)
{
    assert (dim_ == 3 || (dim_ == 2 && direction == WellDirection::Z));

    // directions perpendicular to the direction of the well
    const int perp_dir1 = (direction + 1) % 3;
    const int perp_dir2 = (direction + 2) % 3;
    SetDirectionVectors(direction, perp_dir1, perp_dir2);

    mfem::Vector perm_inv;

    auto EquivalentRadius = [&](int cell_index)
    {
        double perp_h1 = mesh_.GetElementSize(cell_index, perp_dir_vec1_);
        double perp_h2 = mesh_.GetElementSize(cell_index, perp_dir_vec2_);
        double p2_to_p1 = perm_inv[perp_dir1] / perm_inv[perp_dir2];
        double p1_to_p2 = perm_inv[perp_dir2] / perm_inv[perp_dir1];
        double numerator = 0.28 * sqrt(sqrt(p2_to_p1) * perp_h1 * perp_h1 +
                                       sqrt(p1_to_p2) * perp_h2 * perp_h2);
        return numerator / (pow(p2_to_p1, 0.25) + pow(p1_to_p2, 0.25));
    };

    auto EquivalentRadius2 = [&](int cell_index, double p1_inv, double p2_inv)
    {
        double h1 = mesh_.GetElementSize(cell_index, perp_dir_vec1_);
        double h2 = mesh_.GetElementSize(cell_index, perp_dir_vec2_);
        double numerator = 0.28 * sqrt(p1_inv * h1 * h1 + p2_inv * h2 * h2);
        return numerator / (sqrt(p1_inv) + sqrt(p1_inv));
    };

    auto WellCoefficient = [&](double effect_perm, double cell_size, double r_e)
    {
        double numerator = 2 * M_PI * density * effect_perm * cell_size;
        return numerator / (viscosity * std::log(r_e / r_w));
    };

    std::vector<double> well_coeffs;
    well_coeffs.reserve(cells.size());

    for (const auto& cell : cells)
    {
        auto Tr = mesh_.GetElementTransformation(cell);
        Tr->SetIntPoint(&ip_);
        perm_inv_coeff_.Eval(perm_inv, *Tr, ip_);

        auto effect_perm = 1. / sqrt(perm_inv[perp_dir1] * perm_inv[perp_dir2]);
        auto size = dim_ == 2 ? 2.0 * ft : mesh_.GetElementSize(cell, dir_vec_);
        auto r_e = EquivalentRadius(cell);
        assert(fabs(r_e - EquivalentRadius2(cell, perm_inv[perp_dir1], perm_inv[perp_dir2]))<1e-12);
        well_coeffs.push_back(WellCoefficient(effect_perm, size, r_e));
    }

    wells_.push_back({type, value, cells, well_coeffs});
    num_well_cells_ += cells.size();
}

unique_ptr<mfem::HypreParMatrix> ConcatenateIdentity(
    const mfem::HypreParMatrix& pmat, const int id_size)
{
    mfem::SparseMatrix diag, offd;
    HYPRE_Int* old_colmap;
    pmat.GetDiag(diag);
    pmat.GetOffd(offd, old_colmap);

    const int nrows = diag.NumRows() + id_size;
    const int ncols_diag = diag.NumCols() + id_size;
    const int nnz_diag = diag.NumNonZeroElems() + id_size;

    mfem::Array<HYPRE_Int> row_starts, col_starts;
    mfem::Array<HYPRE_Int>* starts[2] = {&row_starts, &col_starts};
    HYPRE_Int sizes[2] = {nrows, ncols_diag};
    GenerateOffsets(pmat.GetComm(), 2, sizes, starts);

    int myid_;
    int num_procs;
    MPI_Comm comm = pmat.GetComm();
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid_);

    int col_diff = col_starts[0] - pmat.ColPart()[0];

    int global_true_dofs = pmat.N();
    mfem::Array<int> col_change(global_true_dofs);
    col_change = 0;

    int start = pmat.ColPart()[0];
    int end = pmat.ColPart()[1];

    for (int i = start; i < end; ++i)
    {
        col_change[i] = col_diff;
    }

    mfem::Array<int> col_remap(global_true_dofs);
    col_remap = 0;

    // TODO: get rid of GetData
    MPI_Scan(col_change.GetData(), col_remap.GetData(), global_true_dofs, HYPRE_MPI_INT, MPI_SUM, comm);
    MPI_Bcast(col_remap.GetData(), global_true_dofs, HYPRE_MPI_INT, num_procs - 1, comm);

    // Append identity matrix to the bottom left of diag
    int* diag_i = new int[nrows + 1];
    std::copy_n(diag.GetI(), diag.Height() + 1, diag_i);
    std::iota(diag_i + diag.Height(), diag_i + nrows + 1, diag_i[diag.Height()]);

    int* diag_j = new int[nnz_diag];
    std::copy_n(diag.GetJ(), diag.NumNonZeroElems(), diag_j);

    for (int i = 0; i < id_size; i++)
    {
        diag_j[diag.NumNonZeroElems() + i] = diag.Width() + i;
    }

    double* diag_data = new double[nnz_diag];
    std::copy_n(diag.GetData(), diag.NumNonZeroElems(), diag_data);
    std::fill_n(diag_data + diag.NumNonZeroElems(), id_size, 1.0);

    // Append zero matrix to the bottom of offd
    const int ncols_offd = offd.Width();
    const int nnz_offd = offd.NumNonZeroElems() + id_size;

    int* offd_i = new int[nrows + 1];
    std::copy_n(offd.GetI(), offd.Height() + 1, offd_i);
    std::fill_n(offd_i + offd.Height() + 1, id_size, offd_i[offd.Height()]);

    int* offd_j = new int[nnz_offd];
    std::copy_n(offd.GetJ(), offd.NumNonZeroElems(), offd_j);

    double* offd_data = new double[nnz_offd];
    std::copy_n(offd.GetData(), offd.NumNonZeroElems(), offd_data);

    HYPRE_Int* colmap = new HYPRE_Int[ncols_offd]();
    std::copy_n(old_colmap, ncols_offd, colmap);

    for (int i = 0; i < ncols_offd; ++i)
    {
        colmap[i] += col_remap[colmap[i]];
    }

    auto out = make_unique<mfem::HypreParMatrix>(
                   pmat.GetComm(), row_starts.Last(), col_starts.Last(),
                   row_starts, col_starts, diag_i, diag_j, diag_data,
                   offd_i, offd_j, offd_data, ncols_offd, colmap);

    out->CopyRowStarts();
    out->CopyColStarts();

    return out;
}

//void CoarsenVertexEssentialCondition(
//    const int num_wells, const int new_size,
//    mfem::Array<int>& ess_marker, mfem::Vector& ess_data)
//{
//    mfem::Array<int> new_ess_marker(new_size);
//    mfem::Vector new_ess_data(new_size);
//    new_ess_marker = 0;
//    new_ess_data = 0.0;

//    const int old_size = ess_data.Size();

//    for (int i = 0; i < num_wells; i++)
//    {
//        if (ess_marker[old_size - 1 - i])
//        {
//            new_ess_marker[new_size - 1 - i] = 1;
//            new_ess_data(new_size - 1 - i) = ess_data(old_size - 1 - i);
//        }
//    }
//    mfem::Swap(ess_marker, new_ess_marker);
//    ess_data.Swap(new_ess_data);
//}

//void CoarsenSigmaEssentialCondition(
//    const int num_wells, const int new_size,
//    mfem::Array<int>& ess_marker)
//{
//    mfem::Array<int> new_ess_marker(new_size);
//    new_ess_marker = 0;

//    const int old_size = ess_marker.Size();

//    for (int i = 0; i < num_wells; i++)
//    {
//        if (ess_marker[old_size - 1 - i])
//        {
//            new_ess_marker[new_size - 1 - i] = 1;
//        }
//    }

//    mfem::Swap(ess_marker, new_ess_marker);
//}

class TwoPhase : public SPE10Problem
{
public:
    TwoPhase(const char* perm_file, int dim, int spe10_scale,
             int slice, bool metis_parition, const mfem::Array<int>& ess_attr,
             int well_height, double inject_rate, double bottom_hole_pressure);

    virtual Graph GetFVGraph(bool use_local_weight) override;

    const std::vector<Well>& GetWells() { return well_manager_.GetWells(); }

    void PrintMeshWithPartitioning(mfem::Array<int>& partition);

private:
    void SetupPeaceman(int well_height, double inject_rate, double bot_hole_pres);

    // extend edge_bdr_ by adding new boundary for production wells edges
    mfem::SparseMatrix ExtendEdgeBoundary();

    WellManager well_manager_;
};

TwoPhase::TwoPhase(const char* perm_file, int dim, int spe10_scale,
             int slice, bool metis_parition, const mfem::Array<int>& ess_attr,
             int well_height, double inject_rate, double bottom_hole_pressure)
    : SPE10Problem(perm_file, dim, spe10_scale, slice, metis_parition, ess_attr),
      well_manager_(*pmesh_, *kinv_vector_)
{
    // Build wells (Peaceman's five-spot pattern)
    SetupPeaceman(well_height, inject_rate, bottom_hole_pressure);

    rhs_sigma_ = 0.0;
    rhs_u_ = 0.0;
}

void TwoPhase::SetupPeaceman(int well_height, double inject_rate, double bhp)
{
    const int num_wells = 5;
    std::vector<std::vector<int>> well_cells(num_wells);

    const double max_x = 365.76 - ft;
    const double max_y = 670.56 - ft;

    mfem::DenseMatrix point(3, num_wells);
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
            point(2, i) += 2.0 * ft;           // Shift Points for next layer
        }
    }

    for (int i = 0; i < num_wells - 1; ++i)
    {
        well_manager_.AddWell(Producer, bhp, well_cells[i]);
    }
    well_manager_.AddWell(Injector, inject_rate, well_cells.back());
}

mfem::SparseMatrix TwoPhase::ExtendEdgeBoundary()
{
    const int num_edges = edge_bdr_.NumRows() + well_manager_.NumWellCells();
    const int nnz = NNZ(edge_bdr_) + well_manager_.NumWellCells(Producer);

    int* I = new int[num_edges + 1];
    int* J = new int[nnz];
    double* Data = new double[nnz];

    std::copy_n(edge_bdr_.GetI(), edge_bdr_.NumRows() + 1, I);
    std::copy_n(edge_bdr_.GetJ(), edge_bdr_.NumNonZeroElems(), J);
    std::fill_n(Data, nnz, 1.0);

    int* I_ptr = I + edge_bdr_.NumRows() + 1;
    int* J_ptr = J + edge_bdr_.NumNonZeroElems();
    int bdr = edge_bdr_.NumCols();

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
    const int num_well_cells = well_manager_.NumWellCells();
    const int num_injectors = well_manager_.NumWells(Injector);
    int edge_counter = vertex_edge_.NumCols();         // reservoir faces
    int vert_counter = vertex_edge_.NumRows();         // reservoir cells
    const int num_edges = edge_counter + num_well_cells;
    const int num_vertices = vert_counter + num_injectors;

    // Copying the old data
    mfem::Vector new_weight(num_edges);
    std::copy_n(weight_.GetData(), weight_.Size(), new_weight.GetData());
    mfem::Vector new_rhs_sigma(num_edges);
    std::copy_n(rhs_sigma_.GetData(), rhs_sigma_.Size(), new_rhs_sigma.GetData());
    std::fill_n(new_rhs_sigma.GetData() + rhs_sigma_.Size(), num_well_cells, 0.0);
    mfem::Vector new_rhs_u(num_vertices);
    std::copy_n(rhs_u_.GetData(), rhs_u_.Size(), new_rhs_u.GetData());

    mfem::SparseMatrix new_vertex_edge(num_vertices, num_edges);
    {
        int* vertex_edge_i = vertex_edge_.GetI();
        int* vertex_edge_j = vertex_edge_.GetJ();
        for (int i = 0; i < vert_counter; i++)
        {
            for (int j = vertex_edge_i[i]; j < vertex_edge_i[i + 1]; j++)
            {
                new_vertex_edge.Add(i, vertex_edge_j[j], 1.0);
            }
        }
    }

    // Adding connection between reservoir and well to the graph
    for (const Well& well : well_manager_.GetWells())
    {
        const auto& well_cells = well.cells;
        const auto& well_coeff = well.coeffs;

        for (unsigned int j = 0; j < well_cells.size(); j++)
        {
            new_vertex_edge.Add(well_cells[j], edge_counter, 1.0);
            if (well.type == Injector)
            {
                new_vertex_edge.Add(vert_counter, edge_counter, 1.0);
            }
            else
            {
                new_rhs_sigma[edge_counter] = well.value;
            }
            new_weight[edge_counter] = well_coeff[j];
            auto& local_weight_j = local_weight_[well_cells[j]];
            mfem::Vector new_local_weight_j(local_weight_j.Size() + 1);
            for (int k = 0; k < local_weight_j.Size(); k++)
            {
                new_local_weight_j[k] = local_weight_j[k];
            }
            new_local_weight_j[local_weight_j.Size()] = well_coeff[j];
            local_weight_j.Swap(new_local_weight_j);

            edge_counter++;
        }

        if (well.type == Injector)
        {
            mfem::Vector local_weight_j(well_cells.size());
            local_weight_j = 1e10;//INFINITY; // Not sure if this is ok
            local_weight_.push_back(local_weight_j);
            new_rhs_u[vert_counter++] = -1.0 * well.value;
        }
    }
    new_vertex_edge.Finalize();

    rhs_sigma_.Swap(new_rhs_sigma);
    rhs_u_.Swap(new_rhs_u);

    auto edge_trueedge = ConcatenateIdentity(*sigma_fes_->Dof_TrueDof_Matrix(), num_well_cells);

    auto ext_edge_bdr = ExtendEdgeBoundary();
//ext_edge_bdr.Print();
    if (use_local_weight && local_weight_.size() > 0)
    {
        return Graph(new_vertex_edge, *edge_trueedge, local_weight_, &ext_edge_bdr);
    }

    return Graph(new_vertex_edge, *edge_trueedge, new_weight, &ext_edge_bdr);
}

void TwoPhase::PrintMeshWithPartitioning(mfem::Array<int>& partition)
{
    std::stringstream fname;
    fname << "mesh.with_parts." << std::setfill('0') << std::setw(6) << myid_;
    std::ofstream ofid(fname.str().c_str());
    ofid.precision(8);
    pmesh_->PrintWithPartitioning(partition.GetData(), ofid, 1);
}
