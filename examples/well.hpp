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

#include "../src/smoothG.hpp"
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
          dim_(mesh_.Dimension()), dir_vec(dim_), perp_dir_vec1_(dim_),
          perp_dir_vec2_(dim_), num_producers_(0), num_injectors_(0) { }

    void AddWell(const WellType type,
                 const double value,
                 const std::vector<int>& cells,
                 const WellDirection direction = WellDirection::Z,
                 const double r_w = 0.01,
                 const double density = 1.0,
                 const double viscosity = 1.0);

    const std::vector<Well>& GetWells() { return wells_; }
private:
    void SetDirectionVectors(int dir, int perp_dir1, int perp_dir2);

    mfem::Mesh& mesh_;
    const mfem::IntegrationRule& ir_;
    const mfem::IntegrationPoint& ip_;
    mfem::VectorCoefficient& perm_inv_coeff_;

    const int dim_;
    mfem::Vector dir_vec;
    mfem::Vector perp_dir_vec1_;
    mfem::Vector perp_dir_vec2_;

    int num_producers_;
    int num_injectors_;

    std::vector<Well> wells_;
};

void WellManager::SetDirectionVectors(int dir, int perp_dir1, int perp_dir2)
{
    dir_vec = perp_dir_vec1_ = perp_dir_vec2_ = 0.0;
    dir_vec[dir] = perp_dir_vec1_[perp_dir1] = perp_dir_vec2_[perp_dir2] = 1.0;
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

    const unsigned int num_well_cells = cells.size();
    mfem::Vector perm_inv;

    // directions perpendicular to the direction of the well
    const int perp_dir1 = (direction + 1) % 3;
    const int perp_dir2 = (direction + 2) % 3;

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

    wells_.push_back({type, value, cells, std::vector<double>(num_well_cells)});

    for (unsigned int i = 0; i < num_well_cells; i++)
    {
        auto Tr = mesh_.GetElementTransformation(cells[i]);
        Tr->SetIntPoint(&ip_);
        perm_inv_coeff_.Eval(perm_inv, *Tr, ip_);

        auto effect_perm = 1. / sqrt(perm_inv[perp_dir1] * perm_inv[perp_dir2]);
        auto size = dim_ == 2 ? 2.0 * ft : mesh_.GetElementSize(cells[i], dir_vec);
        auto r_e = EquivalentRadius(cells[i]);
        assert(fabs(r_e - EquivalentRadius2(cells[i], perm_inv[perp_dir1], perm_inv[perp_dir2]))<1e-12);
        wells_.back().coeffs[i] = WellCoefficient(effect_perm, size, r_e);
    }

    num_producers_ += (type == WellType::Producer);
    num_injectors_ += (type == WellType::Injector);
}

unique_ptr<mfem::HypreParMatrix> ConcatenateIdentity(
    const mfem::HypreParMatrix& pmat, const int id_size)
{
    mfem::SparseMatrix diag, offd;
    HYPRE_Int* old_colmap;
    pmat.GetDiag(diag);
    pmat.GetOffd(offd, old_colmap);

    const int nrows = diag.Height() + id_size;
    const int ncols_diag = diag.Width() + id_size;
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

void RemoveWellDofs(const std::vector<Well>& well_list,
                    const mfem::BlockVector& vec,
                    mfem::Array<int>& offset, mfem::BlockVector& new_vec)
{
    int num_well_cells = 0;
    for (const auto& well : well_list)
        num_well_cells += well.cell_indices.size();

    offset.SetSize(3);
    offset[0] = 0;
    offset[1] = vec.GetBlock(0).Size() - num_well_cells;
    offset[2] = offset[1] + vec.GetBlock(1).Size() - well_list.size();

    double* data = new double[offset[2]];
    new_vec.Update(data, offset);
    new_vec.MakeDataOwner();

    for (int k = 0; k < 2; k++)
        for (int i = 0; i < new_vec.BlockSize(k); i++)
            new_vec.GetBlock(k)[i] = vec.GetBlock(k)[i];
}

void MakeWellMaps(const std::vector<Well>& well_list,
                  const mfem::BlockVector& vec,
                  mfem::Array<int>& no_well_offset,
                  mfem::Array<int>& no_well_map,
                  mfem::Array<int>& well_offset,
                  mfem::Array<int>& well_map)
{
    int num_well_cells = 0;
    for (auto& well : well_list)
        num_well_cells += well.cell_indices.size();

    int num_injectors = 0;
    for (const auto& well : well_list)
    {
        num_injectors += (well.type == WellType::Injector);
    }

    no_well_offset.SetSize(3);
    no_well_offset[0] = 0;
    no_well_offset[1] = vec.GetBlock(0).Size() - num_well_cells;
    no_well_offset[2] = no_well_offset[1] + vec.GetBlock(1).Size() - num_injectors;

    well_offset.SetSize(3);
    well_offset[0] = 0;
    well_offset[1] = num_well_cells;
    well_offset[2] = well_offset[1] + num_injectors;

    no_well_map.SetSize(no_well_offset[2]);
    std::iota(no_well_map.begin(), no_well_map.begin() + no_well_offset[1], 0);
    std::iota(no_well_map.begin() + no_well_offset[1], no_well_map.end(), vec.GetBlock(0).Size());

    well_map.SetSize(num_well_cells + num_injectors);
    std::iota(well_map.begin(), well_map.begin() + num_well_cells, no_well_offset[1]);
    std::iota(well_map.begin() + num_well_cells, well_map.end(), no_well_offset[2] + num_well_cells);
}

void WritePoints(double time, std::ofstream& output, const mfem::Vector& values)
{
    output << time;

    for (int i = 0; i < values.Size(); ++i)
    {
        output << "\t" << values[i];
    }

    output << std::endl;
}


void PartitionVerticesByMetis(
    const mfem::SparseMatrix& vertex_edge,
    const mfem::Vector& edge_weight,
    const mfem::Array<int>& isolate_vertices,
    int num_partitions,
    mfem::Array<int>& partition,
    int degree = 1,
    bool use_edge_weight = false)
{
    mfem::SparseMatrix e_v = smoothg::Transpose(vertex_edge);
    e_v.ScaleRows(edge_weight);
    mfem::SparseMatrix vert_vert = smoothg::Mult(vertex_edge, e_v);

    MetisGraphPartitioner partitioner;
    partitioner.setUnbalanceTol(2.0);

    mfem::SparseMatrix vert_vert_ext(vert_vert);

    for (int i = 1; i < degree; ++i)
    {
        auto tmp = smoothg::Mult(vert_vert_ext, vert_vert);
        vert_vert_ext.Swap(tmp);
    }

    //    mfem::Array<int> connected_vertices;
    //    for (auto i : isolate_vertices)
    //    {
    //        GetTableRow(vert_vert_ext, i, connected_vertices);

    //        for (auto connection : connected_vertices)
    //        {
    //            if (connection != i)
    //            {
    //                partitioner.SetPostIsolateVertices(i);
    //            }
    //        }
    //    }

    partitioner.SetPostIsolateVertices(isolate_vertices);

    partitioner.doPartition(vert_vert, num_partitions, partition, use_edge_weight);
}

// extend edge_boundaryattr by adding empty rows corresponding to wells edges
void ExtendEdgeBoundaryattr(const std::vector<Well>& well_list,
                            mfem::SparseMatrix& edge_boundaryattr)
{
    const int old_nedges = edge_boundaryattr.Height();
    int num_well_cells = 0;
    for (auto& well : well_list)
        num_well_cells += well.cell_indices.size();

    int* new_i = new int[old_nedges + num_well_cells + 1];
    std::copy_n(edge_boundaryattr.GetI(), old_nedges + 1, new_i);
    std::fill_n(new_i + old_nedges + 1, num_well_cells, new_i[old_nedges]);

    mfem::SparseMatrix new_edge_boundaryattr(
        new_i, edge_boundaryattr.GetJ(), edge_boundaryattr.GetData(),
        old_nedges + num_well_cells, edge_boundaryattr.Width());

    edge_boundaryattr.Swap(new_edge_boundaryattr);
    new_edge_boundaryattr.SetGraphOwner(false);
    new_edge_boundaryattr.SetDataOwner(false);
    delete[] new_edge_boundaryattr.GetI();
}

// extend edge_bdr by adding new boundary to rows corresponding to wells edges
void AddEdgeBoundary(const std::vector<Well>& well_list,
                     mfem::SparseMatrix& edge_bdr,
                     mfem::Array<int>& well_marker)
{
    const int old_nedges = edge_bdr.Height();
    const int old_nnz = edge_bdr.NumNonZeroElems();

    int num_well_cells = 0;
    int new_nnz = old_nnz;

    for (Well& well : well_list)
    {
        int num_well_cells_i = well.cell_indices.size();
        num_well_cells += num_well_cells_i;
        new_nnz += num_well_cells_i;
    }

    int* new_i = new int[old_nedges + num_well_cells + 1];
    int* new_j = new int[new_nnz];
    double* new_data = new double[new_nnz];
    well_marker.SetSize(old_nedges + num_well_cells);
    well_marker = 0;

    std::fill_n(new_j, old_nnz, -5);
    std::copy_n(edge_bdr.GetI(), old_nedges + 1, new_i);
    std::copy_n(edge_bdr.GetJ(), old_nnz, new_j);
    std::copy_n(edge_bdr.GetData(), old_nnz, new_data);

    //int counter = old_nedges;
    int counter = 0;
    int new_attr = edge_bdr.Width();

    for (const Well& well : well_list)
    {
        int num_cells = well.cell_indices.size();

        if (well.type == WellType::Producer)
        {
            for (int j = 0; j < num_cells; ++j)
            {
                new_j[old_nnz + counter + j] = new_attr;
                new_data[old_nnz + counter + j] = 1.0;
                new_i[old_nedges + counter + j + 1] = new_i[old_nedges + counter] + j + 1;
                well_marker[old_nedges + counter + j] = 1;
            }
        }
        else
        {
            for (int j = 0; j < num_cells; ++j)
            {
                new_i[old_nedges + counter + j + 1] = new_i[old_nedges + counter];
            }
        }

        counter += num_cells;
    }

    assert(new_i[old_nedges + num_well_cells] == new_nnz);

    mfem::SparseMatrix new_edge_bdr(
        new_i, new_j, new_data,
        old_nedges + num_well_cells, edge_bdr.Width() + 1);

    edge_bdr.Swap(new_edge_bdr);
}

void CoarsenVertexEssentialCondition(
    const int num_wells, const int new_size,
    mfem::Array<int>& ess_marker, mfem::Vector& ess_data)
{
    mfem::Array<int> new_ess_marker(new_size);
    mfem::Vector new_ess_data(new_size);
    new_ess_marker = 0;
    new_ess_data = 0.0;

    const int old_size = ess_data.Size();

    for (int i = 0; i < num_wells; i++)
    {
        if (ess_marker[old_size - 1 - i])
        {
            new_ess_marker[new_size - 1 - i] = 1;
            new_ess_data(new_size - 1 - i) = ess_data(old_size - 1 - i);
        }
    }
    mfem::Swap(ess_marker, new_ess_marker);
    ess_data.Swap(new_ess_data);
}

void CoarsenSigmaEssentialCondition(
    const int num_wells, const int new_size,
    mfem::Array<int>& ess_marker)
{
    mfem::Array<int> new_ess_marker(new_size);
    new_ess_marker = 0;

    const int old_size = ess_marker.Size();

    for (int i = 0; i < num_wells; i++)
    {
        if (ess_marker[old_size - 1 - i])
        {
            new_ess_marker[new_size - 1 - i] = 1;
        }
    }

    mfem::Swap(ess_marker, new_ess_marker);
}

class TwoPhase : public SPE10Problem
{
public:
    TwoPhase(const char* permFile, int nDimensions, int spe10_scale,
             int slice, bool metis_parition, const mfem::Array<int>& ess_attr,
             int well_height = 5, double inject_rate = 1.0, double bottom_hole_pressure = 0.0);

    static double CellVolume()
    {
        return (20.0 * 10.0 * 2.0);// (nDimensions == 2 ) ? (20.0 * 10.0) : (20.0 * 10.0 * 2.0);
    }

    const std::vector<Well>& GetWells()
    {
        return well_manager_->GetWells();
    }

    const mfem::SparseMatrix& GetEdgeBoundaryAttributeTable() const
    {
        return edge_bdr_att_;
    }

    const mfem::Array<int>& GetEssentialEdgeDofsMarker() const
    {
        return ess_edof_marker_;
    }
    void PrintMeshWithPartitioning(mfem::Array<int>& partition);

private:
    void SetupPeaceman(int well_height, double inject_rate, double bot_hole_pres);

    Graph CombineReservoirAndWell();

    unique_ptr<WellManager> well_manager_;

    mfem::Vector bbmin_;
    mfem::Vector bbmax_;

    mfem::SparseMatrix edge_bdr_att_;
    mfem::Array<int> ess_edof_marker_;
};

TwoPhase::TwoPhase(const char* permFile, int nDimensions, int spe10_scale,
             int slice, bool metis_parition, const mfem::Array<int>& ess_attr,
             int well_height, double inject_rate, double bottom_hole_pressure)
    : SPE10Problem(permFile, nDimensions, spe10_scale, slice, metis_parition, ess_attr)
{
//    assert(num_procs_ == 1); // only works in serial for now
//    pmesh_->GetBoundingBox(bbmin_, bbmax_, 1);
    bbmin_ = 0.0;
    bbmax_[0] = 365.76;
    bbmax_[1] = 670.56;

    // Build wells (Peaceman's five-spot pattern)
    well_manager_ = make_unique<WellManager>(*pmesh_, *kinv_vector_);
    well_height = nDimensions == 2 ? 1 : std::min(N_[2], well_height);
    SetupPeaceman(well_height, inject_rate, bottom_hole_pressure);

    rhs_sigma_ = 0.0;
    rhs_u_ = 0.0;

    CombineReservoirAndWell();

    auto edge_bdr_att_tmp = GenerateBoundaryAttributeTable(pmesh_.get());
    edge_bdr_att_.Swap(edge_bdr_att_tmp);
    sigma_fes_->GetEssentialVDofs(ess_attr, ess_edof_marker_);

    mfem::Array<int> well_marker;
    ExtendEdgeBoundaryattr2(well_manager_->GetWells(), edge_bdr_att_, well_marker);

    well_marker = 0;
    std::copy_n(ess_edof_marker_.GetData(), ess_edof_marker_.Size(), well_marker.GetData());
    mfem::Swap(ess_edof_marker_, well_marker);
}

void TwoPhase::SetupPeaceman(int well_height, double inject_rate, double bot_hole_pres)
{
    const int num_wells = 5;

    std::vector<std::vector<int>> producer_well_cells(num_wells - 1);
    std::vector<std::vector<int>> injector_well_cells(1);

    mfem::DenseMatrix point(3, num_wells);
    point = ft;

    // Producers
    point(0, 0) = bbmin_[0] + ft;
    point(1, 0) = bbmin_[1] + ft;

    point(0, 1) = bbmax_[0] - ft;
    point(1, 1) = bbmin_[1] + ft;

    point(0, 2) = bbmin_[0] + ft;
    point(1, 2) = bbmax_[1] - ft;

    point(0, 3) = bbmax_[0] - ft;
    point(1, 3) = bbmax_[1] - ft;

    // Injector, Shifted to avoid middle
    point(0, 4) = ((bbmax_[0] - bbmin_[0]) / 2.0) + ft;
    point(1, 4) = ((bbmax_[1] - bbmin_[1]) / 2.0) + ft;

    for (int j = 0; j < well_height; ++j)
    {
        mfem::Array<int> ids;
        mfem::Array<mfem::IntegrationPoint> ips;

        pmesh_->FindPoints(point, ids, ips, false);

        // Producers
        for (int i = 0; i < num_wells - 1; ++i)
        {
            if (ids[i] >= 0)
            {
                producer_well_cells[i].push_back(ids[i]);
            }
        }

        // Injector
        if (ids[num_wells - 1] >= 0)
        {
            injector_well_cells[0].push_back(ids[num_wells - 1]);
        }

        // Shift Points for next layer
        for (int i = 0; i < 5; ++i)
        {
            point(2, i) += 2 * ft;
        }
    }

    for (const auto& cells : producer_well_cells)
    {
        if (cells.size() > 0)
        {
            well_manager_->AddWell(WellType::Producer, bot_hole_pres, cells);
        }
    }

    for (const auto& cells : injector_well_cells)
    {
        if (cells.size() > 0)
        {
            well_manager_->AddWell(WellType::Injector, inject_rate, cells);
        }
    }
}

Graph TwoPhase::CombineReservoirAndWell()
{
    const std::vector<Well>& well_list = well_manager_->GetWells();
    int num_well_cells = 0;
    for (const auto& well : well_list)
        num_well_cells += well.cell_indices.size();

    int num_injectors = 0;
    for (const auto& well : well_list)
    {
        num_injectors += (well.type == WellType::Injector);
    }

    const int num_reservoir_cells = vertex_edge_.Height();
    const int num_reservoir_faces = vertex_edge_.Width();
    const int new_nedges = num_reservoir_faces + num_well_cells;
    const int new_nvertices = num_reservoir_cells + num_injectors;

    // Copying the old data
    mfem::Vector new_weight(new_nedges);
    std::copy_n(weight_.GetData(), weight_.Size(), new_weight.GetData());
    mfem::Vector new_rhs_sigma(new_nedges);
    std::copy_n(rhs_sigma_.GetData(), rhs_sigma_.Size(), new_rhs_sigma.GetData());
    std::fill_n(new_rhs_sigma.GetData() + rhs_sigma_.Size(), num_well_cells, 0.0);
    mfem::Vector new_rhs_u(new_nvertices);
    std::copy_n(rhs_u_.GetData(), rhs_u_.Size(), new_rhs_u.GetData());

    mfem::SparseMatrix new_vertex_edge(new_nvertices, new_nedges);
    {
        int* vertex_edge_i = vertex_edge_.GetI();
        int* vertex_edge_j = vertex_edge_.GetJ();
        for (int i = 0; i < num_reservoir_cells; i++)
        {
            for (int j = vertex_edge_i[i]; j < vertex_edge_i[i + 1]; j++)
            {
                new_vertex_edge.Add(i, vertex_edge_j[j], 1.0);
            }
        }
    }

    // Adding well equations to the system
    int edge_counter = num_reservoir_faces;
    int injector_counter = 0;
    for (unsigned int i = 0; i < well_list.size(); i++)
    {
        const auto& well_cells = well_list[i].cell_indices;
        const auto& well_coeff = well_list[i].well_coefficients;

        if (well_list[i].type == WellType::Producer)
        {
            for (unsigned int j = 0; j < well_cells.size(); j++)
            {
                new_vertex_edge.Add(well_cells[j], edge_counter, 1.0);
                new_weight[edge_counter] = well_coeff[j];
                new_rhs_sigma[edge_counter] = well_list[i].value;

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
        }
        else
        {
            auto& local_weight_j = local_weight_[num_reservoir_cells + injector_counter];
            local_weight_j.SetSize(well_cells.size());
            local_weight_j = 1e10;//INFINITY; // Not sure if this is ok
            for (unsigned int j = 0; j < well_cells.size(); j++)
            {
                new_vertex_edge.Add(well_cells[j], edge_counter, 1.0);
                new_vertex_edge.Add(num_reservoir_cells + injector_counter, edge_counter, 1.0);
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
            new_rhs_u[num_reservoir_cells + injector_counter] = -1.0 * well_list[i].value;
            injector_counter++;
        }
    }
    new_vertex_edge.Finalize();

    vertex_edge_.Swap(new_vertex_edge);
    weight_.Swap(new_weight);
    rhs_sigma_.Swap(new_rhs_sigma);
    rhs_u_.Swap(new_rhs_u);

    auto edge_trueedge = ConcatenateIdentity(*sigma_fes_->Dof_TrueDof_Matrix(), num_well_cells);
    return Graph(vertex_edge_, edge_trueedge, weight_, &edge_bdratt_);
}

void TwoPhase::PrintMeshWithPartitioning(mfem::Array<int>& partition)
{
    std::stringstream fname;
    fname << "mesh.with_parts." << std::setfill('0') << std::setw(6) << myid_;
    std::ofstream ofid(fname.str().c_str());
    ofid.precision(8);
    pmesh_->PrintWithPartitioning(partition.GetData(), ofid, 1);
}
