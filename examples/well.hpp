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

enum WellType { Injector = 0x01, Producer = 0x02, Any = 0x03 };
enum WellDirection { X = 0, Y, Z };

struct Well
{
    WellType type; // injector or producer
    double value; // injector: total inject rate, producer: bottom hole pressure
    std::vector<int> cells; // cells in reservoir that contain this well
    std::vector<double> well_indices; // well indices of well cells
};

class WellManager
{
public:
    WellManager(mfem::Mesh& mesh, mfem::VectorCoefficient& perm_inv_coeff)
        : mesh_(mesh), ir_(mfem::IntRules.Get(mesh_.GetElementType(0), 1)),
          ip_(ir_.IntPoint(0)), perm_inv_coeff_(perm_inv_coeff),
          dim_(mesh_.Dimension()), dir_vec_(dim_), perp_dir_vec1_(dim_),
          perp_dir_vec2_(dim_) { }

    void AddWell(const WellType type,
                 const double value,
                 const std::vector<int>& cells,
                 const WellDirection direction = Z,
                 const double r_w = 0.01);

    const std::vector<Well>& GetWells() const { return wells_; }

    int NumWellCells(WellType type = Any) const;
    int NumWells(WellType type = Any) const;
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
};

int WellManager::NumWellCells(WellType type) const
{
    int num_well_cells = 0;
    for (const Well& well : wells_)
    {
        num_well_cells += well.type & type ? well.cells.size() : 0;
    }
    return num_well_cells;
}

int WellManager::NumWells(WellType type) const
{
    int num_wells = 0;
    for (const Well& well : wells_) { num_wells += well.type & type ? 1 : 0; }
    return num_wells;
}

void WellManager::SetDirectionVectors(int dir, int perp_dir1, int perp_dir2)
{
    dir_vec_ = perp_dir_vec1_ = perp_dir_vec2_ = 0.0;
    dir_vec_[dim_ == 3 ? dir : 0] = 1.0;
    perp_dir_vec1_[perp_dir1] = perp_dir_vec2_[perp_dir2] = 1.0;
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

        double h3 = dim_ < 3 ? sqrt(h1 * h2) : mesh_.GetElementSize(cell, dir_vec_);
        return 2 * M_PI * h3 * sqrt(k11 * k22) / std::log(equiv_radius / r_w);
    };

    std::vector<double> well_indices;
    well_indices.reserve(cells.size());

    if (type == WellType::Injector)
        std::cout<<"Injector:\n";
    else
        std::cout<<"Producer:\n";

    for (const auto& cell : cells)
    {
        auto Tr = mesh_.GetElementTransformation(cell);
        Tr->SetIntPoint(&ip_);
        mfem::Vector perm_inv;
        perm_inv_coeff_.Eval(perm_inv, *Tr, ip_);

        double k11 = 1. / perm_inv[perp_dir1];
        double k22 = 1. / perm_inv[perp_dir2];
        well_indices.push_back(WellIndex(cell, k11, k22));
//        std::cout<< "WI at "<<cell<<":"<<well_indices.back()<<"\n";
    }

    wells_.push_back({type, value, cells, well_indices});
}

enum AppendType { IOTA, FILL };

template<typename T>
T* Append(const T* in, int in_size, int append_size, T value, AppendType type)
{
    T* out = new T[in_size + append_size];
    std::copy_n(in, in_size, out);
    if (type == FILL) { std::fill_n(out + in_size, append_size, value); }
    else { std::iota(out + in_size, out + in_size + append_size, value); }
    return out;
}

unique_ptr<mfem::HypreParMatrix> ConcatenateIdentity(
    const mfem::HypreParMatrix& mat, const int id_size)
{
    MPI_Comm comm = mat.GetComm();

    mfem::SparseMatrix diag, offd;
    HYPRE_Int* old_colmap;
    mat.GetDiag(diag);
    mat.GetOffd(offd, old_colmap);

    // Append identity matrix diagonally to the bottom left of diag
    int* d_i = Append(diag.GetI(), diag.NumRows() + 1, id_size, NNZ(diag) + 1, IOTA);
    int* d_j = Append(diag.GetJ(), NNZ(diag), id_size, diag.NumCols(), IOTA);
    double* d_data = Append(diag.GetData(), NNZ(diag), id_size, 1.0, FILL);

    // Append zero matrix to the bottom of offd
    int* o_i = Append(offd.GetI(), offd.NumRows() + 1, id_size, NNZ(offd), FILL);
    int* o_j = Append(offd.GetJ(), NNZ(offd), 0, 0, FILL);
    double* o_data = Append(offd.GetData(), NNZ(offd), 0, 0.0, FILL);

    mfem::Array<HYPRE_Int> row_starts, col_starts;
    GenerateOffsets(comm, diag.NumRows() + id_size, row_starts);
    GenerateOffsets(comm, diag.NumCols() + id_size, col_starts);

    const int old_start = mat.ColPart()[0];
    mfem::Array<int> col_change(mat.N());
    col_change = 0;
    std::fill_n(col_change + old_start, mat.NumCols(), col_starts[0] - old_start);

    mfem::Array<int> col_remap(mat.N());
    col_remap = 0;
    MPI_Allreduce(col_change, col_remap, mat.N(), HYPRE_MPI_INT, MPI_SUM, comm);

    HYPRE_Int* colmap = new HYPRE_Int[offd.NumCols()];
    for (int i = 0; i < offd.NumCols(); ++i)
    {
        colmap[i] = old_colmap[i] + col_remap[old_colmap[i]];
    }

    auto out = new mfem::HypreParMatrix(comm, row_starts.Last(), col_starts.Last(),
                                        row_starts, col_starts, d_i, d_j, d_data,
                                        o_i, o_j, o_data, offd.NumCols(), colmap);
    out->CopyRowStarts();
    out->CopyColStarts();

    return unique_ptr<mfem::HypreParMatrix>(out);
}

mfem::SparseMatrix ExtendVertexEdge(const mfem::SparseMatrix& vert_edge,
                                    const WellManager& well_manager)
{
    const int num_well_cells = well_manager.NumWellCells();
    const int num_injector_cells = well_manager.NumWellCells(Injector);

    int num_verts = vert_edge.NumRows();         // number of reservoir cells
    int num_edges = vert_edge.NumCols();         // number of reservoir faces

    int* I = new int[num_verts + well_manager.NumWells(Injector) + 1]();
    int* J = new int[NNZ(vert_edge) + num_well_cells + num_injector_cells];

    for (const Well& well : well_manager.GetWells())
    {
        for (int cell : well.cells) { I[cell + 1] = 1; }
    }

    for (int i = 0; i < vert_edge.NumRows(); ++i)
    {
        I[i + 1] += I[i] + vert_edge.RowSize(i);
        std::copy_n(vert_edge.GetRowColumns(i), vert_edge.RowSize(i), J + I[i]);
    }

    for (const Well& well : well_manager.GetWells())
    {
        if (well.type == Injector)
        {
            I[num_verts + 1] = I[num_verts] + well.cells.size();
            std::iota(J + I[num_verts], J + I[num_verts + 1], num_edges);
            num_verts++;
        }
        for (int cell : well.cells) { J[I[cell + 1] - 1] = num_edges++; }
    }

    double* Data = new double[I[num_verts]];
    std::fill_n(Data, I[num_verts], 1.0);

    return mfem::SparseMatrix(I, J, Data, num_verts, num_edges);
}

std::vector<mfem::Vector> AppendWellIndex(const std::vector<mfem::Vector>& loc_weight,
                                          const WellManager& well_manager)
{
    std::vector<mfem::Vector> new_loc_weight;
    new_loc_weight.reserve(loc_weight.size() + well_manager.NumWells(Injector));
    for (const auto& loc_w : loc_weight) { new_loc_weight.push_back(loc_w); }

    for (const Well& well : well_manager.GetWells())
    {
        for (unsigned int j = 0; j < well.cells.size(); j++)
        {
            const mfem::Vector& loc_w = loc_weight[well.cells[j]];
            mfem::Vector& new_loc_w = new_loc_weight[well.cells[j]];
            new_loc_w.SetSize(loc_w.Size() + 1);
            std::copy_n(loc_w.GetData(), loc_w.Size(), new_loc_w.GetData());
            new_loc_w[loc_w.Size()] = well.well_indices[j];
        }

        if (well.type == Injector)
        {
            mfem::Vector new_loc_w(well.cells.size());
            new_loc_w = 1e10;//INFINITY; // Not sure if this is ok
            new_loc_weight.push_back(new_loc_w);
        }
    }

    return new_loc_weight;
}

mfem::Vector AppendWellData(const mfem::Vector& vec,
                            const WellManager& well_manager,
                            WellType type)
{
    const int append_size = type & Producer ? well_manager.NumWellCells()
                            : well_manager.NumWells(Injector);

    mfem::Vector combined_vec(vec.Size() + append_size);
    std::copy_n(vec.GetData(), vec.Size(), combined_vec.GetData());

    double* data = combined_vec.GetData() + vec.Size();
    for (const Well& w : well_manager.GetWells())
    {
        const int size = w.cells.size();
        switch (type)
        {
            case Any: std::copy_n(w.well_indices.data(), size, data); break;
            case Injector: if (w.type == type) *(data++) = w.value; break;
            case Producer: std::fill_n(data, size, w.type == type ? w.value : 0.0);
        }
        data += type & Producer ? size : 0;
    }

    return combined_vec;
}

mfem::SparseMatrix ExtendEdgeBoundary(const mfem::SparseMatrix& edge_bdr,
                                      const WellManager& well_manager)
{
    const int num_edges = edge_bdr.NumRows() + well_manager.NumWellCells();
    const int nnz = NNZ(edge_bdr) + well_manager.NumWellCells(Producer);

    int* I = new int[num_edges + 1];
    int* J = new int[nnz];
    double* Data = new double[nnz];

    std::copy_n(edge_bdr.GetI(), edge_bdr.NumRows() + 1, I);
    std::copy_n(edge_bdr.GetJ(), NNZ(edge_bdr), J);
    std::fill_n(Data, nnz, 1.0);

    int* I_ptr = I + edge_bdr.NumRows() + 1;
    int* J_ptr = J + NNZ(edge_bdr);
    int bdr = edge_bdr.NumCols();

    for (const Well& well : well_manager.GetWells())
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

class TwoPhase : public SPE10Problem
{
public:
    TwoPhase(const char* perm_file, int dim, int spe10_scale,
             int slice, bool metis_parition, const mfem::Array<int>& ess_attr,
             int well_height, double inject_rate, double bottom_hole_pressure);

    const mfem::Array<int>& BlockOffsets() const { return block_offsets_; }
private:
    // Set up well model (Peaceman's five-spot pattern)
    void SetWells(int well_height, double inject_rate, double bot_hole_pres);
    void SetWells(const std::vector<std::vector<int>>& inj_well_cells,
                  const std::vector<std::vector<int>>& prod_well_cells,
                  double inject_rate, double bhp);
    void CombineReservoirAndWellModel();
    void MetisPart(const mfem::Array<int>& coarsening_factor, mfem::Array<int>& partitioning) const;

    mfem::Vector ComputeVertWeight();

    unique_ptr<mfem::HypreParMatrix> combined_edge_trueedge_;
    WellManager well_manager_;
};

TwoPhase::TwoPhase(const char* perm_file, int dim, int spe10_scale, int slice,
                   bool metis_parition, const mfem::Array<int>& ess_attr,
                   int well_height, double inject_rate, double bottom_hole_pressure)
    : SPE10Problem(perm_file, dim, spe10_scale, slice, metis_parition, ess_attr),
      well_manager_(*mesh_, *kinv_vector_)
{
    rhs_sigma_ = 0.0;
    rhs_u_ = 0.0;

    SetWells(well_height, inject_rate, bottom_hole_pressure);
    CombineReservoirAndWellModel();

    block_offsets_.SetSize(4);
    block_offsets_[0] = 0;
    block_offsets_[1] = vertex_edge_.NumCols();
    block_offsets_[2] = block_offsets_[1] + vertex_edge_.NumRows();
    block_offsets_[3] = block_offsets_[2] + vertex_edge_.NumRows();

    vert_weight_ = ComputeVertWeight();
}

mfem::Vector TwoPhase::ComputeVertWeight()
{
    mfem::Array<int> max_N(3);
    max_N[0] = 60;
    max_N[1] = 220;
    max_N[2] = 85;//85;

    double poro_min = 0.01;

    mfem::Vector vert_weight(vertex_edge_.NumRows());
    vert_weight_ = 1.0;

    // Read Porosity data file
    std::ifstream poro_file("spe_phi.dat");
    double* ip = vert_weight.GetData();
    double unneeded;

//    // map from mfem cell id to mrst cell id
//    mfem::SparseMatrix mfem_to_mrst_map(mesh_->GetNE(), mesh_->GetNE());
//    mfem::Array<int> ids;
//    mfem::Array<mfem::IntegrationPoint> ips;
//    mfem::DenseMatrix point(mesh_->Dimension(), 1);
//    int count = 0;

    for (int k = 0; k < N_[2]; k++)
    {
//        point(2, 0) = k*2.0*ft_ + ft_;
        for (int j = 0; j < N_[1]; j++)
        {
//            point(1, 0) = j*10.0*ft_ + 5.0*ft_;
            for (int i = 0; i < N_[0]; i++)
            {
                poro_file >> *ip;
                *ip = std::max(*ip, poro_min);
                ip++;

//                point(0, 0) = i*20.0*ft_ + 10*ft_;
//                if (count % 1000 == 0) { std::cout<<"have found IDs for " << count << " cells\n"; }
//                mesh_->FindPoints(point, ids, ips, false);
//                if (count < 20) { std::cout << count << " " << ids[0] << "\n"; }
//                mfem_to_mrst_map.Add(count, ids[0], 1.0);
//                count++;
            }
            for (int i = 0; i < max_N[0] - N_[0]; i++)
                poro_file >> unneeded; // skip unneeded part
        }
        for (int j = 0; j < max_N[1] - N_[1]; j++)
            for (int i = 0; i < max_N[0]; i++)
                poro_file >> unneeded;  // skip unneeded part
    }

//    mfem_to_mrst_map.Finalize();
//    std::ofstream mat_file("mfem_to_mrst_map.txt");
//    mfem_to_mrst_map.PrintMatlab(mat_file);

    std::cout << " sum(poro) = " << vert_weight.Sum() << "\n";

    vert_weight *= 20.0 * 10.0 * 2.0 * std::pow(ft_, 3); // cell volume

    return vert_weight;
}

void TwoPhase::SetWells(const std::vector<std::vector<int>>& inj_well_cells,
                        const std::vector<std::vector<int>>& prod_well_cells,
                        double inject_rate, double bhp)
{
    for (unsigned int i = 0; i < inj_well_cells.size(); ++i)
    {
        well_manager_.AddWell(Injector, inject_rate, inj_well_cells[i]);
    }
    for (unsigned int i = 0; i < prod_well_cells.size(); ++i)
    {
        well_manager_.AddWell(Producer, bhp, prod_well_cells[i]);
    }
}

void TwoPhase::SetWells(int well_height, double inject_rate, double bhp)
{
    const int num_wells = 5;
    std::vector<std::vector<int>> cells(num_wells);

    const double max_x = 365.76 - ft_;
    const double max_y = 670.56 - ft_;

    mfem::DenseMatrix point(mesh_->Dimension(), num_wells);
    point = ft_;
//    point(0, 1) = max_x;
//    point(1, 2) = max_y;
//    point(0, 3) = max_x;
//    point(1, 3) = max_y;
////    point(0, 4) = ((max_x + ft_) / 2.0) + ft_;
////    point(1, 4) = ((max_y + ft_) / 2.0) + ft_;
//    point(0, 4) = 185.5; // 182.5;
//    point(1, 4) = 336.5; // 335.0; // 258.0; //
    point(0, 2) = max_x;
    point(1, 4) = max_y;
    point(0, 3) = max_x;
    point(1, 3) = max_y;
    point(0, 0) = 182.5; // 185.5; // 182.5;
    point(1, 0) = 335.0; // 336.5; // 335.0; // 258.0; //


    for (int j = 0; j < well_height; ++j)
    {
        mfem::Array<int> ids;
        mfem::Array<mfem::IntegrationPoint> ips;
        mesh_->FindPoints(point, ids, ips, false);

        for (int i = 0; i < num_wells; ++i)
        {
            if (ids[i] >= 0) { cells[i].push_back(ids[i]); }
            if (mesh_->Dimension() == 3) { point(2, i) += 2.0 * ft_; }
        }
    }

    for (int i = 0; i < num_wells; ++i)
    {
//        WellType type = (i == num_wells - 1) ? Injector : Producer;
//        double value = (i == num_wells - 1) ? inject_rate : bhp;
        WellType type = (i == 0) ? Injector : Producer;
        double value = (i == 0) ? inject_rate : bhp;
        if (cells[i].size())
        {
            well_manager_.AddWell(type, value, cells[i], WellDirection::Z, 0.127);
        }
    }

//    for (int i = 0; i < num_wells; ++i)
//    {
//        double value = (i == num_wells - 1) ? inject_rate : (inject_rate / -4.0);
//        if (cells[i].size()) { well_manager_.AddWell(Injector, value, cells[i]); }
//    }

    //    for (int i = 0; i < num_wells; ++i)
    //    {
    //        double value = (i == num_wells - 1) ? 1e6 : 1e5;
    //        if (cells[i].size()) { well_manager_.AddWell(Producer, value, cells[i]); }
    //    }
}

void TwoPhase::CombineReservoirAndWellModel()
{
    vertex_edge_ = ExtendVertexEdge(vertex_edge_, well_manager_);
    local_weight_ = AppendWellIndex(local_weight_, well_manager_);
    weight_ = AppendWellData(weight_, well_manager_, Any);
    edge_bdr_ = ExtendEdgeBoundary(edge_bdr_, well_manager_);

    rhs_sigma_ = AppendWellData(rhs_sigma_, well_manager_, Producer);
    rhs_u_ = AppendWellData(rhs_u_, well_manager_, Injector);

    combined_edge_trueedge_ = ConcatenateIdentity(*sigma_fes_->Dof_TrueDof_Matrix(),
                                                  well_manager_.NumWellCells());
    edge_trueedge_ = combined_edge_trueedge_.get();

    mfem::Array<int> producer_attr(well_manager_.NumWells(Producer));
    producer_attr = 0;                // treat producer as "natural boundary"
    ess_attr_.Append(producer_attr);
}

void TwoPhase::MetisPart(const mfem::Array<int>& coarsening_factor,
                         mfem::Array<int>& partitioning) const
{
    const int dim = mesh_->Dimension();

    mfem::SparseMatrix scaled_vert_edge(vertex_edge_);
    if (dim == 3)
    {
        mfem::Vector weight_sqrt(weight_);
        for (int i = 0; i < weight_.Size(); ++i)
        {
            weight_sqrt[i] = std::sqrt(weight_[i]);
        }
        scaled_vert_edge.ScaleColumns(weight_sqrt);
    }

    const int xy_cf = coarsening_factor[0] * coarsening_factor[1];
    const int metis_cf = xy_cf * (dim > 2 ? coarsening_factor[2] : 1);

    std::vector<std::vector<int>> iso_verts;
    iso_vert_count_ = well_manager_.NumWells(Injector);
    for (int i = 0; i < iso_vert_count_; ++i)
    {
        iso_verts.push_back(std::vector<int>(1, mesh_->GetNE() + i));
    }

    PartitionAAT(scaled_vert_edge, partitioning, metis_cf, dim > 2, iso_verts);
}

class NorneModel : public DarcyProblem
{
    void SetupMesh();
public:
    NorneModel(MPI_Comm comm, const mfem::Array<int>& ess_attr)
        : DarcyProblem(comm, 3, ess_attr) { SetupMesh(); InitGraph(); }
};

void NorneModel::SetupMesh()
{
    std::ifstream imesh("/Users/lee1029/Downloads/norne/new_norne_HEX.vtk");
    mfem::Mesh serial_mesh(imesh, 1, 1);
    mesh_.reset(new mfem::ParMesh(comm_, serial_mesh));
}

class SaigupModel : public DarcyProblem
{
    void SetupMesh(bool refined);
public:
    SaigupModel(MPI_Comm comm, bool refined, const mfem::Array<int>& ess_attr)
        : DarcyProblem(comm, 3, ess_attr) { SetupMesh(refined); InitGraph(); }
};

void SaigupModel::SetupMesh(bool refined)
{
    std::string file = refined ? "refined_saigup/refined_saigup_HEX.vtk" : "saigup/saigup_HEX.vtk";
    std::ifstream imesh("/Users/lee1029/Downloads/"+file);
    mfem::Mesh serial_mesh(imesh, 1, 1);
    mesh_.reset(new mfem::ParMesh(comm_, serial_mesh));
}


class TwoPhaseEGG : public EggModel
{
public:
    TwoPhaseEGG(int well_height, double inject_rate, double bottom_hole_pressure);

    const mfem::Array<int>& BlockOffsets() const { return block_offsets_; }
private:
    void SetWells(int well_height, double inject_rate, double bot_hole_pres);
    void SetWells(const std::vector<std::vector<int>>& inj_well_cells,
                  const std::vector<std::vector<int>>& prod_well_cells,
                  double inject_rate, double bhp);
    void CombineReservoirAndWellModel();
    void MetisPart(const mfem::Array<int>& coarsening_factor, mfem::Array<int>& partitioning) const;

    mfem::Vector ComputeVertWeight();

    unique_ptr<mfem::HypreParMatrix> combined_edge_trueedge_;
    WellManager well_manager_;
};

mfem::Vector TwoPhaseEGG::ComputeVertWeight()
{
    mfem::Vector vert_weight(vert_weight_);
    vert_weight *= 0.2;
    return vert_weight;
}

void TwoPhaseEGG::SetWells(const std::vector<std::vector<int>>& inj_well_cells,
                        const std::vector<std::vector<int>>& prod_well_cells,
                        double inject_rate, double bhp)
{
    for (unsigned int i = 0; i < inj_well_cells.size(); ++i)
    {
        well_manager_.AddWell(Injector, inject_rate, inj_well_cells[i]);
    }
    for (unsigned int i = 0; i < prod_well_cells.size(); ++i)
    {
        well_manager_.AddWell(Producer, bhp, prod_well_cells[i]);
    }
}

void TwoPhaseEGG::SetWells(int well_height, double inject_rate, double bhp)
{
    const int num_wells = 5;
    std::vector<std::vector<int>> cells(num_wells);

    const double max_x = 365.76 - ft_;
    const double max_y = 670.56 - ft_;

    mfem::DenseMatrix point(mesh_->Dimension(), num_wells);
    point = ft_;
    point(0, 2) = max_x;
    point(1, 4) = max_y;
    point(0, 3) = max_x;
    point(1, 3) = max_y;
    point(0, 0) = 182.5; // 185.5; // 182.5;
    point(1, 0) = 335.0; // 336.5; // 335.0; // 258.0; //


    for (int j = 0; j < well_height; ++j)
    {
        mfem::Array<int> ids;
        mfem::Array<mfem::IntegrationPoint> ips;
        mesh_->FindPoints(point, ids, ips, false);

        for (int i = 0; i < num_wells; ++i)
        {
            if (ids[i] >= 0) { cells[i].push_back(ids[i]); }
            if (mesh_->Dimension() == 3) { point(2, i) += 2.0 * ft_; }
        }
    }

    for (int i = 0; i < num_wells; ++i)
    {
//        WellType type = (i == num_wells - 1) ? Injector : Producer;
//        double value = (i == num_wells - 1) ? inject_rate : bhp;
        WellType type = (i == 0) ? Injector : Producer;
        double value = (i == 0) ? inject_rate : bhp;
        if (cells[i].size())
        {
            well_manager_.AddWell(type, value, cells[i], WellDirection::Z, 0.127);
        }
    }
}

void TwoPhaseEGG::CombineReservoirAndWellModel()
{
    vertex_edge_ = ExtendVertexEdge(vertex_edge_, well_manager_);
    local_weight_ = AppendWellIndex(local_weight_, well_manager_);
    weight_ = AppendWellData(weight_, well_manager_, Any);
    edge_bdr_ = ExtendEdgeBoundary(edge_bdr_, well_manager_);

    rhs_sigma_ = AppendWellData(rhs_sigma_, well_manager_, Producer);
    rhs_u_ = AppendWellData(rhs_u_, well_manager_, Injector);

    combined_edge_trueedge_ = ConcatenateIdentity(*sigma_fes_->Dof_TrueDof_Matrix(),
                                                  well_manager_.NumWellCells());
    edge_trueedge_ = combined_edge_trueedge_.get();

    mfem::Array<int> producer_attr(well_manager_.NumWells(Producer));
    producer_attr = 0;                // treat producer as "natural boundary"
    ess_attr_.Append(producer_attr);
}

void TwoPhaseEGG::MetisPart(const mfem::Array<int>& coarsening_factor,
                            mfem::Array<int>& partitioning) const
{
    const int dim = mesh_->Dimension();

    mfem::SparseMatrix scaled_vert_edge(vertex_edge_);
    if (dim == 3)
    {
        mfem::Vector weight_sqrt(weight_);
        for (int i = 0; i < weight_.Size(); ++i)
        {
            weight_sqrt[i] = std::sqrt(weight_[i]);
        }
        scaled_vert_edge.ScaleColumns(weight_sqrt);
    }

    const int xy_cf = coarsening_factor[0] * coarsening_factor[1];
    const int metis_cf = xy_cf * (dim > 2 ? coarsening_factor[2] : 1);

    std::vector<std::vector<int>> iso_verts;
    iso_vert_count_ = well_manager_.NumWells(Injector);
    for (int i = 0; i < iso_vert_count_; ++i)
    {
        iso_verts.push_back(std::vector<int>(1, mesh_->GetNE() + i));
    }

    PartitionAAT(scaled_vert_edge, partitioning, metis_cf, dim > 2, iso_verts);
}


class LocalProblem : public DarcyProblem
{
public:
    LocalProblem(MPI_Comm comm, int dim, const mfem::Array<int>& ess_attr);

    const mfem::Array<int>& BlockOffsets() const { return block_offsets_; }
private:
    void SetWells();
    void SetWells(const std::vector<std::vector<int>>& inj_well_cells,
                  const std::vector<std::vector<int>>& prod_well_cells,
                  double inject_rate, double bhp);
    void CombineReservoirAndWellModel();

    unique_ptr<mfem::HypreParMatrix> combined_edge_trueedge_;
    unique_ptr<WellManager> well_manager_ptr;
    mfem::Vector coef_;
};

LocalProblem::LocalProblem(
        MPI_Comm comm, int dim, const mfem::Array<int>& ess_attr)
    : DarcyProblem(comm, dim, ess_attr)
{
    mfem::Mesh mesh(29,29, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
    mesh_.reset(new mfem::ParMesh(comm, mesh));

    for (int i = 0; i < mesh_->GetNE(); ++i)
    {
        mesh_->SetAttribute(i, i+1);
    }

    InitGraph();

    coef_.SetSize(mesh_->GetNE());
    coef_ = 1;
    double low_perm_value = 1e5;
    int row_size = 29;
    std::fill_n(coef_.GetData()+7*row_size, 3*row_size, low_perm_value);
    std::fill_n(coef_.GetData()+13*row_size, 3*row_size, low_perm_value);
    std::fill_n(coef_.GetData()+19*row_size, 3*row_size, low_perm_value);

    kinv_vector_ = make_unique<mfem::VectorArrayCoefficient>(dim);
    for (int d = 0; d < dim; ++d)
    {
        auto kinv = new mfem::PWConstCoefficient(coef_);
        dynamic_cast<mfem::VectorArrayCoefficient&>(*kinv_vector_).Set(d, kinv);
    }

    ComputeGraphWeight();

    rhs_sigma_ = 0.0;
    rhs_u_ = 0.0;

    well_manager_ptr.reset(new WellManager(*mesh_, *kinv_vector_));
    SetWells();
    CombineReservoirAndWellModel();

    block_offsets_.SetSize(3);
    block_offsets_[0] = 0;
    block_offsets_[1] = vertex_edge_.NumCols();
    block_offsets_[2] = block_offsets_[1] + vertex_edge_.NumRows();

    // print perm
    {
        mfem::Vector coef_print(coef_);
        low_perm_value = 1e-5;
        std::fill_n(coef_print.GetData()+7*row_size, 3*row_size, low_perm_value);
        std::fill_n(coef_print.GetData()+13*row_size, 3*row_size, low_perm_value);
        std::fill_n(coef_print.GetData()+19*row_size, 3*row_size, low_perm_value);
        //    std::iota(coef.GetData(), coef.GetData()+coef.Size(), 0);

        mfem::socketstream souts;
        VisSetup(souts, coef_print, 0.0, 0.0, "Permeability", false, true);
    }

//    vert_weight_ = ComputeVertWeight();
}

void LocalProblem::SetWells()
{
    const int num_wells = 1;
    std::vector<std::vector<int>> cells(num_wells);

    int well_height = 19;
    double h = 1.0/29;
    mfem::DenseMatrix point(mesh_->Dimension(), num_wells);
    point(0, 0) = 0.5;
    point(1, 0) = 5*h + h/2;

    for (int j = 0; j < well_height; ++j)
    {
        mfem::Array<int> ids;
        mfem::Array<mfem::IntegrationPoint> ips;
        mesh_->FindPoints(point, ids, ips, false);

        for (int i = 0; i < num_wells; ++i)
        {
            if (ids[i] >= 0)
            {
                cells[i].push_back(ids[i]);
            }
            { point(1, i) += h; }
        }
    }

    double inject_rate = 1.0;
    double bhp = 0.0;
    for (int i = 0; i < num_wells; ++i)
    {
        WellType type = (i == 0) ? Injector : Producer;
        double value = (i == 0) ? inject_rate : bhp;
        if (cells[i].size())
        {
            well_manager_ptr->AddWell(type, value, cells[i], WellDirection::Z, 0.05/29);
        }
    }

    mfem::Vector well_loc(mesh_->GetNE());
    well_loc = 0.0;
    for (int cell : cells[0]) { well_loc[cell] = 1; }
    mfem::socketstream souts;
    VisSetup(souts, well_loc, 0.0, 0.0, "Well cells", false, true);
}

void LocalProblem::CombineReservoirAndWellModel()
{
    WellManager& well_manager_ = *well_manager_ptr;
    vertex_edge_ = ExtendVertexEdge(vertex_edge_, well_manager_);
    local_weight_ = AppendWellIndex(local_weight_, well_manager_);
    weight_ = AppendWellData(weight_, well_manager_, Any);
    edge_bdr_ = ExtendEdgeBoundary(edge_bdr_, well_manager_);

    rhs_sigma_ = AppendWellData(rhs_sigma_, well_manager_, Producer);
    rhs_u_ = AppendWellData(rhs_u_, well_manager_, Injector);

    combined_edge_trueedge_ = ConcatenateIdentity(*sigma_fes_->Dof_TrueDof_Matrix(),
                                                  well_manager_.NumWellCells());
    edge_trueedge_ = combined_edge_trueedge_.get();

    mfem::Array<int> producer_attr(well_manager_.NumWells(Producer));
    producer_attr = 0;                // treat producer as "natural boundary"
    ess_attr_.Append(producer_attr);
}
