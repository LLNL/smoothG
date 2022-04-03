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
   @file singlephase.cpp
   @brief This is an example for upscaling a graph Laplacian coming from a finite
   volume discretization of a single phase flow and transport model in parallel.

   A simple way to run the example:

   mpirun -n 4 ./singlephase
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "mfem.hpp"
#include "well.hpp"

#include "../src/picojson.h"
#include "../src/smoothG.hpp"

using namespace smoothg;

double mu_o = 3e-3; //0.005; //0.0002; //
double mu_w = 3e-4; // 0.3*centi*poise //1e-3;
int relperm_order = 2;
//mfem::Array<int> well_cells;
mfem::Array<int> well_perforations;
mfem::Vector well_cell_gf;

mfem::Vector PWConstProject(const MixedMatrix& darcy_system, const mfem::Vector& x)
{
    mfem::Vector S;
//    if (PWConst_S_)
//    {
////        S.SetDataAndSize(x.GetData(), x.Size());
//        S = darcy_system.PWConstProjectS(x);
//    }
//    else
//    {
        S = darcy_system.PWConstProject(x);
//    }
    return S;
}

/// Constructs a solver which is a combination of a given pair of solvers
/// TwoStageSolver * x = solver2 * (I - A * solver1 ) * x + solver1 * x
class TwoStageSolver : public mfem::Solver
{
protected:
    const mfem::Operator& solver1_;
    const mfem::Operator& solver2_;
    const mfem::Operator& op_;
    // additional memory for storing intermediate results
    mutable mfem::Vector tmp1;
    mutable mfem::Vector tmp2;

public:
    virtual void SetOperator(const Operator &op) override { }
    TwoStageSolver(const mfem::Operator& solver1, const mfem::Operator& solver2, const mfem::Operator& op) :
        solver1_(solver1), solver2_(solver2), op_(op),  tmp1(op.NumRows()), tmp2(op.NumRows()) { }

    void Mult(const mfem::Vector & x, mfem::Vector & y) const override
    {
        solver1_.Mult(x, y);
        op_.Mult(y, tmp1);
        tmp1 -= x;
        solver2_.Mult(tmp1, tmp2);
        y -= tmp2;
    }
};

/// Hypre ILU Preconditioner
class HypreILU : public mfem::HypreSolver
{
   HYPRE_Solver ilu_precond;
public:
   HypreILU(mfem::HypreParMatrix &A, int type = 0, int fill_level = 1)
       : HypreSolver(&A)
   {
      HYPRE_ILUCreate(&ilu_precond);
      HYPRE_ILUSetMaxIter( ilu_precond, 1 );
      HYPRE_ILUSetTol( ilu_precond, 0.0 );
      HYPRE_ILUSetType( ilu_precond, type );
      HYPRE_ILUSetLevelOfFill( ilu_precond, fill_level );
      HYPRE_ILUSetDropThreshold( ilu_precond, 1e-2 );
      HYPRE_ILUSetMaxNnzPerRow( ilu_precond, 100 );
      HYPRE_ILUSetLocalReordering(ilu_precond, type == 0 ? false : true);
   }

   virtual void SetOperator(const mfem::Operator &op) { }

   virtual operator HYPRE_Solver() const { return ilu_precond; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ILUSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ILUSolve; }

   virtual ~HypreILU() { HYPRE_ILUDestroy(ilu_precond); }
};

class TwoPhaseHybrid : public HybridSolver
{
    mfem::Array<int> offsets_;
    unique_ptr<mfem::BlockOperator> op_;
    unique_ptr<mfem::SparseMatrix> mono_mat_;
    unique_ptr<mfem::HypreParMatrix> monolithic_;
    unique_ptr<mfem::Solver> A00_inv_;
    unique_ptr<mfem::Solver> A11_inv_;
    unique_ptr<mfem::BlockLowerTriangularPreconditioner> stage1_prec_;
    unique_ptr<HypreILU> stage2_prec_;

    // B00_ and B01_ are the (0,0) and (0,1)-block of [M D^T; D 0]^{-1}
    std::vector<mfem::DenseMatrix> B00_;  // M^{-1} - B01_ D M^{-1}
    std::vector<mfem::DenseMatrix> B01_;  // M^{-1} D^T (DM^{-1}D^T)^{-1}

    double dt_density_;

    mfem::Array<int> ess_redofs_;
    unique_ptr<mfem::SparseMatrix> op3_;
    unique_ptr<mfem::UMFPackSolver> solver3_;
    unique_ptr<mfem::HypreParMatrix> schur_;
    unique_ptr<mfem::HypreParMatrix> schur22_;
    unique_ptr<mfem::HypreParMatrix> schur33_;

    const std::vector<mfem::DenseMatrix>* dTdsigma_;
    const std::vector<mfem::DenseMatrix>* dMdS_;

    mutable double resid_norm_;

    void Init();
    mfem::BlockVector MakeHybridRHS(const mfem::BlockVector& rhs) const;
    void BackSubstitute(const mfem::BlockVector& rhs,
                        const mfem::BlockVector& sol_hb,
                        mfem::BlockVector& sol) const;
public:
    TwoPhaseHybrid(const MixedMatrix& mgL, const mfem::Array<int>* ess_attr)
        : HybridSolver(mgL, ess_attr), offsets_(3), B00_(nAggs_), B01_(nAggs_)
    { Init(); }

    void AssembleSolver(mfem::Vector elem_scaling_inverse,
                        const std::vector<mfem::DenseMatrix>& dMdS,
                        const std::vector<mfem::DenseMatrix>& dTdsigma,
                        const mfem::HypreParMatrix& dTdS,
                        double dt_density_ = 1.0);

    void Mult(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const override;
    double GetResidualNorm() const { return resid_norm_; }
};

enum SteppingScheme { IMPES = 1, SequentiallyImplicit, FullyImplcit };

struct EvolveParamenters
{
    double total_time = 10.0;    // Total time
    double dt = 1.0;   // Time step size
    int vis_step = 0;
    SteppingScheme scheme = IMPES;
};

void SetOptions(FASParameters& param, bool use_vcycle, int num_backtrack, double diff_tol);

mfem::Vector TotalMobility(const mfem::Vector& S);
mfem::Vector dTMinv_dS(const mfem::Vector& S);
mfem::Vector FractionalFlow(const mfem::Vector& S);
mfem::Vector dFdS(const mfem::Vector& S);

DarcyProblem* problem_ptr;
std::vector<mfem::socketstream> sout_resid_(50); // this should not be needed (only for debug)

/**
   This computes dS/dt that solves W dS/dt + Adv F(S) = b, which is the
   semi-discrete form of dS/dt + div(vF(S)) = b, where W and Adv are the mass
   and advection matrices, F is a nonlinear function, b is the influx source.
 */
class TwoPhaseSolver
{
    const int level_;
    const EvolveParamenters& evolve_param_;
    const FASParameters& solver_param_;
    const DarcyProblem& problem_;
    Hierarchy* hierarchy_;
    std::vector<mfem::BlockVector> blk_helper_;

    unique_ptr<Hierarchy> new_hierarchy_;

    mfem::Array<int> blk_offsets_;
    unique_ptr<mfem::BlockVector> source_;
    unique_ptr<mfem::HypreParMatrix> D_te_e_;
    int nonlinear_iter_;
    bool step_converged_;
    int num_coarse_lin_iter_ = 0;
    int num_coarse_lin_solve_ = 0;

    std::vector<mfem::Vector> micro_upwind_flux_;

    std::vector<int> step_nonlinear_iter_;
    std::vector<int> step_num_backtrack_;
    std::vector<int> step_coarsest_nonlinear_iter_;
    std::vector<int> step_average_coarsest_linear_iter_;
    std::vector<int> step_average_fine_linear_iter_;
    std::vector<int> step_average_mid_linear_iter_;
    std::vector<double> step_time_;
    std::vector<double> cumulative_step_time_;
    std::vector<double> step_CFL_const_;

    // TODO: these should be defined in / extracted from the problem, not here
    const double density_ = 1.0252e3;  // 64*pound/(ft^3)
    const double porosity_ = problem_ptr->CellVolume() == 256.0 ? 0.2 : 0.05; // egg 0.2, spe10 0.05
    mfem::SparseMatrix weight_;

    double EvalCFL(double dt, const mfem::BlockVector& x) const;
public:
    TwoPhaseSolver(const DarcyProblem& problem, Hierarchy& hierarchy,
                   const int level, const EvolveParamenters& evolve_param,
                   const FASParameters& solver_param);

    void TimeStepping(const double dt, mfem::BlockVector& x);
    mfem::BlockVector Solve(const mfem::BlockVector& init_val);
};

class CoupledSolver : public NonlinearSolver
{
    mfem::Vector sol_previous_iter_;

    const Hierarchy& hierarchy_;
    int level_;
    const MixedMatrix& darcy_system_;
    const mfem::Array<int>& ess_attr_;
    mfem::GMRESSolver gmres_;
    unique_ptr<mfem::HypreParMatrix> D_;
    unique_ptr<mfem::HypreParMatrix> DT_;
    std::vector<mfem::DenseMatrix> local_dMdS_;
    mfem::SparseMatrix Ms_;
    unique_ptr<mfem::HypreParMatrix> Ds_;

    mfem::Array<int> blk_offsets_;
    mfem::Array<int> true_blk_offsets_;
    const mfem::Array<int>& ess_dofs_;
    mfem::Array<int> sdof_starts_;
    mfem::Array<int> true_edof_starts_;
//    const std::vector<mfem::DenseMatrix>& traces_;
    const mfem::SparseMatrix& micro_upwind_flux_;

    const double dt_;
    const double density_;
    mfem::SparseMatrix weight_;

    mfem::Vector normalizer_;
//    bool is_first_resid_eval_;
    mfem::Vector scales_;

    mfem::Vector initial_flux_;

    mfem::SparseMatrix D_fine_; // this should not be needed (only for testing)

    bool exact_flow_RAP_ = true;

    void Step(const mfem::Vector& rhs, mfem::Vector& x, mfem::Vector& dx) override;
    void Build_dMdS(const MixedMatrix& darcy_system, const mfem::Vector& flux, const mfem::Vector& S);
    mfem::SparseMatrix Assemble_dMdS(const MixedMatrix& darcy_system, const mfem::Vector& flux, const mfem::Vector& S);
    mfem::Vector AssembleTrueVector(const mfem::Vector& vec) const;

//    void BuildHybridRHS(mfem::BlockOperator& op);
//    void HybridSolve(const mfem::Vector& resid, mfem::Vector& dx);

    void MixedSolve(const mfem::BlockOperator& op_ref,
                    const mfem::BlockVector& true_resid,
                    mfem::BlockVector& true_dx);

    void HybridSolve(const mfem::HypreParMatrix& dTdS, const mfem::Vector& U_FS,
                     const mfem::Vector& flux, const mfem::Vector& S,
                     const mfem::BlockVector& true_resid, mfem::BlockVector& true_dx);

    void PrimalSolve(const mfem::BlockOperator& op_ref,
                     const mfem::BlockVector& true_resid,
                     mfem::BlockVector& true_dx);
public:
    CoupledSolver(const Hierarchy& hierarchy,
                  int level,
                  const MixedMatrix& darcy_system,
                  const mfem::Array<int>& ess_attr,
//                  const std::vector<mfem::DenseMatrix>& edge_traces,
                  const mfem::SparseMatrix& micro_upwind_flux,
                  const double dt,
                  mfem::SparseMatrix weight,
                  const double density,
                  const mfem::Vector& S_prev,
                  NLSolverParameters param);

    mfem::Vector Residual(const mfem::Vector& x, const mfem::Vector& y) override;
    double ResidualNorm(const mfem::Vector& x, const mfem::Vector& y) override;
    double Norm(const mfem::Vector& vec);
    const mfem::Array<int>& BlockOffsets() const { return blk_offsets_; }

    void BackTracking(const mfem::Vector& rhs,  double prev_resid_norm,
                      mfem::Vector& x, mfem::Vector& dx) override;

    const mfem::Vector& GetScales() const { return scales_; }
    const mfem::SparseMatrix& GetMs() const { return Ms_; }
};

class CoupledFAS : public FAS
{
    const Hierarchy& hierarchy_;

    double Norm(int level, const mfem::Vector& vec) const override;
    void Restrict(int level, const mfem::Vector& fine, mfem::Vector& coarse) const override;
    void Interpolate(int level, const mfem::Vector& coarse, mfem::Vector& fine) const override;
    void Project(int level, const mfem::Vector& fine, mfem::Vector& coarse) const override;
//    mfem::Vector ProjectS(int level, const mfem::Vector& S) const;
    std::vector<mfem::SparseMatrix> Qs_;
public:
    CoupledFAS(const Hierarchy& hierarchy,
               const mfem::Array<int>& ess_attr,
               const double dt,
               const mfem::SparseMatrix& weight,
               const double density,
               const mfem::Vector& S_prev,
               FASParameters param);

    const NonlinearSolver& GetLevelSolver(int level) const { return *solvers_[level]; };
};

class TransportSolver : public NonlinearSolver
{
    const MixedMatrix& darcy_system_;
    mfem::Array<int> starts_;
    mfem::Array<int> ess_dofs_;
    mfem::GMRESSolver gmres_;
    const mfem::HypreParMatrix& Adv_;
    mfem::SparseMatrix Ms_;

    virtual void Step(const mfem::Vector& rhs, mfem::Vector& x, mfem::Vector& dx) override;
public:
    TransportSolver(const mfem::HypreParMatrix& Adv_,
                    const MixedMatrix& darcy_system,
                    const double vol_dt_inv,
                    NLSolverParameters param)
        : NonlinearSolver(Adv_.GetComm(), param), darcy_system_(darcy_system),
          gmres_(comm_),
          Adv_(Adv_), Ms_(SparseIdentity(Adv_.NumRows()) *= vol_dt_inv)
    {
        gmres_.SetMaxIter(200);
        gmres_.SetRelTol(1e-9);
//        gmres_.SetPrintLevel(1);

        GenerateOffsets(comm_, Adv_.NumRows(), starts_);
    }

    mfem::Vector Residual(const mfem::Vector& x, const mfem::Vector& y) override;
    double ResidualNorm(const mfem::Vector& x, const mfem::Vector& y) override
    {
        return mfem::ParNormlp(Residual(x, y), 2, comm_);
    }
};

class TwoPhaseFromFile : public DarcyProblem
{
public:
    TwoPhaseFromFile(std::string& path, bool need_getline, int num_vert_res,
                     int nnz_res, int num_edges_res, double inject_rate,
                     double bottom_hole_pressure, const mfem::Array<int>& ess_attr)
        : DarcyProblem(MPI_COMM_WORLD, 2, ess_attr)
    {
        std::string c2f_filename = path+"/cell_to_faces.txt";
//        std::string c2f_filename = path+"/new_cell_to_faces_saigup_refined.txt";
        std::string vol_filename = path+"/cell_volume_porosity.txt";
        if (path == "/Users/lee1029/Downloads/spe10_bottom_layer_3d_no_inactive")
        {
           c2f_filename = path+"/filtered_cell_to_face.txt";
           vol_filename = path+"/filtered_cell_volume_porosity.txt";
        }
        else if (path == "/Users/lee1029/Downloads/test_2"
                 || path == "/Users/lee1029/Downloads/test_5")
        {
            c2f_filename = path+"/cell_to_faces_francois.txt";
            vol_filename = path+"/cell_volumes_porosity_francois_no_text.txt";
        }

        if (num_vert_res == 500931) {
            c2f_filename = path+"/cell_to_faces_3x3x3.txt";
            vol_filename = path+"/cell_volume_porosity_3x3x3.txt";
        }
        if (num_vert_res == 148424) {
            c2f_filename = path+"/cell_to_faces_2x2x2.txt";
            vol_filename = path+"/cell_volume_porosity_2x2x2.txt";
        }

        std::ifstream file(c2f_filename);
        std::ifstream vol_file(vol_filename);

        int num_injectors = num_vert_res == 18553 ? 8 : 1; // egg 8 // other 1
        num_injectors = num_vert_res == 500931 ? 8 : num_injectors;
        num_injectors = num_vert_res == 148424 ? 8 : num_injectors;
        int num_producers = num_vert_res < 50 ? 1 : 4; // small case 1 // other 4

        if (num_edges_res == 155268) { num_injectors = 6; num_producers = 5; }
        if (num_edges_res == 155268) { c2f_filename = path+"/cell_to_faces_positive.txt"; }
        if (num_edges_res == 264305) { num_injectors = 5; num_producers = 5; }
        if (num_edges_res == 2000742) { num_injectors = 5; num_producers = 5; }


        int num_vert_total = num_vert_res + num_injectors;
        int num_edges_total = num_edges_res + num_injectors + num_producers;

        if (!file.is_open())
        {
            std::cerr << "Error in opening file cell_to_faces.txt\n";
            mfem::mfem_error("File does not exist");
        }

        if (num_vert_res == 13200 || num_vert_res == 12321) // SPE10 2D model
        {
            mfem::Array<int> max_N(3);
            max_N[0] = 60;
            max_N[1] = 220;
            max_N[2] = 85;//85;

            const int spe10_scale = 5;
            mfem::Array<int> N_(3);
            N_[0] = 12 * spe10_scale; // 60
            N_[1] = 44 * spe10_scale; // 220
            N_[2] = max_N[2];//17 * spe10_scale; // 85

            // SPE10 grid cell sizes
            mfem::Vector h(3);
            h(0) = 20.0 * ft_; // 365.76 / 60. in meters
            h(1) = 10.0 * ft_; // 670.56 / 220. in meters
            h(2) = 2.0 * ft_; // 51.816 / 85. in meters

            const double Lx = N_[0] * h(0);
            const double Ly = N_[1] * h(1);
            mfem::Mesh mesh(N_[0], N_[1], mfem::Element::QUADRILATERAL, true, Lx, Ly);

            mesh_ = make_unique<mfem::ParMesh>(comm_, mesh);
            InitGraph();

            vertex_reorder_map_.reset(new mfem::SparseMatrix(13200, num_vert_res));

            std::string position_filename = path+"/cell_positions_no_text.txt";
            if (num_vert_res == 12321)
            {
                position_filename = path+"/filtered_cell_positions.txt";
            }
            std::ifstream position_file(position_filename);

            mfem::DenseMatrix point(mesh_->Dimension(), num_vert_res);
            mfem::Array<int> ids;
            mfem::Array<mfem::IntegrationPoint> ips;

            for (int i = 0; i < num_vert_res; ++i)
            {
                position_file >> trash_;
                position_file >> point(0, i);
                position_file >> point(1, i);
                position_file >> trash_; //point(2, i);
            }
            std::cout << "last cell center x-coordinate: " << point(0, num_vert_res-1) <<"\n";
            position_file >> trash_;
            position_file >> trash_;
            if (trash_ != -1)
            {
                std::cout << "first well cell x-coordinate: " << trash_ <<"\n";
            }
            assert(trash_ == -1);

            mesh_->FindPoints(point, ids, ips, false);

            for (int i = 0; i < num_vert_res; ++i)
            {
                assert((ids[i] >= 0));
                vertex_reorder_map_->Set(ids[i], i, 1.0);
            }
            vertex_reorder_map_->Finalize();
        }

        local_weight_.resize(num_vert_total);
        mfem::SparseMatrix vert_edge(num_vert_total, num_edges_total);

        std::string str;
        if (need_getline && (num_vert_res != 18553)) std::getline(file, str);

        int vert, edge;
        double half_trans;

        auto save_one_line = [&](int edge_offset = 0)
        {
            file >> vert;
            file >> edge;
            file >> half_trans;
//            if (half_trans < 0.0) { std::cout<< "negative data " << vert<<" "<<edge<<" "<<half_trans<<"\n";}
            vert_edge.Set(vert - 1, edge - 1 + edge_offset, std::fabs(half_trans));//half_trans);//
        };

        // TODO: make it to work for elements of mixed types
        for (int i = 0; i < nnz_res; ++i)
        {
            save_one_line();
        }
        std::cout<<" debug print: " << vert<<" "<<edge<<" "<<half_trans<<"\n";

        if (need_getline)
        {
            std::getline(file, str);
            std::getline(file, str);
            std::getline(file, str);
            std::getline(file, str);
            std::getline(file, str);
        }

        for (int i = 0; i < num_producers; ++i)
        {
            save_one_line(num_injectors);
        }
        std::cout<<" debug print: " << vert<<" "<<edge<<" "<<half_trans<<"\n";

        if (need_getline)
        {
            std::getline(file, str);
            std::getline(file, str);
        }

        for (int i = 0; i < num_injectors; ++i)
        {
            save_one_line(-num_producers);
            save_one_line(-num_producers);
            vert_edge.Set(vert - 1, edge - 1 - num_producers, 1e10); // TODO: this is to match well.hpp, not sure if necessary
        }
        std::cout<<" debug print: " << vert<<" "<<edge<<" "<<half_trans<<"\n";


        vert_edge.Finalize();

        for (int i = 0; i < vert_edge.NumRows(); i++)
        {
            local_weight_[i].SetSize(vert_edge.RowSize(i));
            std::copy_n(vert_edge.GetRowEntries(i), vert_edge.RowSize(i), local_weight_[i].GetData());
        }

        vert_edge = 1.0;

        mfem::SparseMatrix edge_bdr(num_edges_total, 1 + num_producers); // 1 is the reservoir bdr
        {
            auto edge_vert = smoothg::Transpose(vert_edge);
            int edge_count = 0;
            for (int i = 0; i < num_edges_res; ++i)
            {
                if (edge_vert.RowSize(i) != 1 && edge_vert.RowSize(i) != 2 )
                {
                    std::cout<<"edge "<<i<<": row nnz = "<<edge_vert.RowSize(i)<<"\n";
                    edge_count++;
                }

                if (edge_vert.RowSize(i) == 1 && i < num_edges_res)
                {
                    edge_bdr.Set(i, 0, 1.0);
                }
            }
            std::cout<<"abnormal edge_count = "<<edge_count<<"\n";
            assert(edge_count == 0);
        }

        std::cout<<" num_producers = "<<num_producers<<"\n";
        for (int i = 0; i < num_producers; ++i)
        {
            edge_bdr.Set(num_edges_res + num_injectors + i, 1 + i, 1.0);
        }
        edge_bdr.Finalize();

        auto e_te = SparseIdentity(num_edges_total);
        edge_trueedge_read_.reset(ToParMatrix(MPI_COMM_WORLD, e_te));

        // fit data
        vertex_edge_.Swap(vert_edge);
        edge_bdr_.Swap(edge_bdr);
        edge_trueedge_ = edge_trueedge_read_.get();

        // block offset
        block_offsets_.SetSize(4);
        block_offsets_[0] = 0;
        block_offsets_[1] = num_edges_total;
        block_offsets_[2] = block_offsets_[1] + num_vert_total;
        block_offsets_[3] = block_offsets_[2] + num_vert_total;

        // rhs
        rhs_sigma_.SetSize(num_edges_total);
        rhs_sigma_ = 0.0;
        for (int i = 0; i < num_producers; ++i)
        {
            rhs_sigma_[num_edges_res + num_injectors + i] = bottom_hole_pressure;
        }

        rhs_u_.SetSize(num_vert_total);
        rhs_u_ = 0.0;
        for (int i = 0; i < num_injectors; ++i)
        {
            rhs_u_[num_vert_res + i] = inject_rate;
        }

        // read cell volume and porosity
        vert_weight_.SetSize(num_vert_total);
        vert_weight_ = 0.2;

        volumes_.SetSize(num_vert_res);

        if (!vol_file.is_open())
        {
            std::cerr << "Error in opening file cell_volume_porosity.txt\n";
            mfem::mfem_error("File does not exist");
        }
        if (need_getline && (num_vert_res != 18553)) std::getline(vol_file, str);
        for (int i = 0; i < num_vert_res; ++i)
        {
            vol_file >> vert;
            vol_file >> cell_volume_;
            vol_file >> porosity_;
            vol_file >> trash_;
            if (i < 1) std::cout << "volume: " << cell_volume_ << ", porosity: " << porosity_ <<"\n";
            vert_weight_[i] = cell_volume_ * porosity_;
            volumes_[i] = cell_volume_;
        }

        std::cout << "sum of cell volumes = " << volumes_.Sum() << "\n";
        volumes_ = vert_weight_;
        volumes_.SetSize(num_vert_res);
        std::cout << "sum of pore volumes = " << volumes_.Sum() << "\n";

        if (need_getline)
        {
            std::getline(vol_file, str);
            std::getline(vol_file, str);
            std::getline(vol_file, str);
            std::getline(vol_file, str);
            std::getline(vol_file, str);
        }

        std::cout << "volume: " << cell_volume_ << ", porosity: " << porosity_ <<"\n";
        for (int i = num_vert_res; i < num_vert_total; ++i)
        {
            vert_weight_[i] = cell_volume_ * porosity_;
        }
        vol_file >> vert;
        vol_file >> cell_volume_;
        vol_file >> porosity_;
        std::cout << "volume: " << cell_volume_ << ", porosity: " << porosity_ <<"\n";
        assert(cell_volume_ == -1.0 && porosity_ == -1.0);
    }

    virtual double CellVolume() const { return cell_volume_; }
private:
    unique_ptr<mfem::HypreParMatrix> edge_trueedge_read_;
    double cell_volume_;
    double porosity_;
    double trash_;
    mfem::Vector volumes_;
};

void ShowAggregates(const std::vector<Graph>& graphs,
                    const std::vector<GraphTopology>& topos,
                    const DarcyProblem& problem,
                    int num_injectors)
{
    mfem::ParMesh& mesh = const_cast<mfem::ParMesh&>(problem.GetMesh());
    mfem::L2_FECollection fec(0, mesh.SpaceDimension());
    mfem::ParFiniteElementSpace fespace(const_cast<mfem::ParMesh*>(&mesh), &fec);
    mfem::ParGridFunction attr(&fespace);

    mfem::socketstream sol_sock;
    for (unsigned int i = 0; i < topos.size(); i++)
    {
        // Compute partitioning vector on level i+1
        mfem::SparseMatrix Agg_vertex = topos[0].Agg_vertex_;
        for (unsigned int j = 1; j < i + 1; j++)
        {
            auto tmp = smoothg::Mult(topos[j].Agg_vertex_, Agg_vertex);
            Agg_vertex.Swap(tmp);
        }
        auto vertex_Agg = smoothg::Transpose(Agg_vertex);
        int* partitioning = vertex_Agg.GetJ();

        // Make better coloring (better with serial run)
        mfem::SparseMatrix Agg_Agg = AAt(graphs[i + 1].VertexToEdge());
        mfem::Array<int> colors;
        GetElementColoring(colors, Agg_Agg);
        const int num_colors = std::max(colors.Max() + 1, mesh.GetNRanks());

        mfem::Vector parts(vertex_Agg.Height()-num_injectors);

        for (int j = 0; j < vertex_Agg.Height()-num_injectors; j++)
        {
            attr(j) = (colors[partitioning[j]] + mesh.GetMyRank()) % num_colors;

            parts[j] = partitioning[j];
        }

        if (problem.VertReorderMap())
        {
            mfem::Vector attr_graph(attr);
            problem.VertReorderMap()->Mult(attr_graph, attr);

            mfem::Vector parts_graph(parts);
            problem.VertReorderMap()->Mult(parts_graph, parts);
        }

        for (int j = 0; j < vertex_Agg.Height()-num_injectors; j++)
        {
            mesh.SetAttribute(j, parts[j]);
        }

        char vishost[] = "localhost";
        int  visport   = 19916;
        sol_sock.open(vishost, visport);
        if (sol_sock.is_open())
        {
            sol_sock.precision(8);
            sol_sock << "parallel " << mesh.GetNRanks() << " " << mesh.GetMyRank() << "\n";
            if (mesh.SpaceDimension() == 2)
            {
                sol_sock << "fem2d_gf_data_keys\n";
            }
            else
            {
                sol_sock << "fem3d_gf_data_keys\n";
            }

            mesh.PrintWithPartitioning(partitioning, sol_sock, 1);
            attr.Save(sol_sock);

//            sol_sock << "solution\n" << mesh << attr;


            sol_sock << "window_size 1000 800\n";
            sol_sock << "window_title 'Level " << i + 1 << " aggregation'\n";
            if (mesh.SpaceDimension() == 2)
            {
                sol_sock << "view 0 0\n"; // view from top
                sol_sock << "keys j\n";  // turn off perspective and light
                sol_sock << "keys ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]\n";  // increase size
                sol_sock << "keys b\n";  // draw interface
            }
            else
            {
                sol_sock << "keys ]]]]]]]]]]]]]\n";  // increase size
            }
            if (mesh.GetGlobalNE() != 13200) // non-SPE10 (e.g., Egg model)
            {
                sol_sock << "keys RRRRRR\n"; // angle
            }
            sol_sock << "keys m\n"; // show mesh

            sol_sock << "plot_caption '" << "two-layer isolation" << "'\n";

            MPI_Barrier(mesh.GetComm());
        }
    }
}

void SetAttrForAggPrint(const DarcyProblem& problem, const mfem::Array<int>& parts)
{
    mfem::ParMesh& mesh = const_cast<mfem::ParMesh&>(problem.GetMesh());
    for (int j = 0; j < mesh.GetNE(); j++)
    {
        mesh.SetAttribute(j, parts[j]);
    }
}


int main(int argc, char* argv[])
{
    int num_procs, myid;

    // 1. Initialize MPI
    mpi_session session(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // program options from command line
    EvolveParamenters evolve_param;
    mfem::OptionsParser args(argc, argv);
    std::string base_dir = "/Users/lee1029/Downloads/";
    const char* problem_dir = "";
    args.AddOption(&problem_dir, "-pd", "--problem-directory",
                   "Directory where data files are located");
    const char* perm_file = "spe_perm.dat";
    args.AddOption(&perm_file, "-p", "--perm", "SPE10 permeability file data.");
    int dim = 3;
    args.AddOption(&dim, "-d", "--dim", "Dimension of the physical space.");
    int slice = 0;
    args.AddOption(&slice, "-s", "--slice", "Slice of SPE10 data for 2D run.");
    int well_height = 1;
    args.AddOption(&well_height, "-wh", "--well-height", "Well Height.");
    double inject_rate = 5.6544e-04;//0.00005;// * 0.6096;
    args.AddOption(&inject_rate, "-ir", "--inject-rate", "Injector rate.");
    double bhp = -2.7579e07;//-1.0e6;
    args.AddOption(&bhp, "-bhp", "--bottom-hole-pressure", "Bottom Hole Pressure.");
    args.AddOption(&evolve_param.dt, "-dt", "--delta-t", "Time step.");
    args.AddOption(&evolve_param.total_time, "-time", "--total-time",
                   "Total time to step.");
    args.AddOption(&evolve_param.vis_step, "-vs", "--vis-step",
                   "Step size for visualization.");
    int scheme = 3;
    args.AddOption(&scheme, "-scheme", "--stepping-scheme",
                   "Time stepping: 1. IMPES, 2. sequentially implicit, 3. fully implicit. ");
    bool use_vcycle = true;
    args.AddOption(&use_vcycle, "-VCycle", "--use-VCycle", "-FMG",
                   "--use-FMG", "Use V-cycle or FMG-cycle.");
    int num_backtrack = 0;
    args.AddOption(&num_backtrack, "--num-backtrack", "--num-backtrack",
                   "Maximum number of backtracking steps.");
    double diff_tol = -1.0;
    args.AddOption(&diff_tol, "--diff-tol", "--diff-tol",
                   "Tolerance for coefficient change.");
    int print_level = -1;
    args.AddOption(&print_level, "-print-level", "--print-level",
                   "Solver print level (-1 = no to print, 0 = final error, 1 = all errors.");
    bool smeared_front = true;
    args.AddOption(&smeared_front, "-smeared-front", "--smeared-front", "-sharp-front",
                   "--sharp-front", "Control density to produce smeared or sharp saturation front.");
    args.AddOption(&relperm_order, "-ro", "--relperm-order",
                   "Exponent of relperm function.");
    UpscaleParameters upscale_param;
    upscale_param.spect_tol = 1.0;
    upscale_param.max_evects = 1;
    upscale_param.max_traces = 1;
    upscale_param.max_levels = 1;
    upscale_param.RegisterInOptionsParser(args);
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

    evolve_param.scheme = static_cast<SteppingScheme>(scheme);

    dim = 2;
    mfem::Array<int> ess_attr(dim == 3 ? 6 : 4);
    ess_attr = 1;

    // Setting up finite volume discretization problem
    unique_ptr<DarcyProblem> problem;
    problem.reset(new LocalProblem(comm, dim, ess_attr));

    Graph graph = problem->GetFVGraph(true);


    unique_ptr<mfem::Array<int>> partition(nullptr);
//    Hierarchy hierarchy(std::move(graph), upscale_param, partition.get(), &ess_attr);
//    hierarchy.PrintInfo();
    Upscale upscale(std::move(graph), upscale_param, partition.get(), &ess_attr);
    upscale.PrintInfo();

    mfem::BlockVector rhs(upscale.BlockOffsets(0));
    rhs.GetBlock(0) = 0.0;
    rhs.GetBlock(1) = 1.0/(rhs.BlockSize(1)-1);
    rhs[rhs.Size()-1] = -1.0;

    mfem::BlockVector sol = upscale.Solve(0, rhs);


//    mfem::Vector p_wc(19);
//    for (int i = 0; i < 19; ++i) { p_wc[i] = sol.GetBlock(1)[sol.BlockSize(1)-19+i]; }
//    p_wc.Print();
    std::cout<<"P max: " << sol.GetBlock(1).Max() <<", P min: " << sol.GetBlock(1).Min() <<"\n";
    std::cout<<"P mean: " << sol.GetBlock(1).Sum() / sol.BlockSize(1) <<"\n";

    mfem::socketstream vis_v;
    problem->VisSetup(vis_v, sol.GetBlock(1), 0.0, 0.0, "Pressure");

    return EXIT_SUCCESS;
}


//mfem::Vector TotalMobility(const mfem::Vector& S)
//{
//    mfem::Vector LamS(S.Size());
//    LamS = 1000.;
//    return LamS;
//}

//mfem::Vector dTMinv_dS(const mfem::Vector& S)
//{
//    mfem::Vector out(S.Size());
//    out = 0.0;
//    return out;
//}

//mfem::Vector FractionalFlow(const mfem::Vector& S)
//{
//    mfem::Vector FS(S);
//    return FS;
//}

//mfem::Vector dFdS(const mfem::Vector& S)
//{
//    mfem::Vector out(S.Size());
//    out = 1.0;
//    return out;
//}

//mfem::Vector TotalMobility(const mfem::Vector& S)
//{
//    mfem::Vector LamS(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        LamS(i)  = S_w * S_w + S_o * S_o / 5.0;
//    }
//    return LamS;
//}

//mfem::Vector dTMinv_dS(const mfem::Vector& S)
//{
//    mfem::Vector out(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        out(i)  = 2.0 * (S_w - S_o / 5.0);
//        double Lam_S  = S_w * S_w + S_o * S_o / 5.0;
//        out(i) = -1.0 * out(i) / (Lam_S * Lam_S);
//    }
//    return out;
//}

//mfem::Vector FractionalFlow(const mfem::Vector& S)
//{
//    mfem::Vector FS(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        double Lam_S  = S_w * S_w + S_o * S_o / 5.0;
//        FS(i) = S_w * S_w / Lam_S;
//    }
//    return FS;
//}

//mfem::Vector dFdS(const mfem::Vector& S)
//{
//    mfem::Vector out(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        double Lam_S  = S_w * S_w + S_o * S_o / 5.0;
//        out(i) = 0.4 * (S_w - S_w * S_w) / (Lam_S * Lam_S);
//    }
//    return out;
//}

// case 1
//mfem::Vector TotalMobility(const mfem::Vector& S)
//{
//    mfem::Vector LamS(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        LamS(i)  = S_w * S_w / 1e-3 + std::pow(S_o, 1.5) / 1e-4;
//    }
//    return LamS;
//}

//mfem::Vector dTMinv_dS(const mfem::Vector& S)
//{
//    mfem::Vector out(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        out(i)  = 2. * S_w / 1e-3 - 1.5 * std::pow(S_o, 0.5) / 1e-4;
//        double Lam_S  = S_w * S_w / 1e-3 + std::pow(S_o, 1.5) / 1e-4;
//        out(i) = -1.0 * out(i) / (Lam_S * Lam_S);
//    }
//    return out;
//}

//mfem::Vector FractionalFlow(const mfem::Vector& S)
//{
//    mfem::Vector FS(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        double Lam_S  = S_w * S_w / 1e-3 + std::pow(S_o, 1.5) / 1e-4;
//        FS(i) = S_w * S_w / 1e-3 / Lam_S;
//    }
//    return FS;
//}

//mfem::Vector dFdS(const mfem::Vector& S)
//{
//    mfem::Vector out(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        double dLw_dS = 2. * S_w / 1e-3;
//        double Lam_S  = S_w * S_w / 1e-3 + std::pow(S_o, 1.5) / 1e-4;
//        double dLam_dS = 2. * S_w / 1e-3 - 1.5 * std::pow(S_o, 0.5) / 1e-4;
//        out(i) = (dLw_dS * Lam_S - dLam_dS * S_w * S_w / 1e-3) / (Lam_S * Lam_S);
//    }
//    return out;
//}

// case 2
//mfem::Vector TotalMobility(const mfem::Vector& S)
//{
//    mfem::Vector LamS(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        LamS(i)  = S_w * S_w / 1e-3 + std::pow(S_o, 3.0) / 1e-2;
//    }
//    return LamS;
//}

//mfem::Vector dTMinv_dS(const mfem::Vector& S)
//{
//    mfem::Vector out(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        out(i)  = 2. * S_w / 1e-3 - 3.0 * std::pow(S_o, 2.0) / 1e-2;
//        double Lam_S  = S_w * S_w / 1e-3 + std::pow(S_o, 3.0) / 1e-2;
//        out(i) = -1.0 * out(i) / (Lam_S * Lam_S);
//    }
//    return out;
//}

//mfem::Vector FractionalFlow(const mfem::Vector& S)
//{
//    mfem::Vector FS(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        double Lam_S  = S_w * S_w / 1e-3 + std::pow(S_o, 3.0) / 1e-2;
//        FS(i) = S_w * S_w / 1e-3 / Lam_S;
//    }
//    return FS;
//}

//mfem::Vector dFdS(const mfem::Vector& S)
//{
//    mfem::Vector out(S.Size());
//    for (int i = 0; i < S.Size(); i++)
//    {
//        double S_w = S(i);
//        double S_o = 1.0 - S_w;
//        double dLw_dS = 2. * S_w / 1e-3;
//        double Lam_S  = S_w * S_w / 1e-3 + std::pow(S_o, 3.0) / 1e-2;
//        double dLam_dS = 2. * S_w / 1e-3 - 3.0 * std::pow(S_o, 2.0) / 1e-2;
//        out(i) = (dLw_dS * Lam_S - dLam_dS * S_w * S_w / 1e-3) / (Lam_S * Lam_S);
//    }
//    return out;
//}


double _max_S = 0.8;
double _min_S = 0.2;

//// case 5 any exponent
double RelPerm(double S, double mu) // relative permeability
{
//    return std::pow(S, relperm_order) / mu;

    // CoreyRelPerm
    double modified_S = (S-0.2) / (0.6);
    modified_S = modified_S < 0.0 ? 0.0 : (modified_S > 1.0 ? 1.0 : modified_S);
    return std::pow(modified_S, relperm_order) / mu;

}

double RelPermDerivative(double S, double mu)
{
//    return relperm_order * std::pow(S, relperm_order - 1) / mu;

    // CoreyRelPerm
    double modified_S = (S-0.2) / (0.6);
    modified_S = modified_S < 0.0 ? 0.0 : (modified_S > 1.0 ? 1.0 : modified_S);
    return relperm_order * std::pow(modified_S, relperm_order - 1) / mu / (0.6);
}

double CellMobility(double S_w, double S_o)
{
   return RelPerm(S_w, mu_w) + RelPerm(S_o, mu_o);
}

double CellMobilityDerivative(double S_w, double S_o)
{
   return RelPermDerivative(S_w, mu_w) - RelPermDerivative(S_o, mu_o);
}

mfem::Vector TotalMobility(const mfem::Vector& S)
{
    mfem::Vector LamS(S.Size());
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);

        S_w = S_w < _min_S ? _min_S : (S_w > _max_S ? _max_S : S_w);

//        double S_o = 1.0 - S_w;

        // CoreyRelPerm
        double S_o = 1.0 - S_w;

        LamS(i)  = CellMobility(S_w, S_o);
    }
    return LamS;
}

mfem::Vector dTMinv_dS(const mfem::Vector& S)
{
    mfem::Vector out(S.Size());
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);

//        if (S_w < 0.0 || S_w > 1.0) { out(i) = 0.0; continue; }

//        double S_o = 1.0 - S_w;

        // CoreyRelPerm
        if (S_w < 0.2 || S_w > 0.8) { out(i) = 0.0; continue; }
        double S_o = 1.0 - S_w;

        double Lam_S  = CellMobility(S_w, S_o);
        out(i) = -1.0 * CellMobilityDerivative(S_w, S_o) / (Lam_S * Lam_S);
    }
    return out;
}

mfem::Vector FractionalFlow(const mfem::Vector& S)
{
    mfem::Vector FS(S.Size());
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);
        S_w = S_w < _min_S ? _min_S : (S_w > _max_S ? _max_S : S_w);

        //        double S_o = 1.0 - S_w;

        // CoreyRelPerm
        double S_o = 1.0 - S_w;

        double Lam_S  = CellMobility(S_w, S_o);
        FS(i) = RelPerm(S_w, mu_w) / Lam_S;
    }
    return FS;
}

mfem::Vector dFdS(const mfem::Vector& S)
{
    mfem::Vector out(S.Size());
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);
//        if (S_w < 0.0 || S_w > 1.0) { out(i) = 0.0; continue; }

        //        double S_o = 1.0 - S_w;

        // CoreyRelPerm
        if (S_w < 0.2 || S_w > 0.8) { out(i) = 0.0; continue; }
        double S_o = 1.0 - S_w;

        double Lw = RelPerm(S_w, mu_w);
        double dLw_dS = RelPermDerivative(S_w, mu_w);
        double Lam_S  = CellMobility(S_w, S_o);
        double dLam_dS = CellMobilityDerivative(S_w, S_o);
        out(i) = (dLw_dS * Lam_S - dLam_dS * Lw) / (Lam_S * Lam_S);
    }
    return out;
}
