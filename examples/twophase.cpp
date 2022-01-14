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
double mu_w = 3e-4; // 3*centi*poise //1e-3;
int relperm_order = 2;
mfem::Array<int> well_cells;
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

    const int max_iter = upscale_param.max_levels > 1 ? 100 : 100;
//    mu_o = smeared_front ? 0.005 : 0.0002;

    FASParameters fas_param;
    fas_param.fine.max_num_iter = use_vcycle ? 1 : max_iter;
    fas_param.mid.max_num_iter = use_vcycle ? 1 : max_iter;
    fas_param.coarse.max_num_iter = use_vcycle ? 10 : max_iter;
    fas_param.coarse.print_level = use_vcycle ? print_level : print_level;
    fas_param.fine.print_level = use_vcycle ? -1 : print_level;
    fas_param.mid.print_level = use_vcycle ? -1 : print_level;
//    fas_param.coarse.rtol = 1e-10;
//    fas_param.coarse.atol = 1e-12;
    fas_param.nl_solve.print_level = use_vcycle ? 1 : -1;
    fas_param.nl_solve.max_num_iter = use_vcycle ? max_iter : 1;
    fas_param.nl_solve.rtol = 0e-6;
    fas_param.nl_solve.atol = 1e-6;
    fas_param.coarse.rtol = 0e-6;
    fas_param.coarse.atol = 1e-6;
    fas_param.mid.rtol = 0e-6;
    fas_param.mid.atol = 1e-6;
    fas_param.fine.rtol = 0e-6;
    fas_param.fine.atol = 1e-6;
    SetOptions(fas_param, use_vcycle, num_backtrack, diff_tol);

    bool read_from_file = false;
    const bool use_metis = true;
    mfem::Array<int> ess_attr(dim == 3 ? 6 : 4);
    ess_attr = 1;

    int num_attr_from_file = 5;
    upscale_param.num_iso_verts = 1; // TODO: this should be read from file

    // Setting up finite volume discretization problem
    std::string path(base_dir + problem_dir);
    unique_ptr<DarcyProblem> problem, problem_for_plot;
    if (path == "/Users/lee1029/Downloads/")
    {
        problem.reset(new TwoPhase(perm_file, dim, 5, slice, use_metis, ess_attr,
                                   well_height, inject_rate, bhp));
    }
    else
    {
        //    std::string path = "/Users/lee1029/Downloads/spe10_bottom_layer_2d";
        bool need_getline = true;
        int num_vert_res = 13200;
        int nnz_res = num_vert_res * 4;
        int num_edges_res = 26680;

        if (path == "/Users/lee1029/Downloads/spe10_bottom_layer_3d_no_inactive")
        {
            //    std::string path = "/Users/lee1029/Downloads/spe10_bottom_layer_3d_constrast_10^5/";
            //    std::string path = "/Users/lee1029/Downloads/spe10_bottom_layer_3d_constrast_10^3";
            need_getline = false;
            num_vert_res = 12321;
            nnz_res = num_vert_res * 6;
            num_edges_res = 50438;
            inject_rate *= 0.6096;
        }
        else if (path == "/Users/lee1029/Downloads/spe10_top_layer_3d" ||
                 path == "/Users/lee1029/Downloads/spe10_bottom_layer_3d")
        {
            //    std::string path = "/Users/lee1029/Downloads/spe10_bottom_layer_3d";
            num_vert_res = 13200;
            nnz_res = num_vert_res * 6;
            num_edges_res = 53080;
            inject_rate *= 0.6096;
        }
        else if (path == "/Users/lee1029/Downloads/test_2")
        {
            num_vert_res = 16;
            num_edges_res = 40;
            inject_rate = 1e-7;
            num_attr_from_file = 2;
        }
        else if (path == "/Users/lee1029/Downloads/test_5")
        {
            num_vert_res = 25;
            num_edges_res = 60;
            inject_rate = 1e-7;
            num_attr_from_file = 2;
        }
        else if (path == "/Users/lee1029/Downloads/egg")
        {
            num_vert_res = 18553;
            nnz_res = num_vert_res * 6;
            num_edges_res = 59205;
//            inject_rate = 5e-4;
//            inject_rate = 2e-3;
            upscale_param.num_iso_verts = 8; // TODO: this should be read from file

//            // refined 3x3x3
//            num_vert_res = 500931;
//            nnz_res = num_vert_res * 6;
//            num_edges_res = 1534707;

            // refined 2x2x2
//            num_vert_res = 148424;
//            nnz_res = num_vert_res * 6;
//            num_edges_res = 459456;

            ess_attr.SetSize(3, 1);
            problem_for_plot.reset(new EggModel(0, 0, ess_attr));
        }
        else if (path == "/Users/lee1029/Downloads/norne")
        {
            num_vert_res = 44915;
            nnz_res = 291244;
            num_edges_res = 155268;
            inject_rate *= 1.0;
            num_attr_from_file = 6;
            upscale_param.num_iso_verts = 6; // TODO: this should be read from file

            ess_attr.SetSize(1, 1);
            problem_for_plot.reset(new NorneModel(comm, ess_attr));
        }
        else if (path == "/Users/lee1029/Downloads/saigup")
        {
            num_vert_res = 78720;
            nnz_res = 505918;
            num_edges_res = 264305;
            inject_rate *= 1.0;
            num_attr_from_file = 6;
            upscale_param.num_iso_verts = 5; // TODO: this should be read from file

            mfem::Array<int> ess_attr2(1); ess_attr2 = 1;
            problem_for_plot.reset(new SaigupModel(comm, false, ess_attr2));
        }
        else if (path == "/Users/lee1029/Downloads/refined_saigup")
        {
            num_vert_res = 629760;
            nnz_res = 3913208;
            num_edges_res = 2000742;
            inject_rate *= 1.0;
            num_attr_from_file = 6;
            upscale_param.num_iso_verts = 5; // TODO: this should be read from file

            ess_attr.SetSize(1, 1);
            problem_for_plot.reset(new SaigupModel(comm, true, ess_attr));
        }
        else if (path != "/Users/lee1029/Downloads/spe10_bottom_layer_2d")
        {
            mfem::mfem_error("Unknown model problem!");
        }

        ess_attr.SetSize(num_attr_from_file, 0);
        ess_attr[0] = 1;

        read_from_file = true;
        std::cout << "Read data from " << path << "... \n";
        problem.reset(new TwoPhaseFromFile(path, need_getline, num_vert_res, nnz_res,
                                           num_edges_res, inject_rate, bhp, ess_attr));
    }
    std::cout << "Injection rate: " << inject_rate << "\n";

    problem_ptr = problem_for_plot ? problem_for_plot.get() : problem.get();

    Graph graph = problem->GetFVGraph(true);
    auto& ess_attr_final = problem->EssentialAttribute();

    unique_ptr<mfem::Array<int>> partition(read_from_file ? nullptr : new mfem::Array<int>);

    if (read_from_file == false)
    {
        mfem::Array<int> coarsening_factors(dim);
        coarsening_factors = 1;
        coarsening_factors[0] = upscale_param.coarse_factor;
        coarsening_factors[0] = 6;
        coarsening_factors[1] = 11;
        problem->Partition(use_metis, coarsening_factors, *partition);
        if (!use_metis) { partition->Append(partition->Max()+1); }
        if (use_metis) { upscale_param.num_iso_verts = problem->NumIsoVerts(); }
        SetAttrForAggPrint(*problem_ptr, *partition);
    }


//    std::ifstream sin_fs("results_for_paper_new/refined_saigup_final_sat.txt");
//    std::ifstream sin_fs("results_for_paper_new/refined_egg_2x2x2_r4_final_sat.txt");
//    mfem::socketstream sout_fs;
//    mfem::Vector final_sat;
//    final_sat.Load(sin_fs, graph.NumVertices());
//    problem_for_plot->VisSetup(sout_fs, final_sat, 0.0, 1.0);

//    std::ofstream mesh_out_file("spe10_1to5.vtk");
//    const_cast<mfem::ParMesh&>(problem_ptr->GetMesh()).PrintVTK(mesh_out_file);


    const int num_injectors = 1;
    const int num_producers = 4;
    const mfem::SparseMatrix&  vert_edge = graph.VertexToEdge();
    mfem::SparseMatrix edge_vert = smoothg::Transpose(vert_edge);
//    if (0)
    {

//        mfem::Vector well_cells(graph.NumVertices() - upscale_param.num_iso_verts);
        well_cell_gf.SetSize(graph.NumVertices() - upscale_param.num_iso_verts);
        well_cell_gf = 0.0;
//        well_cells.SetSize((num_producers+num_injectors)*well_height);

        mfem::SparseMatrix vert_vert = smoothg::Mult(vert_edge, edge_vert);

        for (int i = vert_edge.NumRows() - num_injectors; i < vert_edge.NumRows(); ++i)
        {
//            assert(vert_vert.RowSize(i) == 2);
            for (int j = 0; j < vert_vert.RowSize(i); ++j)
            {
                int i_friend = vert_vert.GetRowColumns(i)[j];
                if (i_friend == i) { continue; }
//                i_friend = (i_friend == i) ? vert_vert.GetRowColumns(i)[1] : i_friend;
                well_cell_gf[i_friend] = 1.0;
                well_cells.Append(i_friend);
//                std::cout<<"injection well cell: "<<i_friend<<"\n";
            }
        }

//        int num_injectors = upscale_param.num_iso_verts;
//        int num_wells = num_attr_from_file - 1 + num_injectors;
        for (int i = edge_vert.NumRows() - num_producers*well_height; i < edge_vert.NumRows(); ++i)
        {
            assert(edge_vert.RowSize(i) == 1);
            well_cell_gf[edge_vert.GetRowColumns(i)[0]] = -1.0;
            well_cells.Append(edge_vert.GetRowColumns(i)[0]);
//            std::cout<<"production well cell: "<<edge_vert.GetRowColumns(i)[0]<<"\n";
        }

//        int num_perforations = (num_producers+num_injectors)*well_height;
//        for (int i = edge_vert.NumRows() - num_perforations; i < edge_vert.NumRows(); ++i)
//        {
//            well_perforations.Append(i);
//        }

//        mfem::socketstream sout;
//        problem_ptr->VisSetup(sout, well_cells, 0.0, 0.0, "well cells");
    }

//    return 0;

    Hierarchy hierarchy(std::move(graph), upscale_param, partition.get(), &ess_attr_final);
    hierarchy.PrintInfo();

    // Fine scale transport based on fine flux
    std::vector<mfem::Vector> Ss(upscale_param.max_levels);
    std::vector<mfem::Vector> flux_s(upscale_param.max_levels);
    std::vector<mfem::Vector> pres_s(upscale_param.max_levels);

//        int l = 0;

    int perf_offset = edge_vert.NumRows() - (num_injectors+num_producers)*well_height;
    for (int w = 0; w < num_injectors+num_producers; ++w)
    {
        for (int i = perf_offset; i < perf_offset+well_height; ++i)
        {
            well_perforations.Append(i);
        }
        perf_offset += well_height;
    }

    for (int l = 0; l < upscale_param.max_levels; ++l)
    {
        mfem::BlockVector initial_value(problem->BlockOffsets());
        initial_value = 0.0;
        initial_value.GetBlock(2) = 0.2;
        initial_value[initial_value.Size()-1] = 0.8;

        mfem::BlockVector sol(initial_value);
//        if (l == 0)
//        {
//            std::ifstream sol_file("fine_sol.txt");
//            sol.Load(sol_file, initial_value.Size());
//        }
//        else
        {
            fas_param.num_levels = upscale_param.max_levels; //l + 1;
            TwoPhaseSolver solver(*problem, hierarchy, l, evolve_param, fas_param);



            mfem::StopWatch chrono;
            chrono.Start();
            sol = solver.Solve(initial_value);

            if (myid == 0)
            {
                std::cout << "Level " << l << ":\n    Time stepping done in "
                          << chrono.RealTime() << "s.\n";
            }
        }

        Ss[l] = sol.GetBlock(2);
//        Ss[l].SetSize(sol.BlockSize(2) - upscale_param.num_iso_verts);
        for (int ii = 0; ii < upscale_param.num_iso_verts; ++ii)
        {
            Ss[l][sol.BlockSize(2) - 1 - ii] = 0.0;
        }

//        double norm = mfem::ParNormlp(Ss[l], 1, comm);
        double norm = (Ss[l] * hierarchy.GetGraph(0).VertexWeight());
        if (myid == 0) { std::cout << "    || S ||_1 = " << norm << "\n"; }

//        std::ofstream s_file("final_sat_active.txt");
//        Ss[l].Print(s_file, 1);


        flux_s[l] = sol.GetBlock(0);
        pres_s[l] = sol.GetBlock(1);

        auto ShowRelErr = [&](std::vector<mfem::Vector>& sols, std::string name)
        {
            sols[l] -= sols[0];
            double fine_sol_norm = mfem::ParNormlp(sols[0], 2, comm);
            double rel_diff = mfem::ParNormlp(sols[l], 2, comm) / fine_sol_norm;
            if (myid == 0) { std::cout << "   " << name << " relative error: " << rel_diff << "\n"; }
        };

        auto base_msg = "level "+std::to_string(l)+", ne = "+std::to_string(upscale_param.max_evects)
                +", nt = "+std::to_string(upscale_param.max_traces);
//        if (l)
        {
//            mfem::socketstream soutf;
//            problem_ptr->VisSetup(soutf, flux_s[l], 0.0, 0.0, "Flux, "+base_msg, false, false, partition->GetData());
//            mfem::socketstream soutp;
//            problem_ptr->VisSetup(soutp, pres_s[l], 0.0, 0.0, "Pressure, "+base_msg, false, true, partition->GetData());
            mfem::socketstream souts;
            problem_ptr->VisSetup(souts, Ss[l], 0.0, 0.0, "Saturation, "+base_msg, false, true, partition->GetData());
        }


        if (l)
        {
            ShowRelErr(flux_s, "Flux");
            ShowRelErr(pres_s, "Pressure");
            ShowRelErr(Ss, "Saturation");

//            mfem::socketstream sout;
//            problem_ptr->VisSetup(sout, Ss[l], 0.0, 0.0, "Diff "+std::to_string(l));
        }
//        else
//        {
//            std::ofstream sol_file("fine_sol.txt");
//            sol.Print(sol_file, 1);
//        }
    }
    return EXIT_SUCCESS;
}

void SetOptions(FASParameters& param, bool use_vcycle, int num_backtrack, double diff_tol)
{
    param.cycle = use_vcycle ? V_CYCLE : FMG;
    param.nl_solve.linearization = Newton;
    param.coarse_correct_tol = 1e-2;
    param.fine.check_converge = use_vcycle ? false : true;
    param.fine.linearization = param.nl_solve.linearization;
    param.mid.linearization = param.nl_solve.linearization;
    param.coarse.linearization = param.nl_solve.linearization;
    param.fine.num_backtrack = num_backtrack;
    param.mid.num_backtrack = num_backtrack;
    param.coarse.num_backtrack = num_backtrack;
    param.fine.diff_tol = diff_tol;
    param.mid.diff_tol = diff_tol/1.;
    param.coarse.diff_tol = diff_tol/1.;
    param.nl_solve.diff_tol = diff_tol;
}

//mfem::Vector ComputeFaceFlux(const MixedMatrix& darcy_system,
//                             const mfem::Vector& flux)
//{
//    mfem::Vector out(darcy_system.GetTraceFluxes());
//    RescaleVector(flux, out);
//    return out;
//}


mfem::SparseMatrix BuildWeightedUpwindPattern(const GraphSpace& graph_space,
                                              const mfem::Vector& flux)
{
    const int alpha_option = 1;
    const double alpha_tol = std::numeric_limits<double>::min(); // 1e-6;//
    const double alpha_gamma = 1e15;
    double alpha;

    const Graph& graph = graph_space.GetGraph();
    const mfem::SparseMatrix& edge_vert = graph.EdgeToVertex();
    const mfem::SparseMatrix e_te_diag = GetDiag(graph.EdgeToTrueEdge());

    mfem::SparseMatrix upwind_pattern(graph.NumEdges(), graph.NumVertices());

    for (int i = 0; i < graph.NumEdges(); ++i)
    {
        if (edge_vert.RowSize(i) == 2) // edge is interior
        {
//            const int upwind_vert = flux[i] > 0.0 ? 0 : 1;
//            upwind_pattern.Set(i, edge_vert.GetRowColumns(i)[upwind_vert], 1.0);

            if (alpha_option == 1)
            {
                alpha = flux[i] > 0.0 ? 1.0 : 0.0;
            }
            else if (alpha_option == 2)
            {
                if (flux[i] > alpha_tol)
                    alpha = 1.0;
                else if (flux[i] < -alpha_tol)
                    alpha = 0.0;
                else
                    alpha = 0.5;
            }
            else if (alpha_option == 3)
            {
                if (flux[i] > alpha_tol)
                    alpha = 1.0;
                else if (flux[i] < -alpha_tol)
                    alpha = 0.0;
                else
                    alpha = (flux[i] + alpha_tol) / (2 * alpha_tol);
            }
            else if (alpha_option == 4)
            {
                alpha = 0.5 + (1.0 / M_PI) * std::atan(alpha_gamma * flux[i]);
            }

            upwind_pattern.Set(i, edge_vert.GetRowColumns(i)[0], alpha);
            upwind_pattern.Set(i, edge_vert.GetRowColumns(i)[1], 1.0 - alpha);
        }
        else
        {
            assert(edge_vert.RowSize(i) == 1);
            const bool edge_is_owned = e_te_diag.RowSize(i);

            if ((flux[i] > 0.0 && edge_is_owned) || (flux[i] <= 0.0 && !edge_is_owned))
            {
                upwind_pattern.Set(i, edge_vert.GetRowColumns(i)[0], 1.0);
            }
        }
    }
    upwind_pattern.Finalize(); // TODO: use sparsity pattern of DT and update the values

    return upwind_pattern;
}


mfem::SparseMatrix BuildUpwindPattern(const GraphSpace& graph_space,
                                      const mfem::Vector& flux)
{
    const Graph& graph = graph_space.GetGraph();
    const mfem::SparseMatrix& edge_vert = graph.EdgeToVertex();
    const mfem::SparseMatrix e_te_diag = GetDiag(graph.EdgeToTrueEdge());

    const int num_edofs = graph_space.VertexToEDof().NumCols();
    mfem::SparseMatrix upwind_pattern(num_edofs, graph.NumVertices());

    for (int i = 0; i < graph.NumEdges(); ++i)
    {
        if (edge_vert.RowSize(i) == 2) // edge is interior
        {
            const int upwind_vert = flux[i] > 0.0 ? 0 : 1;
            upwind_pattern.Set(i, edge_vert.GetRowColumns(i)[upwind_vert], 1.0);
        }
        else
        {
            assert(edge_vert.RowSize(i) == 1);
            const bool edge_is_owned = e_te_diag.RowSize(i);

            if ((flux[i] > 0.0 && edge_is_owned) || (flux[i] <= 0.0 && !edge_is_owned))
            {
                upwind_pattern.Set(i, edge_vert.GetRowColumns(i)[0], 1.0);
            }
        }
    }
    upwind_pattern.Finalize(); // TODO: use sparsity pattern of DT and update the values

    return upwind_pattern;
}

mfem::SparseMatrix BuildUpwindPattern(const GraphSpace& graph_space,
                                      const mfem::SparseMatrix& micro_upwind_fluxes,
                                      const mfem::Vector& flux)
{
    const Graph& graph = graph_space.GetGraph();
    const mfem::SparseMatrix& edge_vert = graph.EdgeToVertex();
    const mfem::SparseMatrix e_te_diag = GetDiag(graph.EdgeToTrueEdge());

    const int num_edofs = graph_space.VertexToEDof().NumCols();
    mfem::SparseMatrix upwind_pattern(num_edofs, graph.NumVertices());

    for (int i = 0; i < graph.NumEdges(); ++i)
    {
        if (edge_vert.RowSize(i) == 2) // edge is interior
        {
            double weight, weight2;
            if (flux[i] > 0.0)
            {
                const int upwind_vert = edge_vert.GetRowColumns(i)[0];
                const int downwind_vert = edge_vert.GetRowColumns(i)[1];
                if (upwind_vert == micro_upwind_fluxes.GetRowColumns(i)[0])
                {
                    weight = micro_upwind_fluxes.GetRowEntries(i)[0];
                }
                else
                {
                    weight = micro_upwind_fluxes.GetRowEntries(i)[1];
                }
                upwind_pattern.Set(i, upwind_vert, weight);

                if (micro_upwind_fluxes.RowSize(i) == 2)
                {
                    if (upwind_vert == micro_upwind_fluxes.GetRowColumns(i)[0])
                    {
                        weight2 = micro_upwind_fluxes.GetRowEntries(i)[1];
                    }
                    else
                    {
                        weight2 = micro_upwind_fluxes.GetRowEntries(i)[0];
                    }
                    upwind_pattern.Set(i, downwind_vert, weight2);
                }
            }

            if (flux[i] <= 0.0)
            {
                const int upwind_vert = edge_vert.GetRowColumns(i)[1];
                const int downwind_vert = edge_vert.GetRowColumns(i)[0];
                weight = micro_upwind_fluxes.GetRowEntries(i)[0];
                if (upwind_vert == micro_upwind_fluxes.GetRowColumns(i)[0])
                {
                    upwind_pattern.Set(i, downwind_vert, weight);
                }
                else
                {
                    upwind_pattern.Set(i, upwind_vert, weight);
                }

                if (micro_upwind_fluxes.RowSize(i) == 2)
                {
                    weight2 = micro_upwind_fluxes.GetRowEntries(i)[1];
                    if (upwind_vert == micro_upwind_fluxes.GetRowColumns(i)[1])
                    {
                        upwind_pattern.Set(i, downwind_vert, weight2);
                    }
                    else
                    {
                        upwind_pattern.Set(i, upwind_vert, weight2);
                    }
                }
            }
        }
        else
        {
            assert(edge_vert.RowSize(i) == 1);
            const bool edge_is_owned = e_te_diag.RowSize(i);
            if ((flux[i] > 0.0 && edge_is_owned) || (flux[i] <= 0.0 && !edge_is_owned))
            {
                const double weight = micro_upwind_fluxes.GetRowEntries(i)[0];
                upwind_pattern.Set(i, edge_vert.GetRowColumns(i)[0], weight);
            }
        }
    }
    upwind_pattern.Finalize(); // TODO: use sparsity pattern of DT and update the values

    return upwind_pattern;
}

std::vector<mfem::DenseMatrix> Build_dTdsigma(const GraphSpace& graph_space,
                                              const mfem::SparseMatrix& micro_upwind_fluxes,
                                              const mfem::SparseMatrix& D,
                                              const mfem::Vector& flux,
                                              mfem::Vector FS)
{
    const Graph& graph = graph_space.GetGraph();
    const mfem::SparseMatrix& vert_edof = graph_space.VertexToEDof();
    const mfem::SparseMatrix& edge_vert = graph.EdgeToVertex();
    const mfem::SparseMatrix e_te_diag = GetDiag(graph.EdgeToTrueEdge());

    std::vector<mfem::DenseMatrix> out(graph.NumVertices());
    for (int i = 0; i < graph.NumVertices(); ++i)
    {
        const int num_edofs = vert_edof.RowSize(i);
        out[i].SetSize(1, num_edofs);
        out[i] = 0.0;
        for (int j = 0; j < num_edofs; ++j)
        {
            const int edge = vert_edof.GetRowColumns(i)[j];

            if (edge_vert.RowSize(edge) == 2) // edge is interior
            {
                double weight, weight2;
                double FS_up = 0.0;
                if (flux[edge] > 0.0)
                {
                    const int upwind_vert = edge_vert.GetRowColumns(edge)[0];
                    const int downwind_vert = edge_vert.GetRowColumns(edge)[1];
                    if (upwind_vert == micro_upwind_fluxes.GetRowColumns(edge)[0])
                    {
                        weight = micro_upwind_fluxes.GetRowEntries(edge)[0];
                    }
                    else
                    {
                        weight = micro_upwind_fluxes.GetRowEntries(edge)[1];
                    }
                    FS_up += FS[upwind_vert] * weight;

                    if (micro_upwind_fluxes.RowSize(edge) == 2)
                    {
                        if (upwind_vert == micro_upwind_fluxes.GetRowColumns(edge)[0])
                        {
                            weight2 = micro_upwind_fluxes.GetRowEntries(edge)[1];
                        }
                        else
                        {
                            weight2 = micro_upwind_fluxes.GetRowEntries(edge)[0];
                        }
                        FS_up += FS[downwind_vert] * weight2;
                    }
                }

                if (flux[edge] <= 0.0)
                {
                    const int upwind_vert = edge_vert.GetRowColumns(edge)[1];
                    const int downwind_vert = edge_vert.GetRowColumns(edge)[0];
                    weight = micro_upwind_fluxes.GetRowEntries(edge)[0];
                    if (upwind_vert == micro_upwind_fluxes.GetRowColumns(edge)[0])
                    {
                        FS_up += FS[downwind_vert] * weight;
                    }
                    else
                    {
                        FS_up += FS[upwind_vert] * weight;
                    }

                    if (micro_upwind_fluxes.RowSize(edge) == 2)
                    {
                        weight2 = micro_upwind_fluxes.GetRowEntries(edge)[1];
                        if (upwind_vert == micro_upwind_fluxes.GetRowColumns(edge)[1])
                        {
                            FS_up += FS[downwind_vert] * weight2;
                        }
                        else
                        {
                            FS_up += FS[upwind_vert] * weight2;
                        }
                    }
                }

                out[i](0, j) = D(i, edge) * FS_up;
            }
            else
            {
                assert(edge_vert.RowSize(edge) == 1);
                const bool edge_is_owned = e_te_diag.RowSize(edge);

                if ((flux[edge] > 0.0 && edge_is_owned) || (flux[edge] <= 0.0 && !edge_is_owned))
                {
                    if (edge_vert.GetRowColumns(edge)[0] == i)
                    {
                        double weight = micro_upwind_fluxes.GetRowEntries(edge)[0];
                        out[i](0, j) = D(i, edge) * FS[i] * weight;
                    }
                }
            }
        }

    }

    return out;
}


std::vector<mfem::DenseMatrix> Build_dTdsigma(const GraphSpace& graph_space,
                                              const mfem::SparseMatrix& D,
                                              const mfem::Vector& FS_up)
{
    const Graph& graph = graph_space.GetGraph();
    const mfem::SparseMatrix& vert_edof = graph_space.VertexToEDof();

    std::vector<mfem::DenseMatrix> out(graph.NumVertices());
    for (int i = 0; i < graph.NumVertices(); ++i)
    {
        const int num_edofs = vert_edof.RowSize(i);
        out[i].SetSize(1, num_edofs);
        out[i] = 0.0;
        for (int j = 0; j < num_edofs; ++j)
        {
            const int edge = vert_edof.GetRowColumns(i)[j];
            out[i](0, j) = D(i, edge) * FS_up[edge];
        }
    }

    return out;
}

std::vector<mfem::DenseMatrix> Build_dTdsigma(const GraphSpace& graph_space,
                                              const mfem::SparseMatrix& D,
                                              const mfem::Vector& flux,
                                              mfem::Vector FS)
{
    const Graph& graph = graph_space.GetGraph();
    const mfem::SparseMatrix& vert_edof = graph_space.VertexToEDof();
    const mfem::SparseMatrix& edge_vert = graph.EdgeToVertex();
    const mfem::SparseMatrix e_te_diag = GetDiag(graph.EdgeToTrueEdge());

    std::vector<mfem::DenseMatrix> out(graph.NumVertices());
    for (int i = 0; i < graph.NumVertices(); ++i)
    {
        const int num_edofs = vert_edof.RowSize(i);
        out[i].SetSize(1, num_edofs);
        out[i] = 0.0;
        for (int j = 0; j < num_edofs; ++j)
        {
            const int edge = vert_edof.GetRowColumns(i)[j];

            if (edge_vert.RowSize(edge) == 2) // edge is interior
            {
                const int upwind_vert = flux[edge] > 0.0 ? 0 : 1;
                const int upwind_i = edge_vert.GetRowColumns(edge)[upwind_vert];
                out[i](0, j) = D(i, edge) * FS[upwind_i];
            }
            else
            {
                assert(edge_vert.RowSize(edge) == 1);
                const bool edge_is_owned = e_te_diag.RowSize(edge);

                if ((flux[edge] > 0.0 && edge_is_owned) ||
                        (flux[edge] <= 0.0 && !edge_is_owned))
                {
                    if (edge_vert.GetRowColumns(edge)[0] == i)
                    {
                        out[i](0, j) = D(i, edge) * FS[i];
                    }
                }
            }
        }
    }

    return out;
}

TwoPhaseSolver::TwoPhaseSolver(const DarcyProblem& problem, Hierarchy& hierarchy,
                               const int level, const EvolveParamenters& evolve_param,
                               const FASParameters& solver_param)
    : level_(level), evolve_param_(evolve_param), solver_param_(solver_param),
      problem_(problem), hierarchy_(&hierarchy), blk_offsets_(4), nonlinear_iter_(0),
      step_converged_(true), weight_(SparseDiag(hierarchy.GetMatrix(0).GetGraph().VertexWeight()))
{
    weight_ *= density_;
    for (int l = 0; l < level-1; ++l)
    {
        unique_ptr<mfem::SparseMatrix> coarse_weight(mfem::RAP(hierarchy.GetPs(l), weight_, hierarchy.GetPs(l)));
        weight_.Swap(*coarse_weight);
    }

    blk_helper_.reserve(level + 1);
    blk_helper_.emplace_back(hierarchy.BlockOffsets(0));
    blk_helper_[0].GetBlock(0) = problem_.GetEdgeRHS();
    blk_helper_[0].GetBlock(1) = problem_.GetVertexRHS();

    for (int l = 0; l < level_; ++l)
    {
        blk_helper_.push_back(hierarchy.Restrict(l, blk_helper_[l]));
    }

    auto& darcy_system = hierarchy.GetMatrix(level ? level-1 : 0);
    const int S_size = darcy_system.GetPWConstProj().NumCols();
    blk_offsets_[0] = 0;
    blk_offsets_[1] = hierarchy.BlockOffsets(level)[1];
    blk_offsets_[2] = hierarchy.BlockOffsets(level)[2];
    blk_offsets_[3] = blk_offsets_[2] + S_size;

    source_.reset(new mfem::BlockVector(blk_offsets_));
    source_->GetBlock(0) = blk_helper_[level].GetBlock(0);
    source_->GetBlock(1) = blk_helper_[level].GetBlock(1);
//    if (S_size != darcy_system.NumVDofs())
    {
        mfem::Vector tmp_helper(problem_.GetVertexRHS());
        for (int l = 0; l < level_-1; ++l)
        {
            tmp_helper = smoothg::MultTranspose(hierarchy.GetPs(l), tmp_helper);
        }
        source_->GetBlock(2)  = tmp_helper;
    }
//    else
//    {
//        source_->GetBlock(2) = blk_helper_[level].GetBlock(1);
//    }

//    for (int i = 0; i < source_->BlockSize(2); ++i)
//    {
//        source_->GetBlock(2)[i] = std::min(source_->GetBlock(2)[i], 0.0);
//    }

    unique_ptr<mfem::HypreParMatrix> e_te_e(
                mfem::ParMult(&hierarchy.GetMatrix(level).GetGraphSpace().EDofToTrueEDof(),
                              &hierarchy.GetMatrix(level).GetGraphSpace().TrueEDofToEDof()));
    *e_te_e = 1.0;
//    auto& starts = hierarchy.GetMatrix(level).GetGraph().VertexStarts();
    mfem::SparseMatrix D = hierarchy.GetMatrix(level).GetD();
//    D_te_e_ = ParMult(D, *e_te_e, starts);
    D_te_e_.reset(ToParMatrix(hierarchy.GetComm(), std::move(D)));

    cumulative_step_time_.push_back(0.0);
}

int step_global = 0;
int step_local = 0;
mfem::Vector pres_resid;
mfem::Vector sat_resid;

mfem::Vector CFL_plot;
mfem::Vector CFL_influx;
mfem::Vector CFL_outflux;
mfem::Vector CFL_dfds;
mfem::Vector CFL_pore_vol;

mfem::BlockVector TwoPhaseSolver::Solve(const mfem::BlockVector& init_val)
{
    int myid;
    MPI_Comm_rank(hierarchy_->GetComm(), &myid);

    mfem::BlockVector x(blk_offsets_);

    blk_helper_[0].GetBlock(0) = init_val.GetBlock(0);
    blk_helper_[0].GetBlock(1) = init_val.GetBlock(1);
    mfem::Vector x_blk2 = init_val.GetBlock(2);


    mfem::socketstream sout;
    std::string msg = "Level "+std::to_string(level_);
    if (evolve_param_.vis_step) { problem_ptr->VisSetup(sout, x_blk2, 0.0, 0.0, msg); }

    for (int l = 0; l < level_; ++l)
    {
        hierarchy_->Project(l, blk_helper_[l], blk_helper_[l + 1]);
        if (l)
        {
            x_blk2 = MultTranspose(hierarchy_->GetPs(l-1), x_blk2);
        }
    }

    x.GetBlock(0) = blk_helper_[level_].GetBlock(0);
    x.GetBlock(1) = blk_helper_[level_].GetBlock(1);
    x.GetBlock(2) = x_blk2;

    double dt_multiplier = 2.0;
    double time = 0.0;
//    double dt_real = std::min(evolve_param_.dt, evolve_param_.total_time - time) / dt_multiplier;
    double dt_real = std::min(evolve_param_.dt, evolve_param_.total_time - time);

    std::vector<double> cumulative_time;
    bool done = false;
    int step;
    for (step = 1; !done; step++)
    {
        step_global = step;

        mfem::BlockVector previous_x(x);
//        dt_real = std::min(std::min(dt_real * 2.0, evolve_param_.total_time - time), 345600.);
        if (step > 2)
            dt_real = std::min(std::min(dt_real * 2.0, evolve_param_.total_time - time), 2592000.0);
//        dt_real = std::min(dt_real * dt_multiplier, evolve_param_.total_time - time);
//        dt_real = std::min(dt_real, evolve_param_.total_time - time);

        step_converged_ = false;

        TimeStepping(dt_real, x);
        while (!step_converged_)
        {
            x = previous_x;
            dt_real /= 2.0;
            TimeStepping(dt_real, x);
        }

//        auto S_tmp = hierarchy_->PWConstProject(level_, x.GetBlock(2));
//        x.GetBlock(2) = hierarchy_->PWConstInterpolate(level_, S_tmp);

        time += dt_real;
        done = (time >= evolve_param_.total_time);
        cumulative_time.push_back(time / 86400.0);

        mfem::Vector well_flux;
        {
            x.GetSubVector(well_perforations, well_flux);
            std::cout<< "flux at wells" << ":\n";
            well_flux.Print(std::cout, 5);

            std::cout<< "pressure at wells: "<< x.GetBlock(1)[x.BlockSize(1)-1] << "\n";
            std::cout<< "saturation at wells: "<< x[x.Size()-1] << "\n";
            mfem::Vector Ws(x.BlockSize(2));
            weight_.Mult(x.GetBlock(2), Ws);
            std::cout<< "sum(Ws) / sum(W): "<<
                        Ws.Sum() / 11781.6 / density_ / (400.0*std::pow(0.3048, 3)) << "\n";
        }

        if (myid == 0)
        {
            std::cout << "Time step " << step << ": step size = " << dt_real
                      << ", time = " << time << ".\n\n";
        }

        if (evolve_param_.vis_step && (done || step % evolve_param_.vis_step == 0))
        {
            x_blk2 = x.GetBlock(2);
            for (int l = level_-1; l > 0; --l)
            {
                x_blk2 = MatVec(hierarchy_->GetPs(l-2), x_blk2);
            }
            problem_ptr->VisUpdate(sout, x_blk2);
        }

        x_blk2 = x.GetBlock(1);
        x_blk2 *= -1.;
        x_blk2.SetSize(x_blk2.Size() - hierarchy_->GetUpscaleParameters().num_iso_verts);

        std::ofstream all_file("saigup_CFL_"+std::to_string(step)+".vtk");
        all_file.precision(8);
        all_file <<  "\nSCALARS pressure double 1\nLOOKUP_TABLE default\n";
        all_file << std::fixed;
        x_blk2.Print(all_file, 1);

        x_blk2 = x.GetBlock(2);
        x_blk2.SetSize(x_blk2.Size() - hierarchy_->GetUpscaleParameters().num_iso_verts);
//        std::ofstream s_file("egg_sat_"+std::to_string(step)+".vtk");
//        s_file.precision(8);
        all_file << "\nSCALARS saturation double 1\nLOOKUP_TABLE default\n";
        all_file << std::fixed;
        x_blk2.Print(all_file, 1);

        if (level_ == 0)
        {
            step_CFL_const_.push_back(EvalCFL(dt_real, x));
        }

    }

    if (myid == 0)
    {
        std::cout << "Total nonlinear iterations: " << nonlinear_iter_ << "\n";
        std::cout << "Average nonlinear iterations per time step: "
                  << double(nonlinear_iter_) / double(step-1) << "\n";
        std::cout << "Total coarsest linear iterations: " << num_coarse_lin_iter_ << "\n";
        std::cout << "Average linear iterations per coarsest level linear solve: "
                  << num_coarse_lin_iter_ / double(num_coarse_lin_solve_) << "\n";

        PrintTable(step_nonlinear_iter_, "# nonlinear iter");
        PrintTable(step_coarsest_nonlinear_iter_, "# coarsest nonlinear iter");
        PrintTable(step_average_coarsest_linear_iter_, "# average coarsest linear iter");
        PrintTable(step_average_mid_linear_iter_, "# average mid linear iter");
        PrintTable(step_average_fine_linear_iter_, "# average fine linear iter");
        PrintTable(step_time_, "solving_time", false);
        std::vector<double> cumu_step_time(cumulative_step_time_.begin()+1, cumulative_step_time_.end());
        PrintTable(cumu_step_time, "cumulative_solving_time", false);
        PrintTable(cumulative_time, "cumulative_time", false);
        PrintTable(step_CFL_const_, "CFL constants");
//        PrintForLatexTable(step_num_backtrack_, "# backtrack");
    }

    blk_helper_[level_].GetBlock(0) = x.GetBlock(0);
    blk_helper_[level_].GetBlock(1) = x.GetBlock(1);
    x_blk2 = x.GetBlock(2);

    for (int l = level_; l > 0; --l)
    {
        hierarchy_->Interpolate(l, blk_helper_[l], blk_helper_[l - 1]);
        if (l > 1)
        {
            x_blk2 = MatVec(hierarchy_->GetPs(l-2), x_blk2);
        }
    }

    mfem::BlockVector out(problem_.BlockOffsets());
    out.GetBlock(0) = blk_helper_[0].GetBlock(0);
    out.GetBlock(1) = blk_helper_[0].GetBlock(1);
    out.GetBlock(2) = x_blk2;

    return out;
}


double TwoPhaseSolver::EvalCFL(double dt, const mfem::BlockVector& x) const
{
    const auto& darcy_system = hierarchy_->GetMatrix(0);
    const auto& D = darcy_system.GetD();
    const auto& graph = darcy_system.GetGraph();
    const auto& edge_vert = graph.EdgeToVertex();
    auto& pore_volume = graph.VertexWeight();
    const int num_injectors = hierarchy_->GetUpscaleParameters().num_iso_verts;

    double CFL_const = 0.0;

    mfem::Vector S = PWConstProject(darcy_system, x.GetBlock(2));
    for (int ii = 0; ii < S.Size(); ++ii)
    {
        if (S[ii] < 0.0) { S[ii]  = 0.0; }
        if (S[ii] > 1.0) { S[ii] =  1.0; }
    }
    mfem::Vector df_ds = dFdS(S);

    mfem::Array<int> edofs;
    mfem::Vector local_flux(D.NumRows());
    for (int i = 0; i < darcy_system.NumVDofs() - num_injectors; ++i)
    {
        GetTableRow(D, i, edofs);
        x.GetSubVector(edofs, local_flux);
        double local_CFL_const = df_ds[i] * dt * local_flux.Norml1() / pore_volume(i);
        CFL_const = std::max(CFL_const, local_CFL_const);
    }

    mfem::Vector inflow_CFL_consts(S.Size()), outflow_CFL_consts(S.Size());
    inflow_CFL_consts = 0.0;
    outflow_CFL_consts = 0.0;

    double CFL_const2 = 0.0;
    for (int i = 0; i < darcy_system.NumEDofs(); ++i)
    {
        if (edge_vert.RowSize(i) == 2) // edge is interior
        {
            const int upwind_vert = x[i] > 0.0 ? 0 : 1;
            const int downwind_vert = 1 - upwind_vert;
//            upwind_pattern.Set(i, edge_vert.GetRowColumns(i)[upwind_vert], 1.0);
            outflow_CFL_consts[edge_vert.GetRowColumns(i)[upwind_vert]] += std::abs(x[i]);
            inflow_CFL_consts[edge_vert.GetRowColumns(i)[downwind_vert]] += std::abs(x[i]);
        }
        else
        {
            assert(edge_vert.RowSize(i) == 1);
            // TODO: need to also look at e_te_diag when in parallel
            if (x[i] > 0.0) // the cell in the domain is upwind
            {
                outflow_CFL_consts[edge_vert.GetRowColumns(i)[0]] += std::abs(x[i]);
            }
            else if (x[i] < 0.0) // downwind
            {
                inflow_CFL_consts[edge_vert.GetRowColumns(i)[0]] += std::abs(x[i]);
            }
        }
    }

    for (int i = 0; i < darcy_system.NumVDofs() - num_injectors; ++i)
    {
        double outflow_CFL_const = df_ds[i] * dt * outflow_CFL_consts[i] / pore_volume(i);
        double inflow_CFL_const = df_ds[i] * dt * inflow_CFL_consts[i] / pore_volume(i);
        double local_CFL_const = std::max(inflow_CFL_const, outflow_CFL_const);


        CFL_const2 = std::max(CFL_const2, local_CFL_const);
    }
    std::cout<< "CFL const = "<<CFL_const<<", CFL const (separate inflow and outflow) = "<<CFL_const2<<"\n";

    return CFL_const2;
}

void TwoPhaseSolver::TimeStepping(const double dt, mfem::BlockVector& x)
{
    const MixedMatrix& system = hierarchy_->GetMatrix(level_);
    std::vector<mfem::DenseMatrix> traces;

    if (evolve_param_.scheme == FullyImplcit) // coupled: solve all unknowns together
    {
        const MixedMatrix& system_s = level_ ? hierarchy_->GetMatrix(level_-1) : system;
        auto S = PWConstProject(system_s, x.GetBlock(2));
        CoupledSolver solver(*hierarchy_, level_, system, problem_.EssentialAttribute(),
                             hierarchy_->GetUpwindFlux(level_), dt, weight_,
                             density_, S, solver_param_.nl_solve);
//        CoupledFAS solver(*hierarchy_, problem_.EssentialAttribute(), dt, weight_,
//                          density_, x.GetBlock(2), solver_param_);

        mfem::Vector res(x.GetBlock(2));
        res = 0.0;
        if (0 && step_global >= 6)
        {
            problem_ptr->VisSetup(sout_resid_[step_global], res, 0.0, 0.0, "Resid "+std::to_string(step_global));
        }

        mfem::BlockVector rhs(*source_);
        rhs.GetBlock(0) *= (1. / dt / density_);
        rhs.GetBlock(1) *= (dt * density_);
//        add(dt * density_, rhs.GetBlock(2), weight_, x.GetBlock(2), rhs.GetBlock(2));
        rhs.GetBlock(2) *= (dt * density_);
        rhs[rhs.Size()-1] = 0.8;
        weight_.AddMult(x.GetBlock(2), rhs.GetBlock(2));

//        rhs.GetBlock(2) *= (1. / dt / density_);

        solver.Solve(rhs, x);

//        step_coarsest_nonlinear_iter_.push_back(solver.GetNumCoarsestIterations());
////        step_num_backtrack_.push_back(num_backtrack_debug);
        step_nonlinear_iter_.push_back(solver.GetNumIterations());
//        step_nonlinear_iter_.push_back(solver_param_.cycle == V_CYCLE ? solver.GetNumIterations()
//                                                : solver.GetLevelSolver(0).GetNumIterations());
//        int num_coarse_lin_iter = solver.GetLevelSolver(hierarchy_->NumLevels()-1).GetNumLinearIterations();
        int num_fine_lin_iter = solver.GetNumLinearIterations();
//        int num_mid_lin_iter = hierarchy_->NumLevels() > 1 ? solver.GetLevelSolver(1).GetNumLinearIterations() : 0;
////        int num_coarse_lin_iter = solver.GetNumLinearIterations();
//        double avg_lin = solver.GetNumCoarsestIterations() == 0 ? 0.0 :
//                    num_coarse_lin_iter / ((double)solver.GetNumCoarsestIterations());
////                num_coarse_lin_iter / ((double)solver.GetNumIterations());
//        step_average_coarsest_linear_iter_.push_back(std::round(avg_lin));
        nonlinear_iter_ += step_nonlinear_iter_.back();
        step_converged_ = solver.IsConverged();
//        num_coarse_lin_iter_ += num_coarse_lin_iter;
//        num_coarse_lin_solve_ += step_coarsest_nonlinear_iter_.back();
        step_time_.push_back(solver.GetTiming());
//        cumulative_step_time_.push_back(cumulative_step_time_.back()+solver.GetTiming());

////        double avg_lin_f = solver.GetNumCoarsestIterations() == 0 ? 0.0 :
////                    num_fine_lin_iter / ((double)solver.GetNumCoarsestIterations());
////        double avg_lin_m = solver.GetNumCoarsestIterations() == 0 ? 0.0 :
////                    num_mid_lin_iter / ((double)solver.GetNumCoarsestIterations());
        step_average_fine_linear_iter_.push_back(num_fine_lin_iter);
//        step_average_mid_linear_iter_.push_back(num_mid_lin_iter);
    }
    else // sequential: solve for flux and pressure first, and then saturation
    {
        mfem::mfem_error("usage of weight need to be fixed!\n");

        const mfem::Vector S = PWConstProject(system, x.GetBlock(2));
        hierarchy_->RescaleCoefficient(level_, TotalMobility(S));
        mfem::BlockVector flow_rhs(*source_, hierarchy_->BlockOffsets(level_));
        mfem::BlockVector flow_sol(x, hierarchy_->BlockOffsets(level_));
        hierarchy_->Solve(level_, flow_rhs, flow_sol);

        unique_ptr<mfem::HypreParMatrix> Adv_c;
        mfem::SparseMatrix upwind = BuildUpwindPattern(system.GetGraphSpace(), x.GetBlock(0));
        upwind.ScaleRows(x.GetBlock(0));

        if (evolve_param_.scheme == IMPES) // explcict: new_S = S + dt W^{-1} (b - Adv F(S))
        {
            mfem::Vector dSdt(source_->GetBlock(2));
            D_te_e_->Mult(-1.0, MatVec(upwind, FractionalFlow(S)), 1.0, dSdt);
//            x.GetBlock(2).Add(dt * density_ / weight_, dSdt);
            step_converged_ = true;
        }
        else // implicit: new_S solves new_S = S + dt W^{-1} (b - Adv F(new_S))
        {
            auto Adv = ParMult(*D_te_e_, upwind, system.GetGraph().VertexStarts());

            const double scaling = weight_(0, 0) / density_ / dt;
            TransportSolver solver(*Adv, system, scaling, solver_param_.nl_solve);

            mfem::Vector rhs(source_->GetBlock(2));
//            rhs.Add(weight_ / density_ / dt, x.GetBlock(2));
            solver.Solve(rhs, x.GetBlock(2));
            step_converged_ = solver.IsConverged();
            nonlinear_iter_ += solver.GetNumIterations();
        }
    }
}

double ParNorm(const mfem::Vector& vec, MPI_Comm comm)
{
    return mfem::ParNormlp(vec, mfem::infinity(), comm);
}

void LocalChopping(const MixedMatrix& darcy_system, mfem::Vector& x)
{
    const mfem::Vector S = PWConstProject(darcy_system, x);
    for (int i = 0; i < x.Size(); ++i)
    {
        if (S[i] < 0.0) { x[i] = 0.0; }
        if (S[i] > 1.0) { x[i] /= S[i]; }
    }
}

CoupledSolver::CoupledSolver(const Hierarchy& hierarchy,
                             int level,
                             const MixedMatrix& darcy_system,
                             const mfem::Array<int>& ess_attr,
//                             const std::vector<mfem::DenseMatrix>& edge_traces,
                             const mfem::SparseMatrix& micro_upwind_flux,
                             const double dt,
                             mfem::SparseMatrix weight,
                             const double density,
                             const mfem::Vector& S_prev,
                             NLSolverParameters param)
    : NonlinearSolver(darcy_system.GetComm(), param), hierarchy_(hierarchy),
      level_(level), darcy_system_(darcy_system), ess_attr_(ess_attr),
      gmres_(comm_), local_dMdS_(darcy_system.GetGraph().NumVertices()),
      Ms_(std::move(weight)), blk_offsets_(4), true_blk_offsets_(4),
      ess_dofs_(darcy_system.GetEssDofs()),
//      vert_starts_(darcy_system.GetGraph().VertexStarts()),
//      traces_(edge_traces),
      micro_upwind_flux_(micro_upwind_flux),
      dt_(dt), density_(density), //is_first_resid_eval_(false),
      scales_(3), D_fine_(hierarchy_.GetMatrix(0).GetD())
{
    tag_ = "Level " + std::to_string(level) + " coupled solver";

    mfem::SparseMatrix D_proc(darcy_system_.GetD());
    if (ess_dofs_.Size()) { D_proc.EliminateCols(ess_dofs_); }
    D_.reset(darcy_system_.MakeParallelD(D_proc));
    DT_.reset(D_->Transpose());
    *D_ *= (dt_ * density_);
    *DT_ *= (1. / dt_ / density_);

    GenerateOffsets(comm_, D_->NumCols(), true_edof_starts_);

    const int S_size = level ? hierarchy.GetMatrix(level-1).GetPWConstProj().NumCols() :
                               darcy_system.GetPWConstProj().NumCols();
    blk_offsets_[0] = 0;
    blk_offsets_[1] = darcy_system.NumEDofs();
    blk_offsets_[2] = blk_offsets_[1] + darcy_system.NumVDofs();
    blk_offsets_[3] = blk_offsets_[2] + S_size;

    true_blk_offsets_[0] = 0;
    true_blk_offsets_[1] = D_->NumCols();
    true_blk_offsets_[2] = true_blk_offsets_[1] + darcy_system.NumVDofs();
    true_blk_offsets_[3] = true_blk_offsets_[2] + S_size;

    gmres_.SetMaxIter(500);
    gmres_.SetAbsTol(1e-15);
    gmres_.SetRelTol(1e-12);
    gmres_.SetPrintLevel(0);
    gmres_.SetKDim(100);

    GenerateOffsets(comm_, S_size, sdof_starts_);

    mfem::SparseMatrix Ds_proc(hierarchy.GetMatrix(level? level-1 : 0).GetDs());

    const mfem::Array<int>& ess_dofs_f = level ? hierarchy.GetMatrix(level-1).GetEssDofs()
                                               : ess_dofs_;
    if (ess_dofs_f.Size()) { Ds_proc.EliminateCols(ess_dofs_f); }
    auto& edof_trueedof = hierarchy.GetMatrix(level? level-1 : 0).GetGraphSpace().EDofToTrueEDof();
    Ds_ = ParMult(Ds_proc, edof_trueedof, sdof_starts_);
    *Ds_ *= (dt_ * density_);

    normalizer_.SetSize(Ds_->NumRows()); // TODO: need to have one for each block
//    Ms_.GetDiag(normalizer_);
    normalizer_ = 800.0 * (1.0 / density_);

//    if (false) // TODO: define a way to normalize for higher order coarsening
//    {
//        mfem::Vector normalizer_help(normalizer_.Size());
//        normalizer_help = S_prev;
//        normalizer_help -= 1.0;
//        normalizer_help *= -800.0;
//        normalizer_help.Add(1000.0, S_prev);
////        normalizer_ *= (weight(0, 0) / density_); // weight_ / density_ = vol * porosity
//        Ms_.Mult(normalizer_help, normalizer_);
//        normalizer_ /= density_;
//    }

    scales_ = 1.0;

    if (ess_dofs_.Size()) { D_fine_.EliminateCols(hierarchy_.GetMatrix(0).GetEssDofs()); }
    D_fine_ *= (dt_ * density_);
}

mfem::Vector CoupledSolver::AssembleTrueVector(const mfem::Vector& vec) const
{
    mfem::Vector true_v(true_blk_offsets_.Last());
    mfem::BlockVector blk_v(vec.GetData(), blk_offsets_);
    mfem::BlockVector blk_true_v(true_v.GetData(), true_blk_offsets_);
    blk_true_v = 0.0;

    auto& truedof_dof = darcy_system_.GetGraphSpace().TrueEDofToEDof();
    truedof_dof.Mult(blk_v.GetBlock(0), blk_true_v.GetBlock(0));
    blk_true_v.GetBlock(1) = blk_v.GetBlock(1);
    blk_true_v.GetBlock(2) = blk_v.GetBlock(2);

    return true_v;
}

int is_odd=0;
mfem::Vector tmp_save(44921);

mfem::Vector CoupledSolver::Residual(const mfem::Vector& x, const mfem::Vector& y)
{
    mfem::BlockVector blk_x(x.GetData(), blk_offsets_);
    mfem::BlockVector out(blk_offsets_);
    out = 0.0;


    mfem::Vector x_blk_2_coarse = level_ ? MultTranspose(hierarchy_.GetPs(level_-1), blk_x.GetBlock(2)) :
                                           blk_x.GetBlock(2);
    const mfem::Vector S = PWConstProject(darcy_system_, x_blk_2_coarse);

    auto& darcy_system_0 = hierarchy_.GetMatrix(0);
    mfem::Vector fine_flux = blk_x.GetBlock(0);
    for (int i = level_-1; i >= 0 ; --i)
    {
        fine_flux = MatVec(hierarchy_.GetPsigma(i), fine_flux);
    }
    mfem::Vector fine_S = blk_x.GetBlock(2);
    for (int i = level_-2; i >= 0 ; --i)
    {
        fine_S = MatVec(hierarchy_.GetPs(i), fine_S);
    }

    if (exact_flow_RAP_)
    {
        mfem::Vector fine_p = blk_x.GetBlock(1);
        for (int i = level_-1; i >= 0 ; --i)
        {
            fine_p = MatVec(hierarchy_.GetPu(i), fine_p);
        }

        mfem::BlockVector fine_darcy_x(darcy_system_0.BlockOffsets());
        fine_darcy_x.GetBlock(0) = fine_flux;
        fine_darcy_x.GetBlock(1) = fine_p;

        mfem::BlockVector fine_darcy_Rx(darcy_system_0.BlockOffsets());
        fine_darcy_Rx = 0.0;
        darcy_system_0.Mult(TotalMobility(fine_S), fine_darcy_x, fine_darcy_Rx);

        mfem::Vector Rflux = fine_darcy_Rx.GetBlock(0);
        mfem::Vector Rp = fine_darcy_Rx.GetBlock(1);
        for (int i = 0; i < level_ ; ++i)
        {
            Rflux = MultTranspose(hierarchy_.GetPsigma(i), Rflux);
            Rp = MultTranspose(hierarchy_.GetPu(i), Rp);
        }

        out.GetBlock(0) = Rflux;
        out.GetBlock(1) = Rp;
    }
    else
    {
        mfem::BlockVector darcy_x(x.GetData(), darcy_system_.BlockOffsets());
        mfem::BlockVector darcy_Rx(out.GetData(), darcy_system_.BlockOffsets());
        darcy_system_.Mult(TotalMobility(S), darcy_x, darcy_Rx);
    }


    out.GetBlock(0) *= (1. / dt_ / density_);
    out.GetBlock(1) *= (dt_ * density_);

    auto& up_param = hierarchy_.GetUpscaleParameters();
    if (level_ == 0)// || (up_param.max_traces == 1 && (up_param.max_evects == 1 || up_param.add_Pvertices_pwc)))
    {
        const GraphSpace& space = darcy_system_.GetGraphSpace();
        auto upwind = BuildUpwindPattern(space, micro_upwind_flux_, blk_x.GetBlock(0));
//        auto upwind = BuildUpwindPattern(space, blk_x.GetBlock(0));

        auto upw_FS = MatVec(upwind, FractionalFlow(S));
        RescaleVector(blk_x.GetBlock(0), upw_FS);
        auto U_FS = MatVec(space.TrueEDofToEDof(), upw_FS);

        out.GetBlock(2) = MatVec(Ms_, blk_x.GetBlock(2));
        Ds_->Mult(1.0, U_FS, 1.0, out.GetBlock(2));
    }
    else
    {
//        mfem::Vector fine_flux = blk_x.GetBlock(0);
//        for (int i = level_-1; i >= 0 ; --i)
//        {
//            fine_flux = MatVec(hierarchy_.GetPsigma(i), fine_flux);
//        }
//        mfem::Vector fine_S = blk_x.GetBlock(2);
//        for (int i = level_-2; i >= 0 ; --i)
//        {
//            fine_S = MatVec(hierarchy_.GetPs(i), fine_S);
//        }

        auto fine_upwind = BuildWeightedUpwindPattern(darcy_system_0.GetGraphSpace(), fine_flux);

        auto upw_FS = MatVec(fine_upwind, FractionalFlow(fine_S));
        RescaleVector(fine_flux, upw_FS);

//        auto fine_D_upw_FS = MatVec(D_fine_, upw_FS);
//        auto D_upw_FS = MultTranspose(hierarchy_.GetPs(0), fine_D_upw_FS);

        auto D_upw_FS = MatVec(D_fine_, upw_FS);
        for (int i = 0; i < level_-1 ; ++i)
        {
            D_upw_FS = MultTranspose(hierarchy_.GetPs(i), D_upw_FS);
        }

        out.GetBlock(2) = MatVec(Ms_, blk_x.GetBlock(2));
        out.GetBlock(2) += D_upw_FS;
    }

    out -= y;
    SetZeroAtMarker(ess_dofs_, out.GetBlock(0));

    {
        out.GetBlock(0) *= scales_[0];
        out.GetBlock(1) *= scales_[1];
        out.GetBlock(2) *= scales_[2];
    }

    out[out.Size()-1] = 0.0;

    return out;
}

double CoupledSolver::ResidualNorm(const mfem::Vector& x, const mfem::Vector& y)
{
    return Norm(Residual(x, y));
}

double CoupledSolver::Norm(const mfem::Vector& vec)
{
    auto true_resid = AssembleTrueVector(vec);
    mfem::BlockVector blk_resid(true_resid.GetData(), true_blk_offsets_);

//    InvRescaleVector(normalizer_, blk_resid.GetBlock(1));
//    InvRescaleVector(normalizer_, blk_resid.GetBlock(2));
    blk_resid.GetBlock(1) /= (800.0 / density_);
    blk_resid.GetBlock(2) /= (800.0 / density_);

    return ParNorm(blk_resid, comm_);
}

void CoupledSolver::Build_dMdS(const MixedMatrix& darcy_system,
        const mfem::Vector& flux, const mfem::Vector& S)
{
    // TODO: saturation is only 1 dof per cell
    auto& vert_edof = darcy_system.GetGraphSpace().VertexToEDof();
    auto& vert_vdof = darcy_system.GetGraphSpace().VertexToVDof();

    auto& MB = dynamic_cast<const ElementMBuilder&>(darcy_system.GetMBuilder());
    auto& M_el = MB.GetElementMatrices();
    auto& proj_pwc = darcy_system.GetPWConstProj();

    mfem::Array<int> local_edofs, local_vdofs, vert(1);
    mfem::Vector sigma_loc, Msigma_vec;
    mfem::DenseMatrix proj_pwc_loc;

    const mfem::Vector dTMinv_dS_vec = dTMinv_dS(S);

    local_dMdS_.resize(vert_edof.NumRows());
    const int S_size = darcy_system.GetPWConstProj().NumCols();
    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        if (S_size == darcy_system.NumVDofs())
        {
            GetTableRow(vert_vdof, i, local_vdofs);
        }
        else
        {
            local_vdofs.SetSize(1, i);
        }
        vert[0] = i;

        flux.GetSubVector(local_edofs, sigma_loc);
        Msigma_vec.SetSize(local_edofs.Size());
        M_el[i].Mult(sigma_loc, Msigma_vec);
        mfem::DenseMatrix Msigma_loc(Msigma_vec.GetData(), M_el[i].Size(), 1);

        proj_pwc_loc.SetSize(1, local_vdofs.Size());
        proj_pwc_loc = 0.0;
        proj_pwc.GetSubMatrix(vert, local_vdofs, proj_pwc_loc);
        proj_pwc_loc *= dTMinv_dS_vec[i];

        local_dMdS_[i].SetSize(local_edofs.Size(), local_vdofs.Size());
        mfem::Mult(Msigma_loc, proj_pwc_loc, local_dMdS_[i]);
    }
}

mfem::SparseMatrix CoupledSolver::Assemble_dMdS(
        const MixedMatrix& darcy_system, const mfem::Vector& flux, const mfem::Vector& S)
{
    Build_dMdS(darcy_system, flux, S); // local_dMdS_ is constructed here

    auto& vert_edof = darcy_system.GetGraphSpace().VertexToEDof();
    auto& vert_vdof = darcy_system.GetGraphSpace().VertexToVDof();
    mfem::Array<int> local_edofs, local_vdofs;

    const int S_size = darcy_system.GetPWConstProj().NumCols();
    mfem::SparseMatrix out(darcy_system.NumEDofs(), S_size);
    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        if (S_size == darcy_system_.NumVDofs())
        {
            GetTableRow(vert_vdof, i, local_vdofs);
        }
        else
        {
            local_vdofs.SetSize(1, i);
        }
        out.AddSubMatrix(local_edofs, local_vdofs, local_dMdS_[i]);
    }
    out.Finalize();
    return out;
}

unique_ptr<mfem::SparseMatrix> BlockGetDiag(const mfem::BlockOperator& op_ref)
{
    auto& op = const_cast<mfem::BlockOperator&>(op_ref);
    mfem::BlockMatrix op_diag(op.RowOffsets(), op.ColOffsets());
    op_diag.owns_blocks = true;
    for (int i = 0; i < op.NumRowBlocks(); ++i)
    {
        for (int j = 0; j < op.NumColBlocks(); ++j)
        {
            if (op.IsZeroBlock(i, j)) { continue; }
            auto& block = static_cast<mfem::HypreParMatrix&>(op.GetBlock(i, j));
            auto block_diag = new mfem::SparseMatrix;
            block_diag->MakeRef(GetDiag(block));
            op_diag.SetBlock(i, j, block_diag);
        }
    }
    return unique_ptr<mfem::SparseMatrix>(op_diag.CreateMonolithic());
}

void CoupledSolver::Step(const mfem::Vector& rhs, mfem::Vector& x, mfem::Vector& dx)
{
    mfem::BlockVector blk_x(x.GetData(), blk_offsets_);
    const GraphSpace& space = darcy_system_.GetGraphSpace();

    if (level_ == 0)
    {
        LocalChopping(darcy_system_, blk_x.GetBlock(2));
    }

    mfem::Vector x_blk_2_coarse = level_ ? MultTranspose(hierarchy_.GetPs(level_-1), blk_x.GetBlock(2)) :
                                           blk_x.GetBlock(2);
    const mfem::Vector S = PWConstProject(darcy_system_, x_blk_2_coarse);
//    const mfem::Vector S = PWConstProject(darcy_system_, blk_x.GetBlock(2));

    mfem::Vector true_resid = AssembleTrueVector(Residual(x, rhs));
    true_resid *= -1.0;
    mfem::BlockVector true_blk_resid(true_resid.GetData(), blk_offsets_);

    mfem::BlockVector true_blk_dx(true_blk_offsets_);
    true_blk_dx = 0.0;

//    auto M_proc = darcy_system_.GetMBuilder().BuildAssembledM(TotalMobility(S));
//    auto dMdS_proc = Assemble_dMdS(blk_x.GetBlock(0), S);

    mfem::SparseMatrix M_proc, dMdS_proc;
    auto& system_0 = hierarchy_.GetMatrix(0);
    mfem::Vector fine_flux = blk_x.GetBlock(0);
    for (int i = level_-1; i >= 0 ; --i)
    {
        fine_flux = MatVec(hierarchy_.GetPsigma(i), fine_flux);
    }
    mfem::Vector fine_S = blk_x.GetBlock(2);
    for (int i = level_-2; i >= 0 ; --i)
    {
        fine_S = MatVec(hierarchy_.GetPs(i), fine_S);
    }

    // exact RAP
    if (level_ == 0 || !exact_flow_RAP_)
    {
        M_proc = darcy_system_.GetMBuilder().BuildAssembledM(TotalMobility(S));
        dMdS_proc = Assemble_dMdS(darcy_system_, blk_x.GetBlock(0), S);
    }
    else
    {
        unique_ptr<mfem::SparseMatrix> mat_help;
        auto M_fine = system_0.GetMBuilder().BuildAssembledM(TotalMobility(fine_S));
        mat_help.reset(new mfem::SparseMatrix(M_fine));
        for (int i = 0; i < level_; ++i)
        {
            mat_help.reset(mfem::RAP(hierarchy_.GetPsigma(i), *mat_help, hierarchy_.GetPsigma(i)));
        }
        M_proc = *mat_help;

        auto dMdS_fine = Assemble_dMdS(system_0, fine_flux, fine_S);
        auto PsigmaT = smoothg::Transpose(hierarchy_.GetPsigma(0));
        mat_help.reset( mfem::Mult(PsigmaT, dMdS_fine) );
        for (int i = 1; i < level_-1; ++i)
        {
            mat_help.reset(mfem::RAP(hierarchy_.GetPsigma(i), *mat_help, hierarchy_.GetPs(i-1)));
        }
        dMdS_proc = *mat_help;
    }


    for (int mm = 0; mm < ess_dofs_.Size(); ++mm)
    {
        if (ess_dofs_[mm])
        {
            M_proc.EliminateRowCol(mm, mfem::Matrix::DIAG_KEEP); // assume essential data = 0
            dMdS_proc.EliminateRow(mm);
        }
    }

    unique_ptr<mfem::HypreParMatrix> M(darcy_system_.MakeParallelM(M_proc));
    auto dMdS = ParMult(space.TrueEDofToEDof(), dMdS_proc, sdof_starts_);

    *M *= (1. / dt_ / density_);
    *dMdS *= (1. / dt_ / density_);

    unique_ptr<mfem::HypreParMatrix> dTdS;
    unique_ptr<mfem::SparseMatrix> dTdS_RAP;
    unique_ptr<mfem::HypreParMatrix> dTdsigma;
    mfem::Vector U_FS;

    auto& up_param = hierarchy_.GetUpscaleParameters();
//    bool pwc_sat = (up_param.max_evects == 1 || up_param.add_Pvertices_pwc);
    const bool lowest_coarse = (up_param.max_traces == 1 && up_param.max_evects == 1);
    const bool lowest_order = (level_ == 0);// || lowest_coarse);
    if (lowest_order)
    {
        auto upwind = BuildUpwindPattern(space, micro_upwind_flux_, blk_x.GetBlock(0));
//        auto upwind = BuildUpwindPattern(space, blk_x.GetBlock(0));

        U_FS = MatVec(space.TrueEDofToEDof(), MatVec(upwind, FractionalFlow(S)));

        upwind.ScaleRows(blk_x.GetBlock(0));
        upwind.ScaleColumns(dFdS(S));

        auto U = ParMult(space.TrueEDofToEDof(), upwind, sdof_starts_);
        auto U_pwc = ParMult(*U, darcy_system_.GetPWConstProj(), sdof_starts_);
        dTdS.reset(mfem::ParMult(Ds_.get(), U_pwc.get()));
    }
    else
    {
//        mfem::Vector fine_flux = blk_x.GetBlock(0);
//        for (int i = level_-1; i >= 0 ; --i)
//        {
//            fine_flux = MatVec(hierarchy_.GetPsigma(i), fine_flux);
//        }

//        mfem::Vector fine_S = blk_x.GetBlock(2);
//        for (int i = level_-2; i >= 0 ; --i)
//        {
//            fine_S = MatVec(hierarchy_.GetPs(i), fine_S);
//        }

//        auto& Ps = hierarchy_.GetPs(0);
//        auto fine_S = MatVec(Ps, blk_x.GetBlock(2));
        LocalChopping(hierarchy_.GetMatrix(0), fine_S);

        auto fine_upwind = BuildUpwindPattern(hierarchy_.GetMatrix(0).GetGraphSpace(), fine_flux);
//        auto fine_upwind = BuildWeightedUpwindPattern(hierarchy_.GetMatrix(0).GetGraphSpace(), fine_flux);

//        auto& Psigma = hierarchy_.GetPsigma(0);
        auto U_FS_fine = MatVec(fine_upwind, FractionalFlow(fine_S));

//        if (level_ > 1)
//        {
//            auto dTdsigma_fine = smoothg::Mult(D_fine_, SparseDiag(std::move(U_FS_fine)));
//            dTdS_RAP.reset(mfem::RAP(hierarchy_.GetPs(0), dTdsigma_fine,
//                                     hierarchy_.GetPsigma(0)));
//        }
//        else
        {
            auto dTdsigma_fine = smoothg::Mult(D_fine_, SparseDiag(std::move(U_FS_fine)));
            dTdS_RAP.reset(mfem::Mult(dTdsigma_fine, hierarchy_.GetPsigma(0)));
        }

        for (int i = 1; i < level_-1; ++i)
        {
            dTdS_RAP.reset(mfem::RAP(hierarchy_.GetPs(i-1), *dTdS_RAP, hierarchy_.GetPsigma(i)));
        }

//        dTdS_RAP.reset(mfem::RAP(Ps, dTdsigma_fine, Psigma));
        dTdsigma.reset(ToParMatrix(comm_, *dTdS_RAP));

        fine_upwind.ScaleRows(fine_flux);
        fine_upwind.ScaleColumns(dFdS(fine_S));

//        if (level_ > 1)
//        {
//            auto fine_dTdS = smoothg::Mult(D_fine_, fine_upwind);
//            dTdS_RAP.reset(mfem::RAP(hierarchy_.GetPs(0), fine_dTdS, hierarchy_.GetPs(0)));
//        }
//        else
        {
            dTdS_RAP.reset( mfem::Mult(D_fine_, fine_upwind) );
        }

        for (int i = 0; i < level_-1; ++i)
        {
            dTdS_RAP.reset(mfem::RAP(hierarchy_.GetPs(i), *dTdS_RAP, hierarchy_.GetPs(i)));
        }

//        dTdS_RAP.reset(mfem::RAP(Ps, fine_dTdS, Ps));

        if (dTdS)
        {
            auto dTdS_diag = GetDiag(*dTdS);

            double dtds_diff = OperatorsRelDiff(dTdS_diag, *dTdS_RAP);

            unique_ptr<mfem::SparseMatrix> diff_diag(mfem::Add(1.0, dTdS_diag, -1.0, *dTdS_RAP));
            *diff_diag *= (1.0 / FroNorm(*diff_diag));

            if (dtds_diff > 1e-13)
            {
                for (int ii = 0; ii < dTdS_diag.NumRows(); ++ii)
                {
                    for (int jj = 0; jj < dTdS_diag.RowSize(ii); ++jj)
                    {
                        if (diff_diag->GetRowEntries(ii)[jj] > 1e-2 )
                        {
                            const int col = dTdS_diag.GetRowColumns(ii)[jj];
                            std::cout<< "Row: "<<ii<<", Column: "<< col
                                        << ", Entry:" <<dTdS_diag(ii, col)
                            << ", Entry (RAP):" <<dTdS_RAP->Elem(ii, col)<<"\n";
                        }
                    }
                }
                std::cout<<"|| upwind - upwind_RAP ||_F = "<< dtds_diff <<"\n";
            }
//            assert(dtds_diff < 1e-13);
        }
        dTdS.reset(ToParMatrix(comm_, *dTdS_RAP));
    }

//    GetDiag(*dTdS) += Ms_;
    dTdS_RAP.reset(mfem::Add(GetDiag(*dTdS), Ms_));
    dTdS.reset(ToParMatrix(comm_, *dTdS_RAP));


    const bool use_hybrid_solver = (up_param.hybridization && lowest_order && level_);
    if (!use_hybrid_solver)
    {
        if (lowest_order)
        {
            dTdsigma = ParMult(*Ds_, SparseDiag(std::move(U_FS)), true_edof_starts_);
        }

        {
            mfem::SparseMatrix diag;
            dTdsigma->GetDiag(diag);
            diag.EliminateRow(diag.NumRows()-1);
            dTdS->GetDiag(diag);
            diag.EliminateRowCol(diag.NumRows()-1, mfem::Matrix::DIAG_KEEP);

            dMdS->GetDiag(diag);
            diag.EliminateCol(diag.NumCols()-1);
        }


        *M *= scales_[0];
        *dMdS *= scales_[0];
        *dTdsigma *= scales_[2];
        *dTdS *= scales_[2];

        mfem::BlockOperator op(true_blk_offsets_);
        op.SetBlock(0, 0, M.get());
        op.SetBlock(0, 1, DT_.get());
        op.SetBlock(1, 0, D_.get());
        op.SetBlock(0, 2, dMdS.get());
        op.SetBlock(2, 0, dTdsigma.get());
        op.SetBlock(2, 2, dTdS.get());

//        auto fix = SparseIdentity(D_->NumRows());
//        fix = 0.0;
//        fix(0,0) = 1.0;
//        auto pfix = ToParMatrix(comm_, std::move(fix));
//        op.SetBlock(1, 1, pfix);

        const bool use_direct_solver = !lowest_order;
        const bool solve_on_primal_form = (level_ == 0);

        if (solve_on_primal_form == false)
        {
            if (use_direct_solver)
            {
                auto mono_op = BlockGetDiag(op);
//                mono_op->EliminateRowCol(true_blk_dx.BlockSize(0));
//                true_resid[true_blk_dx.BlockSize(0)] = 0.0;
                mfem::UMFPackSolver direct_solve(*mono_op);
                direct_solve.Mult(true_resid, true_blk_dx);
                if (!myid_ && param_.print_level > 1)
                {
                    std::cout << "    Direct solver used\n";
                }
            }
            else { MixedSolve(op, true_blk_resid, true_blk_dx); }
        }
        else { PrimalSolve(op, true_blk_resid, true_blk_dx); }

        linear_iter_ += gmres_.GetNumIterations();
        if (!myid_ && param_.print_level > 1 && (solve_on_primal_form || !use_direct_solver))
        {
            std::cout << "    Level " << level_;
            std::string name = solve_on_primal_form? "Primal" : "Mixed";
            std::cout << "    " << name << " solver: GMRES took " << gmres_.GetNumIterations()
                      << " iterations, residual = " << gmres_.GetFinalNorm() << "\n";
        }
    }
    else
    {
        *dTdS *= (1. / dt_ / density_);
        true_blk_resid.GetBlock(0) *= (dt_ * density_);
        true_blk_resid.GetBlock(1) /= (dt_ * density_);
        true_blk_resid.GetBlock(2) /= (dt_ * density_);
        HybridSolve(*dTdS, U_FS, blk_x.GetBlock(0), S, true_blk_resid, true_blk_dx);
    }

    mfem::BlockVector blk_dx(dx.GetData(), blk_offsets_);
    blk_dx = 0.0;
    auto& dof_truedof = darcy_system_.GetGraphSpace().EDofToTrueEDof();
    dof_truedof.Mult(true_blk_dx.GetBlock(0), blk_dx.GetBlock(0));
    blk_dx.GetBlock(1) = true_blk_dx.GetBlock(1);
    blk_dx.GetBlock(2) = true_blk_dx.GetBlock(2);

    const MixedMatrix& darcy_system_s = level_ ? hierarchy_.GetMatrix(level_-1) : darcy_system_;
    const mfem::Vector dS = PWConstProject(darcy_system_s, blk_dx.GetBlock(2));
    blk_dx *= std::min(1.0, param_.diff_tol / mfem::ParNormlp(dS, mfem::infinity(), comm_));

    x += blk_dx;

    if (level_ == 0)
    {
        LocalChopping(darcy_system_, blk_x.GetBlock(2));
    }

    sol_previous_iter_ = x;
}

mfem::Array2D<mfem::HypreParMatrix*> To2DArray(mfem::BlockOperator& op)
{
    const int num_blks = op.NumRowBlocks();
    mfem::Array2D<mfem::HypreParMatrix*> out(num_blks, num_blks);
    for (int i = 0; i < num_blks; ++i)
    {
        for (int j = 0; j < num_blks; ++j)
        {
            if (op.IsZeroBlock(i, j)) { out(i, j) = nullptr; continue; }
            out(i, j) = static_cast<mfem::HypreParMatrix*>(&op.GetBlock(i, j));
        }
    }
    return out;
}

mfem::Array2D<mfem::HypreParMatrix*>
ApproximateSchurComplement(mfem::Array2D<mfem::HypreParMatrix*>& op)
{
    mfem::Vector op_00_diag;
    op(0, 0)->GetDiag(op_00_diag);
    op_00_diag *= -1.0;

    mfem::Array2D<mfem::HypreParMatrix*> out(op.NumRows(), op.NumCols());
    for (int j = 1; j < op.NumCols(); ++j)
    {
        op(0, j)->InvScaleRows(op_00_diag);
        for (int i = 1; i < op.NumRows(); ++i)
        {
            unique_ptr<mfem::HypreParMatrix> tmp(mfem::ParMult(op(i, 0), op(0, j)));
            if (op(i, j) == nullptr) { out(i, j) = tmp.release(); continue; }
            out(i, j) = ParAdd(*op(i, j), *tmp);
        }
        op(0, j)->ScaleRows(op_00_diag);
    }
    return out;
}

void CoupledSolver::MixedSolve(const mfem::BlockOperator& op,
                               const mfem::BlockVector& true_resid,
                               mfem::BlockVector& true_dx)
{
    auto op_array = To2DArray(const_cast<mfem::BlockOperator&>(op));
    auto schur = ApproximateSchurComplement(op_array);

    auto smooth_t = mfem::HypreSmoother::Type::l1Jacobi;
    mfem::BlockLowerTriangularPreconditioner prec(true_blk_offsets_);
    prec.SetDiagonalBlock(0, new mfem::HypreDiagScale(*op_array(0, 0)));
    prec.SetDiagonalBlock(1, BoomerAMG(*schur(1, 1)));
    prec.SetDiagonalBlock(2, new mfem::HypreSmoother(*schur(2, 2), smooth_t));
    prec.SetBlock(1, 0, op_array(1, 0));
    prec.SetBlock(2, 0, op_array(2, 0));
    prec.SetBlock(2, 1, schur(2,1));

    unique_ptr<mfem::SparseMatrix> mono_op = BlockGetDiag(op);
    auto par_mono_op = ToParMatrix(comm_, std::move(*mono_op));
    HypreILU ILU_smoother(*par_mono_op, 0); // equiv to Euclid
    TwoStageSolver prec_prod(prec, ILU_smoother, op);

    gmres_.SetOperator(op);
    gmres_.SetPreconditioner(prec_prod);
    gmres_.Mult(true_resid, true_dx);
}

void CoupledSolver::HybridSolve(const mfem::HypreParMatrix& dTdS, const mfem::Vector& U_FS,
                                const mfem::Vector &flux, const mfem::Vector& S,
                                const mfem::BlockVector& true_resid, mfem::BlockVector& true_dx)
{
    mfem::Array<int> offset_hb(3);
    offset_hb[0] = 0;
    offset_hb[1] = darcy_system_.NumEDofs();
    offset_hb[2] = offset_hb[1] + darcy_system_.GetPWConstProj().NumCols();

    auto& space = darcy_system_.GetGraphSpace();
    auto& D = darcy_system_.GetD();
//    auto local_dTdsigma = Build_dTdsigma(space, D, flux, FractionalFlow(S));
//    auto local_dTdsigma = Build_dTdsigma(space, micro_upwind_flux_, D, flux, FractionalFlow(S));
    auto local_dTdsigma = Build_dTdsigma(space, D, U_FS);

    TwoPhaseHybrid solver(darcy_system_, &ess_attr_);
    solver.AssembleSolver(TotalMobility(S), local_dMdS_, local_dTdsigma, dTdS);

    solver.Mult(true_resid, true_dx);
    linear_iter_ += solver.GetNumIterations();
    if (!myid_ && param_.print_level > 1)
    {
        std::cout << "    Level " << level_;
        std::cout << "    Hybrid solver: GMRES took " << solver.GetNumIterations()
                  << " iterations, residual = " << solver.GetResidualNorm() << "\n";
    }
}

void CoupledSolver::PrimalSolve(const mfem::BlockOperator& op,
                                const mfem::BlockVector& true_resid,
                                mfem::BlockVector& true_dx)
{
    auto op_array = To2DArray(const_cast<mfem::BlockOperator&>(op));
    auto schur = ApproximateSchurComplement(op_array);

    mfem::Vector op_00_diag;
    op_array(0, 0)->GetDiag(op_00_diag);

    mfem::Array<int> primal_offsets(3);
    primal_offsets[0] = 0;
    primal_offsets[1] = true_blk_offsets_[2] - true_blk_offsets_[1];
    primal_offsets[2] = true_blk_offsets_[3] - true_blk_offsets_[1];

    mfem::BlockOperator schur_op(primal_offsets);
    schur_op.SetBlock(0, 0, schur(1, 1));
    schur_op.SetBlock(0, 1, schur(1, 2));
    schur_op.SetBlock(1, 0, schur(2, 1));
    schur_op.SetBlock(1, 1, schur(2, 2));

    auto smooth_t = mfem::HypreSmoother::Type::l1Jacobi;
    mfem::BlockLowerTriangularPreconditioner prec(primal_offsets);
    prec.SetDiagonalBlock(0, BoomerAMG(*schur(1, 1)));
    prec.SetDiagonalBlock(1, new mfem::HypreSmoother(*schur(2, 2), smooth_t));
    prec.SetBlock(1, 0, schur(2, 1));

    auto mono_schur = BlockGetDiag(schur_op);
    auto par_mono_schur = ToParMatrix(comm_, std::move(*mono_schur));
    HypreILU ILU_smoother(*par_mono_schur, 0);
    TwoStageSolver prec_prod(prec, ILU_smoother, schur_op);
    gmres_.SetOperator(schur_op);
    gmres_.SetPreconditioner(prec_prod);

    mfem::BlockVector primal_resid(primal_offsets);
    primal_resid.GetBlock(0) = true_resid.GetBlock(1);
    primal_resid.GetBlock(1) = true_resid.GetBlock(2);

    InvRescaleVector(op_00_diag, const_cast<mfem::Vector&>(true_resid.GetBlock(0)));
    op_array(1, 0)->Mult(-1.0, true_resid.GetBlock(0), 1.0, primal_resid.GetBlock(0));
    op_array(2, 0)->Mult(-1.0, true_resid.GetBlock(0), 1.0, primal_resid.GetBlock(1));
    RescaleVector(op_00_diag, const_cast<mfem::Vector&>(true_resid.GetBlock(0)));

    mfem::BlockVector primal_dx(true_dx.GetBlock(1), primal_offsets);
    primal_dx = 0.0;
    gmres_.Mult(primal_resid, primal_dx);

    true_dx.GetBlock(0) = true_resid.GetBlock(0);
    op_array(0, 1)->Mult(-1.0, true_dx.GetBlock(1), 1.0, true_dx.GetBlock(0));
    op_array(0, 2)->Mult(-1.0, true_dx.GetBlock(2), 1.0, true_dx.GetBlock(0));
    InvRescaleVector(op_00_diag, true_dx.GetBlock(0));
}

void CoupledSolver::BackTracking(const mfem::Vector& rhs,  double prev_resid_norm,
                                 mfem::Vector& x, mfem::Vector& dx)
{
    if (param_.num_backtrack == 0) { return; }

    mfem::BlockVector blk_x(x, true_blk_offsets_);
    mfem::BlockVector blk_dx(dx, true_blk_offsets_);
    LocalChopping(darcy_system_, blk_x.GetBlock(2));
}

CoupledFAS::CoupledFAS(const Hierarchy& hierarchy,
                       const mfem::Array<int>& ess_attr,
                       const double dt,
                       const mfem::SparseMatrix& weight,
                       const double density,
                       const mfem::Vector& S_prev,
                       FASParameters param)
    : FAS(hierarchy.GetComm(), param), hierarchy_(hierarchy)
{
    mfem::Vector S_prev_l(S_prev);
    unique_ptr<mfem::SparseMatrix> weight_l(new mfem::SparseMatrix(weight));
    Qs_.reserve(hierarchy.NumLevels()-1);

    for (int l = 0; l < param_.num_levels; ++l)
    {
        if (l > 0)
        {
            S_prev_l = smoothg::MultTranspose(hierarchy.GetPs(l-1), S_prev_l);
            auto PsT = smoothg::Transpose(hierarchy.GetPs(l-1));
            Qs_.push_back(smoothg::Mult(PsT, *weight_l));
            weight_l.reset(mfem::RAP(hierarchy.GetPs(l-1), *weight_l, hierarchy.GetPs(l-1)));

            mfem::Vector Wc_inv;
            weight_l->GetDiag(Wc_inv);
            for (int i = 0; i < Wc_inv.Size(); ++i)
            {
                Wc_inv[i] = 1.0 / Wc_inv[i];
            }
            Qs_[l-1].ScaleRows(Wc_inv);
        }

        auto& system_l = hierarchy.GetMatrix(l);

        const mfem::Vector S = PWConstProject(system_l, S_prev_l);

        auto& upwind_flux_l = hierarchy.GetUpwindFlux(l);
        auto& param_l = l ? (l < param.num_levels - 1 ? param.mid : param.coarse) : param.fine;
        solvers_[l].reset(new CoupledSolver(hierarchy, l, system_l, ess_attr,
                                            upwind_flux_l, dt, *weight_l, density, S, param_l));
//        solvers_[l]->SetPrintLevel(param_.cycle == V_CYCLE ? -1 : 1);

        const int S_size = l ? hierarchy.GetMatrix(l-1).GetPWConstProj().NumCols()
                             : system_l.GetPWConstProj().NumCols();
        if (l > 0)
        {
            rhs_[l].SetSize(system_l.NumTotalDofs() + S_size);
            sol_[l].SetSize(system_l.NumTotalDofs() + S_size);
            rhs_[l] = 0.0;
            sol_[l] = 0.0;
        }
        help_[l].SetSize(system_l.NumTotalDofs() + S_size);
        help_[l] = 0.0;
    }

    step_local = 0;
}

double CoupledFAS::Norm(int level, const mfem::Vector& vec) const
{
    return static_cast<CoupledSolver&>(*solvers_[level]).Norm(vec);
}

void CoupledFAS::Restrict(int level, const mfem::Vector& fine, mfem::Vector& coarse) const
{
    auto& solver_f = static_cast<CoupledSolver&>(*solvers_[level]);
    auto& solver_c = static_cast<CoupledSolver&>(*solvers_[level + 1]);
    mfem::BlockVector blk_fine(fine.GetData(), solver_f.BlockOffsets());
    mfem::BlockVector blk_coarse(coarse.GetData(), solver_c.BlockOffsets());
    hierarchy_.Restrict(level, blk_fine, blk_coarse);

    if (level == 0)
    {
        blk_coarse.GetBlock(2) = blk_fine.GetBlock(2);
    }
    else
    {
        hierarchy_.GetPs(level - 1).MultTranspose(blk_fine.GetBlock(2), blk_coarse.GetBlock(2));
    }
//    auto S = hierarchy_.PWConstProject(level + 1, blk_coarse.GetBlock(2));
//    blk_coarse.GetBlock(2) = hierarchy_.PWConstInterpolate(level + 1, S);
}

void CoupledFAS::Interpolate(int level, const mfem::Vector& coarse, mfem::Vector& fine) const
{
    auto& solver_f = static_cast<CoupledSolver&>(*solvers_[level - 1]);
    auto& solver_c = static_cast<CoupledSolver&>(*solvers_[level]);
    mfem::BlockVector blk_fine(fine.GetData(), solver_f.BlockOffsets());
    mfem::BlockVector blk_coarse(coarse.GetData(), solver_c.BlockOffsets());
    hierarchy_.Interpolate(level, blk_coarse, blk_fine);

//    auto S = hierarchy_.PWConstProject(level, blk_coarse.GetBlock(2));
//    blk_coarse.GetBlock(2) = hierarchy_.PWConstInterpolate(level, S);
    if (level == 1)
    {
        blk_fine.GetBlock(2) = blk_coarse.GetBlock(2);
    }
    else
    {
        hierarchy_.GetPs(level - 2).Mult(blk_coarse.GetBlock(2), blk_fine.GetBlock(2));
    }
}

//mfem::Vector CoupledFAS::ProjectS(int level, const mfem::Vector& x) const
//{
//    const auto& darcy_system = hierarchy_.GetMatrix(level);
//    const auto& agg_vert = hierarchy_.GetAggVert(level);
//    const mfem::Vector S = darcy_system.PWConstProjectS(x);

//    mfem::Vector S_loc, S_coarse(agg_vert.NumRows());
//    mfem::Array<int> verts;
//    for (int i = 0; i < agg_vert.NumRows(); ++i)
//    {
//        GetTableRow(agg_vert, i, verts);
//        S.GetSubVector(verts, S_loc);
//        S_coarse[i] = S_loc.Max();
//    }

//    return hierarchy_.GetMatrix(level + 1).PWConstInterpolate(S_coarse);
//}

//mfem::Vector CoupledFAS::ProjectS(int level, const mfem::Vector& x) const
//{

//}

void CoupledFAS::Project(int level, const mfem::Vector& fine, mfem::Vector& coarse) const
{
    auto& solver_f = static_cast<CoupledSolver&>(*solvers_[level]);
    auto& solver_c = static_cast<CoupledSolver&>(*solvers_[level + 1]);
    mfem::BlockVector blk_fine(fine.GetData(), solver_f.BlockOffsets());
    mfem::BlockVector blk_coarse(coarse.GetData(), solver_c.BlockOffsets());
    hierarchy_.Project(level, blk_fine, blk_coarse);


    if (level == 0)
    {
        blk_coarse.GetBlock(2) = blk_fine.GetBlock(2);
    }
    else
    {
        hierarchy_.GetPs(level - 1).MultTranspose(blk_fine.GetBlock(2), blk_coarse.GetBlock(2));
    }

//    auto S = hierarchy_.PWConstProject(level + 1, blk_coarse.GetBlock(2));
//    blk_coarse.GetBlock(2) = hierarchy_.PWConstInterpolate(level + 1, S);

//    blk_coarse.GetBlock(2) = ProjectS(level, blk_fine.GetBlock(2));

//    Qs_[level].Mult(blk_fine.GetBlock(2), blk_coarse.GetBlock(2));
}

mfem::Vector TransportSolver::Residual(const mfem::Vector& x, const mfem::Vector& y)
{
    mfem::Vector out(x);
    mfem::Vector S = PWConstProject(darcy_system_, x);

    auto FS = FractionalFlow(S);
    Adv_.Mult(1.0, FS, Ms_(0, 0), out);
    out -= y;
    return out;
}

void TransportSolver::Step(const mfem::Vector& rhs, mfem::Vector& x, mfem::Vector& dx)
{
    mfem::SparseMatrix df_ds = darcy_system_.GetPWConstProj();
    df_ds.ScaleRows(dFdS(PWConstProject(darcy_system_, x)));

    auto A = ParMult(Adv_, df_ds, starts_);
    GetDiag(*A) += Ms_;

    unique_ptr<mfem::HypreBoomerAMG> solver(BoomerAMG(*A));
    gmres_.SetPreconditioner(*solver);
    gmres_.SetOperator(*A);

    dx = 0.0;
    auto resid = Residual(x, rhs);
    gmres_.Mult(resid, dx);
//    if (!myid_) std::cout << "GMRES took " << gmres_.GetNumIterations() << " iterations\n";

    const mfem::Vector dS = PWConstProject(darcy_system_, dx);
    dx *= std::min(1.0, param_.diff_tol / mfem::ParNormlp(dS, mfem::infinity(), comm_));
    x -= dx;
}

void TwoPhaseHybrid::Init()
{
    offsets_[0] = 0;
    offsets_[1] = multiplier_d_td_->NumCols();
    offsets_[2] = offsets_[1] + mgL_.NumVDofs();

    op_.reset(new mfem::BlockOperator(offsets_));
    op_->owns_blocks = true;

    stage1_prec_.reset(new mfem::BlockLowerTriangularPreconditioner(offsets_));

    for (int agg = 0; agg < nAggs_; ++agg)
    {
        mfem::DenseMatrix AinvDMinv = smoothg::Mult(Ainv_[agg], DMinv_[agg]);
        B01_[agg].Transpose(AinvDMinv);
        B00_[agg] = smoothg::Mult(B01_[agg], DMinv_[agg]);
        B00_[agg] -= Minv_ref_[agg];
        B00_[agg] *= -1.0;
    }

    solver_ = InitKrylovSolver(GMRES);
    solver_->SetAbsTol(1e-10);
    solver_->SetRelTol(1e-8);
}

void TwoPhaseHybrid::AssembleSolver(mfem::Vector elem_scaling_inverse,
                                    const std::vector<mfem::DenseMatrix>& dMdS,
                                    const std::vector<mfem::DenseMatrix>& dTdsigma,
                                    const mfem::HypreParMatrix& dTdS,
                                    double dt_density)
{
    dt_density_ = dt_density;

    const auto& agg_vdof = mgL_.GetGraphSpace().VertexToVDof();

    mfem::SparseMatrix A00(num_multiplier_dofs_);
    mfem::SparseMatrix A01(num_multiplier_dofs_, mgL_.NumVDofs());
    mfem::SparseMatrix A10(mgL_.NumVDofs(), num_multiplier_dofs_);
    mfem::SparseMatrix A11_tmp(mgL_.NumVDofs());

    mfem::DenseMatrix A00_el, A01_el, A10_el, A11_el, help;
    mfem::Array<int> local_vdof, local_mult;

    for (int agg = 0; agg < nAggs_; ++agg)
    {
        elem_scaling_[agg] = 1.0 / elem_scaling_inverse[agg];

        GetTableRow(agg_vdof, agg, local_vdof);
        GetTableRow(Agg_multiplier_, agg, local_mult);

        A00_el = Hybrid_el_[agg];
        A00_el *= elem_scaling_inverse[agg];
//        A00_el *= dt_density_;

        help = smoothg::Mult(C_[agg], B00_[agg]);
        help *= elem_scaling_inverse[agg];
        A01_el = smoothg::Mult(help, dMdS[agg]);

        help.Transpose();
        A10_el = smoothg::Mult(dTdsigma[agg], help);
//        A10_el *= (dt_density_ * dt_density_);

        help = smoothg::Mult(dTdsigma[agg], B00_[agg]);
        A11_el = smoothg::Mult(help, dMdS[agg]);
        A11_el *= elem_scaling_inverse[agg];
//        A11_el *= (dt_density_);

        A00.AddSubMatrix(local_mult, local_mult, A00_el);
        A01.AddSubMatrix(local_mult, local_vdof, A01_el);
        A10.AddSubMatrix(local_vdof, local_mult, A10_el);
        A11_tmp.AddSubMatrix(local_vdof, local_vdof, A11_el);
    }

    A00.Finalize();
    A01.Finalize();
    A10.Finalize();
    A11_tmp.Finalize();

    auto dTdS_diag = GetDiag(dTdS);
    unique_ptr<mfem::SparseMatrix> A11(Add(1.0, A11_tmp, -1.0, dTdS_diag));
    A11->MoveDiagonalFirst();
    unique_ptr<mfem::HypreParMatrix> pA11(ToParMatrix(comm_, *A11));


    BuildParallelSystemAndSolver(A00); // pA00 and A00_inv store in H_ and prec_

    auto Scale = VectorToMatrix(diagonal_scaling_);
    mfem::HypreParMatrix pScale(comm_, H_->N(), H_->GetColStarts(), &Scale);

    for (auto mult : ess_true_multipliers_)
    {
        A01.EliminateRow(mult);
        A10.EliminateCol(mult);
    }

    auto pA01_tmp = ParMult(*multiplier_td_d_, A01, mgL_.GetGraph().VertexStarts());
    auto pA10_tmp = ParMult(A10, *multiplier_d_td_, mgL_.GetGraph().VertexStarts());

    auto pA01 = mfem::ParMult(&pScale, pA01_tmp.get());
    auto pA10 = mfem::ParMult(pA10_tmp.get(), &pScale);

    A11_inv_.reset(new mfem::HypreSmoother(*pA11, mfem::HypreSmoother::l1Jacobi));
    A00_inv_.reset(prec_.release());
    stage1_prec_->SetDiagonalBlock(0, A00_inv_.get());
    stage1_prec_->SetDiagonalBlock(1, A11_inv_.get());
    stage1_prec_->SetBlock(1, 0, pA10);

    op_->SetBlock(0, 0, H_.release());
    op_->SetBlock(0, 1, pA01);
    op_->SetBlock(1, 0, pA10);
    op_->SetBlock(1, 1, pA11.release());

    mono_mat_ = BlockGetDiag(*op_);
    monolithic_.reset(ToParMatrix(comm_, *mono_mat_));
    stage2_prec_.reset(new HypreILU(*monolithic_, 0));

//    prec_ = std::move(stage1_prec_);//
    prec_.reset(new TwoStageSolver(*stage1_prec_, *stage2_prec_, *op_));

    solver_->SetPreconditioner(*prec_);
    solver_->SetOperator(*op_);
    dynamic_cast<mfem::GMRESSolver*>(solver_.get())->SetKDim(100);

    dTdsigma_ = &dTdsigma;
    dMdS_ = &dMdS;
}

mfem::BlockVector TwoPhaseHybrid::MakeHybridRHS(const mfem::BlockVector& rhs) const
{
    const auto& agg_vdof = mgL_.GetGraphSpace().VertexToVDof();
    const auto& agg_edof = mgL_.GetGraphSpace().VertexToEDof();

    mfem::BlockVector out(offsets_);
    out.GetBlock(0) = 0.0;
    out.GetBlock(1).Set(-1.0, rhs.GetBlock(2));

    mfem::Array<int> local_vdof, local_edof, local_mult;
    mfem::Vector local_rhs, sub_vec, helper; // helper = B00 * rhs0 + B01 * rhs1
    for (int agg = 0; agg < nAggs_; ++agg)
    {
        GetTableRow(agg_vdof, agg, local_vdof);
        GetTableRow(agg_edof, agg, local_edof);
        GetTableRow(Agg_multiplier_, agg, local_mult);

        rhs.GetSubVector(local_edof, sub_vec);
        for (int i = 0; i < local_edof.Size(); ++i)
        {
            if (edof_needs_averaging_[local_edof[i]])
            {
                sub_vec[i] /= 2.0;
            }
        }

        helper.SetSize(local_edof.Size());
        B00_[agg].Mult(sub_vec, helper);
        helper /= elem_scaling_[agg];
//        helper *= dt_density_;

        rhs.GetBlock(1).GetSubVector(local_vdof, sub_vec);
        B01_[agg].AddMult_a(1.0 / dt_density_, sub_vec, helper);

        local_rhs.SetSize(local_mult.Size());
        C_[agg].Mult(helper, local_rhs);
        out.AddElementVector(local_mult, local_rhs);

        local_rhs.SetSize(local_vdof.Size());
        (*dTdsigma_)[agg].Mult(helper, local_rhs);
//        local_rhs *= dt_density_;
        out.GetBlock(1).AddElementVector(local_vdof, local_rhs);
    }

    return out;
}

void TwoPhaseHybrid::BackSubstitute(const mfem::BlockVector& rhs,
                                    const mfem::BlockVector& sol_hb,
                                    mfem::BlockVector& sol) const
{
    const auto& agg_vdof = mgL_.GetGraphSpace().VertexToVDof();
    const auto& agg_edof = mgL_.GetGraphSpace().VertexToEDof();

    sol.GetBlock(0) = 0.0;
    sol.GetBlock(1) = 0.0;
    sol.GetBlock(2) = sol_hb.GetBlock(1);

    mfem::Array<int> local_vdof, local_edof, local_mult;
    mfem::Vector local_sol0, local_sol1, sub_vec, helper;
    for (int agg = 0; agg < nAggs_; ++agg)
    {
        GetTableRow(agg_vdof, agg, local_vdof);
        GetTableRow(agg_edof, agg, local_edof);
        GetTableRow(Agg_multiplier_, agg, local_mult);

        local_sol0.SetSize(local_edof.Size());
        local_sol1.SetSize(local_vdof.Size());

        rhs.GetBlock(1).GetSubVector(local_vdof, sub_vec);
        B01_[agg].Mult(sub_vec, local_sol0);
        Ainv_[agg].Mult(sub_vec, local_sol1);
        local_sol1 *= elem_scaling_[agg];

//        local_sol0 /= dt_density_;
//        local_sol1 /= (-1.0 * dt_density_);

        rhs.GetSubVector(local_edof, helper);
        for (int i = 0; i < local_edof.Size(); ++i)
        {
            if (edof_needs_averaging_[local_edof[i]])
            {
                helper[i] /= 2.0;
            }
        }

        sol_hb.GetBlock(1).GetSubVector(local_vdof, sub_vec);
        (*dMdS_)[agg].AddMult_a(-1.0 / dt_density_, sub_vec, helper);

        sol_hb.GetSubVector(local_mult, sub_vec);
        C_[agg].AddMultTranspose(sub_vec, helper, -1.0);

        B00_[agg].AddMult_a(dt_density_* 1.0 / elem_scaling_[agg], helper, local_sol0);
        B01_[agg].AddMultTranspose_a(dt_density_, helper, local_sol1);

        for (int i = 0; i < local_edof.Size(); ++i)
        {
            if (edof_needs_averaging_[local_edof[i]])
            {
                local_sol0[i] /= 2.0;
            }
        }

        sol.AddElementVector(local_edof, local_sol0);
        sol.GetBlock(1).AddElementVector(local_vdof, local_sol1);
    }
}

void TwoPhaseHybrid::Mult(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const
{
    mfem::BlockVector sol_hb(offsets_);
    sol_hb = 0.0;
    const mfem::BlockVector rhs_hb = MakeHybridRHS(rhs);
    solver_->Mult(rhs_hb, sol_hb);
    BackSubstitute(rhs, sol_hb, sol);
    num_iterations_ = solver_->GetNumIterations();
    resid_norm_ = solver_->GetFinalNorm();
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
