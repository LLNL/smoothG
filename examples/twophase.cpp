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
    TwoStageSolver(const mfem::Operator & solver1, const mfem::Operator& solver2, const mfem::Operator& op) :
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
                        double dt_density_);

    void Mult(const mfem::BlockVector& rhs, mfem::BlockVector& sol) const override;
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
    const TwoPhase& problem_;
    Hierarchy& hierarchy_;
    std::vector<mfem::BlockVector> blk_helper_;

    mfem::Array<int> blk_offsets_;
    unique_ptr<mfem::BlockVector> source_;
    unique_ptr<mfem::HypreParMatrix> D_te_e_;
    int nonlinear_iter_;
    int linear_iter_;
    bool step_converged_;

    std::vector<mfem::Vector> micro_upwind_flux_;

    // TODO: these should be defined in / extracted from the problem, not here
    const double density_ = 1e3;
    const double porosity_ = 0.3;
    const double weight_;
public:
    TwoPhaseSolver(const TwoPhase& problem, Hierarchy& hierarchy,
                   const int level, const EvolveParamenters& evolve_param,
                   const FASParameters& solver_param);

    void TimeStepping(const double dt, mfem::BlockVector& x);
    mfem::BlockVector Solve(const mfem::BlockVector& init_val);
};

class CoupledSolver : public NonlinearSolver
{
    const MixedMatrix& darcy_system_;
    mfem::GMRESSolver gmres_;
    unique_ptr<mfem::HypreParMatrix> D_;
    unique_ptr<mfem::HypreParMatrix> DT_;
    std::vector<mfem::DenseMatrix> local_dMdS_;
    mfem::SparseMatrix Ms_;

    mfem::Array<int> blk_offsets_;
    mfem::Array<int> true_blk_offsets_;
    const mfem::Array<int>& ess_dofs_;
    const mfem::Array<int>& vert_starts_;
    mfem::Array<int> true_edof_starts_;
//    const std::vector<mfem::DenseMatrix>& traces_;

    const double dt_;
    const double weight_;
    const double density_;

    mfem::Vector normalizer_;
    bool is_first_resid_eval_;
    mfem::Vector scales_;

    void Step(const mfem::Vector& rhs, mfem::Vector& x, mfem::Vector& dx) override;
    void Build_dMdS(const mfem::Vector& flux, const mfem::Vector& S);
    mfem::SparseMatrix Assemble_dMdS(const mfem::Vector& flux, const mfem::Vector& S);
    mfem::Vector AssembleTrueVector(const mfem::Vector& vec) const;


    void BuildHybridSystem(mfem::BlockOperator& op);
    void BuildHybridRHS(mfem::BlockOperator& op);
    void HybridSolve(const mfem::Vector& resid, mfem::Vector& dx);
public:
    CoupledSolver(const MixedMatrix& darcy_system,
//                  const std::vector<mfem::DenseMatrix>& edge_traces,
                  const double dt,
                  const double weight,
                  const double density,
                  const mfem::Vector& S_prev,
                  NLSolverParameters param);

    mfem::Vector Residual(const mfem::Vector& x, const mfem::Vector& y) override;
    double ResidualNorm(const mfem::Vector& x, const mfem::Vector& y) override;
    double Norm(const mfem::Vector& vec);
    const mfem::Array<int>& BlockOffsets() const { return blk_offsets_; }

    void BackTracking(const mfem::Vector& rhs,  double prev_resid_norm,
                      mfem::Vector& x, mfem::Vector& dx) override;

    const mfem::Vector& GetScales() const { return  scales_; }
};

class CoupledFAS : public FAS
{
    const Hierarchy& hierarchy_;

    double Norm(int level, const mfem::Vector& vec) const override;
    void Restrict(int level, const mfem::Vector& fine, mfem::Vector& coarse) const override;
    void Interpolate(int level, const mfem::Vector& coarse, mfem::Vector& fine) const override;
    void Project(int level, const mfem::Vector& fine, mfem::Vector& coarse) const override;
    mfem::Vector ProjectS(int level, const mfem::Vector& S) const;
public:
    CoupledFAS(const Hierarchy& hierarchy,
               const double dt,
               const double weight,
               const double density,
               const mfem::Vector& S_prev,
               FASParameters param);

    const NonlinearSolver& GetLevelSolver(int level) const { return *solvers_[level]; };
};

class TransportSolver : public NonlinearSolver
{
    const MixedMatrix& darcy_system_;
    const mfem::Array<int>& starts_;
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
          starts_(darcy_system.GetGraph().VertexStarts()), gmres_(comm_),
          Adv_(Adv_), Ms_(SparseIdentity(Adv_.NumCols()) *= vol_dt_inv)
    {
        gmres_.SetMaxIter(200);
        gmres_.SetRelTol(1e-9);
//        gmres_.SetPrintLevel(1);
    }

    mfem::Vector Residual(const mfem::Vector& x, const mfem::Vector& y) override;
    double ResidualNorm(const mfem::Vector& x, const mfem::Vector& y) override
    {
        return mfem::ParNormlp(Residual(x, y), 2, comm_);
    }
};

TwoPhase* problem_ptr;
int count = 0;
int num_coarse_lin_iter = 0;
int num_coarse_lin_solve = 0;

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
    const char* perm_file = "spe_perm.dat";
    args.AddOption(&perm_file, "-p", "--perm", "SPE10 permeability file data.");
    int dim = 3;
    args.AddOption(&dim, "-d", "--dim", "Dimension of the physical space.");
    int slice = 0;
    args.AddOption(&slice, "-s", "--slice", "Slice of SPE10 data for 2D run.");
    int well_height = 1;
    args.AddOption(&well_height, "-wh", "--well-height", "Well Height.");
    double inject_rate = 0.001;
    args.AddOption(&inject_rate, "-ir", "--inject-rate", "Injector rate.");
    double bhp = 1.0e5;
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

//    evolve_param.dt = evolve_param.total_time;

    const int max_iter = 100;

    FASParameters fas_param;
    fas_param.fine.max_num_iter = fas_param.mid.max_num_iter = use_vcycle ? 1 : max_iter;
    fas_param.mid.max_num_iter = use_vcycle ? 1 : max_iter;
    fas_param.coarse.max_num_iter = use_vcycle ? 20 : max_iter;
    fas_param.coarse.print_level = use_vcycle ? -1 : 1;
    fas_param.fine.print_level = use_vcycle ? -1 : 1;
    fas_param.mid.print_level = use_vcycle ? -1 : 1;
//    fas_param.coarse.rtol = 1e-10;
//    fas_param.coarse.atol = 1e-12;
    fas_param.nl_solve.print_level = 1;
    fas_param.nl_solve.max_num_iter = use_vcycle ? max_iter : 1;
    fas_param.nl_solve.atol = 1e-10;
    fas_param.nl_solve.rtol = 1e-8;
    SetOptions(fas_param, use_vcycle, num_backtrack, diff_tol);

    mfem::Array<int> ess_attr(dim == 3 ? 6 : 4);
    ess_attr = 1;

    // Setting up finite volume discretization problem
    bool use_metis = true;
    TwoPhase problem(perm_file, dim, 5, slice, use_metis, ess_attr,
                     well_height, inject_rate, bhp);
problem_ptr = &problem;
    mfem::Array<int> part;
    mfem::Array<int> coarsening_factors(dim);
    coarsening_factors = 1;
    coarsening_factors[0] = upscale_param.coarse_factor;
    problem.Partition(use_metis, coarsening_factors, part);
    upscale_param.num_iso_verts = problem.NumIsoVerts();


    Hierarchy hierarchy(problem.GetFVGraph(true), upscale_param,
                        &part, &problem.EssentialAttribute());
    hierarchy.PrintInfo();

//    if (upscale_param.hybridization)
//    {
//        hierarchy.SetAbsTol(1e-18);
//        hierarchy.SetRelTol(1e-15);
//    }

    // Fine scale transport based on fine flux
    std::vector<mfem::Vector> Ss(upscale_param.max_levels);

    //    int l = 0;
    for (int l = upscale_param.max_levels-1; l < upscale_param.max_levels; ++l)
    {
        fas_param.num_levels = l + 1;
        TwoPhaseSolver solver(problem, hierarchy, 0, evolve_param, fas_param);

        mfem::BlockVector initial_value(problem.BlockOffsets());
        initial_value = 0.0;

        mfem::StopWatch chrono;
        chrono.Start();
        mfem::BlockVector sol = solver.Solve(initial_value);

        if (myid == 0)
        {
            std::cout << "Level " << l << ":\n    Time stepping done in "
                      << chrono.RealTime() << "s.\n";
        }

        Ss[l] = sol.GetBlock(2);
//        Ss[l].SetSize(sol.BlockSize(2) - 5);

        double norm = mfem::ParNormlp(Ss[l], 1, comm);
        if (myid == 0) { std::cout << "    || S ||_1 = " << norm << "\n"; }

//        if (l) { Ss[l] -= Ss[0]; }
//        double diff = mfem::ParNormlp(Ss[l], 2, comm);
//        norm = mfem::ParNormlp(Ss[0], 2, comm);
//        if (myid == 0) { std::cout << "    rel err = " << diff / norm << "\n"; }

//        mfem::socketstream sout;
//        if (l && evolve_param.vis_step)
//        {
//            problem.VisSetup(sout, Ss[l], 0.0, 0.0, "Solution difference");
//        }
    }
    return EXIT_SUCCESS;
}

void SetOptions(FASParameters& param, bool use_vcycle, int num_backtrack, double diff_tol)
{
    param.cycle = use_vcycle ? V_CYCLE : FMG;
    param.nl_solve.linearization = Newton;
    param.coarse_correct_tol = 1e-6;
    param.fine.check_converge = use_vcycle ? false : true;
    param.fine.linearization = param.nl_solve.linearization;
    param.mid.linearization = param.nl_solve.linearization;
    param.coarse.linearization = param.nl_solve.linearization;
    param.fine.num_backtrack = num_backtrack;
    param.mid.num_backtrack = num_backtrack;
    param.coarse.num_backtrack = num_backtrack;
    param.fine.diff_tol = diff_tol;
    param.mid.diff_tol = diff_tol;
    param.coarse.diff_tol = diff_tol;
    param.nl_solve.diff_tol = diff_tol;
}

//mfem::Vector ComputeFaceFlux(const MixedMatrix& darcy_system,
//                             const mfem::Vector& flux)
//{
//    mfem::Vector out(darcy_system.GetTraceFluxes());
//    RescaleVector(flux, out);
//    return out;
//}

mfem::SparseMatrix BuildUpwindPattern(const GraphSpace& graph_space,
                                      const mfem::Vector& flux)
{
    const Graph& graph = graph_space.GetGraph();
    const mfem::SparseMatrix& edge_vert = graph.EdgeToVertex();
    const mfem::SparseMatrix e_te_diag = GetDiag(graph.EdgeToTrueEdge());

    mfem::SparseMatrix upwind_pattern(graph.NumEdges(), graph.NumVertices());

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
        for (int j = 0; j < num_edofs; ++j)
        {
            const int edge = vert_edof.GetRowColumns(i)[j];

            if (edge_vert.RowSize(edge) == 2) // edge is interior
            {
                const int upwind_vert = flux[edge] > 0.0 ? 0 : 1;
//                if (edge_vert.GetRowColumns(edge)[upwind_vert] == i)
//                {
//                    out[i](0, j) = D(i, edge) * FS[i];
//                }
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

TwoPhaseSolver::TwoPhaseSolver(const TwoPhase& problem, Hierarchy& hierarchy,
                               const int level, const EvolveParamenters& evolve_param,
                               const FASParameters& solver_param)
    : level_(level), evolve_param_(evolve_param), solver_param_(solver_param),
      problem_(problem), hierarchy_(hierarchy), blk_offsets_(4), nonlinear_iter_(0),
      step_converged_(true), weight_(problem.CellVolume() * porosity_ * density_)
{
    linear_iter_ = 0;

    blk_helper_.reserve(level + 1);
    blk_helper_.emplace_back(hierarchy.BlockOffsets(0));
    blk_helper_[0].GetBlock(0) = problem_.GetEdgeRHS();
    blk_helper_[0].GetBlock(1) = problem_.GetVertexRHS();

    for (int l = 0; l < level_; ++l)
    {
        blk_helper_.push_back(hierarchy.Restrict(l, blk_helper_[l]));
    }

    blk_offsets_[0] = 0;
    blk_offsets_[1] = hierarchy.BlockOffsets(level)[1];
    blk_offsets_[2] = hierarchy.BlockOffsets(level)[2];
    blk_offsets_[3] = 2 * blk_offsets_[2] - blk_offsets_[1];

    source_.reset(new mfem::BlockVector(blk_offsets_));
    source_->GetBlock(0) = blk_helper_[level].GetBlock(0);
    source_->GetBlock(1) = blk_helper_[level].GetBlock(1);
    source_->GetBlock(2) = blk_helper_[level].GetBlock(1);

    auto& e_te_e = hierarchy.GetMatrix(level).GetGraph().EdgeToTrueEdgeToEdge();
    auto& starts = hierarchy.GetMatrix(level).GetGraph().VertexStarts();
    D_te_e_ = ParMult(hierarchy.GetMatrix(level).GetD(), e_te_e, starts);
}

mfem::BlockVector TwoPhaseSolver::Solve(const mfem::BlockVector& init_val)
{
    int myid;
    MPI_Comm_rank(hierarchy_.GetComm(), &myid);

    mfem::BlockVector x(blk_offsets_);

    blk_helper_[0].GetBlock(0) = init_val.GetBlock(0);
    blk_helper_[0].GetBlock(1) = init_val.GetBlock(1);
    mfem::Vector x_blk2 = init_val.GetBlock(2);

    mfem::socketstream sout;
    if (evolve_param_.vis_step) { problem_.VisSetup(sout, x_blk2, -0.0, 1.0, "Fine scale"); }

    for (int l = 0; l < level_; ++l)
    {
        hierarchy_.Project(l, blk_helper_[l], blk_helper_[l + 1]);
        x_blk2 = hierarchy_.Project(l, x_blk2);
    }

    x.GetBlock(0) = blk_helper_[level_].GetBlock(0);
    x.GetBlock(1) = blk_helper_[level_].GetBlock(1);
    x.GetBlock(2) = x_blk2;

    double time = 0.0;
    double dt_real = std::min(evolve_param_.dt, evolve_param_.total_time - time) / 2.0;

    bool done = false;
    int step;
    for (step = 1; !done; step++)
    {
        mfem::BlockVector previous_x(x);
//        dt_real = std::min(std::min(dt_real * 2.0, evolve_param_.total_time - time), 345600.);
        dt_real = std::min(dt_real * 2.0, evolve_param_.total_time - time);
        step_converged_ = false;

        TimeStepping(dt_real, x);
        while (!step_converged_)
        {
            x = previous_x;
            dt_real /= 2.0;
            TimeStepping(dt_real, x);
        }

        time += dt_real;
        done = (time >= evolve_param_.total_time);

        if (myid == 0)
        {
            std::cout << "Time step " << step << ": step size = " << dt_real
                      << ", time = " << time << "\n\n";
        }
        if (evolve_param_.vis_step && (done || step % evolve_param_.vis_step == 0))
        {
            x_blk2 = x.GetBlock(2);
            for (int l = level_; l > 0; --l)
            {
                x_blk2 = hierarchy_.Interpolate(l, x_blk2);
            }

            problem_.VisUpdate(sout, x_blk2);
        }
    }

    if (myid == 0)
    {
        std::cout << "Total nonlinear iterations: " << nonlinear_iter_ << "\n";
        std::cout << "Average nonlinear iterations per time step: "
                  << double(nonlinear_iter_) / double(step-1) << "\n";
        std::cout << "Total coarsest linear iterations: " << num_coarse_lin_iter << "\n";
        std::cout << "Average linear iterations per coarsest level linear solve: "
                  << num_coarse_lin_iter / double(num_coarse_lin_solve) << "\n";
        std::cout << "Average linear iterations per time step: "
                  << num_coarse_lin_iter / double(step-1) << "\n";
    }

    blk_helper_[level_].GetBlock(0) = x.GetBlock(0);
    blk_helper_[level_].GetBlock(1) = x.GetBlock(1);
    x_blk2 = x.GetBlock(2);

    for (int l = level_; l > 0; --l)
    {
        hierarchy_.Interpolate(l, blk_helper_[l], blk_helper_[l - 1]);
        x_blk2 = hierarchy_.Interpolate(l, x_blk2);
    }

    mfem::BlockVector out(problem_.BlockOffsets());
    out.GetBlock(0) = blk_helper_[0].GetBlock(0);
    out.GetBlock(1) = blk_helper_[0].GetBlock(1);
    out.GetBlock(2) = x_blk2;

    return out;
}

void TwoPhaseSolver::TimeStepping(const double dt, mfem::BlockVector& x)
{
    const MixedMatrix& system = hierarchy_.GetMatrix(level_);
    std::vector<mfem::DenseMatrix> traces;

    if (evolve_param_.scheme == FullyImplcit) // coupled: solve all unknowns together
    {
//        CoupledSolver solver(system, dt, weight_, density_, solver_param_.nl_solve);
        CoupledFAS solver(hierarchy_, dt, weight_, density_, x.GetBlock(2), solver_param_);

//solver.SetAbsTol(1e-9);
//solver.SetRelTol(1e-12);
        mfem::BlockVector rhs(*source_);
        rhs.GetBlock(0) *= (1. / dt / density_);

        rhs.GetBlock(1) *= (dt * density_);
        add(dt * density_, rhs.GetBlock(2), weight_, x.GetBlock(2), rhs.GetBlock(2));

//        rhs.GetBlock(2) *= (1. / dt / density_);

//        {
//            mfem::Vector x_copy(x);
//            solver.SetMaxIter(1);
//            solver.Solve(rhs, x_copy);
//            solver.SetMaxIter(100);

//            auto& scales = ((CoupledSolver&)solver.GetLevelSolver(0)).GetScales();
//            rhs.GetBlock(0) *= scales[0];
//            rhs.GetBlock(1) *= scales[1];
//            rhs.GetBlock(2) *= scales[2];
//        }

//        x = 0.0;
        solver.Solve(rhs, x);
        step_converged_ = solver.IsConverged();
        nonlinear_iter_ += solver.GetNumIterations();
        linear_iter_ += solver.GetLevelSolver(solver_param_.num_levels-1).GetNumLinearIterations();
    }
    else // sequential: solve for flux and pressure first, and then saturation
    {
        const mfem::Vector S = system.PWConstProject(x.GetBlock(2));
        hierarchy_.RescaleCoefficient(level_, TotalMobility(S));
        mfem::BlockVector flow_rhs(*source_, hierarchy_.BlockOffsets(level_));
        mfem::BlockVector flow_sol(x, hierarchy_.BlockOffsets(level_));
        hierarchy_.Solve(level_, flow_rhs, flow_sol);

        auto upwind = BuildUpwindPattern(system.GetGraphSpace(), x.GetBlock(0));
        assert(mfem::ParNormlp(x.GetBlock(0), 2, D_te_e_->GetComm()) < mfem::infinity());
        upwind.ScaleRows(x.GetBlock(0));

        if (evolve_param_.scheme == IMPES) // explcict: new_S = S + dt W^{-1} (b - Adv F(S))
        {
            mfem::Vector dSdt(source_->GetBlock(2));
            D_te_e_->Mult(-1.0, Mult(upwind, FractionalFlow(S)), 1.0, dSdt);
            x.GetBlock(2).Add(dt * density_ / weight_, dSdt);
            step_converged_ = true;
        }
        else // implicit: new_S solves new_S = S + dt W^{-1} (b - Adv F(new_S))
        {
            auto Adv = ParMult(*D_te_e_, upwind, system.GetGraph().VertexStarts());
            const double scaling = weight_ / density_ / dt;
            TransportSolver solver(*Adv, system, scaling, solver_param_.nl_solve);

            mfem::Vector rhs(source_->GetBlock(2));
            rhs.Add(weight_ / density_ / dt, x.GetBlock(2));
            solver.Solve(rhs, x.GetBlock(2));
            step_converged_ = solver.IsConverged();
            nonlinear_iter_ += solver.GetNumIterations();
        }
    }
}

CoupledSolver::CoupledSolver(const MixedMatrix& darcy_system,
//                             const std::vector<mfem::DenseMatrix>& edge_traces,
                             const double dt,
                             const double weight,
                             const double density,
                             const mfem::Vector& S_prev,
                             NLSolverParameters param)
    : NonlinearSolver(darcy_system.GetComm(), param), darcy_system_(darcy_system),
      gmres_(comm_), local_dMdS_(darcy_system.GetGraph().NumVertices()),
      Ms_(SparseIdentity(darcy_system.GetGraph().NumVertices()) *= weight),
      blk_offsets_(4), true_blk_offsets_(4), ess_dofs_(darcy_system.GetEssDofs()),
      vert_starts_(darcy_system.GetGraph().VertexStarts()),
//      traces_(edge_traces),
      dt_(dt), weight_(weight), density_(density), is_first_resid_eval_(false),
      scales_(3)
{
    mfem::SparseMatrix D_proc(darcy_system_.GetD());
    if (ess_dofs_.Size()) { D_proc.EliminateCols(ess_dofs_); }
    D_.reset(darcy_system_.MakeParallelD(D_proc));
    DT_.reset(D_->Transpose());
    *D_ *= (dt_ * density_);
    *DT_ *= (1. / dt_ / density_);

    GenerateOffsets(comm_, D_->NumCols(), true_edof_starts_);

    blk_offsets_[0] = 0;
    blk_offsets_[1] = darcy_system.NumEDofs();
    blk_offsets_[2] = blk_offsets_[1] + darcy_system.NumVDofs();
    blk_offsets_[3] = blk_offsets_[2] + Ms_.NumCols();

    true_blk_offsets_[0] = 0;
    true_blk_offsets_[1] = D_->NumCols();
    true_blk_offsets_[2] = true_blk_offsets_[1] + darcy_system.NumVDofs();
    true_blk_offsets_[3] = true_blk_offsets_[2] + Ms_.NumCols();

    gmres_.SetMaxIter(10000);
//    gmres_.SetAbsTol(1e-15);
    gmres_.SetRelTol(1e-9);
    gmres_.SetPrintLevel(0);
    gmres_.SetKDim(100);

    normalizer_.SetSize(Ms_.NumCols());
//    normalizer_ = 800. * (weight_ / density_);

    {
        normalizer_ = S_prev;
        normalizer_ -= 1.0;
        normalizer_ *= -800.0;
        normalizer_.Add(1000.0, S_prev);
        normalizer_ *= (weight_ / density_); // weight_ / density_ = vol * porosity
    }

    scales_ = 1.0;
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

mfem::Vector CoupledSolver::Residual(const mfem::Vector& x, const mfem::Vector& y)
{
    mfem::BlockVector blk_x(x.GetData(), blk_offsets_);
    mfem::BlockVector out(blk_offsets_);
    out = 0.0;

    mfem::BlockVector darcy_x(x.GetData(), darcy_system_.BlockOffsets());
    mfem::BlockVector darcy_Rx(out.GetData(), darcy_system_.BlockOffsets());

    const mfem::Vector S = darcy_system_.PWConstProject(blk_x.GetBlock(2));
    darcy_system_.Mult(TotalMobility(S), darcy_x, darcy_Rx);

    darcy_Rx.GetBlock(0) *= (1. / dt_ / density_);
    darcy_Rx.GetBlock(1) *= (dt_ * density_);

    out.GetBlock(2) = blk_x.GetBlock(2);

//    out.GetBlock(2) *= (1. / dt_ / density_);


    const GraphSpace& space = darcy_system_.GetGraphSpace();
    auto upwind = BuildUpwindPattern(space, blk_x.GetBlock(0));
    auto upw_FS = Mult(upwind, FractionalFlow(S));
    RescaleVector(blk_x.GetBlock(0), upw_FS);
    auto U_FS = Mult(space.TrueEDofToEDof(), upw_FS);
    D_->Mult(1.0, U_FS, Ms_(0, 0), out.GetBlock(2)); //TODO: Ms_

    out -= y;
    SetZeroAtMarker(ess_dofs_, out.GetBlock(0));

    {
        out.GetBlock(0) *= scales_[0];
        out.GetBlock(1) *= scales_[1];
        out.GetBlock(2) *= scales_[2];
    }


//    if (Norm(out) > 1e-8)
//    {
//        std::cout << "|| resid 0|| " << mfem::ParNormlp(out.GetBlock(0), 2, comm_) << "\n";
//        std::cout << "|| resid 1|| " << mfem::ParNormlp(out.GetBlock(1), 2, comm_) << "\n";
//        std::cout << "|| resid 2|| " << mfem::ParNormlp(out.GetBlock(2), 2, comm_) << "\n";
//    }

//    if (is_first_resid_eval_)
//    {
//        normalizer_ = S;
//        normalizer_ -= 1.0;
//        normalizer_ *= -800.0;
//        normalizer_.Add(1000.0, S);
//        normalizer_ *= (weight_ / density_);
//        is_first_resid_eval_ = false;
//    }

    return out;
}

double CoupledSolver::ResidualNorm(const mfem::Vector& x, const mfem::Vector& y)
{
//    std::cout<<"Resid norm: num of dofs = " << true_blk_offsets_.Last()<<"\n";
    return Norm(Residual(x, y));
}

double CoupledSolver::Norm(const mfem::Vector& vec)
{
    auto true_resid = AssembleTrueVector(vec);
    mfem::BlockVector blk_resid(true_resid.GetData(), true_blk_offsets_);

    InvRescaleVector(normalizer_, blk_resid.GetBlock(1));
    InvRescaleVector(normalizer_, blk_resid.GetBlock(2));

    return mfem::ParNormlp(blk_resid, mfem::infinity(), comm_);
}

void CoupledSolver::Build_dMdS(const mfem::Vector& flux, const mfem::Vector& S)
{
    // TODO: saturation is only 1 dof per cell
    auto& vert_edof = darcy_system_.GetGraphSpace().VertexToEDof();
    auto& vert_vdof = darcy_system_.GetGraphSpace().VertexToVDof();

    auto& MB = dynamic_cast<const ElementMBuilder&>(darcy_system_.GetMBuilder());
    auto& M_el = MB.GetElementMatrices();
    auto& proj_pwc = const_cast<mfem::SparseMatrix&>(darcy_system_.GetPWConstProj());

    mfem::Array<int> local_edofs, local_vdofs, vert(1);
    mfem::Vector sigma_loc, Msigma_vec;
    mfem::DenseMatrix proj_pwc_loc;

    const mfem::Vector dTMinv_dS_vec = dTMinv_dS(S);

    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        GetTableRow(vert_vdof, i, local_vdofs);
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

mfem::SparseMatrix CoupledSolver::Assemble_dMdS(const mfem::Vector& flux, const mfem::Vector& S)
{
    Build_dMdS(flux, S); // local_dMdS_ is constructed here

    auto& vert_edof = darcy_system_.GetGraphSpace().VertexToEDof();
    mfem::Array<int> local_edofs, local_vert(1);
    mfem::SparseMatrix out(vert_edof.NumCols(), vert_edof.NumRows());
    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        local_vert[0] = i;
        out.AddSubMatrix(local_edofs, local_vert, local_dMdS_[i]);
    }
    out.Finalize();
    return out;
}

void CoupledSolver::Step(const mfem::Vector& rhs, mfem::Vector& x, mfem::Vector& dx)
{
    mfem::BlockVector blk_x(x.GetData(), blk_offsets_);
    const GraphSpace& space = darcy_system_.GetGraphSpace();


//    if (false)
    {
        for (int ii = 0; ii < blk_x.BlockSize(2); ++ii)
        {
            if (blk_x.GetBlock(2)[ii] < 0.0)
            {
                blk_x.GetBlock(2)[ii]  = 0.0;
            }
        }
    }

    mfem::Vector true_resid = AssembleTrueVector(Residual(x, rhs));
    true_resid *= -1.0;

    mfem::BlockVector true_blk_dx(true_blk_offsets_);
    true_blk_dx = 0.0;

    const mfem::Vector S = darcy_system_.PWConstProject(blk_x.GetBlock(2));
    auto M_proc = darcy_system_.GetMBuilder().BuildAssembledM(TotalMobility(S));
    auto dMdS_proc = Assemble_dMdS(blk_x.GetBlock(0), S);

//    std::cout<< " before: min(S) max(S) = "<< S.Min() << " " << S.Max() <<"\n";

    for (int mm = 0; mm < ess_dofs_.Size(); ++mm)
    {
        if (ess_dofs_[mm])
        {
            M_proc.EliminateRowCol(mm, mfem::Matrix::DIAG_KEEP); // assume essential data = 0
            dMdS_proc.EliminateRow(mm);
        }
    }

    unique_ptr<mfem::HypreParMatrix> M(darcy_system_.MakeParallelM(M_proc));
    auto dMdS = ParMult(space.TrueEDofToEDof(), dMdS_proc, vert_starts_);

    *M *= (1. / dt_ / density_);
    *dMdS *= (1. / dt_ / density_);


    auto upwind = BuildUpwindPattern(space, blk_x.GetBlock(0));

    auto U_FS = Mult(space.TrueEDofToEDof(), Mult(upwind, FractionalFlow(S)));
    auto dTdsigma = ParMult(*D_, SparseDiag(std::move(U_FS)), true_edof_starts_);

    upwind.ScaleRows(blk_x.GetBlock(0));
    upwind.ScaleColumns(dFdS(S));

    auto U = ParMult(space.TrueEDofToEDof(), upwind, vert_starts_);
    auto U_pwc = ParMult(*U, darcy_system_.GetPWConstProj(), vert_starts_);
    unique_ptr<mfem::HypreParMatrix> dTdS(mfem::ParMult(D_.get(), U_pwc.get()));
    GetDiag(*dTdS) += Ms_;

    const bool hybrid = true;
    if (!hybrid)
    {

    mfem::BlockOperator op(true_blk_offsets_);
    op.SetBlock(0, 0, M.get());
    op.SetBlock(0, 1, DT_.get());
    op.SetBlock(1, 0, D_.get());
    op.SetBlock(0, 2, dMdS.get());
    op.SetBlock(2, 0, dTdsigma.get());
    op.SetBlock(2, 2, dTdS.get());


    if (is_first_resid_eval_ == false)
    {
        mfem::SparseMatrix M_diag = GetDiag(*M);
        mfem::SparseMatrix dTdS_diag = GetDiag(*dTdS);
//        scales_[0] = 1. / FroNorm(M_diag) * dt_ * density_;
//        scales_[0] = 1. * dt_ * density_ * dt_ * density_;
//        scales_[0] = 1. / FroNorm(M_diag);
//        scales_[1] = scales_[0];
//        scales_[1] = 1.0 / dt_ / density_;
//        scales_[2] = 1. / FroNorm(dTdS_diag) / dt_ / density_;

        *DT_ *= scales_[0];
        *D_ *= scales_[1];

        is_first_resid_eval_ = true;
    }

    *M *= scales_[0];
    *dMdS *= scales_[0];
    *dTdsigma *= scales_[2];
    *dTdS *= scales_[2];

    // preconditioner

    mfem::Vector Md;
    M->GetDiag(Md);
    Md *= -1.0;
    DT_->InvScaleRows(Md);
    unique_ptr<mfem::HypreParMatrix> schur(mfem::ParMult(D_.get(), DT_.get()));
    DT_->ScaleRows(Md);


    auto M_inv = make_unique<mfem::HypreDiagScale>(*M);
    unique_ptr<mfem::HypreBoomerAMG> schur_inv(BoomerAMG(*schur));

//    unique_ptr<mfem::HypreBoomerAMG> dTdS_inv(BoomerAMG(*dTdS));

    dMdS->InvScaleRows(Md);
    unique_ptr<mfem::HypreParMatrix> tmp11(mfem::ParMult(dTdsigma.get(), dMdS.get()));
    dMdS->ScaleRows(Md);
    unique_ptr<mfem::HypreParMatrix> schur22(ParAdd(*dTdS, *tmp11));

    auto type22 = mfem::HypreSmoother::Type::l1Jacobi;
    auto dTdS_inv = make_unique<mfem::HypreSmoother>(*schur22, type22);
//    auto dTdS_inv = make_unique<HypreILU>(*schur22, 0);

//    mfem::BlockVector diff(true_blk_dx);
    const bool use_direct_solver = true;
    unique_ptr<mfem::SparseMatrix> mono_mat;

    bool true_cpr = false;

    mfem::BlockVector sol_diff(true_blk_dx);

    if (!true_cpr)
    {
//        if (use_direct_solver)
        {
            mfem::SparseMatrix M_diag = GetDiag(*M);
            mfem::SparseMatrix DT_diag = GetDiag(*DT_);
            mfem::SparseMatrix D_diag = GetDiag(*D_);
            mfem::SparseMatrix dMdS_diag = GetDiag(*dMdS);
            mfem::SparseMatrix dTdsigma_diag = GetDiag(*dTdsigma);
            mfem::SparseMatrix dTdS_diag = GetDiag(*dTdS);

            mfem::BlockMatrix mat(true_blk_offsets_);
            mat.SetBlock(0, 0, &M_diag);
            mat.SetBlock(0, 1, &DT_diag);
            mat.SetBlock(1, 0, &D_diag);
            mat.SetBlock(0, 2, &dMdS_diag);
            mat.SetBlock(2, 0, &dTdsigma_diag);
            mat.SetBlock(2, 2, &dTdS_diag);

            mono_mat.reset(mat.CreateMonolithic());

//            if (mat.NumRows() < 30000)
//            {
//                mfem::SparseMatrix mono_copy(*mono_mat);
//                mfem::UMFPackSolver direct_solve(mono_copy);
//                mfem::BlockVector direct_sol(true_blk_dx);
//                direct_sol = 0.0;
//                direct_solve.Mult(true_resid, true_blk_dx);

//                sol_diff = direct_sol;
//            }

//                    std::cout<< "|| M || = " << FroNorm(M_diag) << "\n";
//                    std::cout<< "|| D || = " << FroNorm(GetDiag(*D_)) << "\n";
//                    std::cout<< "|| DT || = " << FroNorm(GetDiag(*DT_)) << "\n";
//                    std::cout<< "|| dMdS || = " << FroNorm(dMdS_diag) << "\n";
//                    std::cout<< "|| dTdsigma || = " << FroNorm(dTdsigma_diag) << "\n";
//                    std::cout<< "|| dTdS || = " << FroNorm(dTdS_diag) << "\n";

            //        mfem::BlockVector blk_resid(true_resid.GetData(), blk_offsets_);

            //        std::cout << "|| rhs0 || " << mfem::ParNormlp(blk_resid.GetBlock(0), 2, comm_) << "\n";
            //        std::cout << "|| rhs1 || " << mfem::ParNormlp(blk_resid.GetBlock(1), 2, comm_) << "\n";
            //        std::cout << "|| rhs2 || " << mfem::ParNormlp(blk_resid.GetBlock(2), 2, comm_) << "\n";

        }
//        else
        {
            mfem::BlockLowerTriangularPreconditioner prec(true_blk_offsets_);
            prec.SetDiagonalBlock(0, M_inv.get());
            prec.SetDiagonalBlock(1, schur_inv.get());
            prec.SetDiagonalBlock(2, dTdS_inv.get());
            prec.SetBlock(1, 0, D_.get());

            DT_->InvScaleRows(Md);
            unique_ptr<mfem::HypreParMatrix> schur21(mfem::ParMult(dTdsigma.get(), DT_.get()));
            DT_->ScaleRows(Md);
            prec.SetBlock(2, 1, schur21.get());
            prec.SetBlock(2, 0, dTdsigma.get());
//            mfem::IdentityOperator Id(dTdS->NumRows());
//            prec.SetDiagonalBlock(2, &Id);


            mfem::Array<int> A_starts;
            GenerateOffsets(darcy_system_.GetComm(), mono_mat->NumRows(), A_starts);
            mfem::HypreParMatrix pMonoMat(darcy_system_.GetComm(), mono_mat->NumRows(), A_starts, mono_mat.get());


            unique_ptr<mfem::HypreSolver> ILU_smoother;
//            if (op.NumRows() > 30000)
                ILU_smoother.reset(new HypreILU(pMonoMat, 0)); // equiv to Euclid
//            else
//                ILU_smoother.reset(new HypreILU(pMonoMat, 1));

            TwoStageSolver prec2(prec, *ILU_smoother, op);


//            auto dTdS_inv2 = make_unique<HypreILU>(*schur22, 0);
//            mfem::BlockDiagonalPreconditioner prec3(true_blk_offsets_);
//            prec3.SetDiagonalBlock(2, dTdS_inv2.get());
//            TwoStageSolver prec2(prec, prec3, op);


            gmres_.SetOperator(op);
            gmres_.SetPreconditioner(prec2);

            //    gmres_.iterative_mode = true;
            //        gmres_.SetPrintLevel(1);

//            true_blk_dx = 0.0;

//            if (op.NumRows() > 30000)
                gmres_.Mult(true_resid, true_blk_dx);
//            else
//                true_blk_dx = sol_diff;

//            sol_diff -= true_blk_dx;


//////////////////////////////////

//            mfem::BlockVector linear_resid(true_blk_dx);
//            linear_resid = 0.0;
//            op.Mult(true_blk_dx, linear_resid);
//            linear_resid -= true_resid;

//            std::cout << "    || linear resid 0|| "
//                      << mfem::ParNormlp(sol_diff.GetBlock(0), 2, comm_) / mfem::ParNormlp(true_blk_dx.GetBlock(0), 2, comm_) << "\n";
//            std::cout << "    || linear resid 1|| "
//                      << mfem::ParNormlp(sol_diff.GetBlock(1), 2, comm_) / mfem::ParNormlp(true_blk_dx.GetBlock(1), 2, comm_) << "\n";
//            std::cout << "    || linear resid 2|| "
//                      << mfem::ParNormlp(sol_diff.GetBlock(2), 2, comm_) / mfem::ParNormlp(true_blk_dx.GetBlock(2), 2, comm_) << "\n";



//            if (gmres_.GetNumIterations() == 18)
//            {
//                std::ofstream ofs("good_fine_matrix.txt");
//                mono_mat->PrintMatlab(ofs);
//                std::ofstream rhs_file("good_fine_rhs.txt");
//                true_resid.Print(rhs_file, 1);
//                std::ofstream sol_file("good_fine_sol.txt");
//                true_blk_dx.Print(sol_file, 1);
//            }
//            if (gmres_.GetNumIterations() == 76)
//            {
//                std::ofstream ofs("bad_fine_matrix.txt");
//                mono_mat->PrintMatlab(ofs);
//                std::ofstream rhs_file("bad_fine_rhs.txt");
//                true_resid.Print(rhs_file, 1);
//                std::ofstream sol_file("bad_fine_sol.txt");
//                true_blk_dx.Print(sol_file, 1);
//            }
//            if (gmres_.GetNumIterations() == 87)
//            {
//                std::ofstream ofs("good_coarse_matrix.txt");
//                mono_mat->PrintMatlab(ofs);
//                std::ofstream rhs_file("good_coarse_rhs.txt");
//                true_resid.Print(rhs_file, 1);
//                std::ofstream sol_file("good_coarse_sol.txt");
//                true_blk_dx.Print(sol_file, 1);
//            }
//            if (gmres_.GetNumIterations() == 200)
//            {
//                std::ofstream ofs("bad_coarse_matrix.txt");
//                mono_mat->PrintMatlab(ofs);
//                std::ofstream rhs_file("bad_coarse_rhs.txt");
//                true_resid.Print(rhs_file, 1);
//                std::ofstream sol_file("bad_coarse_sol.txt");
//                true_blk_dx.Print(sol_file, 1);
//            }


////////////////////////////////////



            //        if (!myid_ && !gmres_.GetConverged())
            {
//                std::cout << "this level has " << dTdS->N() << " dofs\n";
            }

            linear_iter_ += gmres_.GetNumIterations();
            if (!myid_) std::cout << "    GMRES took " << gmres_.GetNumIterations()
                                  << " iterations, residual = " << gmres_.GetFinalNorm() << "\n";

            if (darcy_system_.NumVDofs() < 300)
            {
                num_coarse_lin_iter += gmres_.GetNumIterations();
                num_coarse_lin_solve++;
            }
        }
    }
    else
    {
        mfem::GMRESSolver gmres2(comm_);
        gmres2.SetMaxIter(10000);
        gmres2.SetRelTol(1e-9);
        gmres2.SetPrintLevel(0);
        gmres2.SetKDim(100);

        dMdS->InvScaleRows(Md);
        unique_ptr<mfem::HypreParMatrix> tmp11(mfem::ParMult(dTdsigma.get(), dMdS.get()));
        dMdS->ScaleRows(Md);
//        (*tmp11) *= -1.0;
//        dTdS->Add(1.0, *schur11);
        unique_ptr<mfem::HypreParMatrix> schur11(ParAdd(*dTdS, *tmp11));


        DT_->InvScaleRows(Md);
        unique_ptr<mfem::HypreParMatrix> schur10(mfem::ParMult(dTdsigma.get(), DT_.get()));
        DT_->ScaleRows(Md);

        Md *= -1;
        dMdS->InvScaleRows(Md);
        unique_ptr<mfem::HypreParMatrix> schur01(mfem::ParMult(D_.get(), dMdS.get()));
        dMdS->ScaleRows(Md);


        mfem::Array<int> true_blk_offsets_2(3);
        true_blk_offsets_2[0] = 0;
        true_blk_offsets_2[1] = true_blk_offsets_[2] - true_blk_offsets_[1];
        true_blk_offsets_2[2] = true_blk_offsets_[3] - true_blk_offsets_[1];

        *schur *= -1.0;
        mfem::BlockOperator op2(true_blk_offsets_2);
        op2.SetBlock(0, 0, schur.get());
        op2.SetBlock(0, 1, schur01.get());
        op2.SetBlock(1, 0, schur10.get());
        op2.SetBlock(1, 1, schur11.get());

        unique_ptr<mfem::HypreBoomerAMG> schur00_inv(BoomerAMG(*schur));
//        unique_ptr<mfem::HypreBoomerAMG> schur11_inv(BoomerAMG(*schur11));
        auto smoother_type = mfem::HypreSmoother::Type::l1Jacobi;
        auto schur11_inv = make_unique<mfem::HypreSmoother>(*schur11, smoother_type);

        mfem::BlockLowerTriangularPreconditioner prec2(true_blk_offsets_2);

        prec2.SetDiagonalBlock(0, schur00_inv.get());
        prec2.SetDiagonalBlock(1, schur11_inv.get());
        prec2.SetBlock(1, 0, schur10.get());
//        mfem::IdentityOperator Id(dTdS->NumRows());
//        prec2.SetDiagonalBlock(1, &Id);


        mfem::SparseMatrix diag00 = GetDiag(*schur);
        mfem::SparseMatrix diag01 = GetDiag(*schur01);
        mfem::SparseMatrix diag10 = GetDiag(*schur10);
        mfem::SparseMatrix diag11 = GetDiag(*schur11);

        mfem::BlockMatrix mat2(true_blk_offsets_2);
        mat2.SetBlock(0, 0, &diag00);
        mat2.SetBlock(0, 1, &diag01);
        mat2.SetBlock(1, 0, &diag10);
        mat2.SetBlock(1, 1, &diag11);

        mono_mat.reset(mat2.CreateMonolithic());

        mfem::Array<int> A_starts;
        GenerateOffsets(darcy_system_.GetComm(), mono_mat->NumRows(), A_starts);
        mfem::HypreParMatrix pMonoMat(darcy_system_.GetComm(), mono_mat->NumRows(), A_starts, mono_mat.get());

        unique_ptr<mfem::HypreSolver> ILU_smoother;
        ILU_smoother.reset(new HypreILU(pMonoMat, 0));

        TwoStageSolver prec_prod(prec2, *ILU_smoother, op2);
        gmres2.SetOperator(op2);
        gmres2.SetPreconditioner(prec_prod);

        mfem::BlockVector blk_true_resid(true_resid.GetData(), true_blk_offsets_);

        mfem::BlockVector true_resid2(true_blk_offsets_2);
        true_resid2.GetBlock(0) = blk_true_resid.GetBlock(1);
        true_resid2.GetBlock(1) = blk_true_resid.GetBlock(2);

        InvRescaleVector(Md, blk_true_resid.GetBlock(0));
        D_->Mult(1.0, blk_true_resid.GetBlock(0), -1.0, true_resid2.GetBlock(0));
        dTdsigma->Mult(-1.0, blk_true_resid.GetBlock(0), 1.0, true_resid2.GetBlock(1));
        RescaleVector(Md, blk_true_resid.GetBlock(0));

//        mfem::BlockVector true_blk_dx_copy(true_blk_dx);

        mfem::BlockVector true_blk_dx2(true_blk_dx+true_blk_dx.BlockSize(0), true_blk_offsets_2);
        true_blk_dx2 = 0.0;
        gmres2.Mult(true_resid2, true_blk_dx2);

        true_blk_dx.GetBlock(0) = blk_true_resid.GetBlock(0);
        DT_->Mult(-1.0, true_blk_dx.GetBlock(1), 1.0, true_blk_dx.GetBlock(0));
        dMdS->Mult(-1.0, true_blk_dx.GetBlock(2), 1.0, true_blk_dx.GetBlock(0));
        InvRescaleVector(Md, true_blk_dx.GetBlock(0));

//        true_blk_dx_copy -= true_blk_dx;
//        std::cout<<"||diff 0 = ||"<<true_blk_dx_copy.GetBlock(0).Norml2() / true_blk_dx.GetBlock(0).Norml2()<<"\n";
//        std::cout<<"||diff 1 = ||"<<true_blk_dx_copy.GetBlock(1).Norml2() / true_blk_dx.GetBlock(1).Norml2()<<"\n";
//        std::cout<<"||diff 2 = ||"<<true_blk_dx_copy.GetBlock(2).Norml2() / true_blk_dx.GetBlock(2).Norml2()<<"\n";

//        std::cout << "this level has " << dTdS->N() << " dofs\n";
        linear_iter_ += gmres2.GetNumIterations();
        if (!myid_) std::cout << "    GMRES took " << gmres2.GetNumIterations()
                              << " iterations, residual = " << gmres2.GetFinalNorm() << "\n";
    }
    }
    else
    {
        mfem::Array<int> offset_hb(3);
        offset_hb[0] = 0;
        offset_hb[1] = darcy_system_.NumEDofs();
        offset_hb[2] = darcy_system_.NumTotalDofs();
        mfem::BlockOperator op_hb(offset_hb);

        mfem::SparseMatrix D_proc(darcy_system_.GetD());
//        D_proc *= (dt_ * density_);
        auto local_dTdsigma = Build_dTdsigma(space, D_proc, blk_x.GetBlock(0),
                                             FractionalFlow(S));

        // for debug
//        if (false)
//        {
//            mfem::SparseMatrix dTdsig(D_->NumRows(), D_->NumCols());
//            mfem::Array<int> rows, cols;
//            for (unsigned int i = 0; i < local_dTdsigma.size(); ++i)
//            {
//                GetTableRow(space.VertexToVDof(), i, rows);
//                GetTableRow(space.VertexToEDof(), i, cols);
//                dTdsig.AddSubMatrix(rows, cols, local_dTdsigma[i]);
//            }
//            dTdsig.Finalize();
//            dTdsig *= (dt_ * density_);
//            unique_ptr<mfem::SparseMatrix> A_diff(Add(1.0, dTdsig, -1.0, GetDiag(*dTdsigma)));
//            std::cout << "|| dTdsig -= dTdsigma || = "<< FroNorm(*A_diff) <<"\n";
//        }

        (*dTdS) *= (1. / dt_ / density_);

        TwoPhaseHybrid solver(darcy_system_, &(problem_ptr->EssentialAttribute()));

        solver.AssembleSolver(TotalMobility(S), local_dMdS_, local_dTdsigma,
                              *dTdS, 1.0);
        mfem::BlockVector true_blk_resid(true_resid, true_blk_offsets_);

        true_blk_resid.GetBlock(0) *= (dt_ * density_);
        true_blk_resid.GetBlock(1) /= (dt_ * density_);
        true_blk_resid.GetBlock(2) /= (dt_ * density_);


//        mfem::BlockVector true_blk_dx_hb(true_blk_dx);
//        true_blk_dx_hb = 0.0;

        solver.Mult(true_blk_resid, true_blk_dx);
        linear_iter_ += solver.GetNumIterations();

        if (darcy_system_.NumVDofs() < 30000)
        {
            num_coarse_lin_iter += solver.GetNumIterations();
            num_coarse_lin_solve++;
        }

//        mfem::Vector for_print(true_blk_dx_hb.GetBlock(0).GetData(), 10);
//        for_print.Print();
//        mfem::Vector for_print2(true_blk_dx.GetBlock(0).GetData(), 10);
//        for_print2.Print();

//        true_blk_dx_hb -= true_blk_dx;

//        mfem::socketstream sout;
////        true_blk_dx_hb.GetBlock(2) -= true_blk_dx.GetBlock(2);
//        if (true_blk_resid.BlockSize(0) > 10000)
//        {problem_ptr->VisSetup(sout, true_blk_dx_hb.GetBlock(2), 0.0, 0.0, "HB diff"); }

//        SetZeroAtMarker(ess_dofs_, true_blk_dx_hb.GetBlock(0));

//        std::ofstream mfile("full_system.txt");
//            if (mono_mat->NumRows()<50000)
//            mono_mat->PrintMatlab(mfile);


//        std::cout << "    || hb sol diff 0|| "
//                  << mfem::ParNormlp(true_blk_dx_hb.GetBlock(0), 2, comm_) / mfem::ParNormlp(true_blk_dx.GetBlock(0), 2, comm_) << "\n";
//        std::cout << "    || hb sol diff 1|| "
//                  << mfem::ParNormlp(true_blk_dx_hb.GetBlock(1), 2, comm_) / mfem::ParNormlp(true_blk_dx.GetBlock(1), 2, comm_) << "\n";
//        std::cout << "    || hb sol diff 2|| "
//                  << mfem::ParNormlp(true_blk_dx_hb.GetBlock(2), 2, comm_) / mfem::ParNormlp(true_blk_dx.GetBlock(2), 2, comm_) << "\n";

    }

    mfem::BlockVector blk_dx(dx.GetData(), blk_offsets_);
    blk_dx = 0.0;
    auto& dof_truedof = darcy_system_.GetGraphSpace().EDofToTrueEDof();
    dof_truedof.Mult(true_blk_dx.GetBlock(0), blk_dx.GetBlock(0));
    blk_dx.GetBlock(1) = true_blk_dx.GetBlock(1);
    blk_dx.GetBlock(2) = true_blk_dx.GetBlock(2);

    const mfem::Vector dS = darcy_system_.PWConstProject(blk_dx.GetBlock(2));
    blk_dx *= std::min(1.0, param_.diff_tol / mfem::ParNormlp(dS, mfem::infinity(), comm_));

//    std::cout << "|| S ||_inf " << mfem::ParNormlp(dS, mfem::infinity(), comm_) << "\n";

    x += blk_dx;

//    std::cout << "|| x0 || " << mfem::ParNormlp(blk_x.GetBlock(0), 2, comm_) << "\n";
//    std::cout << "|| x1 || " << mfem::ParNormlp(blk_x.GetBlock(1), 2, comm_) << "\n";
//    std::cout << "|| x2 || " << mfem::ParNormlp(blk_x.GetBlock(2), 2, comm_) << "\n";


//    std::cout << "|| dx0 || " << mfem::ParNormlp(blk_dx.GetBlock(0), 2, comm_) << "\n";
//    std::cout << "|| dx1 || " << mfem::ParNormlp(blk_dx.GetBlock(1), 2, comm_) << "\n";
//    std::cout << "|| dx2 || " << mfem::ParNormlp(blk_dx.GetBlock(2), 2, comm_) << "\n";


//    diff -= true_blk_dx;
//    std::cout << "|| diff 0|| " << mfem::ParNormlp(diff.GetBlock(0), 2, comm_) << "\n";
//    std::cout << "|| diff 1|| " << mfem::ParNormlp(diff.GetBlock(1), 2, comm_) << "\n";
//    std::cout << "|| diff 2|| " << mfem::ParNormlp(diff.GetBlock(2), 2, comm_) << "\n";

//    if (darcy_system_.NumVDofs()>10000)
//    {
//        mfem::socketstream sout;
//        mfem::BlockVector true_blk_resid(true_resid, true_blk_offsets_);
//        problem_ptr->VisSetup(sout, true_blk_resid.GetBlock(2), 0.0, 0.0, "diff");
//    }

//    if (darcy_system_.NumVDofs()>10000)
//    std::cout<< " blk_x.last = "<<blk_x[blk_x.Size()-1]<<"\n";

//    if (false)
    {
        for (int ii = 0; ii < blk_x.BlockSize(2); ++ii)
        {
            //        blk_x.GetBlock(2)[ii] = std::fabs(blk_x.GetBlock(2)[ii]);
            if (blk_x.GetBlock(2)[ii] < 0.0)
            {
                blk_x.GetBlock(2)[ii]  = 0.0;
            }
//            if (darcy_system_.NumVDofs()>10000 and blk_x.GetBlock(2)[ii] > 1.0)
//            {
//                blk_x.GetBlock(2)[ii]  = 1.0;
//            }
        }
        //    const mfem::Vector S2 = darcy_system_.PWConstProject(blk_x.GetBlock(2));
        //    std::cout<< " after: min(S) max(S) = "<< S2.Min() << " " << S2.Max() <<"\n";
    }
}

void CoupledSolver::BuildHybridSystem(mfem::BlockOperator& op)
{

}

void CoupledSolver::BackTracking(const mfem::Vector& rhs,  double prev_resid_norm,
                                 mfem::Vector& x, mfem::Vector& dx)
{
    if (param_.num_backtrack == 0) return;

    x -= dx;
    mfem::BlockVector blk_x(x, true_blk_offsets_);
    mfem::BlockVector blk_dx(dx, true_blk_offsets_);

    const mfem::Vector S = darcy_system_.PWConstProject(blk_x.GetBlock(2));
    const mfem::Vector dS = darcy_system_.PWConstProject(blk_dx.GetBlock(2));

    auto violate = (mfem::Vector(1) = mfem::infinity());
    for (int i = 0; i < S.Size(); ++i)
    {
//        if (dS[i] < S[i]) violate = std::min(violate, S[i]);
        if (dS[i] > 1.0 - S[i])
            violate[0] = std::min(violate[0], (1.0 - S[i]) / dS[i] *.9);
    }

    violate = Min(violate = std::min(violate[0], 1.0), comm_);

    if (!myid_ && violate[0] < 1.0)
        std::cout<< "backtracking = " << violate[0] << "\n";

    blk_dx *= violate[0];

    x += blk_dx;

    resid_norm_ = violate[0] < 1e-2 ? 0.0 : ResidualNorm(x, rhs);
}

CoupledFAS::CoupledFAS(const Hierarchy& hierarchy,
                       const double dt,
                       const double weight,
                       const double density,
                       const mfem::Vector& S_prev,
                       FASParameters param)
    : FAS(hierarchy.GetComm(), param), hierarchy_(hierarchy)
{
    mfem::Vector S_prev_l(S_prev);

    for (int l = 0; l < param_.num_levels; ++l)
    {
        if (l > 0) { S_prev_l = hierarchy.Project(l - 1, S_prev_l); }

        auto& system_l = hierarchy.GetMatrix(l);

        const mfem::Vector S = system_l.PWConstProject(S_prev_l);

        auto& param_l = l ? (l < param.num_levels - 1 ? param.mid : param.coarse) : param.fine;
        solvers_[l].reset(new CoupledSolver(system_l, dt, weight, density, S, param_l));
//        solvers_[l]->SetPrintLevel(param_.cycle == V_CYCLE ? -1 : 1);

        if (l > 0)
        {
            rhs_[l].SetSize(system_l.NumTotalDofs() + system_l.NumVDofs());
            sol_[l].SetSize(system_l.NumTotalDofs() + system_l.NumVDofs());
            rhs_[l] = 0.0;
            sol_[l] = 0.0;
        }
        help_[l].SetSize(system_l.NumTotalDofs() + system_l.NumVDofs());
        help_[l] = 0.0;
    }
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
    hierarchy_.Restrict(level, blk_fine.GetBlock(2), blk_coarse.GetBlock(2));
}

void CoupledFAS::Interpolate(int level, const mfem::Vector& coarse, mfem::Vector& fine) const
{
    auto& solver_f = static_cast<CoupledSolver&>(*solvers_[level - 1]);
    auto& solver_c = static_cast<CoupledSolver&>(*solvers_[level]);
    mfem::BlockVector blk_fine(fine.GetData(), solver_f.BlockOffsets());
    mfem::BlockVector blk_coarse(coarse.GetData(), solver_c.BlockOffsets());
    hierarchy_.Interpolate(level, blk_coarse, blk_fine);
    hierarchy_.Interpolate(level, blk_coarse.GetBlock(2), blk_fine.GetBlock(2));
}

mfem::Vector CoupledFAS::ProjectS(int level, const mfem::Vector& x) const
{
    const auto& darcy_system = hierarchy_.GetMatrix(level);
    const auto& agg_vert = hierarchy_.GetAggVert(level);
    const mfem::Vector S = darcy_system.PWConstProject(x);

    mfem::Vector S_loc, S_coarse(agg_vert.NumRows());
    mfem::Array<int> verts;
    for (int i = 0; i < agg_vert.NumRows(); ++i)
    {
        GetTableRow(agg_vert, i, verts);
        S.GetSubVector(verts, S_loc);
        S_coarse[i] = S_loc.Max();
    }

    return hierarchy_.GetMatrix(level + 1).PWConstInterpolate(S_coarse);
}

void CoupledFAS::Project(int level, const mfem::Vector& fine, mfem::Vector& coarse) const
{
    auto& solver_f = static_cast<CoupledSolver&>(*solvers_[level]);
    auto& solver_c = static_cast<CoupledSolver&>(*solvers_[level + 1]);
    mfem::BlockVector blk_fine(fine.GetData(), solver_f.BlockOffsets());
    mfem::BlockVector blk_coarse(coarse.GetData(), solver_c.BlockOffsets());
    hierarchy_.Project(level, blk_fine, blk_coarse);
    hierarchy_.Project(level, blk_fine.GetBlock(2), blk_coarse.GetBlock(2));
//    blk_coarse.GetBlock(2) = ProjectS(level, blk_fine.GetBlock(2));
}

mfem::Vector TransportSolver::Residual(const mfem::Vector& x, const mfem::Vector& y)
{
    mfem::Vector out(x);
    auto FS = FractionalFlow(darcy_system_.PWConstProject(x));
    Adv_.Mult(1.0, FS, Ms_(0, 0), out);
    out -= y;
    return out;
}

void TransportSolver::Step(const mfem::Vector& rhs, mfem::Vector& x, mfem::Vector& dx)
{
    mfem::SparseMatrix df_ds = darcy_system_.GetPWConstProj();
    df_ds.ScaleRows(dFdS(darcy_system_.PWConstProject(x)));

    auto A = ParMult(Adv_, df_ds, starts_);
    GetDiag(*A) += Ms_;

    unique_ptr<mfem::HypreBoomerAMG> solver(BoomerAMG(*A));
    gmres_.SetPreconditioner(*solver);
    gmres_.SetOperator(*A);

    dx = 0.0;
    auto resid = Residual(x, rhs);
    gmres_.Mult(resid, dx);
//    if (!myid_) std::cout << "GMRES took " << gmres_.GetNumIterations() << " iterations\n";

    const mfem::Vector dS = darcy_system_.PWConstProject(dx);
    dx *= std::min(1.0, param_.diff_tol / mfem::ParNormlp(dS, mfem::infinity(), comm_));
    x -= dx;
}

void TwoPhaseHybrid::Init()
{
    offsets_[0] = 0;
    offsets_[1] = multiplier_d_td_->NumCols();
    offsets_[2] = offsets_[1] + mgL_.NumVDofs();

//    offsets_.SetSize(5);
//    offsets_[0] = 0;
//    offsets_[1] = mgL_.GetGraphSpace().VertexToEDof().NumNonZeroElems();
//    offsets_[2] = mgL_.NumVDofs();
//    offsets_[3] = mgL_.NumVDofs();
//    offsets_[4] = multiplier_d_td_->NumCols();
//    offsets_.PartialSum();

    op_.reset(new mfem::BlockOperator(offsets_));
    op_->owns_blocks = true;

    stage1_prec_.reset(new mfem::BlockLowerTriangularPreconditioner(offsets_));
    stage1_prec_->owns_blocks = true;

    for (int agg = 0; agg < nAggs_; ++agg)
    {
        mfem::DenseMatrix AinvDMinv = smoothg::Mult(Ainv_[agg], DMinv_[agg]);
        B01_[agg].Transpose(AinvDMinv);
        B00_[agg] = smoothg::Mult(B01_[agg], DMinv_[agg]);
        B00_[agg] -= Minv_ref_[agg];
        B00_[agg] *= -1.0;
    }

    solver_ = InitKrylovSolver(GMRES);

    solver_->SetAbsTol(1e-12);
    solver_->SetRelTol(1e-9);
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
        A00_el *= dt_density_;

        help = smoothg::Mult(C_[agg], B00_[agg]);
        help *= elem_scaling_inverse[agg];
        A01_el = smoothg::Mult(help, dMdS[agg]);

        help.Transpose();
        A10_el = smoothg::Mult(dTdsigma[agg], help);
        A10_el *= (dt_density_ * dt_density_);

        help = smoothg::Mult(dTdsigma[agg], B00_[agg]);
        A11_el = smoothg::Mult(help, dMdS[agg]);
        A11_el *= elem_scaling_inverse[agg];
        A11_el *= (dt_density_);

        A00.AddSubMatrix(local_mult, local_mult, A00_el);
        A01.AddSubMatrix(local_mult, local_vdof, A01_el);
        A10.AddSubMatrix(local_vdof, local_mult, A10_el);
        A11_tmp.AddSubMatrix(local_vdof, local_vdof, A11_el);
    }

    A00.Finalize();
    A01.Finalize();
    A10.Finalize();
    A11_tmp.Finalize();

//    auto pA11 = Copy(dTdS);
//    *pA11 *= -1.0;
//    GetDiag(*pA11) += A11_tmp;
    //    auto A11 = GetDiag(*pA11);

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

    auto A11_inv = new mfem::HypreSmoother(*pA11, mfem::HypreSmoother::l1Jacobi);

    stage1_prec_->SetDiagonalBlock(0, prec_.release());
    stage1_prec_->SetDiagonalBlock(1, A11_inv);
//    stage1_prec_->SetBlock(1, 0, pA10);

    mfem::BlockMatrix block_A(offsets_);
    block_A.SetBlock(0, 0, &A00);
    block_A.SetBlock(0, 1, &A01);
    block_A.SetBlock(1, 0, &A10);
    block_A.SetBlock(1, 1, A11.get());
    mono_mat_.reset(block_A.CreateMonolithic());
    monolithic_.reset(ToParMatrix(comm_, *mono_mat_));
    stage2_prec_.reset(new HypreILU(*monolithic_, 0));

    op_->SetBlock(0, 0, H_.release());
    op_->SetBlock(0, 1, pA01);
    op_->SetBlock(1, 0, pA10);
    op_->SetBlock(1, 1, pA11.release());

    prec_.reset(new TwoStageSolver(*stage1_prec_, *stage2_prec_, *op_));

    solver_->SetPreconditioner(*prec_);
    solver_->SetOperator(*op_);
    dynamic_cast<mfem::GMRESSolver*>(solver_.get())->SetKDim(100);

    dTdsigma_ = &dTdsigma;
    dMdS_ = &dMdS;



// *******************************************************************************************

//    const auto& agg_vdof = mgL_.GetGraphSpace().VertexToVDof();
//    const auto& agg_edof = mgL_.GetGraphSpace().VertexToEDof();

//    mfem::SparseMatrix M_proc(offsets_[1]);
//    mfem::SparseMatrix D_proc(mgL_.NumVDofs(), offsets_[1]);
//    mfem::SparseMatrix C_proc(num_multiplier_dofs_, offsets_[1]);
//    mfem::SparseMatrix dMdS_proc(offsets_[1], mgL_.NumVDofs());
//    mfem::SparseMatrix dTdsigma_proc(mgL_.NumVDofs(), offsets_[1]);

//    ess_redofs_.SetSize(offsets_[1], 0);

//    int redof_count = 0;

//    mfem::Array<int> redofs, mults, vdofs, edofs;
//    auto& M_el = dynamic_cast<const ElementMBuilder&>(mgL_.GetMBuilder()).GetElementMatrices();
//    for (int agg = 0; agg < nAggs_; ++agg)
//    {
//        redofs.SetSize(agg_edof.RowSize(agg));
//        std::iota(redofs.GetData(), redofs+redofs.Size(), redof_count);
//        GetTableRow(agg_vdof, agg, vdofs);
//        GetTableRow(agg_edof, agg, edofs);
//        GetTableRow(Agg_multiplier_, agg, mults);

//        mfem::DenseMatrix M_a = M_el[agg];
//        mfem::DenseMatrix D_a = smoothg::Mult(DMinv_[agg], M_a);

//        M_a *= 1.0 / elem_scaling_inverse[agg];
//        M_proc.AddSubMatrix(redofs, redofs, M_a);

//        D_proc.AddSubMatrix(vdofs, redofs, D_a);
//        dMdS_proc.AddSubMatrix(redofs, vdofs, dMdS[agg]);
//        dTdsigma_proc.AddSubMatrix(vdofs, redofs, dTdsigma[agg]);

//        mfem::DenseMatrix C_a;
//        Full(C_[agg], C_a);
//        C_proc.AddSubMatrix(mults, redofs, C_a);

//        for (int i = 0; i < edofs.Size(); i++)
//        {
//            ess_redofs_[redofs[i]] = ess_edofs_[edofs[i]];
//        }

//        redof_count += redofs.Size();
//    }

//    M_proc.Finalize();
//    D_proc.Finalize();
//    C_proc.Finalize();
//    dMdS_proc.Finalize();
//    dTdsigma_proc.Finalize();
//    M_proc.MoveDiagonalFirst();

//    mfem::SparseMatrix eliminated(C_proc.NumRows());
//    for (int i = 0; i < ess_true_multipliers_.Size(); ++i)
//    {
//        C_proc.EliminateRow(ess_true_multipliers_[i]);
//        eliminated.Add(ess_true_multipliers_[i], ess_true_multipliers_[i], 1.0);
//    }
//    eliminated.Finalize();


// *******************************************************************************************

//    auto M = ToParMatrix(comm_, M_proc);
//    auto D = ToParMatrix(comm_, D_proc);
//    auto C = ToParMatrix(comm_, C_proc);
//    auto pdMdS = ToParMatrix(comm_, dMdS_proc);
//    auto pdTdsigma = ToParMatrix(comm_, dTdsigma_proc);
//    auto DT = D->Transpose();
//    auto CT = C->Transpose();
//    auto dTdS_copy = Copy(dTdS);


//    mfem::Vector Md;
//    M->GetDiag(Md);
//    Md *= -1.0;
//    pdMdS->InvScaleRows(Md);
//    unique_ptr<mfem::HypreParMatrix> tmp11(mfem::ParMult(pdTdsigma, pdMdS));
//    pdMdS->ScaleRows(Md);
//    schur22_.reset(ParAdd(*dTdS_copy, *tmp11));

//    auto type22 = mfem::HypreSmoother::Type::l1Jacobi;
//    auto dTdS_inv = new mfem::HypreSmoother(*schur22_, type22);

//    op_->SetBlock(0, 0, M);
//    op_->SetBlock(0, 1, DT);
//    op_->SetBlock(0, 2, pdMdS);
//    op_->SetBlock(0, 3, CT);
//    op_->SetBlock(1, 0, D);
//    op_->SetBlock(2, 0, pdTdsigma);
//    op_->SetBlock(2, 2, dTdS_copy.release());
//    op_->SetBlock(3, 0, C);

//    solver_->SetOperator(*op_);

//    auto M_inv = new mfem::HypreDiagScale(*M);

//    DT->InvScaleRows(Md);
//    schur_.reset(mfem::ParMult(D, DT));
//    DT->ScaleRows(Md);
//    mfem::HypreBoomerAMG* schur_inv = BoomerAMG(*schur_);

////    CT->InvScaleRows(Md);
////    schur33_.reset(mfem::ParMult(C, CT));
////    CT->ScaleRows(Md);
////    auto schur33 = new mfem::HypreDiagScale(*schur33_);
//    mfem::IdentityOperator* schur33 = new mfem::IdentityOperator(C->NumRows());

//    stage1_prec_->SetBlock(0, 0, M_inv);
//    stage1_prec_->SetBlock(1, 1, schur_inv);
//    stage1_prec_->SetBlock(2, 2, dTdS_inv);
//    stage1_prec_->SetBlock(3, 3, schur33);

//    solver_->SetPreconditioner(*stage1_prec_);

// *******************************************************************************************

//    auto DT_proc = smoothg::Transpose(D_proc);
//    auto CT_proc = smoothg::Transpose(C_proc);
//    auto dTdS_proc = GetDiag(dTdS);

//    unique_ptr<mfem::BlockMatrix> op2_(new mfem::BlockMatrix(offsets_));

//    M_proc *= 1.0 / dt_density_;
//    DT_proc *= 1.0 / dt_density_;
//    dMdS_proc *= 1.0 / dt_density_;
//    D_proc *= dt_density_;
//    dTdsigma_proc *= dt_density_;
//    CT_proc *= 1.0 / dt_density_;
//    C_proc *= dt_density_;

//    op2_->SetBlock(0, 0, &M_proc);
//    op2_->SetBlock(0, 1, &DT_proc);
//    op2_->SetBlock(0, 2, &dMdS_proc);
//    op2_->SetBlock(0, 3, &CT_proc);
//    op2_->SetBlock(1, 0, &D_proc);
//    op2_->SetBlock(2, 0, &dTdsigma_proc);
//    op2_->SetBlock(2, 2, &dTdS_proc);
//    op2_->SetBlock(3, 0, &C_proc);
//    op2_->SetBlock(3, 3, &eliminated);
//    op3_.reset(op2_->CreateMonolithic());


//    std::ofstream mfile("full_hb_system.txt");
//    if (op3_->NumRows()<50000)
//        op3_->PrintMatlab(mfile);
//    solver3_.reset(new mfem::UMFPackSolver(*op3_));
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
        helper *= dt_density_;

        rhs.GetBlock(1).GetSubVector(local_vdof, sub_vec);
        B01_[agg].AddMult_a(1.0 / dt_density_, sub_vec, helper);

        local_rhs.SetSize(local_mult.Size());
        C_[agg].Mult(helper, local_rhs);
        out.AddElementVector(local_mult, local_rhs);

        local_rhs.SetSize(local_vdof.Size());
        (*dTdsigma_)[agg].Mult(helper, local_rhs);
        local_rhs *= dt_density_;
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

        local_sol0 /= dt_density_;
        local_sol1 /= (-1.0 * dt_density_);

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
    mfem::BlockVector rhs_hb = MakeHybridRHS(rhs);

    mfem::BlockVector sol_hb(offsets_);
//    sol_hb = 0.0;
//    rhs_hb.Randomize(1);

    solver_->Mult(rhs_hb, sol_hb);

    num_iterations_ = solver_->GetNumIterations();
    if (!myid_) std::cout << "          HB: GMRES took " << solver_->GetNumIterations()
                          << " iterations, residual = " << solver_->GetFinalNorm() << "\n";

//    if (solver_->GetNumIterations() == 18)
//    {
//        std::ofstream ofs("good_fine_hybrid_matrix.txt");
//        mono_mat_->PrintMatlab(ofs);
//        std::ofstream rhs_file("good_fine_hybrid_rhs.txt");
//        rhs_hb.Print(rhs_file, 1);
//        std::ofstream sol_file("good_fine_hybrid_sol.txt");
//        sol_hb.Print(sol_file, 1);
//    }
//    if (solver_->GetNumIterations() == 16)
//    {
//        std::ofstream ofs("bad_fine_hybrid_matrix.txt");
//        mono_mat_->PrintMatlab(ofs);
//        std::ofstream rhs_file("bad_fine_hybrid_rhs.txt");
//        rhs_hb.Print(rhs_file, 1);
//        std::ofstream sol_file("bad_fine_hybrid_sol.txt");
//        sol_hb.Print(sol_file, 1);
//    }
//    if (solver_->GetNumIterations() == 37)
//    {
//        std::ofstream ofs("good_coarse_hybrid_matrix.txt");
//        mono_mat_->PrintMatlab(ofs);
//        std::ofstream rhs_file("good_coarse_hybrid_rhs.txt");
//        rhs_hb.Print(rhs_file, 1);
//        std::ofstream sol_file("good_coarse_hybrid_sol.txt");
//        sol_hb.Print(sol_file, 1);
//    }
//    if (solver_->GetNumIterations() == 25)
//    {
//        std::ofstream ofs("bad_coarse_hybrid_matrix.txt");
//        mono_mat_->PrintMatlab(ofs);
//        std::ofstream rhs_file("bad_coarse_hybrid_rhs.txt");
//        rhs_hb.Print(rhs_file, 1);
//        std::ofstream sol_file("bad_coarse_hybrid_sol.txt");
//        sol_hb.Print(sol_file, 1);
//    }


    BackSubstitute(rhs, sol_hb, sol);

//    const auto& agg_edof = mgL_.GetGraphSpace().VertexToEDof();

//    mfem::BlockVector rhs_hb(offsets_);
//    rhs_hb.GetBlock(0) = 0.0;
//    rhs_hb.GetBlock(1) = rhs.GetBlock(1);
//    rhs_hb.GetBlock(2) = rhs.GetBlock(2);
//    rhs_hb.GetBlock(3) = 0.0;

//    int redof_count = 0;

//    mfem::Array<int> redofs, edofs;
//    mfem::Vector sub_vec;
//    for (int agg = 0; agg < nAggs_; ++agg)
//    {
//        redofs.SetSize(agg_edof.RowSize(agg));
//        std::iota(redofs.GetData(), redofs+redofs.Size(), redof_count);
//        GetTableRow(agg_edof, agg, edofs);
//        rhs.GetSubVector(edofs, sub_vec);
//        for (int i = 0; i < edofs.Size(); ++i)
//        {
//            if (edof_needs_averaging_[edofs[i]])
//            {
//                sub_vec[i] /= 2.0;
//            }
//        }
//        rhs_hb.SetSubVector(redofs, sub_vec);
//        redof_count += redofs.Size();
//    }

//    mfem::BlockVector sol_hb(offsets_);
//    solver3_->Mult(rhs_hb, sol_hb);

//    sol.GetBlock(0) = 0.0;
//    sol.GetBlock(1) = sol_hb.GetBlock(1);
//    sol.GetBlock(2) = sol_hb.GetBlock(2);

//    redof_count = 0;
//    for (int agg = 0; agg < nAggs_; ++agg)
//    {
//        redofs.SetSize(agg_edof.RowSize(agg));
//        std::iota(redofs.GetData(), redofs+redofs.Size(), redof_count);
//        GetTableRow(agg_edof, agg, edofs);
//        sol_hb.GetSubVector(redofs, sub_vec);
//        for (int i = 0; i < edofs.Size(); ++i)
//        {
//            if (edof_needs_averaging_[edofs[i]])
//            {
//                sub_vec[i] /= 2.0;
//            }
//        }
//        sol.AddElementVector(edofs, sub_vec);
//        redof_count += redofs.Size();
//    }
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

// case 3
mfem::Vector TotalMobility(const mfem::Vector& S)
{
    mfem::Vector LamS(S.Size());
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);
        double S_o = 1.0 - S_w;
        LamS(i)  = S_w * S_w / 1e-3 + S_o * S_o / 1e-2;
    }
    return LamS;
}

mfem::Vector dTMinv_dS(const mfem::Vector& S)
{
    mfem::Vector out(S.Size());
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);
        double S_o = 1.0 - S_w;
        out(i)  = 2.0 * S_w / 1e-3 - 2.0 * S_o / 1e-2;
        double Lam_S  = S_w * S_w / 1e-3 + S_o * S_o / 1e-2;
        out(i) = -1.0 * out(i) / (Lam_S * Lam_S);
    }
    return out;
}

mfem::Vector FractionalFlow(const mfem::Vector& S)
{
    mfem::Vector FS(S.Size());
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);
        double S_o = 1.0 - S_w;
        double Lam_S  = S_w * S_w / 1e-3 + S_o * S_o / 1e-2;
        FS(i) = S_w * S_w / 1e-3 / Lam_S;
    }
    return FS;
}

mfem::Vector dFdS(const mfem::Vector& S)
{
    mfem::Vector out(S.Size());
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);
        double S_o = 1.0 - S_w;
        double dLw_dS = 2.0 * S_w / 1e-3;
        double Lam_S  = S_w * S_w / 1e-3 + S_o * S_o / 1e-2;
        double dLam_dS = 2.0 * S_w / 1e-3 - 2.0 * S_o / 1e-2;
        out(i) = (dLw_dS * Lam_S - dLam_dS * S_w * S_w / 1e-3) / (Lam_S * Lam_S);
    }
    return out;
}


