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

enum SteppingScheme { IMPES = 1, SequentiallyImplicit, FullyImplcit };

struct EvolveParamenters
{
    double total_time = 10.0;    // Total time
    double dt = 1.0;   // Time step size
    int vis_step = 0;
    SteppingScheme scheme = IMPES;
};

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
    const EvolveParamenters& param_;
    const TwoPhase& problem_;
    Hierarchy& hierarchy_;

    mfem::BlockVector source_;
    unique_ptr<mfem::HypreParMatrix> Winv_D_;
    bool step_converged_;

    // update saturation block only
    void TransportStep(const double dt, mfem::BlockVector& x);
public:
    TwoPhaseSolver(const TwoPhase& problem, Hierarchy& hierarchy,
                   const EvolveParamenters& param);

    void Step(const double dt, mfem::BlockVector& x);
    mfem::BlockVector Solve(const mfem::BlockVector& init_val);
};

class CoupledStepSolver : public NonlinearSolver
{
    const MixedMatrix& darcy_system_;
    mfem::GMRESSolver gmres_;
    mfem::SparseMatrix dt_inv_;
    const mfem::HypreParMatrix& Winv_D_;
    const mfem::Array<int>& starts_;
    mfem::Array<int> block_offsets_;
    mfem::Array<int> true_block_offsets_;

    virtual void Mult(const mfem::Vector& x, mfem::Vector& Rx) override;
    virtual void IterationStep(const mfem::Vector& rhs, mfem::Vector& sol) override;
    mfem::Vector AssembleTrueVector(const mfem::Vector& v) const;
    const mfem::Array<int>& GetEssDofs() const { return darcy_system_.GetEssDofs(); }

    std::vector<mfem::DenseMatrix> Build_dMdS(const mfem::BlockVector& x);
public:
    CoupledStepSolver(const MixedMatrix& darcy_system,
                      const mfem::HypreParMatrix& Winv_D,
                      const mfem::Array<int>& starts,
                      const double dt);
};

class ImplicitTransportStepSolver : public NonlinearSolver
{
    mfem::Array<int> ess_dofs_;
    mfem::GMRESSolver gmres_;
    unique_ptr<mfem::HypreParMatrix> Winv_Adv_;
    mfem::SparseMatrix dt_inv_;
    const mfem::Array<int>& starts_;

    virtual void Mult(const mfem::Vector& x, mfem::Vector& Rx) override;
    virtual void IterationStep(const mfem::Vector& rhs, mfem::Vector& sol) override;
    mfem::Vector AssembleTrueVector(const mfem::Vector& v) const { return v; }
    const mfem::Array<int>& GetEssDofs() const { return ess_dofs_; }

public:
    ImplicitTransportStepSolver(const mfem::HypreParMatrix& Winv_D,
                                const mfem::SparseMatrix& upwind,
                                const mfem::Array<int>& starts,
                                const double dt)
        : NonlinearSolver(Winv_D.GetComm(), upwind.NumCols(), Newton, "", 1e-6),
          gmres_(Winv_D.GetComm()), Winv_Adv_(ParMult(Winv_D, upwind, starts)),
          dt_inv_(SparseIdentity(upwind.NumCols()) *= (1.0 / dt)), starts_(starts)
    {
        gmres_.SetMaxIter(200);
        gmres_.SetRelTol(1e-9);
    }
};

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
    bool use_metis = false;
    args.AddOption(&use_metis, "-ma", "--metis", "-nm", "--no-metis",
                   "Use Metis for partitioning (instead of geometric).");
    int well_height = 1;
    args.AddOption(&well_height, "-wh", "--well-height", "Well Height.");
    double inject_rate = 0.3;
    args.AddOption(&inject_rate, "-ir", "--inject-rate", "Injector rate.");
    double bhp = 175.0;
    args.AddOption(&bhp, "-bhp", "--bottom-hole-pressure", "Bottom Hole Pressure.");
    args.AddOption(&evolve_param.dt, "-dt", "--delta-t", "Time step.");
    args.AddOption(&evolve_param.total_time, "-time", "--total-time",
                   "Total time to step.");
    args.AddOption(&evolve_param.vis_step, "-vs", "--vis-step",
                   "Step size for visualization.");
    int scheme = 1;
    args.AddOption(&scheme, "-scheme", "--stepping-scheme",
                   "Time stepping: 1. IMPES, 2. sequentially implicit, 3. fully implicit. ");
    UpscaleParameters upscale_param;
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

    mfem::Array<int> ess_attr(dim == 3 ? 6 : 4);
    ess_attr = 1;

    // Setting up finite volume discretization problem
    TwoPhase problem(perm_file, dim, 5, slice, use_metis, ess_attr,
                     well_height, inject_rate, bhp);

    Hierarchy hierarchy(problem.GetFVGraph(true), upscale_param,
                        nullptr, &problem.EssentialAttribute());
    hierarchy.PrintInfo();

    // Fine scale transport based on fine flux
    TwoPhaseSolver solver(problem, hierarchy, evolve_param);

    mfem::BlockVector initial_value(problem.BlockOffsets());
    initial_value = 0.0;

    mfem::StopWatch chrono;
    chrono.Start();
    mfem::BlockVector S_fine = solver.Solve(initial_value);

    if (myid == 0)
    {
        std::cout << "Time stepping done in " << chrono.RealTime() << "s.\n";
    }

    double norm = mfem::ParNormlp(S_fine.GetBlock(2), 2, comm);
    if (myid == 0) { std::cout<<"|| S || = "<< norm <<"\n"; }

    return EXIT_SUCCESS;
}

mfem::SparseMatrix BuildUpwindPattern(const Graph& graph, const mfem::Vector& flux)
{
    const mfem::SparseMatrix& e_v = graph.EdgeToVertex();
    const mfem::SparseMatrix e_te_diag = GetDiag(graph.EdgeToTrueEdge());

    mfem::SparseMatrix upwind_pattern(graph.NumEdges(), graph.NumVertices());

    for (int i = 0; i < graph.NumEdges(); ++i)
    {
        if (e_v.RowSize(i) == 2) // edge is interior
        {
            const int upwind_vert = flux(i) > 0.0 ? 0 : 1;
            upwind_pattern.Set(i, e_v.GetRowColumns(i)[upwind_vert], 1.0);
        }
        else
        {
            assert(e_v.RowSize(i) == 1);
            const bool edge_is_owned = e_te_diag.RowSize(i);

            if ((flux(i) > 0.0 && edge_is_owned) || (flux(i) <= 0.0 && !edge_is_owned))
            {
                upwind_pattern.Set(i, e_v.GetRowColumns(i)[0], 1.0);
            }
        }
    }
    upwind_pattern.Finalize(); // TODO: use sparsity pattern of DT and update the values

    return upwind_pattern;
}

TwoPhaseSolver::TwoPhaseSolver(const TwoPhase& problem, Hierarchy& hierarchy,
                               const EvolveParamenters& param)
    : param_(param), problem_(problem), hierarchy_(hierarchy),
      source_(problem.BlockOffsets()), step_converged_(true)
{
    auto Winv = SparseIdentity(source_.BlockSize(2));
    Winv *= 1. / problem.CellVolume() / 0.3; // assume W is diagonal

    const MixedMatrix& system = hierarchy_.GetMatrix(0);
    const GraphSpace& space = system.GetGraphSpace();
    unique_ptr<mfem::HypreParMatrix> D(system.MakeParallelD(system.GetD()));
    auto tmp(ParMult(Winv, *D, space.VDofStarts()));
    Winv_D_.reset(mfem::ParMult(tmp.get(), &space.TrueEDofToEDof()));

    source_.GetBlock(0) = problem_.GetEdgeRHS();
    source_.GetBlock(1) = problem_.GetVertexRHS();
    Winv.Mult(problem_.GetVertexRHS(), source_.GetBlock(2));
}

mfem::BlockVector TwoPhaseSolver::Solve(const mfem::BlockVector& init_val)
{
    int myid;
    MPI_Comm_rank(hierarchy_.GetComm(), &myid);

    mfem::BlockVector x = init_val;

    mfem::socketstream sout;
    if (param_.vis_step) { problem_.VisSetup(sout, x.GetBlock(2), 0.0, 1.0, "Fine scale"); }

    double time = 0.0;
    double dt_real = std::min(param_.dt, param_.total_time - time) / 2.0;

    bool done = false;
    for (int step = 1; !done; step++)
    {
        mfem::BlockVector previous_x(x);
        dt_real = std::min(std::min(dt_real * 2.0, param_.total_time - time), param_.dt);
        step_converged_ = false;

        Step(dt_real, x);
        while (!step_converged_)
        {
            x = previous_x;
            dt_real /= 2.0;
            Step(dt_real, x);
        }

        time += dt_real;
        done = (time >= param_.total_time);

        if (myid == 0)
        {
            std::cout << "Time step " << step << ": step size = " << dt_real
                      << ", time = " << time << "\n";
        }
        if (param_.vis_step && (done || step % param_.vis_step == 0))
        {
            problem_.VisUpdate(sout, x.GetBlock(2));
        }
    }

    return x;
}

void TwoPhaseSolver::Step(const double dt, mfem::BlockVector& x)
{
    if (param_.scheme == FullyImplcit) // coupled: solve all unknowns together
    {
        auto& starts = hierarchy_.GetGraph(0).VertexStarts();
        CoupledStepSolver solver(hierarchy_.GetMatrix(0), *Winv_D_, starts, dt);
        solver.SetPrintLevel(-1);

        mfem::BlockVector rhs(source_);
        rhs.GetBlock(2).Add(1. / dt, x.GetBlock(2));
        solver.Solve(rhs, x);
        step_converged_ = solver.IsConverged();
    }
    else // sequential: solve for flux and pressure first, and then saturation
    {
        hierarchy_.RescaleCoefficient(0, TotalMobility(x.GetBlock(2)));
        mfem::BlockVector flow_rhs(source_.GetData(), hierarchy_.BlockOffsets(0));
        mfem::BlockVector flow_sol(x.GetData(), hierarchy_.BlockOffsets(0));
        hierarchy_.Solve(0, flow_rhs, flow_sol);

        TransportStep(dt, x);
    }
}

void TwoPhaseSolver::TransportStep(const double dt, mfem::BlockVector& x)
{
    const Graph& graph = hierarchy_.GetMatrix(0).GetGraph();
    mfem::SparseMatrix upwind = BuildUpwindPattern(graph, x.GetBlock(0));
    upwind.ScaleRows(x.GetBlock(0));

    if (param_.scheme == IMPES) // explcict: new_S = S + dt W^{-1} (b - Adv F(S))
    {
        mfem::Vector upwind_flux(x.GetBlock(0).Size());
        upwind_flux = 0.0;
        upwind.Mult(FractionalFlow(x.GetBlock(2)), upwind_flux);

        mfem::Vector dSdt(source_.GetBlock(2));
        Winv_D_->Mult(-1.0, upwind_flux, 1.0, dSdt);
        x.GetBlock(2).Add(dt, dSdt);
        step_converged_ = true;
    }
    else // implicit: new_S solves new_S = S + dt W^{-1} (b - Adv F(new_S))
    {
        auto& starts = hierarchy_.GetGraph(0).VertexStarts();
        ImplicitTransportStepSolver solver(*Winv_D_, upwind, starts, dt);
        solver.SetPrintLevel(-1);

        mfem::Vector rhs(source_.GetBlock(2));
        rhs.Add(1. / dt, x.GetBlock(2));
        solver.Solve(rhs, x.GetBlock(2));
        step_converged_ = solver.IsConverged();
    }
}

CoupledStepSolver::CoupledStepSolver(const MixedMatrix& darcy_system,
                                     const mfem::HypreParMatrix& Winv_D,
                                     const mfem::Array<int>& starts,
                                     const double dt)
    : NonlinearSolver(Winv_D.GetComm(), 0, Newton, "", 1e-6),
      darcy_system_(darcy_system), gmres_(Winv_D.GetComm()),
      dt_inv_(SparseIdentity(Winv_D.NumRows()) *= (1.0 / dt)), Winv_D_(Winv_D),
      starts_(starts), block_offsets_(4), true_block_offsets_(4)
{
    block_offsets_[0] = 0;
    block_offsets_[1] = darcy_system.NumEDofs();
    block_offsets_[2] = block_offsets_[1] + darcy_system.NumVDofs();
    block_offsets_[3] = block_offsets_[2] + darcy_system.GetGraph().NumVertices();

    true_block_offsets_[0] = 0;
    true_block_offsets_[1] = darcy_system.GetGraphSpace().EDofToTrueEDof().NumCols();
    true_block_offsets_[2] = true_block_offsets_[1] + darcy_system.NumVDofs();
    true_block_offsets_[3] = true_block_offsets_[2] + darcy_system.GetGraph().NumVertices();

    NonlinearSolver::size_ = block_offsets_[3];
    residual_.SetSize(size_);

    gmres_.SetMaxIter(200);
    gmres_.SetRelTol(1e-9);
}

mfem::Vector CoupledStepSolver::AssembleTrueVector(const mfem::Vector& v) const
{
    mfem::Vector true_v(true_block_offsets_.Last());
    mfem::BlockVector blk_v(v.GetData(), block_offsets_);
    mfem::BlockVector blk_true_v(true_v.GetData(), true_block_offsets_);

    auto& truedof_dof = darcy_system_.GetGraphSpace().TrueEDofToEDof();
    truedof_dof.Mult(blk_v.GetBlock(0), blk_true_v.GetBlock(0));
    blk_true_v.GetBlock(1) = blk_v.GetBlock(1);
    blk_true_v.GetBlock(2) = blk_v.GetBlock(2);

    return true_v;
}

void CoupledStepSolver::Mult(const mfem::Vector& x, mfem::Vector& Rx)
{
    mfem::BlockVector blk_x(x.GetData(), block_offsets_);
    mfem::BlockVector blk_Rx(Rx.GetData(), block_offsets_);

    mfem::BlockVector darcy_x(x.GetData(), darcy_system_.BlockOffsets());
    mfem::BlockVector darcy_Rx(Rx.GetData(), darcy_system_.BlockOffsets());
    darcy_system_.Mult(TotalMobility(blk_x.GetBlock(2)), darcy_x, darcy_Rx);

    blk_Rx.GetBlock(2) = blk_x.GetBlock(2);

    auto upwind = BuildUpwindPattern(darcy_system_.GetGraph(), blk_x.GetBlock(0));
    upwind.ScaleRows(blk_x.GetBlock(0));
    auto Winv_Adv = ParMult(Winv_D_, upwind, starts_);
    Winv_Adv->Mult(1.0, FractionalFlow(blk_x.GetBlock(2)), dt_inv_(0,0), blk_Rx.GetBlock(2));
}

void CoupledStepSolver::IterationStep(const mfem::Vector& rhs, mfem::Vector& sol)
{
    mfem::BlockVector blk_sol(sol.GetData(), block_offsets_);

    const GraphSpace& space = darcy_system_.GetGraphSpace();
    auto& vert_edof = space.VertexToEDof();

    mfem::Vector total_mobility = TotalMobility(blk_sol.GetBlock(2));
    auto M_proc = darcy_system_.GetMBuilder().BuildAssembledM(total_mobility);
    mfem::SparseMatrix D_proc(darcy_system_.GetD());

    std::vector<mfem::DenseMatrix> local_dMdS = Build_dMdS(blk_sol);
    mfem::Array<int> local_edofs, local_vert(1);
    mfem::SparseMatrix dMdS_proc(vert_edof.NumCols(), vert_edof.NumRows());
    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        local_vert[0] = i;
        dMdS_proc.AddSubMatrix(local_edofs, local_vert, local_dMdS[i]);
    }
    dMdS_proc.Finalize();

    auto upwind_pattern = BuildUpwindPattern(space.GetGraph(), blk_sol.GetBlock(0));
    mfem::Vector pattern_FS(blk_sol.BlockSize(0));
    upwind_pattern.Mult(FractionalFlow(blk_sol.GetBlock(2)), pattern_FS);

    upwind_pattern.ScaleRows(blk_sol.GetBlock(0));
    upwind_pattern.ScaleColumns(dFdS(blk_sol.GetBlock(2)));

    for (int mm = 0; mm < GetEssDofs().Size(); ++mm)
    {
        if (GetEssDofs()[mm])
        {
            M_proc.EliminateRowCol(mm); // assume essential data = 0
            dMdS_proc.EliminateRow(mm);
            pattern_FS[mm] = 0.0;
        }
    }
    if (GetEssDofs().Size()) { D_proc.EliminateCols(GetEssDofs()); }

    unique_ptr<mfem::HypreParMatrix> M(darcy_system_.MakeParallelM(M_proc));
    unique_ptr<mfem::HypreParMatrix> D(darcy_system_.MakeParallelD(D_proc));
    unique_ptr<mfem::HypreParMatrix> DT(D->Transpose());

    auto dMdS = ParMult(space.TrueEDofToEDof(), dMdS_proc, starts_);

    auto U_FS = Copy(space.EDofToTrueEDof());
    U_FS->ScaleRows(pattern_FS);
    unique_ptr<mfem::HypreParMatrix> dTdsigma(mfem::ParMult(&Winv_D_, U_FS.get()));

    auto dTdS = ParMult(Winv_D_, upwind_pattern, starts_);
    GetDiag(*dTdS) += dt_inv_;

    mfem::BlockOperator op(true_block_offsets_);
    op.SetBlock(0, 0, M.get());
    op.SetBlock(0, 1, DT.get());
    op.SetBlock(1, 0, D.get());
    op.SetBlock(0, 2, dMdS.get());
    op.SetBlock(2, 0, dTdsigma.get());
    op.SetBlock(2, 2, dTdS.get());

    mfem::Vector Md;
    M->GetDiag(Md);
    DT->InvScaleRows(Md);
    unique_ptr<mfem::HypreParMatrix> schur(mfem::ParMult(D.get(), DT.get()));
    (*schur) *= -1.0;
    DT->ScaleRows(Md);

    mfem::BlockDiagonalPreconditioner prec(true_block_offsets_);
    prec.SetDiagonalBlock(0, new mfem::HypreDiagScale(*M));
    prec.SetDiagonalBlock(1, BoomerAMG(*schur));
    prec.SetDiagonalBlock(2, BoomerAMG(*dTdS));
    prec.owns_blocks = true;

    gmres_.SetPrintLevel(-1);
    gmres_.SetOperator(op);
    gmres_.SetPreconditioner(prec);

    auto true_resid = AssembleTrueVector(residual_);

    mfem::BlockVector true_delta_sol(true_block_offsets_);
    true_delta_sol = 0.0;
    gmres_.Mult(true_resid, true_delta_sol);

    mfem::BlockVector delta_sol(block_offsets_);
    auto& dof_truedof = darcy_system_.GetGraphSpace().EDofToTrueEDof();
    dof_truedof.Mult(true_delta_sol.GetBlock(0), delta_sol.GetBlock(0));
    delta_sol.GetBlock(1) = true_delta_sol.GetBlock(1);
    delta_sol.GetBlock(2) = true_delta_sol.GetBlock(2);

    sol -= delta_sol;
}

std::vector<mfem::DenseMatrix> CoupledStepSolver::Build_dMdS(const mfem::BlockVector& x)
{
    // TODO: saturation is only 1 dof per cell
    auto& vert_edof = darcy_system_.GetGraphSpace().VertexToEDof();
    auto& vert_vdof = darcy_system_.GetGraphSpace().VertexToVDof();

    auto& MB = dynamic_cast<const ElementMBuilder&>(darcy_system_.GetMBuilder());
    auto& M_el = MB.GetElementMatrices();

    auto& proj_pwc = const_cast<mfem::SparseMatrix&>(darcy_system_.GetPWConstProj());

    std::vector<mfem::DenseMatrix> out(M_el.size());
    mfem::Array<int> local_edofs, local_vdofs, vert(1);
    mfem::Vector sigma_loc, Msigma_vec;
    mfem::DenseMatrix proj_pwc_loc;

    mfem::Vector dTMinv_dS_vec = dTMinv_dS(x.GetBlock(2));

    for (int i = 0; i < vert_edof.NumRows(); ++i)
    {
        GetTableRow(vert_edof, i, local_edofs);
        GetTableRow(vert_vdof, i, local_vdofs);
        vert[0] = i;

        x.GetSubVector(local_edofs, sigma_loc);
        Msigma_vec.SetSize(local_edofs.Size());
        M_el[i].Mult(sigma_loc, Msigma_vec);
        mfem::DenseMatrix Msigma_loc(Msigma_vec.GetData(), M_el[i].Size(), 1);

        proj_pwc_loc.SetSize(1, local_vdofs.Size());
        proj_pwc_loc = 0.0;
        proj_pwc.GetSubMatrix(vert, local_vdofs, proj_pwc_loc);
        proj_pwc_loc *= dTMinv_dS_vec[i];

        out[i].SetSize(local_edofs.Size(), local_vdofs.Size());
        mfem::Mult(Msigma_loc, proj_pwc_loc, out[i]);
    }

    return out;
}

void ImplicitTransportStepSolver::Mult(const mfem::Vector& x, mfem::Vector& Rx)
{
    Rx = x;
    Winv_Adv_->Mult(1.0, FractionalFlow(x), dt_inv_(0,0), Rx);
}

void ImplicitTransportStepSolver::IterationStep(const mfem::Vector& rhs, mfem::Vector& sol)
{
    auto A = ParMult(*Winv_Adv_, SparseDiag(dFdS(sol)), starts_);
    GetDiag(*A) += dt_inv_;

    mfem::HypreBoomerAMG solver(*A);
    solver.SetPrintLevel(-1);
    gmres_.SetOperator(*A);
    gmres_.SetPreconditioner(solver);

    mfem::Vector delta_sol(rhs.Size());
    delta_sol = 0.0;
    gmres_.Mult(residual_, delta_sol);
    sol -= delta_sol;
}

mfem::Vector TotalMobility(const mfem::Vector& S)
{
    mfem::Vector LamS(S.Size());
    LamS = 1000.;
    return LamS;
}

mfem::Vector dTMinv_dS(const mfem::Vector& S)
{
    mfem::Vector out(S.Size());
    out = 0.0;
    return out;
}

mfem::Vector FractionalFlow(const mfem::Vector& S)
{
    mfem::Vector FS(S);
    return FS;
}

mfem::Vector dFdS(const mfem::Vector& S)
{
    mfem::Vector out(S.Size());
    out = 1.0;
    return out;
}

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
