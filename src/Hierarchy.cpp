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

/** @file

    @brief Implements Hierarchy class
*/

#include "Hierarchy.hpp"
#include "GraphCoarsen.hpp"
#include <iostream>
#include <fstream>

namespace smoothg
{


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

Hierarchy::Hierarchy(MixedMatrix mixed_system,
                     const UpscaleParameters& param,
                     const mfem::Array<int>* partitioning,
                     const mfem::Array<int>* ess_attr)
    : comm_(mixed_system.GetComm()),
      solvers_(param.max_levels),
      two_phase_solvers_(param.max_levels),
      setup_time_(0.0),
      ess_attr_(ess_attr),
      param_(param)
{
    mfem::StopWatch chrono;
    chrono.Start();

    MPI_Comm_rank(comm_, &myid_);

    upwind_fluxes_.reserve(param.max_levels - 1);

    {
        mfem::Vector trace_vec(mixed_system.NumEDofs());
        trace_vec = 1.0;
        upwind_fluxes_.push_back(BuildUpwindPattern(mixed_system.GetGraphSpace(), trace_vec));
    }

    edge_traces_.reserve(param.max_levels);
    const int num_edges = mixed_system.GetGraph().NumEdges();
    edge_traces_.emplace_back(num_edges, mfem::DenseMatrix(1));
    for (int edge = 0; edge < num_edges; ++edge)
    {
        edge_traces_[0][edge] = 1.0;
    }

    mixed_system.SetDs(mixed_system.GetD());
//    mixed_system.SetMs(SparseDiag(mixed_system.GetGraph().VertexWeight()));
    mixed_systems_.reserve(param.max_levels);
    mixed_systems_.push_back(std::move(mixed_system));
    if (ess_attr) { mixed_systems_.back().SetEssDofs(*ess_attr); }
    MakeSolver(0, param);

    agg_vert_.reserve(param.max_levels - 1);

    for (int level = 0; level < param.max_levels - 1; ++level)
    {
        Coarsen(level, param, level ? nullptr : partitioning);
        MakeSolver(level + 1, param);
        if (ess_attr) { mixed_systems_.back().SetEssDofs(*ess_attr); }
    }

    chrono.Stop();
    setup_time_ = chrono.RealTime();
}

void Hierarchy::Coarsen(int level, const UpscaleParameters& param,
                        const mfem::Array<int>* partitioning)
{
    MixedMatrix& mgL = GetMatrix(level);
    mgL.BuildM();

    GraphTopology topology;
    int coarse_factor = level ? 8 : param.coarse_factor;
    Graph coarse_graph = partitioning ? topology.Coarsen(mgL.GetGraph(), *partitioning) :
                         topology.Coarsen(mgL.GetGraph(), coarse_factor,
                                          param.num_iso_verts);

    DofAggregate dof_agg(topology, mgL.GetGraphSpace());

    LocalMixedGraphSpectralTargets localtargets(mgL, coarse_graph, dof_agg, param);
    auto vert_targets = localtargets.ComputeVertexTargets();
    edge_traces_.push_back(localtargets.ComputeEdgeTargets(vert_targets));

    GraphCoarsen coarsener(mgL, dof_agg, edge_traces_.back(),
                           vert_targets, std::move(coarse_graph));
    Pu_.push_back(coarsener.BuildPVertices());
    Psigma_.push_back(coarsener.BuildPEdges(param.coarse_components));
    Proj_sigma_.push_back(coarsener.BuildEdgeProjection());

    if (param.max_evects > 1 && param.add_Pvertices_pwc)
    {
        Ps_.push_back(MakeMinimalPvertices(Pu_[level], vert_targets));
    }
    mixed_systems_.push_back(coarsener.BuildCoarseMatrix(mgL, Pu_[level], GetPs(level)));

    std::unique_ptr<mfem::SparseMatrix> coarse_Ds(
            mfem::RAP(GetPs(level), mgL.GetDs(), Psigma_[level]));
//    std::unique_ptr<mfem::SparseMatrix> coarse_Ms(
//            mfem::RAP(GetPs(level), mgL.GetMs(), GetPs(level)));
    mixed_systems_[level+1].SetDs(*coarse_Ds);
//    mixed_systems_[level+1].SetMs(*coarse_Ms);

    agg_vert_.push_back(std::move(topology.Agg_vertex_));

//    upwind_fluxes_.push_back(ComputeMicroUpwindFlux(level + 1, dof_agg));
    upwind_fluxes_.push_back(ComputeMicroUpwindFlux2(level + 1));

    face_edge_.push_back(std::move(topology.face_edge_));

#ifdef SMOOTHG_DEBUG
    Debug_tests(level);
#endif
}

mfem::SparseMatrix Hierarchy::MakeMinimalPvertices(
        const mfem::SparseMatrix& Pvertices,
        const std::vector<mfem::DenseMatrix>& vert_targets)
{
    int dof_counter = 0;
    mfem::SparseMatrix select(Pvertices.NumCols(), vert_targets.size());
    for (unsigned int i = 0; i < vert_targets.size(); ++i)
    {
        select.Set(dof_counter, i, 1.0);
        dof_counter += vert_targets[i].NumCols();
    }
    select.Finalize();

    auto out = smoothg::Mult(Pvertices, select);
    if (Selects_.size())
    {
        auto fine_select_T = smoothg::Transpose(Selects_.back());
        out = smoothg::Mult(fine_select_T, out);
    }

    Selects_.push_back(select);
    return out;
}

void Hierarchy::MakeSolver(int level, const UpscaleParameters& param)
{
    if (level && param.hybridization) // Hybridization solver
    {
        SAAMGeParam* sa_param = level ? param.saamge_param : nullptr;
        solvers_[level].reset(new HybridSolver(GetMatrix(level), ess_attr_,
                                               param.rescale_iter, sa_param));

        MakeTwoPhaseHybridSolver(level);
    }
    else // L2-H1 block diagonal preconditioner
    {
        GetMatrix(level).BuildM();
        solvers_[level].reset(new BlockSolverFalse(GetMatrix(level), ess_attr_));
    }
}

void Hierarchy::Solve(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(level >= 0 && level < NumLevels());
    solvers_[level]->Solve(x, y);
}

mfem::BlockVector Hierarchy::Solve(int level, const mfem::BlockVector& x) const
{
    mfem::BlockVector y(BlockOffsets(level));
    y = 0.0;
    Solve(level, x, y);
    return y;
}

void Hierarchy::Solve(int level, const mfem::Vector& x, mfem::Vector& y) const
{
    assert(level >= 0 && level < NumLevels());
    solvers_[level]->Solve(x, y);
}

mfem::Vector Hierarchy::Solve(int level, const mfem::Vector& x) const
{
    mfem::Vector y(GetMatrix(level).NumVDofs());
    y = 0.0;
    Solve(level, x, y);
    return y;
}

void Hierarchy::Interpolate(int level, const mfem::Vector& x, mfem::Vector& y) const
{
    assert(level >= 1 && level < NumLevels());
    Pu_[level - 1].Mult(x, y);
}

mfem::Vector Hierarchy::Interpolate(int level, const mfem::Vector& x) const
{
    mfem::Vector fine_vect(GetMatrix(level - 1).NumVDofs());
    Interpolate(level, x, fine_vect);
    return fine_vect;
}

void Hierarchy::Interpolate(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(level >= 1 && level < NumLevels());
    Psigma_[level - 1].Mult(x.GetBlock(0), y.GetBlock(0));
    Pu_[level - 1].Mult(x.GetBlock(1), y.GetBlock(1));
}

mfem::BlockVector Hierarchy::Interpolate(int level, const mfem::BlockVector& x) const
{
    mfem::BlockVector fine_vect(BlockOffsets(level - 1));
    Interpolate(level, x, fine_vect);
    return fine_vect;
}

void Hierarchy::Restrict(int level, const mfem::Vector& x, mfem::Vector& y) const
{
    assert(level >= 0 && level < NumLevels() - 1);
    Pu_[level].MultTranspose(x, y);
}

mfem::Vector Hierarchy::Restrict(int level, const mfem::Vector& x) const
{
    mfem::Vector coarse_vect(GetMatrix(level + 1).NumVDofs());
    Restrict(level, x, coarse_vect);
    return coarse_vect;
}

void Hierarchy::Restrict(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(level >= 0 && level < NumLevels() - 1);
    Psigma_[level].MultTranspose(x.GetBlock(0), y.GetBlock(0));
    Pu_[level].MultTranspose(x.GetBlock(1), y.GetBlock(1));
}

mfem::BlockVector Hierarchy::Restrict(int level, const mfem::BlockVector& x) const
{
    mfem::BlockVector coarse_vect(BlockOffsets(level + 1));
    Restrict(level, x, coarse_vect);
    return coarse_vect;
}

void Hierarchy::Project(int level, const mfem::Vector& x, mfem::Vector& y) const
{
    Restrict(level, x, y);
//    Ps_[level].MultTranspose(x, y);
}

mfem::Vector Hierarchy::Project(int level, const mfem::Vector& x) const
{
    return Restrict(level, x);
//    mfem::Vector y(Ps_[level].NumCols());
//    y = 0.0;
//    Ps_[level].MultTranspose(x, y);
//    return y;
}

void Hierarchy::Project(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(level >= 0 && level < NumLevels() - 1);
    Proj_sigma_[level].Mult(x.GetBlock(0), y.GetBlock(0));
    Pu_[level].MultTranspose(x.GetBlock(1), y.GetBlock(1));
}

mfem::BlockVector Hierarchy::Project(int level, const mfem::BlockVector& x) const
{
    mfem::BlockVector coarse_vect(BlockOffsets(level + 1));
    Project(level, x, coarse_vect);
    return coarse_vect;
}

mfem::Vector Hierarchy::PWConstProject(int level, const mfem::Vector& x) const
{
    return GetMatrix(level).PWConstProject(x);
}

mfem::Vector Hierarchy::PWConstInterpolate(int level, const mfem::Vector& x) const
{
    return GetMatrix(level).PWConstInterpolate(x);
}

MixedMatrix& Hierarchy::GetMatrix(int level)
{
    assert(level >= 0 && level < NumLevels());
    return mixed_systems_[level];
}

const MixedMatrix& Hierarchy::GetMatrix(int level) const
{
    assert(level >= 0 && level < NumLevels());
    return mixed_systems_[level];
}

void Hierarchy::PrintInfo(std::ostream& out) const
{
    int num_procs;
    MPI_Comm_size(comm_, &num_procs);

    std::stringstream tout;
    {
        tout.precision(3);

        tout << "\n";

        if (num_procs > 1)
        {
            tout << "Processors: " << num_procs << "\n";
            tout << "---------------------\n";
        }

        tout << "\n";

        for (int i = 0; i < NumLevels(); ++i)
        {
            tout << "Level " << i << " Matrix\n";
            tout << "---------------------\n";
            tout << "M Size\t\t" << GetMatrix(i).GetGraphSpace().EDofToTrueEDof().N() << "\n";
            tout << "D Size\t\t" << GetMatrix(i).GetGraphSpace().VDofStarts().Last() << "\n";
            tout << "NonZeros:\t" << solvers_[i]->GetNNZ() << "\n";
            tout << "\n";

            if (i != 0)
            {
                tout << "Op Comp (level " << i - 1 << " to " << i
                     << "):\t" << OperatorComplexityAtLevel(i) << "\n";
                tout << "\n";
            }
        }

        tout << "Total Op Comp:\t"
             << OperatorComplexity(mixed_systems_.size() - 1) << "\n";
        tout << "\n";
    }
    if (myid_ == 0)
    {
        out << tout.str();
    }

    ShowSetupTime(out);
}

double Hierarchy::OperatorComplexity(int level) const
{
    assert(level < NumLevels());

    int nnz_all = 0;
    for (int i = 0; i < level + 1; ++i)
    {
        assert(solvers_[i]);
        nnz_all += solvers_[i]->GetNNZ();
    }

    int nnz_fine = solvers_[0]->GetNNZ();

    return nnz_all / (double) nnz_fine;
}

double Hierarchy::OperatorComplexityAtLevel(int level) const
{
    assert(level < NumLevels());

    if (level == 0)
        return 1.0;

    assert(solvers_[level - 1] && solvers_[level]);
    int nnz_coarse = solvers_[level]->GetNNZ();
    int nnz_fine = solvers_[level - 1]->GetNNZ();

    return 1.0 + nnz_coarse / (double) nnz_fine;
}

void Hierarchy::SetPrintLevel(int print_level)
{
    for (auto& solver : solvers_)
    {
        solver->SetPrintLevel(print_level);
    }
}

void Hierarchy::SetMaxIter(int max_num_iter)
{
    for (auto& solver : solvers_)
    {
        solver->SetMaxIter(max_num_iter);
    }
}

void Hierarchy::SetRelTol(double rtol)
{
    for (auto& solver : solvers_)
    {
        solver->SetRelTol(rtol);
    }
}

void Hierarchy::SetAbsTol(double atol)
{
    for (auto& solver : solvers_)
    {
        solver->SetAbsTol(atol);
    }
}

void Hierarchy::SetPrintLevel(int level, int print_level)
{
    solvers_[level]->SetPrintLevel(print_level);
}

void Hierarchy::SetMaxIter(int level, int max_num_iter)
{
    solvers_[level]->SetMaxIter(max_num_iter);
}

void Hierarchy::SetRelTol(int level, double rtol)
{
    solvers_[level]->SetRelTol(rtol);
}

void Hierarchy::SetAbsTol(int level, double atol)
{
    solvers_[level]->SetAbsTol(atol);
}

void Hierarchy::ShowSetupTime(std::ostream& out) const
{
    if (myid_ == 0)
    {
        out << "Hierarchy Setup Time:      " << setup_time_ << "\n\n";
    }
}

void Hierarchy::DumpDebug(const std::string& prefix) const
{
    int counter = 0;
    for (auto& ml : mixed_systems_)
    {
        std::stringstream s;
        s << prefix << "M" << counter << ".sparsematrix";
        std::ofstream outM(s.str().c_str());
        outM << std::scientific << std::setprecision(15);
        ml.GetM().Print(outM, 1);
        s.str("");
        s << prefix << "D" << counter++ << ".sparsematrix";
        std::ofstream outD(s.str().c_str());
        outD << std::scientific << std::setprecision(15);
        ml.GetD().Print(outD, 1);
    }

    counter = 0;
    for (auto& Psigma : Psigma_)
    {
        std::stringstream s;
        s << prefix << "Psigma" << counter << ".sparsematrix";
        std::ofstream outPsigma(s.str().c_str());
        outPsigma << std::scientific << std::setprecision(15);
        Psigma.Print(outPsigma, 1);
    }

    counter = 0;
    for (auto& Pu : Pu_)
    {
        std::stringstream s;
        s << prefix << "Pu" << counter++ << ".sparsematrix";
        std::ofstream outPu(s.str().c_str());
        outPu << std::scientific << std::setprecision(15);
        Pu.Print(outPu, 1);
    }
}

void Hierarchy::RescaleCoefficient(int level, const mfem::Vector& coeff)
{
    solvers_[level]->UpdateElemScaling(coeff);
}

int Hierarchy::NumVertices(int level) const
{
    return GetMatrix(level).GetGraph().NumVertices();
}

std::vector<int> Hierarchy::GetVertexSizes() const
{
    std::vector<int> out(NumLevels());
    for (int level = 0; level < NumLevels(); ++level)
    {
        out[level] = NumVertices(level);
    }
    return out;
}

void Hierarchy::Debug_tests(int level) const
{
    const double tol = 5e-10;
    mfem::Vector rand_vec(Psigma_[level].NumRows());
    rand_vec.Randomize(myid_);

    auto Check = [&](mfem::Vector & v, mfem::Vector & u, std::string op)
    {
        v -= u;
        double diff = mfem::ParNormlp(v, 2, comm_) / mfem::ParNormlp(u, 2, comm_);
        if (diff > tol && myid_ == 0)
        {
            std::cerr << "\nWarning: || " << op << " || = " << diff << " !!!\n";
        }
    };

    // Compute D * pi_sigma * random vector
    const mfem::SparseMatrix& D = GetMatrix(level).GetD();
    mfem::Vector Proj_sigma_rand = MatVec(Proj_sigma_[level], rand_vec);
    mfem::Vector pi_sigma_rand = MatVec(Psigma_[level], Proj_sigma_rand);
    mfem::Vector D_pi_sigma_rand = MatVec(D, pi_sigma_rand);

    // Compute pi_u * D * random vector
    mfem::Vector PuT_D_rand = Restrict(level, MatVec(D, rand_vec));
    mfem::Vector pi_u_D_rand = MatVec(Pu_[level], PuT_D_rand);
    Check(pi_u_D_rand, D_pi_sigma_rand, "pi_u * D - D * pi_sigma");

    rand_vec.SetSize(Proj_sigma_rand.Size());
    mfem::Vector Psigma_rand = MatVec(Psigma_[level], rand_vec);
    mfem::Vector Proj_sigma_Psigma_rand = MatVec(Proj_sigma_[level], Psigma_rand);
    Check(Proj_sigma_Psigma_rand, rand_vec, "Proj_sigma * Psigma - I");
}


// TODO: look at ComputeEdgeTargets to determine right direction in parallel
mfem::SparseMatrix
Hierarchy::ComputeMicroUpwindFlux(int level, const DofAggregate& dof_agg)
{
    auto& traces = GetTraces(level);
//    assert(GetQsigma(level - 1).NumRows() == traces.size());

    mfem::Array<int> edofs;
    mfem::Vector trace_vec(dof_agg.agg_edof_.NumCols());
    trace_vec = 0.0;
    for (unsigned int i = 0; i < traces.size(); ++i)
    {
        GetTableRow(dof_agg.face_edof_, i, edofs);
        trace_vec.SetSubVector(edofs, traces[i].GetData());
    }

    auto U = BuildUpwindPattern(GetMatrix(level - 1).GetGraphSpace(), trace_vec);
//    auto U = BuildUpwindPattern(GetMatrix(level - 1).GetGraphSpace(),
//                                upwind_fluxes_[level - 1], trace_vec);
//    auto vert_agg = smoothg::Transpose(GetAggVert(level - 1));
//    auto UPu = smoothg::Mult(U, vert_agg);
    auto UPu = smoothg::Mult(U, Ps_.size() ? Ps_[level - 1] : Pu_[level - 1]);
    UPu.ScaleRows(trace_vec);

    auto Q_pwc = smoothg::Transpose(GetMatrix(level).GetPWConstProj());
    Q_pwc.ScaleColumns(GetMatrix(level).GetVertexSizes());
    auto UPuQ_pwc = smoothg::Mult(UPu, Q_pwc);


//    auto U = BuildUpwindPattern(GetMatrix(level - 1).GetGraphSpace(), trace_vec);
//    auto UPu = smoothg::Mult(U, Ps_[level - 1]);

//    mfem::Vector ones(Psigma_[level - 1].NumCols());
//    ones = 1.0;
//    mfem::Vector Psigma_ones = MatVec(Psigma_[level - 1], ones);
//    mfem::SparseMatrix diag_Psigma_ones = SparseDiag(Psigma_ones);
//    UPu.ScaleRows(trace_vec);

    auto out = smoothg::Mult(GetQsigma(level - 1), UPuQ_pwc);
    out.SortColumnIndices();
//    out.Print();
    return out;
}


// TODO: look at ComputeEdgeTargets to determine right direction in parallel
mfem::SparseMatrix
Hierarchy::ComputeMicroUpwindFlux2(int level)
{
    mfem::Vector trace_vec(Psigma_[level - 1].NumCols());
    trace_vec = 1.0;
//    mfem::Vector trace_vec = MatVec(Psigma_[level - 1], ones);
    for (int i = level-1; i >= 0 ; --i)
    {
        trace_vec = MatVec(GetPsigma(i), trace_vec);
    }

    auto UPu = BuildUpwindPattern(GetMatrix(0).GetGraphSpace(), trace_vec);
//    auto U = BuildUpwindPattern(GetMatrix(level - 1).GetGraphSpace(),
//                                upwind_fluxes_[level - 1], trace_vec);
//    auto vert_agg = smoothg::Transpose(GetAggVert(level - 1));
//    auto UPu = smoothg::Mult(U, vert_agg);
//    auto UPu = smoothg::Mult(U, GetPs(level-1));
    for (int i = 0; i < level; ++i)
    {
        UPu = smoothg::Mult(UPu, GetPs(i));;
    }
    UPu.ScaleRows(trace_vec);

    auto Q_pwc = smoothg::Transpose(GetMatrix(level).GetPWConstProj());
    Q_pwc.ScaleColumns(GetMatrix(level).GetVertexSizes());
    auto out = smoothg::Mult(UPu, Q_pwc);

//    auto out = smoothg::Mult(GetQsigma(level - 1), UPuQ_pwc);
    for (int i = 0; i < level; ++i)
    {
        out = smoothg::Mult(GetQsigma(i), out);
    }
    out.SortColumnIndices();
//    out.Print();
    return out;
}

void Hierarchy::MakeTwoPhaseHybridSolver(int level)
{
    auto solver = make_unique<TwoPhaseHybrid>(GetMatrix(level), ess_attr_);
    two_phase_solvers_[level] = std::move(solver);
}


} // namespace smoothg
