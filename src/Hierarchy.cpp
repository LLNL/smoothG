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

    @brief Implements Upscale class
*/

#include "Hierarchy.hpp"
#include "GraphCoarsen.hpp"
#include <iostream>
#include <fstream>

namespace smoothg
{

Hierarchy::Hierarchy(MixedMatrix mixed_system,
                     const UpscaleParameters& param,
                     const mfem::Array<int>* partitioning,
                     const mfem::Array<int>* ess_attr)
    : comm_(mixed_system.GetComm()),
      solvers_(param.max_levels),
      setup_time_(0.0),
      ess_attr_(ess_attr),
      param_(param)
{
    mfem::StopWatch chrono;
    chrono.Start();

    MPI_Comm_rank(comm_, &myid_);

    mixed_systems_.reserve(param_.max_levels);
    mixed_systems_.push_back(std::move(mixed_system));
    MakeSolver(0);

    for (int level = 0; level < param_.max_levels - 1; ++level)
    {
        Coarsen(level, level ? nullptr : partitioning);
        MakeSolver(level + 1);
    }

    chrono.Stop();
    setup_time_ = chrono.RealTime();
}

void Hierarchy::Coarsen(int level, const mfem::Array<int>* partitioning)
{
    MixedMatrix& mgL = GetMatrix(level);
    mgL.BuildM();

    GraphTopology topology(mgL.GetGraph());
    Graph coarse_graph = partitioning ? topology.Coarsen(*partitioning)
                         : topology.Coarsen(param_.coarse_factor);

    DofAggregate dof_agg(topology, mgL.GetGraphSpace());

    std::vector<mfem::DenseMatrix> edge_traces;
    std::vector<mfem::DenseMatrix> vertex_targets;

    LocalMixedGraphSpectralTargets localtargets(mgL, coarse_graph, dof_agg, param_);
    localtargets.Compute(edge_traces, vertex_targets);

    GraphCoarsen graph_coarsen(mgL, dof_agg, edge_traces, vertex_targets, std::move(coarse_graph));

    Pu_.push_back(graph_coarsen.BuildPVertices());
    Psigma_.push_back(graph_coarsen.BuildPEdges(param_.coarse_components));
    Proj_sigma_.push_back(graph_coarsen.BuildEdgeProjection());

    mixed_systems_.push_back(graph_coarsen.BuildCoarseMatrix(mgL, Pu_[level]));

#ifdef SMOOTHG_DEBUG
    Debug_tests(level);
#endif
}

void Hierarchy::MakeSolver(int level)
{
    if (param_.hybridization) // Hybridization solver
    {
        SAAMGeParam* saamge_param = level ? param_.saamge_param : nullptr;
        solvers_[level] = make_unique<HybridSolver>(
                              GetMatrix(level), ess_attr_, 0, saamge_param);
    }
    else // L2-H1 block diagonal preconditioner
    {
        GetMatrix(level).BuildM();
        solvers_[level] = make_unique<MinresBlockSolverFalse>(GetMatrix(level),
                                                              ess_attr_);
    }
}

void Hierarchy::Mult(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(solvers_[level]);
    solvers_[level]->Solve(x, y);
}

mfem::BlockVector Hierarchy::Mult(int level, const mfem::BlockVector& x) const
{
    mfem::BlockVector y(GetMatrix(level).GetBlockOffsets());
    Solve(level, x, y);
    return y;
}

void Hierarchy::Solve(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(level >= 0 && level < NumLevels());
    solvers_[level]->Solve(x, y);
}

mfem::BlockVector Hierarchy::Solve(int level, const mfem::BlockVector& x) const
{
    mfem::BlockVector y(GetMatrix(level).GetBlockOffsets());
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
    mfem::Vector y(GetMatrix(level).GetD().NumRows());
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
    mfem::Vector fine_vect(GetMatrix(level - 1).GetD().NumRows());
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
    mfem::BlockVector fine_vect(GetMatrix(level - 1).GetBlockOffsets());
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
    mfem::Vector coarse_vect(GetMatrix(level + 1).GetD().NumRows());
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
    mfem::BlockVector coarse_vect(GetMatrix(level + 1).GetBlockOffsets());
    Restrict(level, x, coarse_vect);
    return coarse_vect;
}

void Hierarchy::Project(int level, const mfem::Vector& x, mfem::Vector& y) const
{
    Restrict(level, x, y);
}

mfem::Vector Hierarchy::Project(int level, const mfem::Vector& x) const
{
    return Restrict(level, x);
}

void Hierarchy::Project(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(level >= 0 && level < NumLevels() - 1);
    Proj_sigma_[level].Mult(x.GetBlock(0), y.GetBlock(0));
    Pu_[level].MultTranspose(x.GetBlock(1), y.GetBlock(1));
}

mfem::BlockVector Hierarchy::Project(int level, const mfem::BlockVector& x) const
{
    mfem::BlockVector coarse_vect(GetMatrix(level + 1).GetBlockOffsets());
    Project(level, x, coarse_vect);
    return coarse_vect;
}

mfem::Vector Hierarchy::PWConstProject(int level, const mfem::Vector& x) const
{
    mfem::Vector out(GetMatrix(level).GetGraph().NumVertices());
    GetMatrix(level).GetPWConstProj().Mult(x, out);
    return out;
}

mfem::Vector Hierarchy::PWConstInterpolate(int level, const mfem::Vector& x) const
{
    mfem::Vector scaled_x(x);
    RescaleVector(GetMatrix(level).GetVertexSizes(), scaled_x);
    mfem::Vector out(GetMatrix(level).GetD().NumRows());
    GetMatrix(level).GetPWConstProj().MultTranspose(scaled_x, out);
    return out;
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
            tout << "M Size\t\t" << GetMatrix(i).GetParallelD().N() << "\n";
            tout << "D Size\t\t" << GetMatrix(i).GetParallelD().M() << "\n";
            // tout << "+ Size\t\t" << GetMatrix(i).GlobalRows() << "\n";
            tout << "NonZeros:\t" << GetMatrix(i).GlobalNNZ() << "\n";
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

    int nnz_fine = solvers_[0] ? solvers_[0]->GetNNZ() : GetMatrix(0).GlobalNNZ();

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

void Hierarchy::ShowSetupTime(std::ostream& out) const
{
    if (myid_ == 0)
    {
        out << "\n";
        out << "Hierarchy Setup Time:      " << setup_time_ << "\n";
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
    if (!param_.hybridization)
    {
        GetMatrix(level).UpdateM(coeff);
        MakeSolver(level);
    }
    else
    {
        auto hybrid_solver = dynamic_cast<HybridSolver*>(solvers_[level].get());
        assert(hybrid_solver);
        hybrid_solver->UpdateAggScaling(coeff);
    }
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
    const mfem::SparseMatrix& D = GetMatrix(level).GetD();

    mfem::Vector random_vec(Proj_sigma_[level].Height());
    random_vec.Randomize();

    mfem::Vector Psigma_rand(Psigma_[level].Height());
    Psigma_[level].Mult(random_vec, Psigma_rand);
    mfem::Vector out(Proj_sigma_[level].Height());
    Proj_sigma_[level].Mult(Psigma_rand, out);

    out -= random_vec;
    double diff = out.Norml2();
    if (diff >= 1e-10)
    {
        std::cerr << "|| rand - Proj_sigma_ * Psigma_ * rand || = " << diff
                  << "\nEdge projection operator is not a projection!\n";
    }
    assert(diff < 1e-10);

    random_vec.SetSize(Psigma_[level].Height());
    random_vec.Randomize();

    // Compute D * pi_sigma * random vector
    mfem::Vector D_pi_sigma_rand(D.Height());
    {
        mfem::Vector Proj_sigma_rand(Proj_sigma_[level].Height());
        Proj_sigma_[level].Mult(random_vec, Proj_sigma_rand);
        mfem::Vector pi_sigma_rand(Psigma_[level].Height());
        Psigma_[level].Mult(Proj_sigma_rand, pi_sigma_rand);
        D.Mult(pi_sigma_rand, D_pi_sigma_rand);
    }

    // Compute pi_u * D * random vector
    mfem::Vector pi_u_D_rand(D.Height());
    {
        mfem::Vector D_rand(D.Height());
        D.Mult(random_vec, D_rand);
        mfem::Vector PuT_D_rand = Restrict(level, D_rand);
        Pu_[level].Mult(PuT_D_rand, pi_u_D_rand);
    }

    pi_u_D_rand -= D_pi_sigma_rand;
    diff = pi_u_D_rand.Norml2();
    if (diff >= 1e-10)
    {
        std::cerr << "|| pi_u * D * rand - D * pi_sigma * rand || = " << diff
                  << "\nCommutativity does not hold!\n";
    }
    assert(diff < 1e-10);
}

} // namespace smoothg
