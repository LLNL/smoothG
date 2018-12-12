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

#include "Upscale.hpp"
#include <iostream>
#include <fstream>

namespace smoothg
{

Upscale::Upscale(const Graph& graph,
                 const UpscaleParameters& param,
                 const mfem::Array<int>* partitioning,
                 const mfem::Array<int>* ess_attr,
                 const mfem::SparseMatrix& w_block)
    : Operator(graph.NumVertices()), comm_(graph.GetComm()), setup_time_(0.0),
      ess_attr_(ess_attr), param_(param)
{
    mfem::StopWatch chrono;
    chrono.Start();

    mixed_laplacians_.reserve(param_.max_levels);
    mixed_laplacians_.emplace_back(graph, w_block);
    Init(partitioning);

    chrono.Stop();
    setup_time_ += chrono.RealTime();
}

void Upscale::Init(const mfem::Array<int>* partitioning)
{
    MPI_Comm_rank(comm_, &myid_);

    solver_.resize(param_.max_levels);
    rhs_.resize(param_.max_levels);
    sol_.resize(param_.max_levels);

    // coarser levels: matrices
    for (int level = 0; level < param_.max_levels - 1; ++level)
    {
        mixed_laplacians_.back().BuildM();
        coarsener_.emplace_back(make_unique<SpectralAMG_MGL_Coarsener>(param_));
        if (level == 0)
        {
            mixed_laplacians_.push_back(
                coarsener_[level]->Coarsen(GetMatrix(level), partitioning));
        }
        else
        {
            mixed_laplacians_.push_back(coarsener_[level]->Coarsen(GetMatrix(level)));
        }
    }

    // make solver on each level
    for (int level = 0; level < param_.max_levels; ++level)
    {
        MakeSolver(level);
    }
}

void Upscale::MakeSolver(int level)
{
    mfem::SparseMatrix& Dref = GetMatrix(level).GetD();
    mfem::Array<int> marker(Dref.Width());
    marker = 0;
    if (GetMatrix(level).GetGraph().HasBoundary())
    {
        BooleanMult(GetMatrix(level).EDofToBdrAtt(), *ess_attr_, marker);
        marker.SetSize(Dref.Width());
    }

    if (param_.hybridization) // Hybridization solver
    {
        SAAMGeParam* saamge_param = level > 0 ? param_.saamge_param : nullptr;
        solver_[level] = make_unique<HybridSolver>(
                             comm_, GetMatrix(level), &marker, 0, saamge_param);
    }
    else // L2-H1 block diagonal preconditioner
    {
        GetMatrix(level).BuildM();
        mfem::SparseMatrix& Mref = GetMatrix(level).GetM();
        for (int mm = 0; mm < marker.Size(); ++mm)
        {
            // Assume M diagonal, no ess data
            if (marker[mm])
                Mref.EliminateRowCol(mm, true);
        }
        if (marker.Size())
        {
            Dref.EliminateCols(marker);
        }
        solver_[level] = make_unique<MinresBlockSolverFalse>(comm_, GetMatrix(level));
    }
    MakeVectors(level);
}

void Upscale::Mult(int level, const mfem::Vector& x, mfem::Vector& y) const
{
    // restrict right-hand-side x
    rhs_[0]->GetBlock(1) = x;
    for (int i = 0; i < level; ++i)
    {
        coarsener_[i]->Restrict(rhs_[i]->GetBlock(1), rhs_[i + 1]->GetBlock(1));
    }

    // solve
    if (level > 0)
    {
        rhs_[level]->GetBlock(1) *= -1.0;
    }
    solver_[level]->Solve(rhs_[level]->GetBlock(1), sol_[level]->GetBlock(1));
    if (level == 0)
    {
        sol_[level]->GetBlock(1) *= -1.0;
    }
    // orthogonalize at coarse level, every level, or fine level?

    // interpolate solution
    for (int i = level - 1; i >= 0; --i)
    {
        coarsener_[i]->Interpolate(sol_[i + 1]->GetBlock(1), sol_[i]->GetBlock(1));
    }
    y = sol_[0]->GetBlock(1);
    Orthogonalize(0, y);
}

void Upscale::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    Mult(1, x, y);
}

void Upscale::Solve(int level, const mfem::Vector& x, mfem::Vector& y) const
{
    Mult(level, x, y);
}

mfem::Vector Upscale::Solve(int level, const mfem::Vector& x) const
{
    mfem::Vector y(x.Size());

    Solve(level, x, y);

    return y;
}

void Upscale::Solve(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    MFEM_ASSERT(
        rhs_[0], "Multilevel vectors not built, probably because MakeVectors() not called!");

    // restrict right-hand-side x
    *rhs_[0] = x;
    for (int i = 0; i < level; ++i)
    {
        coarsener_[i]->Restrict(*rhs_[i], * rhs_[i + 1]);
    }

    // solve
    rhs_[level]->GetBlock(1) *= -1.0; // for reasons I do not fully understand

    solver_[level]->Solve(*rhs_[level], *sol_[level]);

    // orthogonalize at coarse level, every level, or fine level?
    Orthogonalize(level, sol_[level]->GetBlock(1));

    // interpolate solution
    for (int i = level - 1; i >= 0; --i)
    {
        coarsener_[i]->Interpolate(*sol_[i + 1], *sol_[i]);
    }
    y = *sol_[0];
}

mfem::BlockVector Upscale::Solve(int level, const mfem::BlockVector& x) const
{
    mfem::BlockVector y(GetBlockVector(0));

    Solve(level, x, y);

    return y;
}

void Upscale::SolveAtLevel(int level, const mfem::Vector& x, mfem::Vector& y) const
{
    assert(solver_[level]);

    solver_[level]->Solve(x, y);
    y *= -1.0; // ????
    Orthogonalize(level, y);
}

mfem::Vector Upscale::SolveAtLevel(int level, const mfem::Vector& x) const
{
    mfem::Vector coarse_vect = GetVector(1);
    SolveAtLevel(level, x, coarse_vect);

    return coarse_vect;
}

void Upscale::SolveAtLevel(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(solver_[level]);

    solver_[level]->Solve(x, y);
    y *= -1.0;
    Orthogonalize(level, y); // TODO: temporary literal 1!
}

mfem::BlockVector Upscale::SolveAtLevel(int level, const mfem::BlockVector& x) const
{
    mfem::BlockVector coarse_vect(GetBlockVector(1));
    SolveAtLevel(level, x, coarse_vect);

    return coarse_vect;
}

void Upscale::Interpolate(int level, const mfem::Vector& x, mfem::Vector& y) const
{
    assert(coarsener_[level - 1]);
    coarsener_[level - 1]->Interpolate(x, y);
}

mfem::Vector Upscale::Interpolate(int level, const mfem::Vector& x) const
{
    mfem::Vector fine_vect = GetVector(level - 1);

    Interpolate(level, x, fine_vect);

    return fine_vect;
}

void Upscale::Interpolate(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(coarsener_[level - 1]);

    coarsener_[level - 1]->Interpolate(x, y);
}

mfem::BlockVector Upscale::Interpolate(int level, const mfem::BlockVector& x) const
{
    mfem::BlockVector fine_vect(GetBlockVector(level - 1));

    Interpolate(level, x, fine_vect);

    return fine_vect;
}

void Upscale::Restrict(int level, const mfem::Vector& x, mfem::Vector& y) const
{
    assert(coarsener_[level - 1]);

    coarsener_[level - 1]->Restrict(x, y);
}

mfem::Vector Upscale::Restrict(int level, const mfem::Vector& x) const
{
    mfem::Vector coarse_vect = GetVector(level);
    Restrict(level, x, coarse_vect);

    return coarse_vect;
}

void Upscale::Restrict(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(coarsener_[level - 1]);

    coarsener_[level - 1]->Restrict(x, y);
}

mfem::BlockVector Upscale::Restrict(int level, const mfem::BlockVector& x) const
{
    mfem::BlockVector coarse_vect(GetBlockVector(level));
    Restrict(level, x, coarse_vect);

    return coarse_vect;
}

void Upscale::BlockOffsets(int level, mfem::Array<int>& offsets) const
{
    GetMatrix(level).GetBlockOffsets().Copy(offsets);
}

void Upscale::TrueBlockOffsets(int level, mfem::Array<int>& offsets) const
{
    GetMatrix(level).GetBlockTrueOffsets().Copy(offsets);
}

void Upscale::Orthogonalize(int level, mfem::BlockVector& vect) const
{
    Orthogonalize(level, vect.GetBlock(1));
}

void Upscale::Orthogonalize(int level, mfem::Vector& vect) const
{
    const mfem::Vector& coarse_constant_rep = GetConstantRep(level);
    double local_dot = (vect * coarse_constant_rep);
    double global_dot;
    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm_);

    double local_scale = (coarse_constant_rep * coarse_constant_rep);
    double global_scale;
    MPI_Allreduce(&local_scale, &global_scale, 1, MPI_DOUBLE, MPI_SUM, comm_);

    vect.Add(-global_dot / global_scale, coarse_constant_rep);
}

mfem::Vector Upscale::GetVector(int level) const
{
    const auto& offsets = GetMatrix(level).GetBlockOffsets();
    const int vsize = offsets[2] - offsets[1];

    return mfem::Vector(vsize);
}

mfem::BlockVector Upscale::GetBlockVector(int level) const
{
    const auto& offsets = GetMatrix(level).GetBlockOffsets();

    return mfem::BlockVector(offsets);
}

mfem::BlockVector Upscale::GetTrueBlockVector(int level) const
{
    const auto& offsets = GetMatrix(level).GetBlockTrueOffsets();

    return mfem::BlockVector(offsets);
}

MixedMatrix& Upscale::GetMatrix(int level)
{
    assert(level >= 0 && level < static_cast<int>(mixed_laplacians_.size()));
    return mixed_laplacians_[level];
}

const MixedMatrix& Upscale::GetMatrix(int level) const
{
    assert(level >= 0 && level < static_cast<int>(mixed_laplacians_.size()));
    return mixed_laplacians_[level];
}

mfem::Vector Upscale::PWConstProject(int level, const mfem::Vector& x) const
{
    mfem::Vector out(GetMatrix(level).GetGraph().NumVertices());
    GetMatrix(level).GetPWConstProj().Mult(x, out);
    return out;
}

mfem::Vector Upscale::PWConstInterpolate(int level, const mfem::Vector& x) const
{
    mfem::Vector out(GetMatrix(level).GetD().NumRows());
    GetMatrix(level).GetPWConstInterp().Mult(x, out);
    return out;
}

void Upscale::PrintInfo(std::ostream& out) const
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

        for (unsigned int i = 0; i < mixed_laplacians_.size(); ++i)
        {
            tout << "Level " << i << " Matrix\n";
            tout << "---------------------\n";
            tout << "M Size\t\t" << GetMatrix(i).GetParallelD().N() << "\n";
            tout << "D Size\t\t" << GetMatrix(i).GetParallelD().M() << "\n";
            // tout << "+ Size\t\t" << GetMatrix(i).GlobalRows() << "\n";
            tout << "NonZeros:\t" << GetMatrix(i).GlobalNNZ() << "\n";
            tout << "\n";

            if (i != 0 && solver_[i] && solver_[0])
            {
                double op_comp = 1.0 + (solver_[i]->GetNNZ() / (double) solver_[0]->GetNNZ());

                tout << "Op Comp:\t" << op_comp << "\n";
                tout << "\n";
            }
        }
    }
    if (myid_ == 0)
    {
        out << tout.str();
    }
}

/// @todo multilevel this implementation (relatively easy)
double Upscale::OperatorComplexity() const
{
    assert(solver_[1]);

    int nnz_coarse = solver_[1]->GetNNZ();
    int nnz_fine;

    if (solver_[0])
    {
        nnz_fine = solver_[0]->GetNNZ();
    }
    else
    {
        nnz_fine = GetMatrix(0).GlobalNNZ();
    }


    double op_comp = 1.0 + (nnz_coarse / (double) nnz_fine);

    return op_comp;
}

void Upscale::SetPrintLevel(int print_level)
{
    for (auto& solver : solver_)
    {
        if (solver)
            solver->SetPrintLevel(print_level);
    }
}

void Upscale::SetMaxIter(int max_num_iter)
{
    for (auto& solver : solver_)
    {
        if (solver)
            solver->SetMaxIter(max_num_iter);
    }
}

void Upscale::SetRelTol(double rtol)
{
    for (auto& solver : solver_)
    {
        if (solver)
            solver->SetRelTol(rtol);
    }
}

void Upscale::SetAbsTol(double atol)
{
    for (auto& solver : solver_)
    {
        if (solver)
            solver->SetAbsTol(atol);
    }
}

std::vector<double> Upscale::ComputeErrors(const mfem::BlockVector& upscaled_sol,
                                           const mfem::BlockVector& fine_sol) const
{
    const mfem::SparseMatrix& M = GetMatrix(0).GetM();
    const mfem::SparseMatrix& D = GetMatrix(0).GetD();

    auto info = smoothg::ComputeErrors(comm_, M, D, upscaled_sol, fine_sol);
    info.push_back(OperatorComplexity());

    return info;
}

void Upscale::ShowErrors(const mfem::BlockVector& upscaled_sol,
                         const mfem::BlockVector& fine_sol) const
{
    auto info = ComputeErrors(upscaled_sol, fine_sol);

    if (myid_ == 0)
    {
        smoothg::ShowErrors(info);
    }
}

void Upscale::ShowSolveInfo(int level, std::ostream& out) const
{
    assert(solver_[level]);
    std::string tag;
    if (level == 0)
        tag = "Fine";
    else if (level == 1)
        tag = "Coarse1";
    else
    {
        std::stringstream out;
        out << "Level" << level;
        tag = out.str();
    }
    if (myid_ == 0)
    {
        out << "\n";
        out << tag << " Solve Time:         " << solver_[level]->GetTiming() << "\n";
        out << tag << " Solve Iterations:   " << solver_[level]->GetNumIterations() << "\n";
    }
}

void Upscale::ShowSetupTime(std::ostream& out) const
{
    if (myid_ == 0)
    {
        out << "\n";
        out << "Upscale Setup Time:      " << setup_time_ << "\n";
    }
}

double Upscale::GetSolveTime(int level) const
{
    assert(solver_[level]);
    return solver_[level]->GetTiming();
}

int Upscale::GetSolveIters(int level) const
{
    assert(solver_[level]);
    return solver_[level]->GetNumIterations();
}

double Upscale::GetSetupTime() const
{
    return setup_time_;
}

void Upscale::DumpDebug(const std::string& prefix) const
{
    int counter = 0;
    for (auto& ml : mixed_laplacians_)
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
    for (auto& c : coarsener_)
    {
        std::stringstream s;
        s << prefix << "Psigma" << counter << ".sparsematrix";
        std::ofstream outPsigma(s.str().c_str());
        outPsigma << std::scientific << std::setprecision(15);
        c->GetPsigma().Print(outPsigma, 1);
        s.str("");
        s << prefix << "Pu" << counter++ << ".sparsematrix";
        std::ofstream outPu(s.str().c_str());
        outPu << std::scientific << std::setprecision(15);
        c->GetPu().Print(outPu, 1);
    }
}

void Upscale::RescaleCoefficient(int level, const mfem::Vector& coeff)
{
    if (!param_.hybridization)
    {
        GetMatrix(level).UpdateM(coeff);
        MakeSolver(level);
    }
    else
    {
        auto hybrid_solver = dynamic_cast<HybridSolver*>(solver_[level].get());
        assert(hybrid_solver);
        hybrid_solver->UpdateAggScaling(coeff);
    }
}

int Upscale::GetNumVertices(int level) const
{
    return GetMatrix(level).GetGraph().NumVertices();
}

std::vector<int> Upscale::GetVertexSizes() const
{
    std::vector<int> out(GetNumLevels());
    for (int level = 0; level < GetNumLevels(); ++level)
    {
        out[level] = GetNumVertices(level);
    }
    return out;
}

} // namespace smoothg
