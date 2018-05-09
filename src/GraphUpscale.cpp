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

    @brief Contains GraphUpscale class
*/

#include "GraphUpscale.hpp"

namespace smoothg
{

GraphUpscale::GraphUpscale(Graph graph, double spect_tol, int max_evects, bool hybridization)
    : Operator(graph.vertex_edge_local_.Rows()),
      comm_(graph.edge_true_edge_.GetComm()),
      myid_(graph.edge_true_edge_.GetMyId()),
      global_vertices_(graph.global_vertices_),
      global_edges_(graph.global_edges_),
      setup_time_(0), spect_tol_(spect_tol),
      max_evects_(max_evects), hybridization_(hybridization),
      graph_(std::move(graph))
{
    Timer timer(Timer::Start::True);

    gt_ = GraphTopology(graph_);

    VectorElemMM fine_mm(graph_);
    fine_mm.AssembleM(); // Coarsening requires assembled M, for now

    coarsener_ = GraphCoarsen(fine_mm, gt_, max_evects_, spect_tol_);

    DenseElemMM coarse_mm = coarsener_.Coarsen(gt_, fine_mm);

    mgl_.push_back(make_unique<VectorElemMM>(std::move(fine_mm)));
    mgl_.push_back(make_unique<DenseElemMM>(std::move(coarse_mm)));

    MakeCoarseVectors();
    MakeCoarseSolver();
    MakeFineSolver(); // TODO(gelever1): unset and let user make

    timer.Click();
    setup_time_ += timer.TotalTime();
}

void GraphUpscale::MakeCoarseSolver()
{
    auto& mm = dynamic_cast<DenseElemMM&>(GetCoarseMatrix());

    if (hybridization_)
    {
        coarse_solver_ = make_unique<HybridSolver>(mm, coarsener_);
    }
    else
    {
        mm.AssembleM();
        coarse_solver_ = make_unique<MinresBlockSolver>(mm);
    }
}

void GraphUpscale::MakeFineSolver()
{
    auto& mm = dynamic_cast<VectorElemMM&>(GetFineMatrix());

    if (hybridization_)
    {
        fine_solver_ = make_unique<HybridSolver>(mm);
    }
    else
    {
        mm.AssembleM();
        fine_solver_ = make_unique<MinresBlockSolver>(mm);
    }
}

void GraphUpscale::MakeCoarseSolver(const std::vector<double>& agg_weights)
{
    auto& mm = dynamic_cast<DenseElemMM&>(GetCoarseMatrix());

    if (hybridization_)
    {
        assert(coarse_solver_);

        auto& hb = dynamic_cast<HybridSolver&>(*coarse_solver_);
        hb.UpdateAggScaling(agg_weights);
    }
    else
    {
        mm.AssembleM(agg_weights);
        coarse_solver_ = make_unique<MinresBlockSolver>(mm);
    }
}

void GraphUpscale::MakeFineSolver(const std::vector<double>& agg_weights)
{
    auto& mm = dynamic_cast<VectorElemMM&>(GetFineMatrix());

    if (hybridization_)
    {
        if (!fine_solver_)
        {
            fine_solver_ = make_unique<HybridSolver>(mm);
        }

        auto& hb = dynamic_cast<HybridSolver&>(*fine_solver_);
        hb.UpdateAggScaling(agg_weights);
    }
    else
    {
        mm.AssembleM(agg_weights);
        fine_solver_ = make_unique<MinresBlockSolver>(mm);
    }
}

Vector GraphUpscale::ReadVertexVector(const std::string& filename) const
{
    return ReadVector(filename, graph_.vertex_map_);
}

Vector GraphUpscale::ReadEdgeVector(const std::string& filename) const
{
    return ReadVector(filename, graph_.edge_map_);
}

BlockVector GraphUpscale::ReadVertexBlockVector(const std::string& filename) const
{
    BlockVector vect = GetFineBlockVector();

    vect.GetBlock(0) = 0.0;
    vect.GetBlock(1) = ReadVertexVector(filename);

    return vect;
}

BlockVector GraphUpscale::ReadEdgeBlockVector(const std::string& filename) const
{
    BlockVector vect = GetFineBlockVector();

    vect.GetBlock(0) = ReadEdgeVector(filename);
    vect.GetBlock(1) = 0.0;

    return vect;
}

void GraphUpscale::WriteVertexVector(const VectorView& vect, const std::string& filename) const
{
    WriteVector(comm_, vect, filename, global_vertices_, graph_.vertex_map_);
}

void GraphUpscale::WriteEdgeVector(const VectorView& vect, const std::string& filename) const
{
    WriteVector(comm_, vect, filename, global_edges_, graph_.edge_map_);
}

void GraphUpscale::Mult(const VectorView& x, VectorView y) const
{
    assert(coarse_solver_);

    coarsener_.Restrict(x, rhs_coarse_.GetBlock(1));

    rhs_coarse_.GetBlock(0) = 0.0;
    rhs_coarse_.GetBlock(1) *= -1.0;

    coarse_solver_->Solve(rhs_coarse_, sol_coarse_);

    coarsener_.Interpolate(sol_coarse_.GetBlock(1), y);

    Orthogonalize(y);
}

void GraphUpscale::Solve(const VectorView& x, VectorView y) const
{
    Mult(x, y);
}

Vector GraphUpscale::Solve(const VectorView& x) const
{
    Vector y(x.size());

    Solve(x, y);

    return y;
}

void GraphUpscale::Solve(const BlockVector& x, BlockVector& y) const
{
    assert(coarse_solver_);

    coarsener_.Restrict(x, rhs_coarse_);
    rhs_coarse_.GetBlock(1) *= -1.0;

    coarse_solver_->Solve(rhs_coarse_, sol_coarse_);
    coarsener_.Interpolate(sol_coarse_, y);

    Orthogonalize(y);
}

BlockVector GraphUpscale::Solve(const BlockVector& x) const
{
    BlockVector y = GetFineBlockVector();

    Solve(x, y);

    return y;
}

void GraphUpscale::SolveCoarse(const VectorView& x, VectorView y) const
{
    assert(coarse_solver_);

    coarse_solver_->Solve(x, y);
}

Vector GraphUpscale::SolveCoarse(const VectorView& x) const
{
    Vector coarse_vect = GetCoarseVector();
    SolveCoarse(x, coarse_vect);

    return coarse_vect;
}

void GraphUpscale::SolveCoarse(const BlockVector& x, BlockVector& y) const
{
    assert(coarse_solver_);

    coarse_solver_->Solve(x, y);
    y *= -1.0;
}

BlockVector GraphUpscale::SolveCoarse(const BlockVector& x) const
{
    BlockVector coarse_vect = GetCoarseBlockVector();
    SolveCoarse(x, coarse_vect);

    return coarse_vect;
}

void GraphUpscale::SolveFine(const VectorView& x, VectorView y) const
{
    assert(fine_solver_);

    fine_solver_->Solve(x, y);
    y *= -1.0;

    Orthogonalize(y);
}

Vector GraphUpscale::SolveFine(const VectorView& x) const
{
    Vector y(x.size());

    SolveFine(x, y);

    return y;
}

void GraphUpscale::SolveFine(const BlockVector& x, BlockVector& y) const
{
    assert(fine_solver_);

    fine_solver_->Solve(x, y);
    y *= -1.0;

    Orthogonalize(y);
}

BlockVector GraphUpscale::SolveFine(const BlockVector& x) const
{
    BlockVector y = GetFineBlockVector();

    SolveFine(x, y);

    return y;
}

void GraphUpscale::Interpolate(const VectorView& x, VectorView y) const
{
    coarsener_.Interpolate(x, y);
}

Vector GraphUpscale::Interpolate(const VectorView& x) const
{
    return coarsener_.Interpolate(x);
}

void GraphUpscale::Interpolate(const BlockVector& x, BlockVector& y) const
{
    coarsener_.Interpolate(x, y);
}

BlockVector GraphUpscale::Interpolate(const BlockVector& x) const
{
    return coarsener_.Interpolate(x);
}

void GraphUpscale::Restrict(const VectorView& x, VectorView y) const
{
    coarsener_.Restrict(x, y);
}

Vector GraphUpscale::Restrict(const VectorView& x) const
{
    return coarsener_.Restrict(x);
}

void GraphUpscale::Restrict(const BlockVector& x, BlockVector& y) const
{
    coarsener_.Restrict(x, y);
}

BlockVector GraphUpscale::Restrict(const BlockVector& x) const
{
    return coarsener_.Restrict(x);
}

const std::vector<int>& GraphUpscale::FineBlockOffsets() const
{
    return GetFineMatrix().Offsets();
}

const std::vector<int>& GraphUpscale::CoarseBlockOffsets() const
{
    return GetCoarseMatrix().Offsets();
}

const std::vector<int>& GraphUpscale::FineTrueBlockOffsets() const
{
    return GetFineMatrix().TrueOffsets();
}

const std::vector<int>& GraphUpscale::CoarseTrueBlockOffsets() const
{
    return GetCoarseMatrix().TrueOffsets();
}

void GraphUpscale::Orthogonalize(VectorView vect) const
{
    OrthoConstant(comm_, vect, GetFineMatrix().GlobalD().GlobalRows());
}

void GraphUpscale::Orthogonalize(BlockVector& vect) const
{
    Orthogonalize(vect.GetBlock(1));
}

Vector GraphUpscale::GetCoarseVector() const
{
    int coarse_size = GetCoarseMatrix().LocalD().Rows();

    return Vector(coarse_size);
}

Vector GraphUpscale::GetFineVector() const
{
    int fine_size = GetFineMatrix().LocalD().Rows();

    return Vector(fine_size);
}

BlockVector GraphUpscale::GetCoarseBlockVector() const
{
    return BlockVector(GetCoarseMatrix().Offsets());
}

BlockVector GraphUpscale::GetFineBlockVector() const
{
    return BlockVector(GetFineMatrix().Offsets());
}

BlockVector GraphUpscale::GetCoarseTrueBlockVector() const
{
    return BlockVector(GetCoarseMatrix().TrueOffsets());
}

BlockVector GraphUpscale::GetFineTrueBlockVector() const
{
    return BlockVector(GetFineMatrix().TrueOffsets());
}

MixedMatrix& GraphUpscale::GetMatrix(int level)
{
    assert(level >= 0 && level < static_cast<int>(mgl_.size()));
    assert(mgl_[level]);

    return *mgl_[level];
}

const MixedMatrix& GraphUpscale::GetMatrix(int level) const
{
    assert(level >= 0 && level < static_cast<int>(mgl_.size()));
    assert(mgl_[level]);

    return *mgl_[level];
}

MixedMatrix& GraphUpscale::GetFineMatrix()
{
    return GetMatrix(0);
}

const MixedMatrix& GraphUpscale::GetFineMatrix() const
{
    return GetMatrix(0);
}

MixedMatrix& GraphUpscale::GetCoarseMatrix()
{
    return GetMatrix(1);
}

const MixedMatrix& GraphUpscale::GetCoarseMatrix() const
{
    return GetMatrix(1);
}

void GraphUpscale::PrintInfo(std::ostream& out) const
{
    // Matrix sizes, not solvers
    int nnz_coarse = GetCoarseMatrix().GlobalNNZ();
    int nnz_fine = GetFineMatrix().GlobalNNZ();

    // True dof size
    int size_fine = GetFineMatrix().GlobalRows();
    int size_coarse = GetCoarseMatrix().GlobalRows();

    int num_procs;
    MPI_Comm_size(comm_, &num_procs);

    double op_comp = OperatorComplexity();

    if (myid_ == 0)
    {
        int old_precision = out.precision();
        out.precision(3);

        out << "\n";

        if (num_procs > 1)
        {
            out << "Processors: " << num_procs << "\n";
            out << "---------------------\n";
        }

        out << "Fine Matrix\n";
        out << "---------------------\n";
        out << "Size\t\t" << size_fine << "\n";
        out << "NonZeros:\t" << nnz_fine << "\n";
        out << "\n";
        out << "Coarse Matrix\n";
        out << "---------------------\n";
        out << "Size\t\t" << size_coarse << "\n";
        out << "NonZeros:\t" << nnz_coarse << "\n";
        out << "\n";
        out << "Op Comp:\t" << op_comp << "\n";

        out.precision(old_precision);
    }
}

double GraphUpscale::OperatorComplexity() const
{
    assert(coarse_solver_);

    int nnz_coarse = coarse_solver_->GetNNZ();
    int nnz_fine;

    if (fine_solver_)
    {
        nnz_fine = fine_solver_->GetNNZ();
    }
    else
    {
        nnz_fine = GetFineMatrix().GlobalNNZ();
    }


    double op_comp = 1.0 + (nnz_coarse / (double) nnz_fine);

    return op_comp;
}

void GraphUpscale::SetPrintLevel(int print_level)
{
    assert(coarse_solver_);
    coarse_solver_->SetPrintLevel(print_level);

    if (fine_solver_)
    {
        fine_solver_->SetPrintLevel(print_level);
    }
}

void GraphUpscale::SetMaxIter(int max_num_iter)
{
    assert(coarse_solver_);
    coarse_solver_->SetMaxIter(max_num_iter);

    if (fine_solver_)
    {
        fine_solver_->SetMaxIter(max_num_iter);
    }
}

void GraphUpscale::SetRelTol(double rtol)
{
    assert(coarse_solver_);
    coarse_solver_->SetRelTol(rtol);

    if (fine_solver_)
    {
        fine_solver_->SetRelTol(rtol);
    }
}

void GraphUpscale::SetAbsTol(double atol)
{
    assert(coarse_solver_);
    coarse_solver_->SetAbsTol(atol);

    if (fine_solver_)
    {
        fine_solver_->SetAbsTol(atol);
    }
}

void GraphUpscale::ShowCoarseSolveInfo(std::ostream& out) const
{
    assert(coarse_solver_);

    if (myid_ == 0)
    {
        out << "\n";
        out << "Coarse Solve Time:       " << coarse_solver_->GetTiming() << "\n";
        out << "Coarse Solve Iterations: " << coarse_solver_->GetNumIterations() << "\n";
    }
}

void GraphUpscale::ShowFineSolveInfo(std::ostream& out) const
{
    assert(fine_solver_);

    if (myid_ == 0)
    {
        out << "\n";
        out << "Fine Solve Time:         " << fine_solver_->GetTiming() << "\n";
        out << "Fine Solve Iterations:   " << fine_solver_->GetNumIterations() << "\n";
    }
}

void GraphUpscale::ShowSetupTime(std::ostream& out) const
{
    if (myid_ == 0)
    {
        out << "\n";
        out << "GraphUpscale Setup Time:      " << setup_time_ << "\n";
    }
}

double GraphUpscale::GetCoarseSolveTime() const
{
    assert(coarse_solver_);

    return coarse_solver_->GetTiming();
}

double GraphUpscale::GetFineSolveTime() const
{
    assert(fine_solver_);

    return fine_solver_->GetTiming();
}

int GraphUpscale::GetCoarseSolveIters() const
{
    assert(coarse_solver_);

    return coarse_solver_->GetNumIterations();
}

int GraphUpscale::GetFineSolveIters() const
{
    assert(fine_solver_);

    return fine_solver_->GetNumIterations();
}

double GraphUpscale::GetSetupTime() const
{
    return setup_time_;
}

std::vector<double> GraphUpscale::ComputeErrors(const BlockVector& upscaled_sol,
                                           const BlockVector& fine_sol) const
{
    const SparseMatrix& M = GetFineMatrix().LocalM();
    const SparseMatrix& D = GetFineMatrix().LocalD();

    auto info = smoothg::ComputeErrors(comm_, M, D, upscaled_sol, fine_sol);
    info.push_back(OperatorComplexity());

    return info;
}

void GraphUpscale::ShowErrors(const BlockVector& upscaled_sol,
                         const BlockVector& fine_sol) const
{
    auto info = ComputeErrors(upscaled_sol, fine_sol);

    if (myid_ == 0)
    {
        smoothg::ShowErrors(info);
    }
}

void GraphUpscale::MakeCoarseVectors()
{
    rhs_coarse_ = BlockVector(GetCoarseMatrix().Offsets());
    sol_coarse_ = BlockVector(GetCoarseMatrix().Offsets());
}


} // namespace smoothg
