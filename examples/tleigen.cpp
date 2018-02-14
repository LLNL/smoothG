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

/** @file mleigen.cpp
    @brief Example usage of multilevel eigensolver
*/

#include <mpi.h>

#include "mfem.hpp"
#include "../src/picojson.h"
#include "../src/GraphUpscale.hpp"
#include "../src/smoothG.hpp"

using namespace smoothg;

// Build graph Laplacian matrix (+ I) associated with a graph
mfem::HypreParMatrix* BuildGraphLaplacian(const mfem::SparseMatrix& vertex_edge)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int coarse_factor = 300;

    mfem::SparseMatrix edge_vertex = smoothg::Transpose(vertex_edge);
    mfem::SparseMatrix vertex_vertex = smoothg::Mult(vertex_edge, edge_vertex);

    int num_parts = (vertex_edge.Height() / (double)(coarse_factor)) + 0.5;
    num_parts = std::max(1, num_parts);
    mfem::Array<int> partition;
    Partition(vertex_vertex, partition, num_parts);

    smoothg::ParGraph pgraph(comm, vertex_edge, partition);
    auto& v_e = pgraph.GetLocalVertexToEdge();
    auto& e_te = pgraph.GetEdgeToTrueEdge();

    MixedMatrix mixed_laplacian(v_e, e_te);
    auto& pD = mixed_laplacian.get_pD();
    auto& pM = mixed_laplacian.get_pM();

    auto pDT = pD.Transpose();
    mfem::Vector M_diag;
    pM.GetDiag(M_diag);
    pDT->InvScaleRows(M_diag);
    auto A = mfem::ParMult(&pD, pDT);
    A->CopyColStarts();
    A->CopyRowStarts();

    // Add identity so that the matrix is well-posed
    mfem::SparseMatrix A_diag;
    A->GetDiag(A_diag);
    for (int i = 0; i < A_diag.Width(); i++)
        A_diag(i, i) += 1.0;

    return A;
}

void ExtractLaplacianGraph(const mfem::SparseMatrix& gl,
                           mfem::SparseMatrix& v_e,
                           mfem::Vector& weight)
{
    const int nvertices = gl.Height();
    const int nedges = gl.NumNonZeroElems() - nvertices;

    const int* gl_i =  gl.GetI();
    const int* gl_j =  gl.GetJ();
    const double* gl_data =  gl.GetData();

    weight.SetSize(nedges);

    int* e_v_i = new int[nedges];
    int* e_v_j = new int[nedges * 2];
    double* e_v_data = new double[nedges * 2];
    std::fill_n(e_v_data, nedges * 2, 1.0);

    for (int i = 0; i < nedges+1; i++)
    {
        e_v_i[i] = 2*i;
    }

    int edge_counter = 0;
    for (int i = 0; i < nvertices; i++)
    {
        for (int j = gl_i[i]; j < gl_i[i + 1]; j++)
        {
            e_v_j[2 * edge_counter] = i;
            e_v_j[2 * edge_counter + 1] = gl_j[j];
            weight[edge_counter++] = gl_data[j] * -1.0;
        }
    }
    mfem::SparseMatrix e_v(e_v_i, e_v_j, e_v_data, nedges, nvertices);
    mfem::SparseMatrix v_e_tmp = smoothg::Transpose(e_v);
    v_e.Swap(v_e_tmp);
}

// Two-level V-cycle with default hypre smoother and smoothG coarse space
class MixedSpectralAMGe : public mfem::Solver
{
private:
    const mfem::HypreParMatrix& A_;
    std::unique_ptr<mfem::HypreSmoother> smoother_;
    std::unique_ptr<GraphUpscale> upscaler_;
    mutable mfem::Vector correct;
    mutable mfem::Vector resid;

public:
    MixedSpectralAMGe(mfem::HypreParMatrix& A,
                      const mfem::SparseMatrix& vertex_edge)
        : mfem::Solver(A.Height()), A_(A), correct(A.Height()), resid(A.Height())
    {
        smoother_ = make_unique<mfem::HypreSmoother>(A);
        // needs w_block
        upscaler_ = make_unique<GraphUpscale>(A.GetComm(), vertex_edge, 300,
                                              1, 4, true, false, false, true);
    }

    void SetOperator(const mfem::Operator& op) {}

    void Mult(const mfem::Vector& x, mfem::Vector& y) const
    {
        resid = x;
        y = 0.0;

        smoother_->Mult(resid, correct);
        y += correct;
        A_.Mult(-1.0, correct, 1.0, resid);

        upscaler_->Solve(resid, correct);
        y += correct;
        A_.Mult(-1.0, correct, 1.0, resid);

        smoother_->Mult(resid, correct);
        y += correct;
    }
};

// DMinvDt + shift * I
class ShiftedDMinvDt : public mfem::Operator
{
public:
    ShiftedDMinvDt(mfem::HypreParMatrix& M, mfem::HypreParMatrix& D, double shift = 1.0)
        : Operator(D.GetNumRows(), D.GetNumRows()), M_(M), D_(D), M_prec_(M),
          M_solver_(M), DTx_(D.GetNumCols()), MinvDTx_(D.GetNumCols()), shift_(shift)
    {
        M_prec_.SetSymmetry(1);
        HYPRE_ParaSailsSetLogging(M_prec_, 0);
        HYPRE_ParaSailsSetReuse(M_prec_, 1);
        HYPRE_ParaSailsSetThresh(M_prec_, .1);
        HYPRE_ParaSailsSetFilter(M_prec_, .05);

        M_solver_.SetPrintLevel(0);
        M_solver_.SetMaxIter(M_.Width());
        M_solver_.SetTol(1e-8);
        M_solver_.SetPreconditioner(M_prec_);
    }

    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const
    {
        DTx_ = 0.;
        D_.MultTranspose(x, DTx_);

        MinvDTx_ = 0.0;
        M_solver_.Mult(DTx_, MinvDTx_);

        y = 0.;
        D_.Mult(MinvDTx_, y);

        y.Add(shift_, x);
    }

private:
    const mfem::HypreParMatrix& M_;
    const mfem::HypreParMatrix& D_;

    mfem::HypreParaSails M_prec_;
    mfem::HyprePCG M_solver_;

    mutable mfem::Vector DTx_;
    mutable mfem::Vector MinvDTx_;

    const double shift_;
};

void SetDefaultLOBPCGParameters(mfem::HypreLOBPCG& lobpcg)
{
    lobpcg.SetNumModes(4);
    lobpcg.SetRandomSeed(4);
    lobpcg.SetMaxIter(5000);
    lobpcg.SetTol(1e-12);
    lobpcg.SetPrecondUsageMode(1);
    lobpcg.SetPrintLevel(1);
}

class TLSGLE
{
public:
    TLSGLE(mfem::HypreParMatrix& A,
           const mfem::SparseMatrix& vertex_edge,
           bool use_coarse_approx = true)
        : A_(A), prec_(A), lobpcg_(A.GetComm()), nev_(1), print_level_(1),
          vertex_edge_(vertex_edge), use_coarse_approx_(use_coarse_approx)
    {
        MPI_Comm_rank(A.GetComm(), &myid_);

        prec_.SetPrintLevel(0);

        SetDefaultLOBPCGParameters(lobpcg_);
        lobpcg_.SetNumModes(nev_);
        lobpcg_.SetPrintLevel(print_level_);
        lobpcg_.SetOperator(A_);
        lobpcg_.SetPreconditioner(prec_);
    }

    void SetNumModes(int nev) { nev_ = nev; lobpcg_.SetNumModes(nev); }

    void SetPrintLevel(int print_level)
    {
        print_level_ = print_level; lobpcg_.SetPrintLevel(print_level);
    }

    void Solve()
    {
        mfem::HypreParVector* evects_c[nev_];
        if (use_coarse_approx_)
        {
            GraphUpscale upscaler(A_.GetComm(), vertex_edge_, 300, 1, 4,
                                  true, false, false, true);
            auto& pM_c = upscaler.GetCoarseMatrix().get_pM();
            auto& pD_c = upscaler.GetCoarseMatrix().get_pD();

            ShiftedDMinvDt A_c(pM_c, pD_c);
            UpscaleCoarseSolve prec_c(upscaler);

            mfem::HypreLOBPCG lobpcg_c(A_.GetComm());
            SetDefaultLOBPCGParameters(lobpcg_c);
            lobpcg_c.SetNumModes(nev_);
            lobpcg_c.SetPrintLevel(print_level_);
            lobpcg_c.SetOperator(A_c);
            lobpcg_c.SetPreconditioner(prec_c);
            lobpcg_c.Solve();

            if (print_level_ > 0 && myid_ == 0)
            {
                mfem::Array<double> eval;
                lobpcg_c.GetEigenvalues(eval);
                std::cout<< "Eigenvalues of coarse operator: \n";
                eval.Print();
            }

            auto& Drow_start = upscaler.GetFineMatrix().get_Drow_start();
            for (int i = 0; i< nev_; i++)
            {
                evects_c[i] = new mfem::HypreParVector(
                        A_.GetComm(), Drow_start.Last(), Drow_start);
                (*evects_c[i]) = 0.0;
                upscaler.Interpolate(lobpcg_c.GetEigenvector(i), *(evects_c[i]));
            }
            lobpcg_.SetInitialVectors(nev_, evects_c);
        }

        lobpcg_.Solve();
    }

    void GetEigenvalues(mfem::Array<double>& eval)
    {
        lobpcg_.GetEigenvalues(eval);
    }

private:
    mfem::HypreParMatrix& A_;
    mfem::HypreBoomerAMG prec_;
    mfem::HypreLOBPCG lobpcg_;
    int nev_;
    int print_level_;

    const mfem::SparseMatrix& vertex_edge_;
    bool use_coarse_approx_;

    int myid_;
};


int main(int argc, char* argv[])
{
    // Initialize MPI
    int myid;
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &myid);

    constexpr auto print_level = 0;
    constexpr auto ve_filename = "../../graphdata/vertex_edge_1069524880.txt";
    const auto vertex_edge = ReadVertexEdge(ve_filename);

    // Find eigenpairs of graph Laplacian using LOBPCG + BoomerAMG
    {
        auto A = BuildGraphLaplacian(vertex_edge);

        mfem::StopWatch ch; ch.Clear();ch.Start();

        mfem::HypreBoomerAMG prec1(*A);
        prec1.SetPrintLevel(0);

        mfem::HypreLOBPCG lobpcg(comm);
        SetDefaultLOBPCGParameters(lobpcg);
        lobpcg.SetPrintLevel(print_level);
        lobpcg.SetOperator(*A);
        lobpcg.SetPreconditioner(prec1);
        lobpcg.Solve();

        if (myid==0)
        {
            mfem::Array<double> eval;
            lobpcg.GetEigenvalues(eval);

            std::cout<< "LOBPCG + BoomerAMG took " << ch.RealTime() << "s\n";
            std::cout<< "Eigenvalues: \n";
            eval.Print();
        }
    }

    // Find eigenpairs of graph Laplacian using LOBPCG + Mixed Spectral AMGe
    if (false) // not working well currently
    {
        auto A = BuildGraphLaplacian(vertex_edge);

        mfem::StopWatch ch; ch.Clear();ch.Start();

        MixedSpectralAMGe prec2(*A, vertex_edge);

        mfem::HypreLOBPCG lobpcg(comm);
        SetDefaultLOBPCGParameters(lobpcg);
        lobpcg.SetPrintLevel(print_level);
        lobpcg.SetOperator(*A);
        lobpcg.SetPreconditioner(prec2);
        lobpcg.Solve();

        if (myid==0)
        {
            mfem::Array<double> eval;
            lobpcg.GetEigenvalues(eval);

            std::cout<< "LOBPCG + MS-AMGe took " << ch.RealTime() << "s\n";
            std::cout<< "Eigenvalues: \n";
            eval.Print();
        }
    }

    // Two level spectral graph Laplacian eigensolver
    {
        auto A = BuildGraphLaplacian(vertex_edge);

        mfem::StopWatch ch; ch.Clear();ch.Start();

        TLSGLE tlsgle(*A, vertex_edge);
        tlsgle.SetNumModes(4);
        tlsgle.SetPrintLevel(print_level);
        tlsgle.Solve();

        if (myid==0)
        {
            mfem::Array<double> eval;
            tlsgle.GetEigenvalues(eval);

            std::cout<< "TLSGLE took " << ch.RealTime() << "s\n";
            std::cout<< "Eigenvalues: \n";
            eval.Print();
        }
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
