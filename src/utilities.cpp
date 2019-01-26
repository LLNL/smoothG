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

    @brief Implements some shared code and utility functions.
*/

#include <mfem.hpp>

#include "utilities.hpp"
#include "MatrixUtilities.hpp"
#include "SpectralAMG_MGL_Coarsener.hpp"
#include "MixedLaplacianSolver.hpp"

using std::unique_ptr;

extern "C"
{
    void dgesvd_(const char* jobu, const char* jobvt, const int* m,
                 const int* n, double* a, const int* lda, double* s,
                 double* u, const int* ldu, double* vt, const int* ldvt,
                 double* work, const int* lwork, int* info);
}

namespace smoothg
{

UpscalingStatistics::UpscalingStatistics(int nLevels)
    :
    timings_(nLevels, NSTAGES),
    sigma_weighted_l2_error_square_(nLevels),
    u_l2_error_square_(nLevels),
    Dsigma_l2_error_square_(nLevels),
    help_(nLevels),
    iter_(nLevels),
    ndofs_(nLevels),
    nnzs_(nLevels)
{
    timings_ = 0.0;
    iter_ = 0;
    ndofs_ = 0;
    nnzs_ = 0;
}

UpscalingStatistics::~UpscalingStatistics()
{
}

void UpscalingStatistics::ComputeErrorSquare(
    int k,
    const std::vector<MixedMatrix>& mgL,
    const Mixed_GL_Coarsener& mgLc,
    const std::vector<unique_ptr<mfem::BlockVector> >& sol)
{
    if (k == 0)
    {
        for (unsigned int j = 0; j < help_.size(); ++j)
        {
            help_[j] = make_unique<mfem::BlockVector>(mgL[j].GetBlockOffsets());
            (*help_[j]) = 0.0;
        }
    }

    *(help_[k]) = *(sol[k]);
    for (int j = k; j > 0; --j)
    {
        MFEM_ASSERT(j == 1, "Only a two-level method is considered");
        mgLc.GetPsigma().Mult(help_[j]->GetBlock(0), help_[j - 1]->GetBlock(0));
        mgLc.GetPu().Mult(help_[j]->GetBlock(1), help_[j - 1]->GetBlock(1));
    }

    if (!mgL[0].CheckW())
    {
        par_orthogonalize_from_constant(help_[0]->GetBlock(1),
                                        mgL[0].GetDrowStarts().Last());
    }

    for (int j(0); j <= k; ++j)
    {
        MFEM_ASSERT(help_[j]->Size() == sol[j]->Size() &&
                    sol[j]->GetBlock(0).Size() == mgL[j].GetD().Width(),
                    "Graph Laplacian");

        int sigmasize = sol[0]->GetBlock(0).Size();
        int usize = sol[0]->GetBlock(1).Size();
        mfem::Vector sigma_H(help_[0]->GetBlock(0).GetData(), sigmasize);
        mfem::Vector sigma_h(sigmasize), u_h(usize);
        sigma_h = 0.;
        u_h = 0.;
        if (j == 1)
        {
            sigma_h += help_[0]->GetBlock(0);
            u_h += help_[0]->GetBlock(1);
        }
        else
        {
            sigma_h += sol[j]->GetBlock(0);
            u_h += sol[j]->GetBlock(1);
        }
        mfem::Vector u_H(help_[0]->GetBlock(1).GetData(), usize);
        mfem::Vector sigma_diff(sigmasize), dsigma_diff(usize), u_diff(usize),
             dsigma_h(usize);
        sigma_diff = 0.;
        dsigma_diff = 0.;
        u_diff = 0.;
        dsigma_h = 0.;

        if (j == k)
        {
            mgL[0].GetD().Mult(sigma_h, dsigma_h);

            sigma_weighted_l2_error_square_(k, j)
                = mgL[0].GetM().InnerProduct(sigma_h, sigma_h);
            Dsigma_l2_error_square_(k, j) = dsigma_h * dsigma_h;
            u_l2_error_square_(k, j) = u_h * u_h;
        }
        else
        {
            subtract(sigma_H, sigma_h, sigma_diff);
            mgL[j].GetD().Mult(sigma_diff, dsigma_diff);
            subtract(u_H, u_h, u_diff);

            sigma_weighted_l2_error_square_(k, j)
                = mgL[j].GetM().InnerProduct(sigma_diff, sigma_diff);
            Dsigma_l2_error_square_(k, j) = dsigma_diff * dsigma_diff;
            u_l2_error_square_(k, j) = u_diff * u_diff;
        }
    }
}

void UpscalingStatistics::PrintStatistics(
    MPI_Comm comm,
    picojson::object& serialize)
{
    enum
    {
        TOPOLOGY = 0, SEQUENCE, SOLVER
    };

    int myid;
    MPI_Comm_rank(comm, &myid);

    int nLevels = u_l2_error_square_.Size();
    mfem::DenseMatrix sigma_weighted_l2_error(nLevels);
    mfem::DenseMatrix u_l2_error(nLevels);
    mfem::DenseMatrix Dsigma_l2_error(nLevels);

    MPI_Reduce(sigma_weighted_l2_error_square_.Data(),
               sigma_weighted_l2_error.Data(), nLevels * nLevels,
               MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(u_l2_error_square_.Data(), u_l2_error.Data(), nLevels * nLevels,
               MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(Dsigma_l2_error_square_.Data(), Dsigma_l2_error.Data(),
               nLevels * nLevels, MPI_DOUBLE, MPI_SUM, 0, comm);

    if (myid == 0)
    {
        std::transform(sigma_weighted_l2_error.Data(),
                       sigma_weighted_l2_error.Data() + nLevels * nLevels,
                       sigma_weighted_l2_error.Data(), (double (*)(double)) sqrt);
        std::transform(u_l2_error.Data(), u_l2_error.Data() + nLevels * nLevels,
                       u_l2_error.Data(), (double(*)(double)) sqrt);
        std::transform(Dsigma_l2_error.Data(), Dsigma_l2_error.Data() + nLevels * nLevels,
                       Dsigma_l2_error.Data(), (double(*)(double)) sqrt);
    }
    for (int k(1); k < nLevels; ++k)
    {
        for (int j(0); j < k; ++j)
        {
            sigma_weighted_l2_error(k, j) /= sigma_weighted_l2_error(j, j);
            Dsigma_l2_error(k, j) /= Dsigma_l2_error(j, j);
            u_l2_error(k, j) /= u_l2_error(j, j);
        }
    }

    if (myid == 0)
    {
        serialize["finest-u-error"] = picojson::value(sigma_weighted_l2_error(1, 0));
        serialize["finest-p-error"] = picojson::value(u_l2_error(1, 0));
        serialize["finest-div-error"] = picojson::value(Dsigma_l2_error(1, 0));
        std::cout << "\n{\n";
        int w = 14;
        std::cout << "%level" << std::setw(w) << "Topology" << std::setw(w)
                  << "Sequence\n";
        for (int i(0); i < nLevels; ++i)
            std::cout << std::setw(6) << i << std::setw(w)
                      << timings_(i, TOPOLOGY) << std::setw(w)
                      << timings_(i, SEQUENCE) << "\n";
        std::cout << "}\n";

        std::cout << "\n{\n";
        std::cout << "%level" << std::setw(w) << "size" << std::setw(w) << "nnz"
                  << std::setw(w) << "nit" << std::setw(w) << "Solver\n";
        for (int i(0); i < nLevels; ++i)
            std::cout << std::setw(6) << i << std::setw(w) << ndofs_[i]
                      << std::setw(w) << nnzs_[i] << std::setw(w) << iter_[i]
                      << std::setw(w) << timings_(i, SOLVER) << "\n";
        std::cout << "}\n";
        double operator_complexity = 1.0 + ((double)nnzs_[1]) / ((double)nnzs_[0]);
        serialize["operator-complexity"] = picojson::value(operator_complexity);
        std::cout << "OC = " << operator_complexity << std::endl;
        std::cout << "\nRelative upscaling errors "
                  "(diagonal entries are solution norms):\n";
        std::cout << "{\n";
        std::cout << "% || sigma_h - sigma_H || / || sigma_h ||\n";
        sigma_weighted_l2_error.PrintMatlab(std::cout);
        std::cout << "\n% || u_h - u_H || / || u_h ||\n";
        u_l2_error.PrintMatlab(std::cout);
        std::cout << "\n% || D ( sigma_h - sigma_H ) || / || D sigma_h ||\n";
        Dsigma_l2_error.PrintMatlab(std::cout);

        std::cout << "}\n";
    }

}

void UpscalingStatistics::RegisterSolve(const MixedLaplacianSolver& solver, int level)
{
    iter_[level] = solver.GetNumIterations();
    nnzs_[level] = solver.GetNNZ();
    // ndofs_[k] = mixed_laplacians[k].get_edge_d_td().GetGlobalNumCols()
    //    + mixed_laplacians[k].get_Drow_start().Last();
}

const mfem::BlockVector& UpscalingStatistics::GetInterpolatedSolution()
{
    return *help_[0];
}

void UpscalingStatistics::BeginTiming()
{
    chrono_.Clear();
    chrono_.Start();
}

void UpscalingStatistics::EndTiming(int level, int stage)
{
    chrono_.Stop();
    timings_(level, stage) = chrono_.RealTime();
}

double UpscalingStatistics::GetTiming(int level, int stage)
{
    return timings_(level, stage);
}

void VisualizeSolution(int k,
                       mfem::ParFiniteElementSpace* sigmafespace,
                       mfem::ParFiniteElementSpace* ufespace,
                       const mfem::SparseMatrix& D,
                       const mfem::BlockVector& sol)
{
    int myid;
    MPI_Comm_rank(sigmafespace->GetComm(), &myid);

    mfem::GridFunction sigma(sigmafespace);
    mfem::GridFunction u(ufespace);
    mfem::GridFunction Dsigma(ufespace);

    sigma = sol.GetBlock(0);
    u = sol.GetBlock(1);
    mfem::Vector DsigmaV(sol.GetBlock(1).Size());
    D.Mult(sol.GetBlock(0), DsigmaV);
    Dsigma = DsigmaV;

    std::ostringstream sigma_name, u_name, Dsigma_name;
    sigma_name << "sol_sigma" << k << "." << std::setfill('0') << std::setw(6) << myid;
    u_name << "sol_u" << k << "." << std::setfill('0') << std::setw(6) << myid;
    Dsigma_name << "sol_Dsigma" << k << "." << std::setfill('0') << std::setw(6) << myid;

    std::ofstream sigma_ofs(sigma_name.str().c_str());
    sigma_ofs.precision(8);
    sigma.Save(sigma_ofs);

    std::ofstream u_ofs(u_name.str().c_str());
    u_ofs.precision(8);
    u.Save(u_ofs);

    std::ofstream Dsigma_ofs(Dsigma_name.str().c_str());
    Dsigma_ofs.precision(8);
    Dsigma.Save(Dsigma_ofs);
}

int MarkDofsOnBoundary(
    const mfem::SparseMatrix& face_boundaryatt,
    const mfem::SparseMatrix& face_dof,
    const mfem::Array<int>& bndrAttributesMarker,
    mfem::Array<int>& dofMarker)
{
    dofMarker = 0;
    const int num_faces = face_boundaryatt.Height();

    const int* i_bndr = face_boundaryatt.GetI();
    const int* j_bndr = face_boundaryatt.GetJ();

    mfem::Array<int> dofs;

    for (int i = 0; i < num_faces; ++i)
    {
        int start = i_bndr[i];
        int end = i_bndr[i + 1];

        // Assert one attribute per face. For this to be true on coarse levels,
        // some care must be taken in generating coarse faces (respecting
        // boundary attributes in minimial intersection sets, for example)
        assert(((end - start) == 0) || ((end - start) == 1));

        if ((end - start) == 1 && bndrAttributesMarker[j_bndr[start]])
        {
            GetTableRow(face_dof, i, dofs);

            for (int dof : dofs)
            {
                dofMarker[dof] = 1;
            }
        }
    }

    int num_marked = dofMarker.Sum();

    return num_marked;
}

/**
   This implementation basically taken from
   DofAgglomeration::GetViewAgglomerateDofGlobalNumbering()
   as one step to extracting from Parelag.
*/
void GetTableRow(
    const mfem::SparseMatrix& mat, int rownum, mfem::Array<int>& J)
{
    const int begin = mat.GetI()[rownum];
    const int end = mat.GetI()[rownum + 1];
    const int size = end - begin;
    assert(size >= 0);
    J.MakeRef(mat.GetJ() + begin, size);
}

/// instead of a reference, get a copy
void GetTableRowCopy(
    const mfem::SparseMatrix& mat, int rownum, mfem::Array<int>& J)
{
    const int begin = mat.GetI()[rownum];
    const int end = mat.GetI()[rownum + 1];
    const int size = end - begin;
    mfem::Array<int> temp;
    temp.MakeRef(mat.GetJ() + begin, size);
    temp.Copy(J);
}

void FiniteVolumeMassIntegrator::AssembleElementMatrix(
    const mfem::FiniteElement& el,
    mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat)
{
    int dim = el.GetDim();
    int ndof = el.GetDof();
    elmat.SetSize(ndof);
    elmat = 0.0;

    mq.SetSize(dim);

    int order = 1;
    const mfem::IntegrationRule* ir = &mfem::IntRules.Get(el.GetGeomType(), order);

    MFEM_ASSERT(ir->GetNPoints() == 1, "Only implemented for piecewise "
                "constants!");

    int p = 0;
    const mfem::IntegrationPoint& ip = ir->IntPoint(p);

    if (VQ)
    {
        vq.SetSize(dim);
        VQ->Eval(vq, Trans, ip);
        for (int i = 0; i < dim; i++)
            mq(i, i) = vq(i);
    }
    else if (Q)
    {
        sq = Q->Eval(Trans, ip);
        for (int i = 0; i < dim; i++)
            mq(i, i) = sq;
    }
    else if (MQ)
        MQ->Eval(mq, Trans, ip);
    else
    {
        for (int i = 0; i < dim; i++)
            mq(i, i) = 1.0;
    }

    // Compute face area of each face
    mfem::DenseMatrix vshape;
    vshape.SetSize(ndof, dim);
    Trans.SetIntPoint(&ip);
    el.CalcVShape(Trans, vshape);
    vshape *= 2.;

    mfem::DenseMatrix vshapeT(vshape, 't');
    mfem::DenseMatrix tmp(ndof);
    Mult(vshape, vshapeT, tmp);

    mfem::Vector FaceAreaSquareInv(ndof);
    tmp.GetDiag(FaceAreaSquareInv);
    mfem::Vector FaceArea(ndof);

    for (int i = 0; i < ndof; i++)
        FaceArea(i) = 1. / std::sqrt(FaceAreaSquareInv(i));

    vshape.LeftScaling(FaceArea);
    vshapeT.RightScaling(FaceArea);

    // Compute k_{ii}
    mfem::DenseMatrix nk(ndof, dim);
    Mult(vshape, mq, nk);

    mfem::DenseMatrix nkn(ndof);
    Mult(nk, vshapeT, nkn);

    // this is right for grid-aligned permeability, maybe not for full tensor?
    mfem::Vector k(ndof);
    nkn.GetDiag(k);

    // here assume the input is k^{-1};
    mfem::Vector mii(ndof);
    for (int i = 0; i < ndof; i++)
        // Trans.Weight()/FaceArea(i)=Volume/face area=h (for rectangular grid)
        mii(i) = (Trans.Weight() / FaceArea(i)) * k(i) / FaceArea(i) / 2;
    elmat.Diag(mii.GetData(), ndof);
}

SVD_Calculator::SVD_Calculator():
    jobu_('O'),
    jobvt_('N'),
    lwork_(-1),
    info_(0)
{
}

void SVD_Calculator::Compute(mfem::DenseMatrix& A, mfem::Vector& singularValues)
{
    const int nrows = A.Height();
    const int ncols = A.Width();

    if (nrows < 1 || ncols < 1)
    {
        return;
    }

    // Allocate optimal size
    std::vector<double> tmp(nrows * ncols, 0.);
    lwork_ = -1;
    double qwork = 0.0;
    dgesvd_(&jobu_, &jobvt_, &nrows, &ncols, tmp.data(), &nrows,
            tmp.data(), tmp.data(), &nrows, tmp.data(), &ncols, &qwork,
            &lwork_, &info_);
    lwork_ = (int) qwork;
    std::vector<double>(lwork_).swap(work_);

    // Actual SVD computation
    singularValues.SetSize(std::min(nrows, ncols));
    const int ldA = std::max(nrows, 1);
    dgesvd_(&jobu_, &jobvt_, &nrows, &ncols, A.Data(), &ldA,
            singularValues.GetData(), nullptr, &ldA, nullptr, &ldA, work_.data(),
            &lwork_, &info_);
}

void ReadVertexEdge(std::ifstream& graphFile, mfem::SparseMatrix& out)
{
    int nvertices, nedges;
    if (!graphFile.is_open())
        mfem::mfem_error("Error in opening the graph file");
    graphFile >> nvertices;
    graphFile >> nedges;

    int* vertex_edge_i = new int[nvertices + 1];
    int* vertex_edge_j = new int[nedges * 2];
    double* vertex_edge_data = new double[nedges * 2];
    for (int i = 0; i < nvertices + 1; i++)
        graphFile >> vertex_edge_i[i];
    for (int i = 0; i < nedges * 2; i++)
        graphFile >> vertex_edge_j[i];
    for (int i = 0; i < nedges * 2; i++)
        vertex_edge_data[i] = 1.0;
    //graphFile >> vertex_edge_data[i];
    mfem::SparseMatrix vertex_edge(vertex_edge_i, vertex_edge_j,
                                   vertex_edge_data, nvertices, nedges);
    out.Swap(vertex_edge);
}

mfem::SparseMatrix ReadVertexEdge(const std::string& filename)
{
    std::ifstream graph_file(filename);
    mfem::SparseMatrix out;

    ReadVertexEdge(graph_file, out);

    return out;
}

void ReadCoordinate(std::ifstream& graphFile, mfem::SparseMatrix& out)
{
    int nvertices, nedges;
    if (!graphFile.is_open())
        mfem::mfem_error("Error in opening the graph file");
    graphFile >> nvertices;
    graphFile >> nedges;

    int i, j;
    double val;

    mfem::SparseMatrix mat(nvertices, nedges);

    while (graphFile >> i >> j >> val)
    {
        mat.Add(i, j, val);
    }

    mat.Finalize();

    out.Swap(mat);
}

double DivError(MPI_Comm comm, const mfem::SparseMatrix& D, const mfem::Vector& numer,
                const mfem::Vector& denom)
{
    mfem::Vector sigma_diff = denom;
    sigma_diff -= numer;

    mfem::Vector Dfine(D.Height());
    mfem::Vector Ddiff(D.Height());

    D.Mult(sigma_diff, Ddiff);
    D.Mult(denom, Dfine);

    const double error = mfem::ParNormlp(Ddiff, 2, comm) / mfem::ParNormlp(Dfine, 2, comm);

    return error;
}

double CompareError(MPI_Comm comm, const mfem::Vector& numer, const mfem::Vector& denom)
{
    mfem::Vector diff = denom;
    diff -= numer;

    const double error = mfem::ParNormlp(diff, 2, comm) / ParNormlp(denom, 2, comm);

    return error;
}

void ShowErrors(const std::vector<double>& error_info, std::ostream& out, bool pretty)
{
    assert(error_info.size() >= 3);

    picojson::object serialize;
    serialize["finest-p-error"] = picojson::value(error_info[0]);
    serialize["finest-u-error"] = picojson::value(error_info[1]);
    serialize["finest-div-error"] = picojson::value(error_info[2]);

    if (error_info.size() > 3)
    {
        serialize["operator-complexity"] = picojson::value(error_info[3]);
    }

    out << picojson::value(serialize).serialize(pretty) << std::endl;
}

std::vector<double> ComputeErrors(MPI_Comm comm, const mfem::SparseMatrix& M,
                                  const mfem::SparseMatrix& D,
                                  const mfem::BlockVector& upscaled_sol,
                                  const mfem::BlockVector& fine_sol)
{
    mfem::BlockVector M_scaled_up_sol(upscaled_sol);
    mfem::BlockVector M_scaled_fine_sol(fine_sol);

    const double* M_data = M.GetData();

    const int num_edges = upscaled_sol.GetBlock(0).Size();

    for (int i = 0; i < num_edges; ++i)
    {
        M_scaled_up_sol[i] *= std::sqrt(M_data[i]);
        M_scaled_fine_sol[i] *= std::sqrt(M_data[i]);
    }

    std::vector<double> info(3);

    info[0] = CompareError(comm, M_scaled_up_sol.GetBlock(1), M_scaled_fine_sol.GetBlock(1));  // vertex
    info[1] = CompareError(comm, M_scaled_up_sol.GetBlock(0), M_scaled_fine_sol.GetBlock(0));  // edge
    info[2] = DivError(comm, D, upscaled_sol.GetBlock(0), fine_sol.GetBlock(0));   // div

    return info;
}

double PowerIterate(MPI_Comm comm, const mfem::Operator& A, mfem::Vector& result, int max_iter,
                    double tol, bool verbose)
{
    int myid;
    MPI_Comm_rank(comm, &myid);

    mfem::Vector temp(result.Size());

    auto rayleigh = 0.0;
    auto old_rayleigh = 0.0;

    for (int i = 0; i < max_iter; ++i)
    {
        A.Mult(result, temp);

        rayleigh = mfem::InnerProduct(comm, temp, result) / mfem::InnerProduct(comm, result, result);
        temp /= mfem::ParNormlp(temp, 2, comm);

        mfem::Swap(temp, result);

        if (verbose && myid == 0)
        {
            std::cout << std::scientific;
            std::cout << " i: " << i << " ray: " << rayleigh;
            std::cout << " inverse: " << (1.0 / rayleigh);
            std::cout << " rate: " << (std::fabs(rayleigh - old_rayleigh) / rayleigh) << "\n";
        }

        if (std::fabs(rayleigh - old_rayleigh) / std::fabs(rayleigh) < tol)
        {
            break;
        }

        old_rayleigh = rayleigh;
    }

    return rayleigh;
}

void RescaleVector(const mfem::Vector& scaling, mfem::Vector& vec)
{
    for (int i = 0; i < vec.Size(); i++)
    {
        vec[i] *= scaling[i];
    }
}

void GetElementColoring(mfem::Array<int>& colors, const mfem::SparseMatrix& el_el)
{
    const int el0 = 0;

    int num_el = el_el.Size(), stack_p, stack_top_p, max_num_colors;
    mfem::Array<int> el_stack(num_el);

    const int* i_el_el = el_el.GetI();
    const int* j_el_el = el_el.GetJ();

    colors.SetSize(num_el);
    colors = -2;
    max_num_colors = 1;
    stack_p = stack_top_p = 0;
    for (int el = el0; stack_top_p < num_el; el = (el + 1) % num_el)
    {
        if (colors[el] != -2)
        {
            continue;
        }

        colors[el] = -1;
        el_stack[stack_top_p++] = el;

        for ( ; stack_p < stack_top_p; stack_p++)
        {
            int i = el_stack[stack_p];
            int num_nb = i_el_el[i + 1] - i_el_el[i] - 1; // assume nonzero diagonal
            max_num_colors = std::max(max_num_colors, num_nb + 1);
            for (int j = i_el_el[i]; j < i_el_el[i + 1]; j++)
            {
                int k = j_el_el[j];
                if (j == i)
                {
                    continue; // skip self-interaction
                }
                if (colors[k] == -2)
                {
                    colors[k] = -1;
                    el_stack[stack_top_p++] = k;
                }
            }
        }
    }

    mfem::Array<int> color_marker(max_num_colors);
    for (stack_p = 0; stack_p < stack_top_p; stack_p++)
    {
        int i = el_stack[stack_p], color;
        color_marker = 0;
        for (int j = i_el_el[i]; j < i_el_el[i + 1]; j++)
        {
            if (j_el_el[j] == i)
            {
                continue;          // skip self-interaction
            }
            color = colors[j_el_el[j]];
            if (color != -1)
            {
                color_marker[color] = 1;
            }
        }

        for (color = 0; color < max_num_colors; color++)
        {
            if (color_marker[color] == 0)
            {
                break;
            }
        }

        colors[i] = color;
    }
}

std::set<unsigned> FindNonZeroColumns(const mfem::SparseMatrix& mat)
{
    std::set<unsigned> cols;
    int* mat_j = mat.GetJ();
    int* end = mat_j + mat.NumNonZeroElems();
    for (; mat_j != end; mat_j++)
    {
        cols.insert(*mat_j);
    }

    return cols;
}

} // namespace smoothg
