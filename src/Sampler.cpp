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

    @brief Contains Sampler classes for getting permeability coefficients.
*/

#include "Sampler.hpp"

namespace smoothg
{

NormalDistribution::NormalDistribution(double mean, double stddev, int seed)
    :
    generator_(seed),
    dist_(mean, stddev)
{
}

double NormalDistribution::Sample()
{
    double out = dist_(generator_);
    return out;
}

SimpleSampler::SimpleSampler(std::vector<int>& size)
    :
    sample_(-1), helper_(size.size())
{
    for (unsigned int level = 0; level < size.size(); ++level)
    {
        helper_[level].SetSize(size[level]);
    }
}

void SimpleSampler::NewSample()
{
    sample_++;
}

mfem::Vector& SimpleSampler::GetCoefficient(int level)
{
    MFEM_ASSERT(sample_ >= 0, "SimpleSampler in wrong state (call NewSample() first)!");
    helper_[level] = (1.0 + sample_);
    return helper_[level];
}

PDESampler::PDESampler(int dimension, double kappa, int seed,
                       Hierarchy&& hierarchy)
    : hierarchy_(std::move(hierarchy))
{
    Initialize(dimension, kappa, seed);
}

PDESampler::PDESampler(int dimension, double cell_volume, double kappa, int seed,
                       const Graph& graph,
                       const UpscaleParameters& param,
                       const mfem::Array<int>* partitioning,
                       const mfem::Array<int>* ess_attr)
    : PDESampler(dimension, mfem::Vector(graph.NumVertices()) = cell_volume,
                 kappa, seed, graph, param, partitioning, ess_attr)
{}

PDESampler::PDESampler(int dimension, mfem::Vector cell_volume,
                       double kappa, int seed, const Graph& graph,
                       const UpscaleParameters& param,
                       const mfem::Array<int>* partitioning,
                       const mfem::Array<int>* ess_attr)
{
    MFEM_ASSERT(cell_volume.Size() == graph.NumVertices(), "cell_volume: wrong size!");
    auto W = SparseDiag(std::move(cell_volume)) *= (kappa * kappa);
    hierarchy_ = Hierarchy(graph, param, partitioning, ess_attr, W);
    Initialize(dimension, kappa, seed);
}

void PDESampler::Initialize(int dimension, double kappa, int seed)
{
    normal_distribution_ = NormalDistribution(0.0, 1.0, seed);
    num_aggs_.resize(hierarchy_.NumLevels());
    kappa_ = kappa;
    sampled_ = false;
    rhs_.resize(hierarchy_.NumLevels());
    coefficient_.resize(hierarchy_.NumLevels());

    for (int level = 0; level < hierarchy_.NumLevels(); ++level)
    {
        num_aggs_[level] = hierarchy_.NumVertices(level);
        rhs_[level].SetSize(hierarchy_.GetMatrix(level).NumVDofs());
        coefficient_[level].SetSize(num_aggs_[level]);
    }

    double nu_parameter;
    MFEM_ASSERT(dimension == 2 || dimension == 3, "Invalid dimension!");
    if (dimension == 2)
        nu_parameter = 1.0;
    else
        nu_parameter = 0.5;
    double ddim = static_cast<double>(dimension);
    scalar_g_ = std::pow(4.0 * M_PI, ddim / 4.0) * std::pow(kappa, nu_parameter) *
                std::sqrt( std::tgamma(nu_parameter + ddim / 2.0) / tgamma(nu_parameter) );

    W_sqrt_.resize(hierarchy_.NumLevels());

    const mfem::SparseMatrix& W = hierarchy_.GetMatrix(0).GetW();
    mfem::Vector W_diag(W.GetData(), W.NumRows()); // assume W is diagonal
    mfem::Vector W_sqrt_diag(W.NumRows());
    for (int i = 0; i < W.NumRows(); ++i)
    {
        W_sqrt_diag[i] = std::sqrt(W_diag[i]);
    }
    W_sqrt_[0] = SparseDiag(std::move(W_sqrt_diag));

    // This computes P^T W^{1/2} P, but we may want (P^T W P)^{1/2} instead
    for (int level = 0; level < hierarchy_.NumLevels() - 1; ++level)
    {
        auto& P = hierarchy_.GetPu(level);
        std::unique_ptr<mfem::SparseMatrix> tmp(mfem::RAP(P, W_sqrt_[level], P));
        W_sqrt_[level + 1].Swap(*tmp);
    }
}

PDESampler::~PDESampler()
{
}

void PDESampler::NewSample()
{
    mfem::Vector state(num_aggs_[0]);
    for (int i = 0; i < num_aggs_[0]; ++i)
    {
        state(i) = normal_distribution_.Sample();
    }

    SetSample(state);
}

/// @todo cell_volume should be variable rather than constant
void PDESampler::SetSample(const mfem::Vector& state)
{
    MFEM_ASSERT(state.Size() == num_aggs_[0],
                "state vector is the wrong size!");
    sampled_ = true;

    // build right-hand side for PDE-sampler based on white noise in state
    // (cell_volume is supposed to represent fine-grid W_h)
    const mfem::SparseMatrix& W = hierarchy_.GetMatrix(0).GetW();
    mfem::Vector kappa_sq_cell_volume(W.GetData(), W.NumRows());
    for (int i = 0; i < num_aggs_[0]; ++i)
    {
        rhs_[0](i) = scalar_g_ * std::sqrt(kappa_sq_cell_volume[i]) / kappa_ * state(i);
    }

    for (int level = 0; level < hierarchy_.NumLevels() - 1; ++level)
    {
        hierarchy_.Restrict(level, rhs_[level], rhs_[level + 1]);
    }
}

void PDESampler::SetSampleAtLevel(int level, const mfem::Vector& state)
{
    sampled_ = true;

    W_sqrt_[level].Mult(state, rhs_[level]);
    rhs_[level] *= scalar_g_ / kappa_;
}

mfem::Vector PDESampler::ScaleWhiteNoise(int level, const mfem::Vector& state) const
{
    mfem::Vector out(state.Size());
    W_sqrt_[level].Mult(state, out);
    out *= scalar_g_ / kappa_;
    return out;
}

/**
   Implementation notes:

   c_i comes from solving PDE with white noise on right-hand side
   q_i represents the constant on the coarse mesh

   c_i              : coefficient for coarse basis function, representing ~normal field K
   (c_i / q_i)      : value of ~normal field K on agg i (comes from PWConstProject)
   exp(c_i/q_i)     : value of lognormal field exp(K) on agg i (what this returns, usable to rescale a linear system)
   exp(c_i/q_i) q_i : coefficient for coarse basis function, representing lognormal field exp(K) (what ForVisualization variant returns, usable for projection...)

   indexing: the indexing above is wrong if there is more than one dof / aggregate,
             we consider only the coefficient for the *constant* component i

   @todo: not working multilevel unless restricted to one eigenvector / agg (which maybe is the only sensible case for sampling anyway?)
*/
mfem::Vector& PDESampler::GetCoefficient(int level)
{
    mfem::Vector pw1_coarse_sol = GetLogCoefficient(level);
    for (int i = 0; i < pw1_coarse_sol.Size(); ++i)
    {
        coefficient_[level](i) = std::exp(pw1_coarse_sol(i));
    }
    return coefficient_[level];
}

mfem::Vector PDESampler::GetLogCoefficient(int level)
{
    MFEM_ASSERT(sampled_,
                "PDESampler object in wrong state (call NewSample() first)!");

    mfem::Vector coarse_sol = hierarchy_.Solve(level, rhs_[level]);

    // coarse solution projected to piece-wise constant on aggregates
    mfem::Vector pw1_coarse_sol = hierarchy_.PWConstProject(level, coarse_sol);

    return pw1_coarse_sol;
}

mfem::Vector PDESampler::GetCoefficientForVisualization(int level)
{
    // coarse solution projected to piece-wise constant on aggregates
    mfem::Vector pw1_coarse_sol = GetCoefficient(level);

    // interpolate piece-wise constant function to vertex space
    return hierarchy_.PWConstInterpolate(level, pw1_coarse_sol);
}

mfem::Vector PDESampler::GetLogCoefficientForVisualization(int level)
{
    // coarse solution projected to piece-wise constant on aggregates
    mfem::Vector pw1_coarse_sol = GetLogCoefficient(level);

    // interpolate piece-wise constant function to vertex space
    return hierarchy_.PWConstInterpolate(level, pw1_coarse_sol);
}

}
