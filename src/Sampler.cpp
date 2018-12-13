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

PDESampler::PDESampler(std::shared_ptr<Upscale> fvupscale,
                       int dimension, double cell_volume, double kappa,
                       int seed)
    :
    fvupscale_(fvupscale),
    normal_distribution_(0.0, 1.0, seed),
    num_aggs_(fvupscale->GetNumLevels()),
    cell_volume_(cell_volume),
    sampled_(false),
    rhs_(fvupscale->GetNumLevels()),
    coefficient_(fvupscale->GetNumLevels())
{
    for (int level = 0; level < fvupscale->GetNumLevels(); ++level)
    {
        num_aggs_[level] = fvupscale->GetNumVertices(level);
    }
    Initialize(dimension, kappa);
}

PDESampler::PDESampler(MPI_Comm comm, int dimension,
                       double cell_volume, double kappa, int seed,
                       const mfem::SparseMatrix& vertex_edge,
                       const mfem::Vector& weight,
                       const mfem::Array<int>& partitioning,
                       const mfem::HypreParMatrix& edge_d_td,
                       const mfem::SparseMatrix& edge_boundary_att,
                       const mfem::Array<int>& ess_attr,
                       const UpscaleParameters& param)
    :
    normal_distribution_(0.0, 1.0, seed),
    num_aggs_(param.max_levels),
    cell_volume_(cell_volume),
    sampled_(false),
    rhs_(param.max_levels),
    coefficient_(param.max_levels)
{
    mfem::SparseMatrix W_block = SparseIdentity(vertex_edge.Height());
    W_block *= cell_volume_ * kappa * kappa;

    graph_ = Graph(vertex_edge, edge_d_td, weight, &edge_boundary_att);
    fvupscale_ = std::make_shared<Upscale>(graph_, param, &partitioning,
                                           &ess_attr, W_block);

    for (int level = 0; level < fvupscale_->GetNumLevels(); ++level)
    {
        num_aggs_[level] = fvupscale_->GetNumVertices(level);
    }
    Initialize(dimension, kappa);
}

void PDESampler::Initialize(int dimension, double kappa)
{
    for (int level = 0; level < fvupscale_->GetNumLevels(); ++level)
    {
        rhs_[level] = fvupscale_->GetVector(level);
        // rhs_[level].SetSize(num_aggs_[level]);
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
}

PDESampler::~PDESampler()
{
}

/// @todo cell_volume should be variable rather than constant
void PDESampler::NewSample()
{
    sampled_ = true;

    // construct white noise right-hand side
    // (cell_volume is supposed to represent fine-grid W_h)
    for (int i = 0; i < num_aggs_[0]; ++i)
    {
        rhs_[0](i) = scalar_g_ * std::sqrt(cell_volume_) *
                     normal_distribution_.Sample();
    }
}

mfem::Vector& PDESampler::GetFineCoefficient()
{
    const int level = 0;
    fvupscale_->Solve(level, rhs_[level], coefficient_[level]);
    for (int i = 0; i < coefficient_[level].Size(); ++i)
    {
        coefficient_[level](i) = std::exp(coefficient_[level](i));
    }
    return coefficient_[level];
}

/**
   Implementation notes:

   c_i comes from solving PDE with white noise on right-hand side
   q_i represents the constant on the coarse mesh

   c_i              : coefficient for coarse basis function, representing ~normal field K
   (c_i / q_i)      : value of ~normal field K on agg i
   exp(c_i/q_i)     : value of lognormal field exp(K) on agg i (what this returns)
   exp(c_i/q_i) q_i : coefficient for coarse basis function, representing lognormal field exp(K) (what ForVisualization variant returns)

   indexing: the indexing above is wrong if there is more than one dof / aggregate,
             we consider only the coefficient for the *constant* component i

   @todo: not working multilevel unless restricted to one eigenvector / agg (which maybe is the only sensible case for sampling anyway?)
*/
mfem::Vector& PDESampler::GetCoefficient(int level)
{
    MFEM_ASSERT(sampled_,
                "PDESampler object in wrong state (call NewSample() first)!");

    if (level == 0)
    {
        return GetFineCoefficient();
    }

    for (int k = 0; k < level; ++k)
    {
        fvupscale_->Restrict(k + 1, rhs_[k], rhs_[k + 1]);
    }
    mfem::Vector coarse_sol = fvupscale_->GetVector(level);
    fvupscale_->SolveAtLevel(level, rhs_[level], coarse_sol);

    coefficient_[level] = 0.0;
    mfem::Vector coarse_constant_rep = fvupscale_->GetConstantRep(level);
    MFEM_ASSERT(coarse_constant_rep.Size() == coarse_sol.Size(),
                "PDESampler::GetCoarseCoefficient : Sizes do not match!");
    int agg_index = 0;
    for (int i = 0; i < coarse_sol.Size(); ++i)
    {
        if (std::fabs(coarse_constant_rep(i)) > 1.e-8)
        {
            coefficient_[level](agg_index++) =
                std::exp(coarse_sol(i) / coarse_constant_rep(i));
        }
    }
    MFEM_ASSERT(agg_index == num_aggs_[level], "Something wrong in coarse_constant_rep!");

    return coefficient_[level];
}

mfem::Vector& PDESampler::GetCoefficientForVisualization(int level)
{
    MFEM_ASSERT(sampled_,
                "PDESampler object in wrong state (call NewSample() first)!");
    if (level == 0)
    {
        return GetFineCoefficient();
    }

    for (int i = 0; i < level; ++i)
    {
        fvupscale_->Restrict(i + 1, rhs_[i], rhs_[i + 1]);
    }
    coefficient_[level].SetSize(rhs_[level].Size());
    fvupscale_->SolveAtLevel(level, rhs_[level], coefficient_[level]);

    const mfem::Vector& coarse_constant_rep = fvupscale_->GetConstantRep(level);
    MFEM_ASSERT(coarse_constant_rep.Size() == coefficient_[level].Size(),
                "PDESampler::GetCoarseCoefficient : Sizes do not match!");
    for (int i = 0; i < coefficient_[level].Size(); ++i)
    {
        if (std::fabs(coarse_constant_rep(i)) > 1.e-8)
        {
            coefficient_[level](i) =
                std::exp(coefficient_[level](i) / coarse_constant_rep(i)) * coarse_constant_rep(i);
        }
        else
        {
            coefficient_[level](i) = 0.0;
        }
    }

    return coefficient_[level];
}

}
