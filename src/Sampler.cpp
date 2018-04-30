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

SimpleSampler::SimpleSampler(int fine_size, int coarse_size)
    :
    fine_size_(fine_size), coarse_size_(coarse_size), sample_(-1)
{
    fine_.SetSize(fine_size_);
    coarse_.SetSize(coarse_size_);
}

void SimpleSampler::NewSample()
{
    sample_++;
}

mfem::Vector& SimpleSampler::GetFineCoefficient()
{
    MFEM_ASSERT(sample_ >= 0, "SimpleSampler in wrong state (call NewSample() first)!");
    fine_ = (1.0 + sample_);
    return fine_;
}

mfem::Vector& SimpleSampler::GetCoarseCoefficient()
{
    MFEM_ASSERT(sample_ >= 0, "SimpleSampler in wrong state (call NewSample() first)!");
    coarse_ = (1.0 + sample_);
    return coarse_;
}

PDESampler::PDESampler(const Upscale& fvupscale, int fine_vector_size, int coarse_aggs,
                       int dimension, double cell_volume, double kappa, int seed)
    :
    fvupscale_(fvupscale),
    normal_distribution_(0.0, 1.0, seed),
    fine_vector_size_(fine_vector_size),
    num_coarse_aggs_(coarse_aggs),
    cell_volume_(cell_volume),
    current_state_(NO_SAMPLE)
{
    rhs_fine_.SetSize(fine_vector_size_);
    coefficient_fine_.SetSize(fine_vector_size_);
    rhs_coarse_ = fvupscale_.GetCoarseVector();
    coefficient_coarse_.SetSize(coarse_aggs);

    double nu_parameter;
    MFEM_ASSERT(dimension == 2 || dimension == 3, "Invalid dimension!");
    if (dimension == 2)
        nu_parameter = 1.0;
    else
        nu_parameter = 0.5;
    double ddim = static_cast<double>(dimension);
    scalar_g_ = std::pow(4.0 * M_PI, ddim / 4.0) * std::pow(kappa, nu_parameter) *
                std::sqrt( tgamma(nu_parameter + ddim / 2.0) / tgamma(nu_parameter) );
}

PDESampler::~PDESampler()
{
}

/// @todo cell_volume should be variable rather than constant
void PDESampler::NewSample()
{
    current_state_ = FINE_SAMPLE;

    // construct white noise right-hand side
    // (cell_volume is supposed to represent fine-grid W_h)
    for (int i = 0; i < fine_vector_size_; ++i)
    {
        rhs_fine_(i) = scalar_g_ * std::sqrt(cell_volume_) *
                       normal_distribution_.Sample();
    }
}

void PDESampler::NewCoarseSample()
{
    current_state_ = COARSE_SAMPLE;
    MFEM_ASSERT(false, "Not implemented!");
}

mfem::Vector& PDESampler::GetFineCoefficient()
{
    MFEM_ASSERT(current_state_ == FINE_SAMPLE,
                "PDESampler object in wrong state (call NewSample() first)!");

    fvupscale_.SolveFine(rhs_fine_, coefficient_fine_);
    for (int i = 0; i < coefficient_fine_.Size(); ++i)
        coefficient_fine_(i) = std::exp(coefficient_fine_(i));
    return coefficient_fine_;
}

mfem::Vector& PDESampler::GetCoarseCoefficient()
{
    MFEM_ASSERT(current_state_ == FINE_SAMPLE ||
                current_state_ == COARSE_SAMPLE,
                "PDESampler object in wrong state (call NewSample() first)!");

    if (current_state_ == FINE_SAMPLE)
        fvupscale_.Restrict(rhs_fine_, rhs_coarse_);
    mfem::Vector coarse_sol = fvupscale_.GetCoarseVector();
    fvupscale_.SolveCoarse(rhs_coarse_, coarse_sol);
    coarse_sol *= -1.0;

    coefficient_coarse_ = 0.0;
    const mfem::Vector& coarse_constant_rep = fvupscale_.GetGraphCoarsen().GetCoarseConstantRep();
    MFEM_ASSERT(coarse_constant_rep.Size() == coarse_sol.Size(),
                "PDESampler::GetCoarseCoefficient : Sizes do not match!");
    int agg_index = 0;
    for (int i = 0; i < coarse_sol.Size(); ++i)
    {
        if (std::fabs(coarse_constant_rep(i)) > 1.e-8)
        {
            coefficient_coarse_(agg_index++) =
                std::exp(coarse_sol(i) / coarse_constant_rep(i)) * coarse_constant_rep(i);
        }
    }
    MFEM_ASSERT(agg_index == num_coarse_aggs_, "Something wrong in coarse_constant_rep!");

    return coefficient_coarse_;
}

/// @todo ugly hack for debugging, @deprecated
mfem::Vector& PDESampler::GetCoarseCoefficientForVisualization()
{
    MFEM_ASSERT(current_state_ == FINE_SAMPLE ||
                current_state_ == COARSE_SAMPLE,
                "PDESampler object in wrong state (call NewSample() first)!");

    if (current_state_ == FINE_SAMPLE)
        fvupscale_.Restrict(rhs_fine_, rhs_coarse_);
    coefficient_coarse_.SetSize(rhs_coarse_.Size());
    fvupscale_.SolveCoarse(rhs_coarse_, coefficient_coarse_);
    coefficient_coarse_ *= -1.0; // ??

    const mfem::Vector& coarse_constant_rep = fvupscale_.GetGraphCoarsen().GetCoarseConstantRep();
    MFEM_ASSERT(coarse_constant_rep.Size() == coefficient_coarse_.Size(),
                "PDESampler::GetCoarseCoefficient : Sizes do not match!");
    for (int i = 0; i < coefficient_coarse_.Size(); ++i)
    {
        coefficient_coarse_(i) =
            std::exp(coefficient_coarse_(i) / coarse_constant_rep(i)) * coarse_constant_rep(i);
    }

    return coefficient_coarse_;
}

}
