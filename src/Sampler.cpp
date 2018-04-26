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

PDESampler::PDESampler(const FiniteVolumeUpscale& fvupscale,
                       int fine_vector_size, int dimension, double cell_volume,
                       double kappa, int seed)
    :
    fvupscale_(fvupscale),
    normal_distribution_(0.0, 1.0, seed),
    fine_vector_size_(fine_vector_size),
    cell_volume_(cell_volume),
    current_state_(NO_SAMPLE)
{
    rhs_fine_.SetSize(fine_vector_size_);
    coefficient_fine_.SetSize(fine_vector_size_);
    rhs_coarse_ = fvupscale_.GetCoarseVector();
    coefficient_coarse_ = fvupscale_.GetCoarseVector();

    double nu_parameter;
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
void PDESampler::Sample()
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

void PDESampler::CoarseSample()
{
    current_state_ = COARSE_SAMPLE;
    MFEM_ASSERT(false, "Not implemented!");
}

mfem::Vector& PDESampler::GetFineCoefficient()
{
    MFEM_ASSERT(current_state_ == FINE_SAMPLE,
                "PDESampler object in wrong state (call Sample() first)!");

    fvupscale_.SolveFine(rhs_fine_, coefficient_fine_);
    return coefficient_fine_;
}

mfem::Vector& PDESampler::GetCoarseCoefficient()
{
    MFEM_ASSERT(current_state_ == FINE_SAMPLE ||
                current_state_ == COARSE_SAMPLE,
                "PDESampler object in wrong state (call Sample() first)!");

    if (current_state_ == FINE_SAMPLE)
        fvupscale_.Restrict(rhs_fine_, rhs_coarse_);
    fvupscale_.SolveCoarse(rhs_coarse_, coefficient_coarse_);
    coefficient_coarse_ *= -1.0; // ??
    return coefficient_coarse_;
}

}
