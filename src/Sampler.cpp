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

const mfem::Vector& SimpleSampler::GetFineCoefficient()
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

PDESampler::PDESampler(std::shared_ptr<const Upscale> fvupscale, int fine_vector_size,
                       int coarse_aggs, int dimension, double cell_volume, double kappa,
                       int seed)
    :
    fvupscale_(fvupscale),
    normal_distribution_(0.0, 1.0, seed),
    fine_vector_size_(fine_vector_size),
    num_coarse_aggs_(coarse_aggs),
    cell_volume_(cell_volume),
    current_state_(NO_SAMPLE)
{
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
    fine_vector_size_(vertex_edge.Height()),
    num_coarse_aggs_(partitioning.Max() + 1),
    cell_volume_(cell_volume),
    current_state_(NO_SAMPLE)
{
    mfem::SparseMatrix W_block = SparseIdentity(vertex_edge.Height());
    W_block *= cell_volume_ * kappa * kappa;

    graph_ = Graph(vertex_edge, edge_d_td, weight);
    fvupscale_ = std::make_shared<Upscale>(graph_, &partitioning,
                                           &edge_boundary_att, &ess_attr, param, W_block);
    Initialize(dimension, kappa);
}

void PDESampler::Initialize(int dimension, double kappa)
{
    rhs_fine_.SetSize(fine_vector_size_);
    coefficient_fine_.SetSize(fine_vector_size_);
    rhs_coarse_ = fvupscale_->GetVector(1);
    coefficient_coarse_.SetSize(num_coarse_aggs_);

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

const mfem::Vector& PDESampler::GetFineCoefficient()
{
    MFEM_ASSERT(current_state_ == FINE_SAMPLE,
                "PDESampler object in wrong state (call NewSample() first)!");

    fvupscale_->Solve(0, rhs_fine_, coefficient_fine_);
    for (int i = 0; i < coefficient_fine_.Size(); ++i)
    {
        coefficient_fine_(i) = std::exp(coefficient_fine_(i));
    }
    return coefficient_fine_;
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
*/
mfem::Vector& PDESampler::GetCoarseCoefficient()
{
    MFEM_ASSERT(current_state_ == FINE_SAMPLE ||
                current_state_ == COARSE_SAMPLE,
                "PDESampler object in wrong state (call NewSample() first)!");

    if (current_state_ == FINE_SAMPLE)
        fvupscale_->Restrict(1, rhs_fine_, rhs_coarse_);
    mfem::Vector coarse_sol = fvupscale_->GetVector(1);
    fvupscale_->SolveAtLevel(1, rhs_coarse_, coarse_sol);

    coefficient_coarse_ = 0.0;
    mfem::Vector coarse_constant_rep = fvupscale_->GetConstantRep(1);
    MFEM_ASSERT(coarse_constant_rep.Size() == coarse_sol.Size(),
                "PDESampler::GetCoarseCoefficient : Sizes do not match!");
    int agg_index = 0;
    for (int i = 0; i < coarse_sol.Size(); ++i)
    {
        if (std::fabs(coarse_constant_rep(i)) > 1.e-8)
        {
            coefficient_coarse_(agg_index++) =
                std::exp(coarse_sol(i) / coarse_constant_rep(i));
        }
    }
    MFEM_ASSERT(agg_index == num_coarse_aggs_, "Something wrong in coarse_constant_rep!");

    return coefficient_coarse_;
}

mfem::Vector& PDESampler::GetCoarseCoefficientForVisualization()
{
    MFEM_ASSERT(current_state_ == FINE_SAMPLE ||
                current_state_ == COARSE_SAMPLE,
                "PDESampler object in wrong state (call NewSample() first)!");

    if (current_state_ == FINE_SAMPLE)
        fvupscale_->Restrict(1, rhs_fine_, rhs_coarse_);
    coefficient_coarse_.SetSize(rhs_coarse_.Size());
    fvupscale_->SolveAtLevel(1, rhs_coarse_, coefficient_coarse_);

    const mfem::Vector& coarse_constant_rep = fvupscale_->GetConstantRep(1);
    MFEM_ASSERT(coarse_constant_rep.Size() == coefficient_coarse_.Size(),
                "PDESampler::GetCoarseCoefficient : Sizes do not match!");
    for (int i = 0; i < coefficient_coarse_.Size(); ++i)
    {
        if (std::fabs(coarse_constant_rep(i)) > 1.e-8)
        {
            coefficient_coarse_(i) =
                std::exp(coefficient_coarse_(i) / coarse_constant_rep(i)) * coarse_constant_rep(i);
        }
        else
        {
            coefficient_coarse_(i) = 0.0;
        }
    }

    return coefficient_coarse_;
}

}
