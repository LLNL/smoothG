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

/**
   @file Sampler.hpp
   @brief Contains sampler implementations
*/

#ifndef __SAMPLER_HPP__
#define __SAMPLER_HPP__

#include "smoothG.hpp"

namespace smoothg
{

/** @brief Saves output vectors to file as ("prefix" + index + ".txt")
    @param upscale upscale object to perform permutations
    @param vect local vector to save
    @param prefix filename prefix
    @param index filename suffix
*/
template <typename T>
void SaveOutput(const GraphUpscale& upscale, const T& vect, const std::string& prefix, int index)
{
    std::stringstream ss;
    ss << prefix << std::setw(5) << std::setfill('0') << index << ".txt";

    upscale.WriteVertexVector(vect, ss.str());
}

/** @brief Scalar normal distribution */
class NormalDistribution
{
public:
    /** @brief Constructor setting RNG paramaters
        @param mean mean
        @param stddev standard deviation
        @param seed generator seed
    */
    NormalDistribution(double mean = 0.0, double stddev = 1.0, int seed = 0)
        : generator_(seed), dist_(mean, stddev) { }

    /** @brief Default Destructor */
    ~NormalDistribution() = default;

    /** @brief Generate a random number */
    double Sample() { return dist_(generator_); }

private:
    std::mt19937 generator_;
    std::normal_distribution<double> dist_;
};


/** @brief Provides lognormal random fields with Matern covariance.

    Uses technique from Osborn, Vassilevski, and Villa, A multilevel,
    hierarchical sampling technique for spatially correlated random fields,
    SISC 39 (2017) pp. S543-S562.
*/
class PDESampler
{
public:

    /** @brief Constructor w/ given graph information and upscaling params
        @param graph Graph information
        @param double spect_tol spectral tolerance for upscaling
        @param max_evects maximum number of eigenvectors for upscaling
        @param hybridization use hybridization solver
        @param dimension spatial dimension of the mesh from which the graph originates
        @param cell_volume size of a typical cell
        @param kappa inverse correlation length for Matern covariance
        @param seed seed for random number generator
     */
    PDESampler(Graph graph, double spect_tol, int max_evects, bool hybridization,
               int dimension, double kappa, double cell_volume, int seed);

    /** @brief Default Destructor */
    ~PDESampler() = default;

    /** @brief Generate a new sample.
        @param coarse_sample generate the sample on the coarse level
    */
    void Sample(bool coarse_sample = false);

    /** @brief Access the fine level coefficients */
    const std::vector<double>& GetCoefficientFine() const { return coefficient_fine_; }

    /** @brief Access the coarse level coefficients */
    const std::vector<double>& GetCoefficientCoarse() const { return coefficient_coarse_; }

    /** @brief Access the upscaled coefficients */
    const std::vector<double>& GetCoefficientUpscaled() const { return coefficient_upscaled_; }

    /** @brief Access the GraphUpscale object */
    const GraphUpscale& GetUpscale() const { return upscale_; }

    /** @brief Get the total number of coarse solver iterations. */
    int CoarseTotalIters() const { return total_coarse_iters_; }

    /** @brief Get the total number of fine solver iterations. */
    int FineTotalIters() const { return total_fine_iters_; }

    /** @brief Get the total solve time of the coarse solver. */
    double CoarseTotalTime() const { return total_coarse_time_; }

    /** @brief Get the total solve time of the fine solver. */
    double FineTotalTime() const { return total_fine_time_; }

private:
    GraphUpscale upscale_;

    NormalDistribution normal_dist_;
    double cell_volume_;
    double scalar_g_;

    Vector rhs_fine_;
    Vector rhs_coarse_;

    Vector sol_fine_;
    Vector sol_coarse_;

    std::vector<double> coefficient_fine_;
    std::vector<double> coefficient_coarse_;
    std::vector<double> coefficient_upscaled_;

    Vector constant_coarse_;

    int total_coarse_iters_ = 0;
    int total_fine_iters_ = 0;

    double total_coarse_time_ = 0.0;
    double total_fine_time_ = 0.0;
};


PDESampler::PDESampler(Graph graph, double spect_tol, int max_evects, bool hybridization,
                       int dimension, double kappa, double cell_volume, int seed)
    : upscale_(std::move(graph), spect_tol, max_evects, hybridization),
      normal_dist_(0.0, 1.0, seed),
      cell_volume_(cell_volume),
      rhs_fine_(upscale_.GetFineVector()),
      rhs_coarse_(upscale_.GetCoarseVector()),
      sol_fine_(upscale_.GetFineVector()),
      sol_coarse_(upscale_.GetCoarseVector()),
      coefficient_fine_(upscale_.Rows()),
      coefficient_coarse_(upscale_.NumAggs()),
      coefficient_upscaled_(upscale_.Rows()),
      constant_coarse_(upscale_.GetCoarseConstant())
{
    upscale_.PrintInfo();
    upscale_.ShowSetupTime();

    // Denormalize the coarse constant vector
    constant_coarse_ *= std::sqrt(upscale_.GlobalRows());

    double nu_param = dimension == 2 ? 1.0 : 0.5;
    double ddim = static_cast<double>(dimension);

    scalar_g_ = std::pow(4.0 * M_PI, ddim / 4.0) * std::pow(kappa, nu_param) *
                std::sqrt( std::tgamma(nu_param + ddim / 2.0) / std::tgamma(nu_param) );
}

void PDESampler::Sample(bool coarse_sample)
{
    int fine_size = sol_fine_.size();
    int coarse_size = sol_coarse_.size();

    // Generate Samples
    double g_cell_vol_sqrt = scalar_g_ * std::sqrt(cell_volume_);

    for (int i = 0; i < fine_size; ++i)
    {
        rhs_fine_[i] = g_cell_vol_sqrt * normal_dist_.Sample();
    }

    upscale_.Restrict(rhs_fine_, rhs_coarse_);

    if (coarse_sample)
    {
        // TODO(gelever1): Implement coarse sampling
    }

    // Set Coarse Coefficient
    upscale_.SolveCoarse(rhs_coarse_, sol_coarse_);

    assert(constant_coarse_.size() == coarse_size);

    std::fill(std::begin(coefficient_coarse_), std::end(coefficient_coarse_), 0.0);
    int agg_index = 0;

    for (int i = 0; i < coarse_size; ++i)
    {
        if (std::fabs(constant_coarse_[i]) > 1e-8)
        {
            sol_coarse_[i] = std::exp(sol_coarse_[i] / constant_coarse_[i]);
            coefficient_coarse_[agg_index++] = sol_coarse_[i];
        }
        else
        {
            sol_coarse_[i] = 0.0;
        }
    }

    assert(agg_index == upscale_.NumAggs());

    // Set Fine Coefficient
    upscale_.SolveFine(rhs_fine_, sol_fine_);

    assert(coefficient_fine_.size() == fine_size);

    for (int i = 0; i < fine_size; ++i)
    {
        coefficient_fine_[i] = std::exp(sol_fine_[i]);
    }

    // Set Upscaled Coefficient
    sol_coarse_ *= constant_coarse_;
    VectorView coeff_view(coefficient_upscaled_.data(), coefficient_upscaled_.size());
    upscale_.Interpolate(sol_coarse_, coeff_view);

    // Show/Update Solve Information
    upscale_.ShowCoarseSolveInfo();
    upscale_.ShowFineSolveInfo();

    total_coarse_iters_ += upscale_.GetCoarseSolveIters();
    total_fine_iters_ += upscale_.GetFineSolveIters();

    total_coarse_time_ += upscale_.GetCoarseSolveTime();
    total_fine_time_ += upscale_.GetFineSolveTime();
}


} // namespace smoothg

#endif // __SAMPLER_HPP__
