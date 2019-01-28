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
   @file

   @brief Implements Mixed_GL_Coarsener
*/

#include "Mixed_GL_Coarsener.hpp"
#include <assert.h>

namespace smoothg
{

const mfem::SparseMatrix& Mixed_GL_Coarsener::GetPu() const
{
    check_subspace_construction_("Pu");
    return Pu_;
}

const mfem::SparseMatrix& Mixed_GL_Coarsener::GetPsigma() const
{
    check_subspace_construction_("Psigma");
    return Psigma_;
}

void Mixed_GL_Coarsener::Restrict(const mfem::BlockVector& fine_vect,
                                  mfem::BlockVector& coarse_vect) const
{
    Psigma_.MultTranspose(fine_vect.GetBlock(0), coarse_vect.GetBlock(0));
    Pu_.MultTranspose(fine_vect.GetBlock(1), coarse_vect.GetBlock(1));
}

void Mixed_GL_Coarsener::Interpolate(const mfem::BlockVector& coarse_vect,
                                     mfem::BlockVector& fine_vect) const
{
    Psigma_.Mult(coarse_vect.GetBlock(0), fine_vect.GetBlock(0));
    Pu_.Mult(coarse_vect.GetBlock(1), fine_vect.GetBlock(1));
}

void Mixed_GL_Coarsener::Project(const mfem::BlockVector& fine_vect,
                                 mfem::BlockVector& coarse_vect) const
{
    Proj_sigma_.Mult(fine_vect.GetBlock(0), coarse_vect.GetBlock(0));
    Pu_.MultTranspose(fine_vect.GetBlock(1), coarse_vect.GetBlock(1));
}

void Mixed_GL_Coarsener::Restrict(const mfem::Vector& fine_vect,
                                  mfem::Vector& coarse_vect) const
{
    Pu_.MultTranspose(fine_vect, coarse_vect);
}

void Mixed_GL_Coarsener::Interpolate(const mfem::Vector& coarse_vect,
                                     mfem::Vector& fine_vect) const
{
    Pu_.Mult(coarse_vect, fine_vect);
}

void Mixed_GL_Coarsener::Project(const mfem::Vector& fine_vect,
                                 mfem::Vector& coarse_vect) const
{
    Pu_.MultTranspose(fine_vect, coarse_vect);
}

void Mixed_GL_Coarsener::Debug_tests(const mfem::SparseMatrix& D) const
{
    mfem::Vector random_vec(Proj_sigma_.Height());
    random_vec.Randomize();

    const double error_tolerance = 5e-10;

    mfem::Vector Psigma_rand(Psigma_.Height());
    Psigma_.Mult(random_vec, Psigma_rand);
    mfem::Vector out(Proj_sigma_.Height());
    Proj_sigma_.Mult(Psigma_rand, out);

    out -= random_vec;
    double diff = out.Norml2();
    if (diff >= error_tolerance)
    {
        std::cerr << "|| rand - Proj_sigma_ * Psigma_ * rand || = " << diff
                  << "\nEdge projection operator is not a projection!\n";
    }
    assert(diff < error_tolerance);

    random_vec.SetSize(Psigma_.Height());
    random_vec.Randomize();

    // Compute D * pi_sigma * random vector
    mfem::Vector D_pi_sigma_rand(D.Height());
    {
        mfem::Vector Proj_sigma_rand(Proj_sigma_.Height());
        Proj_sigma_.Mult(random_vec, Proj_sigma_rand);
        mfem::Vector pi_sigma_rand(Psigma_.Height());
        Psigma_.Mult(Proj_sigma_rand, pi_sigma_rand);
        D.Mult(pi_sigma_rand, D_pi_sigma_rand);
    }

    // Compute pi_u * D * random vector
    mfem::Vector pi_u_D_rand(D.Height());
    {
        mfem::Vector D_rand(D.Height());
        D.Mult(random_vec, D_rand);
        mfem::Vector PuT_D_rand(Pu_.Width());
        Pu_.MultTranspose(D_rand, PuT_D_rand);
        Pu_.Mult(PuT_D_rand, pi_u_D_rand);
    }

    pi_u_D_rand -= D_pi_sigma_rand;
    diff = pi_u_D_rand.Norml2();
    if (diff >= error_tolerance)
    {
        std::cerr << "|| pi_u * D * rand - D * pi_sigma * rand || = " << diff
                  << "\nCommutativity does not hold!\n";
    }
    assert(diff < error_tolerance);
}

} // namespace smoothg

