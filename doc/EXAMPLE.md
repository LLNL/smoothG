<!-- BHEADER ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 +
 + Copyright (c) 2018, Lawrence Livermore National Security, LLC.
 + Produced at the Lawrence Livermore National Laboratory.
 + LLNL-CODE-745247. All Rights reserved. See file COPYRIGHT for details.
 +
 + This file is part of smoothG. For more information and source code
 + availability, see https://www.github.com/llnl/smoothG.
 +
 + smoothG is free software; you can redistribute it and/or modify it under the
 + terms of the GNU Lesser General Public License (as published by the Free
 + Software Foundation) version 2.1 dated February 1999.
 +
 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ EHEADER -->

A walkthrough of example code           {#EXAMPLE}
==================

Here we take you through the `generalgraph.cpp` example to give you an idea of how this library works.

### The problem

One important problem in computational graph algorithms is to find the [Fiedler vector](https://en.wikipedia.org/wiki/Algebraic_connectivity),
which can be used to partition a network into communities.
For a connected graph, the Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue of the graph Laplacian.

To keep things simple, in this example we do not solve an eigenvalue problem but instead a single linear system.
`smoothG` ships with an example graph and a precomputed Fiedler vector.
In this example we load that data and compare an upscaled approximation to the fine-scale Fiedler vector.

In mathematical terms, we are given a graph-Laplacian \f$ A \f$ and a Fiedler vector \f$ x \f$ on a fine-scale graph.
We know
\f[
  Ax = \lambda x ,
\f]
and the main purpose of this library is to produce a smaller, easier to solve, *upscaled* version of the graph,
which we can think of as having a graph Laplacian matrix \f$ A_c \f$.
This code then computes a coarse Fiedler vector \f$ x_c \f$ from
\f[
  A_c x_c = \lambda P^T x
\f]
and compares it to \f$ x \f$ to see how accurate our upscaling is.

### The code

After some initialization and reading command line arguments, the first substantive code is to read a (mixed) graph from a file:

\snippet generalgraph.cpp Load graph from file or generate one

We consider graphs in a *vertex-edge* format, which means we interpret them as matrices with as many rows as there are vertices in the graph,
and as many columns as there are edges.
This matrix has a 1 wherever a vertex and an edge are connected, and 0 elsewhere.
As a result, each column has exactly two nonzeros, since each edge is connected to exactly two vertices in a graph.

After loading the graph, we partition the vertices into *agglomerated vertices*, which are key to our upscaling approach.
If you have a partitioning you like, you can provide it in a file.
Otherwise we use Metis to partition the vertices into agglomerates, putting the results into a partitioning vector `global_partitioning` which,
for each vertex, indicates which agglomerate it is in.

\snippet generalgraph.cpp Partitioning

If the user has provided edge weights to use, we load them from file.  Otherwise, all weights are set to 1.

\snippet generalgraph.cpp Load the edge weights

The next step is to create an [GraphUpscale](@ref smoothg::GraphUpscale) object.
Given some user parameters, the [GraphUpscale](@ref smoothg::GraphUpscale) object handles the construction of the coarse space.
\snippet generalgraph.cpp Upscale

The right hand side can be read from file or computed as the Fielder vector of the fine level graph.
It is then set into the appropriate mixed form.
\snippet generalgraph.cpp Right Hand Side

The next major step is to actually solve the system.
Both the upscaled solution and the fine level solution can be computed by the [Upscale](@ref smoothg::Upscale) object.
\snippet generalgraph.cpp Solve

The last few lines of the code compare the solutions on each level, to see how close the upscaled model is to the original fine-scale graph.
\snippet generalgraph.cpp Check Error
