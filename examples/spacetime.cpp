//                       MFEM Example 5 - Parallel Version
//
// Compile with: make ex5p
//
// Sample runs:  mpirun -np 4 ex5p -m ../data/square-disc.mesh
//               mpirun -np 4 ex5p -m ../data/star.mesh
//               mpirun -np 4 ex5p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex5p -m ../data/beam-hex.mesh
//               mpirun -np 4 ex5p -m ../data/escher.mesh
//               mpirun -np 4 ex5p -m ../data/fichera.mesh
//
// Description:  This example code solves a simple 2D/3D mixed Darcy problem
//               corresponding to the saddle point system
//                                 k*u + grad p = f
//                                 - div u      = g
//               with natural boundary condition -p = <given pressure>.
//               Here, we use a given exact solution (u,p) and compute the
//               corresponding r.h.s. (f,g).  We discretize with Raviart-Thomas
//               finite elements (velocity u) and piecewise discontinuous
//               polynomials (pressure p).
//
//               The example demonstrates the use of the BlockMatrix class, as
//               well as the collective saving of several grid functions in a
//               VisIt (visit.llnl.gov) visualization format.
//
//               We recommend viewing examples 1-4 before viewing this example.

#include "mfem.hpp"
#include "../src/smoothG.hpp"

#include <fstream>
#include <iostream>
#include <assert.h>
#include <memory>

using namespace std;
using namespace mfem;
using namespace smoothg;

// Define the analytical solution and forcing terms / boundary conditions
double uFun_ex(const Vector& xt); // Exact Solution
double fFun(const Vector& xt); // Source f
void bfFun_ex(const Vector& xt, Vector& gradf);
void gradfFun_ex(const Vector& xt, Vector& gradf);
void sigmaFun_ex(const Vector& xt, Vector& sigma);
double bTb_ex(const Vector& xt);
void bFun_ex (const Vector& xt, Vector& b);
void Ktilda_ex(const Vector& xt, DenseMatrix& Ktilda);
void bbT_ex(const Vector& xt, DenseMatrix& bbT);

//// Geometric Multigrid
class Multigrid : public Solver
{
public:
    Multigrid(HypreParMatrix &Operator,
              const Array<HypreParMatrix*> &P,
              Solver *CoarsePrec=NULL)
        :
          Solver(Operator.GetNumRows()),
          P_(P),
          Operators_(P.Size()+1),
          Smoothers_(Operators_.Size()),
          current_level(Operators_.Size()-1),
          correction(Operators_.Size()),
          residual(Operators_.Size()),
          CoarseSolver(NULL),
          CoarsePrec_(CoarsePrec)
    {
        if (CoarsePrec)
        {
            CoarseSolver = new CGSolver(Operators_[0]->GetComm());
            CoarseSolver->SetRelTol(1e-8);
            CoarseSolver->SetMaxIter(50);
            CoarseSolver->SetPrintLevel(0);
            CoarseSolver->SetOperator(*Operators_[0]);
            CoarseSolver->SetPreconditioner(*CoarsePrec);
        }

        Operators_.Last() = &Operator;
        for (int l = Operators_.Size()-1; l > 0; l--)
        {
            // Two steps RAP
            unique_ptr<HypreParMatrix> PT( P[l-1]->Transpose() );
            unique_ptr<HypreParMatrix> AP( ParMult(Operators_[l], P[l-1]) );
            Operators_[l-1] = ParMult(PT.get(), AP.get());
            Operators_[l-1]->CopyRowStarts();
        }

        for (int l = 0; l < Operators_.Size(); l++)
        {
            Smoothers_[l] = new HypreSmoother(*Operators_[l]);
            correction[l] = new Vector(Operators_[l]->GetNumRows());
            residual[l] = new Vector(Operators_[l]->GetNumRows());
        }
    }

    virtual void Mult(const Vector & x, Vector & y) const;

    virtual void SetOperator(const Operator &op) { }

    ~Multigrid()
    {
        for (int l = 0; l < Operators_.Size(); l++)
        {
            delete Smoothers_[l];
            delete correction[l];
            delete residual[l];
        }
    }

private:
    void MG_Cycle() const;

    const Array<HypreParMatrix*> &P_;

    Array<HypreParMatrix*> Operators_;
    Array<HypreSmoother*> Smoothers_;

    mutable int current_level;

    mutable Array<Vector*> correction;
    mutable Array<Vector*> residual;

    mutable Vector res_aux;
    mutable Vector cor_cor;
    mutable Vector cor_aux;

    CGSolver *CoarseSolver;
    Solver *CoarsePrec_;
};

void Multigrid::Mult(const Vector & x, Vector & y) const
{
    *residual.Last() = x;
    correction.Last()->SetDataAndSize(y.GetData(), y.Size());
    MG_Cycle();
}

void Multigrid::MG_Cycle() const
{
    // PreSmoothing
    const HypreParMatrix& Operator_l = *Operators_[current_level];
    const HypreSmoother& Smoother_l = *Smoothers_[current_level];

    Vector& residual_l = *residual[current_level];
    Vector& correction_l = *correction[current_level];

    Smoother_l.Mult(residual_l, correction_l);
    Operator_l.Mult(-1.0, correction_l, 1.0, residual_l);

    // Coarse grid correction
    if (current_level > 0)
    {
        const HypreParMatrix& P_l = *P_[current_level-1];

        P_l.MultTranspose(residual_l, *residual[current_level-1]);

        current_level--;
        MG_Cycle();
        current_level++;

        cor_cor.SetSize(residual_l.Size());
        P_l.Mult(*correction[current_level-1], cor_cor);
        correction_l += cor_cor;
        Operator_l.Mult(-1.0, cor_cor, 1.0, residual_l);
    }
    else
    {
        cor_cor.SetSize(residual_l.Size());
        if (CoarseSolver)
        {
            CoarseSolver->Mult(residual_l, cor_cor);
            correction_l += cor_cor;
            Operator_l.Mult(-1.0, cor_cor, 1.0, residual_l);
        }
    }

    // PostSmoothing
    Smoother_l.Mult(residual_l, cor_cor);
    correction_l += cor_cor;
}

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   bool verbose = (myid == 0);

   // 2. Parse command-line options.
   int order = 0;
   bool visualization = 1;
   bool divfree = 0;
   bool GMG = 0;
   int par_ref_levels = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&divfree, "-df", "--divfree", "-no-df",
                  "--no-divfree",
                  "whether to use the divergence free solver or not.");
   args.AddOption(&GMG, "-GMG", "--GeometricMG", "-AMG",
                  "--AlgebraicMG",
                  "whether to use goemetric or algebraic multigrid solver.");
   args.AddOption(&par_ref_levels, "-r", "--ref",
                     "Number of parallel refinement steps.");
   args.Parse();
   if (!args.Good())
   {
      if (verbose)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (verbose)
   {
      args.PrintOptions(cout);
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(2, 2, 2, mfem::Element::HEXAHEDRON, 1);
//   std::ifstream imesh("/Users/lee1029/Codes/meshes/cylinder_mfem.mesh3d");
//   Mesh *mesh = new Mesh(imesh, 1, 1);
//   imesh.close();
   int dim = mesh->Dimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   //   ess_bdr.Last() = 0;

   // Constructing multigrid hierarchy while refining the mesh (if GMG is true)
   auto *hcurl_coll = new ND_FECollection(order+1, dim);
   auto *h1_coll = new H1_FECollection(order+1, dim);
   ParFiniteElementSpace *N_space, *H_space;
   Array<HypreParMatrix*> P(par_ref_levels);
   Array<HypreParMatrix*> DiscreteGrads(par_ref_levels+1);
   if (divfree)
   {
       chrono.Clear();
       chrono.Start();

       if (GMG)
       {
           N_space = new ParFiniteElementSpace(pmesh, hcurl_coll);
           H_space = new ParFiniteElementSpace(pmesh, h1_coll, dim);

           ParDiscreteLinearOperator DiscreteGradForm(H_space, N_space);
           DiscreteGradForm.AddDomainInterpolator(new IdentityInterpolator);
           DiscreteGradForm.Assemble();
           Vector vec1(H_space->GetVSize());
           Vector vec2(N_space->GetVSize());
           DiscreteGradForm.EliminateTrialDofs(ess_bdr, vec1, vec2);
           DiscreteGradForm.Finalize();
           DiscreteGrads[0] = DiscreteGradForm.ParallelAssemble();
           DiscreteGrads[0]->CopyColStarts();
           DiscreteGrads[0]->CopyRowStarts();

           auto coarse_N_space = new ParFiniteElementSpace(pmesh, hcurl_coll);
           const SparseMatrix *P_local;

           for (int l = 0; l < par_ref_levels; l++)
           {
               coarse_N_space->Update();

               pmesh->UniformRefinement();
               P_local = ((const SparseMatrix*)N_space->GetUpdateOperator());

               H_space->Update();
               DiscreteGradForm.Update();
               DiscreteGradForm.Assemble();
               if (l < par_ref_levels)
               {
                   vec1.SetSize(H_space->GetVSize());
                   vec2.SetSize(N_space->GetVSize());
                   DiscreteGradForm.EliminateTrialDofs(ess_bdr, vec1, vec2);
               }
               DiscreteGradForm.Finalize();
               DiscreteGrads[l+1] = DiscreteGradForm.ParallelAssemble();
               DiscreteGrads[l+1]->CopyColStarts();
               DiscreteGrads[l+1]->CopyRowStarts();

               auto P_local_copy = DropSmall(*P_local, 1e-8);

               auto d_td_coarse = coarse_N_space->Dof_TrueDof_Matrix();
               auto RP_local = smoothg::Mult(*N_space->GetRestrictionMatrix(), P_local_copy);

               P[l] = d_td_coarse->LeftDiagMult(RP_local, N_space->GetTrueDofOffsets());
               P[l]->CopyColStarts();
               P[l]->CopyRowStarts();
           }

           delete coarse_N_space;
       }
       else
       {
           for (int l = 0; l < par_ref_levels; l++)
               pmesh->UniformRefinement();
           N_space = new ParFiniteElementSpace(pmesh, hcurl_coll);
       }
       if (verbose)
           cout << "Divergence free hierarchy constructed in "
                << chrono.RealTime() << endl;
   }
   else
   {
       for (int l = 0; l < par_ref_levels; l++)
           pmesh->UniformRefinement();
   }

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
   FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));

   ParFiniteElementSpace *R_space = new ParFiniteElementSpace(pmesh, hdiv_coll);
   ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh, l2_coll);

   HYPRE_Int dimR = R_space->GlobalTrueVSize();
   HYPRE_Int dimW = W_space->GlobalTrueVSize();
   HYPRE_Int dimN = divfree ? N_space->GlobalTrueVSize() : 0;

   if (verbose)
   {
      std::cout << "***********************************************************\n";
      std::cout << "dim(R) = " << dimR << "\n";
      std::cout << "dim(W) = " << dimW << "\n";
      std::cout << "dim(R+W) = " << dimR + dimW << "\n";
      if (divfree)
          std::cout << "dim(N) = " << dimN << "\n";
      std::cout << "***********************************************************\n";
   }

   // 7. Define the two BlockStructure of the problem.  block_offsets is used
   //    for Vector based on dof (like ParGridFunction or ParLinearForm),
   //    block_trueOffstes is used for Vector based on trueDof (HypreParVector
   //    for the rhs and solution of the linear system).  The offsets computed
   //    here are local to the processor.
   Array<int> block_offsets(3); // number of variables + 1
   block_offsets[0] = 0;
   block_offsets[1] = R_space->GetVSize();
   block_offsets[2] = W_space->GetVSize();
   block_offsets.PartialSum();

   Array<int> block_trueOffsets(3); // number of variables + 1
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = R_space->TrueVSize();
   block_trueOffsets[2] = W_space->TrueVSize();
   block_trueOffsets.PartialSum();

   // 8. Define the coefficients, analytical solution, and rhs of the PDE.
   MatrixFunctionCoefficient Ktilde(dim, Ktilda_ex);

   FunctionCoefficient fcoeff(fFun);
   FunctionCoefficient ucoeff(uFun_ex);

   VectorFunctionCoefficient sigmacoeff(dim, sigmaFun_ex);

   // 9. Define the parallel grid function and parallel linear forms, solution
   //    vector and rhs.
   BlockVector x(block_offsets), rhs(block_offsets);
   x = 0.0; rhs = 0.0;
   BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);
   trueX = 0.0; trueRhs = 0.0;

   ParGridFunction sigma_gf(R_space, x.GetBlock(0).GetData());
   sigma_gf.ProjectBdrCoefficientNormal(sigmacoeff, ess_bdr);

   ParLinearForm *gform(new ParLinearForm);
   gform->Update(R_space, rhs.GetBlock(0), 0);
   gform->Assemble();

   ParLinearForm *fform(new ParLinearForm);
   fform->Update(W_space, rhs.GetBlock(1), 0);
   fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
   fform->Assemble();

   // 10. Assemble the finite element matrices for the Darcy operator
   //
   //                            D = [ M  B^T ]
   //                                [ B   0  ]
   //     where:
   //
   //     M = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
   //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
   ParBilinearForm *mVarf(new ParBilinearForm(R_space));
   ParMixedBilinearForm *bVarf(new ParMixedBilinearForm(R_space, W_space));

   mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(Ktilde));
   mVarf->Assemble();
   mVarf->EliminateEssentialBC(ess_bdr, x.GetBlock(0), *gform);
   mVarf->Finalize();
   HypreParMatrix *M = mVarf->ParallelAssemble();

   bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   bVarf->Assemble();
   bVarf->EliminateTrialDofs(ess_bdr, x.GetBlock(0), *fform);
   bVarf->Finalize();
   HypreParMatrix *B = bVarf->ParallelAssemble();
   HypreParMatrix *BT = B->Transpose();

   gform->ParallelAssemble(trueRhs.GetBlock(0));
   fform->ParallelAssemble(trueRhs.GetBlock(1));

   int maxIter(50000);
   double rtol(1.e-9);
   double atol(1.e-12);

   Operator *darcyOp;
   Solver *darcyPr;
   if (divfree)
   {
       chrono.Clear();
       chrono.Start();

       StopWatch chrono_local;
       chrono_local.Clear();
       chrono_local.Start();

       // Find a particular solution for div sigma = f
       auto BBT = ParMult(B, BT);
       trueX.GetBlock(1) = 0.0;
       auto prec_particular = new HypreBoomerAMG(*BBT);
       prec_particular->SetPrintLevel(0);

       CGSolver solver_particular(MPI_COMM_WORLD);
       solver_particular.SetAbsTol(atol);
       solver_particular.SetRelTol(rtol);
       solver_particular.SetMaxIter(maxIter);
       solver_particular.SetOperator(*BBT);
       solver_particular.SetPreconditioner(*prec_particular);
       solver_particular.SetPrintLevel(0);
       solver_particular.Mult(trueRhs.GetBlock(1), trueX.GetBlock(1));
       Vector sol_particular(BT->GetNumRows());
       sol_particular = 0.0;
       BT->Mult(trueX.GetBlock(1), sol_particular);

       //correct essential bc
       R_space->GetRestrictionMatrix()->Mult(x.GetBlock(0), trueX.GetBlock(0));
       sol_particular += trueX.GetBlock(0);

       chrono_local.Stop();

       if (verbose)
       {
          if (solver_particular.GetConverged())
             cout << "CG converged in " << solver_particular.GetNumIterations()
                       << " iterations with a residual norm of " << solver_particular.GetFinalNorm() << ".\n";
          else
             cout << "CG did not converge in " << solver_particular.GetNumIterations()
                       << " iterations. Residual norm is " << solver_particular.GetFinalNorm() << ".\n";
          cout << "Particular solution found in " << chrono_local.RealTime() << "s. \n";
       }

       chrono_local.Clear();
       chrono_local.Start();
       ParDiscreteLinearOperator DiscreteCurl(N_space, R_space);
       DiscreteCurl.AddDomainInterpolator(new CurlInterpolator);
       DiscreteCurl.Assemble();
       DiscreteCurl.Finalize();
       auto C = DiscreteCurl.ParallelAssemble();
       auto MC = ParMult(M, C);
       auto CT = C->Transpose();
       darcyOp = ParMult(CT, MC);
       if (GMG)
           darcyPr = new Multigrid(((HypreParMatrix&)*darcyOp), P);
       else
       {
           darcyPr = new HypreAMS(((HypreParMatrix&)*darcyOp), N_space);
           ((HypreAMS*)darcyPr)->SetSingularProblem();
       }

       // Compute the right hand side for the divergence free solver problem
       Vector rhs_divfree(MC->GetNumCols());
       rhs_divfree = 0.0;
       M->Mult(-1.0, sol_particular, 1.0, trueRhs.GetBlock(0));
       CT->Mult(trueRhs.GetBlock(0), rhs_divfree);

       Array<int> bc_dofs;
       N_space->GetEssentialTrueDofs(ess_bdr, bc_dofs);
       ((HypreParMatrix*)darcyOp)->EliminateRowsCols(bc_dofs);
       for (auto i : bc_dofs)
           rhs_divfree(i) = 0.0;

       // Solve the divergence free solution
       CGSolver solver(MPI_COMM_WORLD);
       solver.SetAbsTol(atol);
       solver.SetRelTol(rtol);
       solver.SetMaxIter(maxIter);
       solver.SetOperator(*darcyOp);
       solver.SetPreconditioner(*darcyPr);
       solver.SetPrintLevel(0);

       Vector sol_potential(darcyOp->Width());
       sol_potential = 0.0;
       solver.Mult(rhs_divfree, sol_potential);

       Vector sol_divfree(C->GetNumRows());
       C->Mult(sol_potential, sol_divfree);

       // Combining the particular solution and the divergence free solution

       trueX.GetBlock(0) = sol_particular;
       trueX.GetBlock(0) += sol_divfree;

       chrono_local.Stop();
       if (verbose)
       {
          if (solver.GetConverged())
             cout << "CG converged in " << solver.GetNumIterations()
                       << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
          else
             cout << "CG did not converge in " << solver.GetNumIterations()
                       << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
          cout << "Divergence free solution found in " << chrono_local.RealTime() << "s. \n";
       }

       // Compute the right hand side for the pressure problem BB^T p = rhs_p
       chrono_local.Clear();
       chrono_local.Start();

       M->Mult(-1.0, sol_divfree, 1.0, trueRhs.GetBlock(0));
       Vector rhs_p(B->GetNumRows());
       B->Mult(trueRhs.GetBlock(0), rhs_p);
       trueX.GetBlock(1) = 0.0;
       solver_particular.Mult(rhs_p, trueX.GetBlock(1));

       chrono_local.Stop();
       if (verbose)
       {
          if (solver_particular.GetConverged())
             cout << "CG converged in " << solver_particular.GetNumIterations()
                       << " iterations with a residual norm of " << solver_particular.GetFinalNorm() << ".\n";
          else
             cout << "CG did not converge in " << solver_particular.GetNumIterations()
                       << " iterations. Residual norm is " << solver_particular.GetFinalNorm() << ".\n";
          cout << "Pressure solution found in " << chrono_local.RealTime() << "s. \n";
       }
       chrono.Stop();
       if (verbose)
           cout << "Divergence free solver overall took " << chrono.RealTime() << "s. \n";
   }
   else
   {
       chrono.Clear();
       chrono.Start();

       darcyOp = new BlockOperator(block_trueOffsets);
       ((BlockOperator*)darcyOp)->SetBlock(0,0, M);
       ((BlockOperator*)darcyOp)->SetBlock(0,1, BT);
       ((BlockOperator*)darcyOp)->SetBlock(1,0, B);

       // 11. Construct the operators for preconditioner
       //
       //                 P = [ diag(M)         0         ]
       //                     [  0       B diag(M)^-1 B^T ]
       //
       //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
       //     pressure Schur Complement.
       HypreParMatrix *MinvBt = B->Transpose();
       Vector Md(M->GetNumRows());
       M->GetDiag(Md);
       MinvBt->InvScaleRows(Md);
       HypreParMatrix *S = ParMult(B, MinvBt);

       HypreSolver *invS;
       auto invM = new HypreSmoother(*M);
       invS = new HypreBoomerAMG(*S);
       static_cast<HypreBoomerAMG*>(invS)->SetPrintLevel(0);

       invM->iterative_mode = false;
       invS->iterative_mode = false;

       darcyPr = new BlockDiagonalPreconditioner(
                   block_trueOffsets);
       ((BlockDiagonalPreconditioner*)darcyPr)->SetDiagonalBlock(0, invM);
       ((BlockDiagonalPreconditioner*)darcyPr)->SetDiagonalBlock(1, invS);

       // 12. Solve the linear system with MINRES.
       //     Check the norm of the unpreconditioned residual.

       MINRESSolver solver(MPI_COMM_WORLD);
       solver.SetAbsTol(atol);
       solver.SetRelTol(rtol);
       solver.SetMaxIter(maxIter);
       solver.SetOperator(*darcyOp);
       solver.SetPreconditioner(*darcyPr);
       solver.SetPrintLevel(0);
       trueX = 0.0;
       solver.Mult(trueRhs, trueX);
       chrono.Stop();

       if (verbose)
       {
          if (solver.GetConverged())
             std::cout << "MINRES converged in " << solver.GetNumIterations()
                       << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
          else
             std::cout << "MINRES did not converge in " << solver.GetNumIterations()
                       << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
          std::cout << "MINRES solver took " << chrono.RealTime() << "s. \n";
       }
       delete invM;
       delete invS;
       delete S;
       delete MinvBt;
   }

   // 13. Extract the parallel grid function corresponding to the finite element
   //     approximation X. This is the local solution on each processor. Compute
   //     L2 error norms.
   ParGridFunction *u(new ParGridFunction);
   u->MakeRef(W_space, x.GetBlock(1), 0);
   u->Distribute(&(trueX.GetBlock(1)));

   ParGridFunction *sigma(new ParGridFunction);
   sigma->MakeRef(R_space, x.GetBlock(0), 0);
   sigma->Distribute(&(trueX.GetBlock(0)));

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double err_u  = u->ComputeL2Error(ucoeff, irs);
   double norm_u = ComputeGlobalLpNorm(2, ucoeff, *pmesh, irs);
   double err_sigma  = sigma->ComputeL2Error(sigmacoeff, irs);
   double norm_sigma = ComputeGlobalLpNorm(2, sigmacoeff, *pmesh, irs);

   if (verbose)
   {
      std::cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
      std::cout << "|| sigma_h - sigma_ex || / || sigma_ex || = "
                << err_sigma / norm_sigma << "\n";
   }

   // 14. Save the refined mesh and the solution in parallel. This output can be
   //     viewed later using GLVis: "glvis -np <np> -m mesh -g sol_*".
   {
      ostringstream mesh_name, u_name, sigma_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      u_name << "sol_u." << setfill('0') << setw(6) << myid;
      sigma_name << "sol_sigma." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream u_ofs(u_name.str().c_str());
      u_ofs.precision(8);
      u->Save(u_ofs);

      ofstream sigma_ofs(sigma_name.str().c_str());
      sigma_ofs.precision(8);
      sigma->Save(sigma_ofs);
   }

   // 15. Save data in the VisIt format
   VisItDataCollection visit_dc("Example5-Parallel", pmesh);
   visit_dc.RegisterField("Primal variable S", u);
   visit_dc.RegisterField("Flux sigma", sigma);
   visit_dc.Save();

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream u_sock(vishost, visport);
      u_sock << "parallel " << num_procs << " " << myid << "\n";
      u_sock.precision(8);
      u_sock << "solution\n" << *pmesh << *u << "window_title 'Primal variable S'"
             << endl;
      // Make sure all ranks have sent their 'u' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());
      socketstream sigma_sock(vishost, visport);
      sigma_sock << "parallel " << num_procs << " " << myid << "\n";
      sigma_sock.precision(8);
      sigma_sock << "solution\n" << *pmesh << *sigma << "window_title 'Flux sigma'"
                 << endl;
   }

   // 17. Free the used memory.
   delete fform;
   delete gform;
   delete u;
   delete sigma;
   delete darcyOp;
   delete darcyPr;
   delete BT;
   delete B;
   delete M;
   delete mVarf;
   delete bVarf;
   delete W_space;
   delete R_space;
   if (divfree)
       delete N_space;
   delete l2_coll;
   delete hdiv_coll;
   delete hcurl_coll;
   delete pmesh;

   MPI_Finalize();

   return 0;
}

// This is actually the function for imposing boundary (or initial) condition
//double GaussianHill(const Vector& xt)
//{
//    return std::exp(-100.0 * (std::pow(xt(0) - 0.5, 2.0) + xt(1) * xt(1)));
//}

////double GaussianHill(const Vector&xvec)
////{
////    double x = xvec(0);
////    double y = xvec(1);
////    return exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y));
////}

//double uFun_ex(const Vector& xt)
//{
//    double x = xt(0);
//    double y = xt(1);
//    double r = sqrt(x*x + y*y);
//    double teta = atan2(y,x);
//    /*
//    if (fabs(x) < MYZEROTOL && y > 0)
//        teta = M_PI / 2.0;
//    else if (fabs(x) < MYZEROTOL && y < 0)
//        teta = - M_PI / 2.0;
//    else
//        teta = atan(y,x);
//    */
//    double t = xt(xt.Size()-1);
//    Vector xvec(2);
//    xvec(0) = r * cos (teta - t);
//    xvec(1) = r * sin (teta - t);
//    return GaussianHill(xvec);
//}

//double fFun(const Vector& xt)
//{
//    return 0.0;
//}

//void bFun_ex(const Vector& xt, Vector& b )
//{
//    b.SetSize(xt.Size());

//    if (xt.Size() == 3)
//    {
//        b(0) = -xt(1);
//        b(1) = xt(0);
//    }
//    else
//    {
//        b(0) = sin(xt(0)*M_PI)*cos(xt(1)*M_PI)*cos(xt(2)*M_PI);
//        b(1) = -0.5*sin(xt(1)*M_PI)*cos(xt(0)*M_PI)*cos(xt(2)*M_PI);
//        b(2) = -0.5*sin(xt(2)*M_PI)*cos(xt(0)*M_PI)*cos(xt(1)*M_PI);
//    }

//    b(xt.Size()-1) = 1.;
//}

//void bfFun_ex(const Vector& xt, Vector& bf)
//{
//    bf.SetSize(xt.Size());
//    bFun_ex(xt, bf);
//    bf *= fFun(xt);
//}

void sigmaFun_ex(const Vector& xt, Vector& sigma)
{
    Vector b;
    bFun_ex(xt, b);
    sigma.SetSize(xt.Size());
    sigma(xt.Size()-1) = uFun_ex(xt);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = b(i) * sigma(xt.Size()-1);
}

void Ktilda_ex(const Vector& xt, DenseMatrix& Ktilda)
{
    int nDim = xt.Size();
    Vector b;
    bFun_ex(xt,b);

    double bTbInv = (-1.0/(b*b));
    Ktilda.Diag(1.0,nDim);
    AddMult_a_VVt(bTbInv,b,Ktilda);
}

double bTb_ex(const Vector& xt)
{
    Vector b;
    bFun_ex(xt,b);
    return 1.*(b*b);
}

void bbT_ex(const Vector& xt, DenseMatrix& bbT)
{
    Vector b;
    bFun_ex(xt,b);
    MultVVt(b, bbT);
}

double uFun_ex(const Vector& xt)
{
    double t = xt(xt.Size()-1);
//    return t;
    return sin(t)*exp(t)*xt(0)*xt(1);
}

double fFun(const Vector& xt)
{
//    double tmp = 0.;
//    for (int i = 0; i < xt.Size()-1; i++)
//        tmp += xt(i);
//    return 1.;//+ (xt.Size()-1-2*tmp) * uFun_ex(xt);
    double t = xt(xt.Size()-1);
    Vector b;
    bFun_ex(xt, b);
    return (cos(t)*exp(t)+sin(t)*exp(t))*xt(0)*xt(1)+
            b(0)*sin(t)*exp(t)*xt(1) + b(1)*sin(t)*exp(t)*xt(0);
}

void bFun_ex(const Vector& xt, Vector& b)
{
    b.SetSize(xt.Size());
    b = 0.;

    b(xt.Size()-1) = 1.;

    if (xt.Size() == 3)
    {
        b(0) = sin(xt(0)*M_PI)*cos(xt(1)*M_PI);
        b(1) = -sin(xt(1)*M_PI)*cos(xt(0)*M_PI);
    }
    else
    {
        b(0) = sin(xt(0)*M_PI)*cos(xt(1)*M_PI)*cos(xt(2)*M_PI);
        b(1) = -0.5*sin(xt(1)*M_PI)*cos(xt(0)*M_PI)*cos(xt(2)*M_PI);
        b(2) = -0.5*sin(xt(2)*M_PI)*cos(xt(0)*M_PI)*cos(xt(1)*M_PI);
    }
}
