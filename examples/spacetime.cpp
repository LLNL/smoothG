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
   int par_ref_levels = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
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
   {
       for (int l = 0; l < par_ref_levels; l++)
           pmesh->UniformRefinement();
   }
   delete mesh;

   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   //   ess_bdr.Last() = 0;

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
   FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));

   ParFiniteElementSpace *R_space = new ParFiniteElementSpace(pmesh, hdiv_coll);
   ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh, l2_coll);

   HYPRE_Int dimR = R_space->GlobalTrueVSize();
   HYPRE_Int dimW = W_space->GlobalTrueVSize();

   if (verbose)
   {
      std::cout << "***********************************************************\n";
      std::cout << "dim(R) = " << dimR << "\n";
      std::cout << "dim(W) = " << dimW << "\n";
      std::cout << "dim(R+W) = " << dimR + dimW << "\n";
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
   delete l2_coll;
   delete hdiv_coll;
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
