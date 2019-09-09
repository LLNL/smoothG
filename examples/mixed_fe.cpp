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
//               We recommend viewing example 5 before viewing this miniapp.

#include "mfem.hpp"
#include "../src/mixed_fe_solvers.hpp"
#include "../src/smoothG.hpp"
#include "pde.hpp"
#include <fstream>
#include <iostream>
#include <assert.h>
#include <memory>

using namespace std;
using namespace mfem;
using namespace smoothg;

// Define the analytical solution and forcing terms / boundary conditions
void uFun_ex(const Vector & x, Vector & u);
double pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
double gFun(const Vector & x);
double f_natural(const Vector & x);
void sigmaFun_ex(const Vector& xt, Vector& sigma);
double sFun_ex(const Vector& xt);
void kFun(const Vector& xt, DenseMatrix& k);
double spacetime_fFun(const Vector& xt);

//     Assemble the finite element matrices for the Darcy problem
//
//                            D = [ M  B^T ]
//                                [ B   0  ]
//     where:
//
//     M = \int_\Omega u_h \cdot v_h d\Omega   u_h, v_h \in R_h
//     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
class FEDarcyProblem
{
    OperatorPtr M_;
    OperatorPtr B_;
    Vector rhs_;
    Vector ess_data_;
    ParGridFunction p_;
    ParGridFunction u_;
    ParMesh& mesh_;
    FunctionCoefficient pcoeff_;
    VectorFunctionCoefficient ucoeff_;
    DFSDataCollector collector_;
    const IntegrationRule *irs_[Geometry::NumGeom];
    unique_ptr<MixedMatrix> mixed_system_;
public:
    FEDarcyProblem(ParMesh& mesh, int num_refines, int order, bool spacetime,
                   Array<int>& ess_bdr, DFSParameters param); // TODO: better design

    HypreParMatrix& GetM() { return *M_.As<HypreParMatrix>(); }
    HypreParMatrix& GetB() { return *B_.As<HypreParMatrix>(); }
    const Vector& GetRHS() { return rhs_; }
    const Vector& GetBC() { return ess_data_; }
    const DFSDataCollector& GetDFSDataCollector() const { return collector_; }
    const MixedMatrix& GetMixedMatrix() const { return *mixed_system_; }

    void ShowError(const Vector& sol, bool spacetime, bool verbose);
};

FEDarcyProblem::FEDarcyProblem(ParMesh &mesh, int num_refines, int order, bool spacetime,
                               Array<int>& ess_bdr, DFSParameters dfs_param)
    : mesh_(mesh), pcoeff_(pFun_ex),
      ucoeff_(mesh_.Dimension(), spacetime ? sigmaFun_ex : uFun_ex),
      collector_(order, num_refines, &mesh_, ess_bdr, dfs_param)
{
    for (int l = 0; l < num_refines; l++)
    {
        mesh_.UniformRefinement();
        collector_.CollectData(&mesh_);
    }

    // Define the coefficients, analytical solution, and rhs of the PDE.
    VectorFunctionCoefficient fcoeff(mesh_.Dimension(), fFun);
    FunctionCoefficient fnatcoeff(f_natural);
    FunctionCoefficient gcoeff(gFun);
    MatrixFunctionCoefficient kcoeff(mesh_.Dimension(), kFun);
    FunctionCoefficient st_fcoeff(spacetime_fFun);

    u_.SetSpace(collector_.hdiv_fes_.get());
    p_.SetSpace(collector_.l2_fes_.get());
    u_ = 0.0;
    u_.ProjectBdrCoefficientNormal(ucoeff_, ess_bdr);

    ParLinearForm fform(collector_.hdiv_fes_.get());
    ParLinearForm gform(collector_.l2_fes_.get());
    ParBilinearForm mVarf(collector_.hdiv_fes_.get());
    ParMixedBilinearForm bVarf(&(*collector_.hdiv_fes_), &(*collector_.l2_fes_));

    if (spacetime)
    {
        gform.AddDomainIntegrator(new DomainLFIntegrator(st_fcoeff));
        mVarf.AddDomainIntegrator(new VectorFEMassIntegrator(kcoeff));
    }
    else
    {
        fform.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
        fform.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fnatcoeff));
        gform.AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
        mVarf.AddDomainIntegrator(new VectorFEMassIntegrator);
    }

    fform.Assemble();
    gform.Assemble();

    mVarf.ComputeElementMatrices();
    mVarf.Assemble();
    mVarf.EliminateEssentialBC(ess_bdr, u_, fform);
    mVarf.Finalize();
    M_.Reset(mVarf.ParallelAssemble());

    bVarf.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
    bVarf.Assemble();
    if (!spacetime) bVarf.SpMat() *= -1.0;
    bVarf.Finalize();
    SparseMatrix D = bVarf.SpMat();
    bVarf.EliminateTrialDofs(ess_bdr, u_, gform);
    B_.Reset(bVarf.ParallelAssemble());

    RT_FECollection RT0_fec(0, mesh_.Dimension());
    ParFiniteElementSpace RT0_fes(&mesh_, &RT0_fec);

    auto edge_bdratt = GenerateBoundaryAttributeTable(&mesh_);
    auto vertex_edge = TableToMatrix(mesh_.ElementToFaceTable());
    auto& edge_trueedge = *(RT0_fes.Dof_TrueDof_Matrix());
    Graph graph(vertex_edge, edge_trueedge, mfem::Vector(), &edge_bdratt);
    GraphSpace graph_space(std::move(graph), *collector_.hdiv_fes_, *collector_.l2_fes_);

    std::vector<mfem::DenseMatrix> M_el(graph_space.GetGraph().NumVertices());
    mfem::Array<int> vdofs;
    for (int i = 0; i < mesh_.GetNE(); ++i)
    {
        mVarf.ComputeElementMatrix(i, M_el[i]);

        DenseMatrix sign_fix(M_el[i].NumRows());
        collector_.hdiv_fes_.get()->GetElementVDofs(i, vdofs);
        for (int j = 0; j < sign_fix.NumRows(); ++j)
        {
            sign_fix(j, j) = vdofs[j] < 0 ? -1.0 : 1.0;
        }

        mfem::DenseMatrix help(sign_fix);
        mfem::Mult(M_el[i], sign_fix, help);
        mfem::Mult(sign_fix, help, M_el[i]);
    }
    auto mbuilder = make_unique<ElementMBuilder>(std::move(M_el), graph_space.VertexToEDof());

    mfem::SparseMatrix W;
    mfem::Vector const_rep(graph_space.VertexToVDof().NumCols());
    const_rep = 1.0 / std::sqrt(collector_.l2_fes_->GlobalTrueVSize());

    mfem::Vector vertex_sizes(graph_space.GetGraph().NumVertices());
    vertex_sizes = 1.0;

    mfem::SparseMatrix P_pwc = SparseIdentity(const_rep.Size());

    mixed_system_.reset(new MixedMatrix(std::move(graph_space), std::move(mbuilder),
                                        std::move(D), std::move(W), std::move(const_rep),
                                        std::move(vertex_sizes), std::move(P_pwc)));
    mixed_system_->SetEssDofs(ess_bdr);
    mixed_system_->BuildM();

    rhs_.SetSize(M_->NumRows()+B_->NumRows());
    rhs_ = 0.0;
    Vector block0_view(rhs_.GetData(), M_->NumRows());
    Vector block1_view(rhs_.GetData()+M_->NumRows(), B_->NumRows());
    fform.ParallelAssemble(block0_view);
    gform.ParallelAssemble(block1_view);

    ess_data_.SetSize(M_->NumRows()+B_->NumRows());
    ess_data_ = 0.0;
    Vector ess_data_block0(ess_data_.GetData(), M_->NumRows());
    u_.ParallelProject(ess_data_block0);

    int order_quad = max(2, 2*order+1);
    for (int i=0; i < Geometry::NumGeom; ++i)
    {
        irs_[i] = &(IntRules.Get(i, order_quad));
    }
}

void FEDarcyProblem::ShowError(const Vector &sol, bool spacetime, bool verbose)
{
    u_.Distribute(Vector(sol.GetData(), M_->NumRows()));
    double err  = u_.ComputeL2Error(ucoeff_, irs_);
    double norm = ComputeGlobalLpNorm(2, ucoeff_, mesh_, irs_);
    if (verbose) cout << "\n|| u_h - u || / || u || = " << err / norm << "\n";
    if (spacetime) return;

    p_.Distribute(Vector(sol.GetData()+M_->NumRows(), B_->NumRows()));
    err  = p_.ComputeL2Error(pcoeff_, irs_);
    norm = ComputeGlobalLpNorm(2, pcoeff_, mesh_, irs_);
    if (verbose) cout << "|| p_h - p || / || p || = " << err / norm << "\n";
}

int main(int argc, char *argv[])
{
    StopWatch chrono;
    auto ResetTimer = [&chrono]() { chrono.Clear(); chrono.Start(); };

    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    bool verbose = (myid == 0);

    // 2. Parse command-line options.
    int order = 0;
    int num_refines = 2;
    bool use_tet_mesh = false;
    bool spacetime = false;
    bool show_error = false;
    OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree).");
    args.AddOption(&num_refines, "-r", "--ref",
                   "Number of parallel refinement steps.");
    args.AddOption(&use_tet_mesh, "-tet", "--tet-mesh", "-hex", "--hex-mesh",
                   "Use a tetrahedral or hexahedral mesh (on unit cube).");
    args.AddOption(&spacetime, "-st", "--spacetime", "-no-st", "--no-spacetime",
                   "Solve spacetime problem or normal Darcy flow.");
    args.AddOption(&show_error, "-se", "--show-error", "-no-se", "--no-show-error",
                   "Show or not show approximation error.");
    args.Parse();
    if (!args.Good())
    {
        if (verbose) args.PrintUsage(cout);
        MPI_Finalize();
        return 1;
    }
    if (verbose) args.PrintOptions(cout);

    auto elem_type = use_tet_mesh ? Element::TETRAHEDRON : Element::HEXAHEDRON;
    Mesh* mesh = new Mesh(2, 2, 2, elem_type, true);
    for (int i = 0; i < (int)(log(num_procs)/log(8)); ++i)
         mesh->UniformRefinement();

    Array<int> ess_bdr(mesh->bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 1;

    IterSolveParameters param;
    param.max_iter = 100000;
    DFSParameters dfs_param;
    dfs_param.MG_type = order > 0 && use_tet_mesh ? AlgebraicMG : GeometricMG;
    dfs_param.B_has_nullity_one = (ess_bdr.Sum() == ess_bdr.Size());
    if (order > 0 && use_tet_mesh) dfs_param.ml_particular = false;

    int num_procs_x = static_cast<int>(std::sqrt(num_procs) + 0.5);
    while (num_procs % num_procs_x)
        num_procs_x -= 1;

    Array<int> np_xyz(mesh->SpaceDimension());
    np_xyz[0] = num_procs_x;
    np_xyz[1] = num_procs / num_procs_x;
    np_xyz[2] = 1;
    assert(np_xyz[0] * np_xyz[1] == num_procs);

    Array<int> partition(mesh->CartesianPartitioning(np_xyz), mesh->GetNE());

    string line = "\n*******************************************************\n";
    {
        ParMesh pmesh(MPI_COMM_WORLD, *mesh, partition);
        delete mesh;
        ResetTimer();
        FEDarcyProblem darcy(pmesh, num_refines, order, spacetime, ess_bdr, dfs_param);
        HypreParMatrix& M = darcy.GetM();
        HypreParMatrix& B = darcy.GetB();
        const DFSDataCollector& collector = darcy.GetDFSDataCollector();

        if (verbose)
        {
            cout << line << "dim(R) = " << M.M() << ", dim(W) = " << B.M() << ", ";
            cout << "dim(N) = " << collector.hcurl_fes_->GlobalTrueVSize() << "\n";
            cout << "System assembled in " << chrono.RealTime() << "s.\n";
        }

        map<const DarcySolver*, double> setup_time;
        map<const DarcySolver*, std::string> solver_to_name;

        ResetTimer();
        DivFreeSolver dfs(M, B, collector.hcurl_fes_.get(), collector.GetData());
        setup_time[&dfs] = chrono.RealTime();
        solver_to_name[&dfs] = "Divergence free";

        ResetTimer();
        BDPMinresSolver bdp(M, B, param);
        setup_time[&bdp] = chrono.RealTime();
        solver_to_name[&bdp] = "Block-diagonal-preconditioned MINRES";

        ResetTimer();
        HybridSolver hybrid(darcy.GetMixedMatrix(), &ess_bdr, -1);
        hybrid.SetMaxIter(100000);
        setup_time[&hybrid] = chrono.RealTime();
        solver_to_name[&hybrid] = "Hybridization";

        for (const auto& solver_pair : solver_to_name)
        {
            auto& solver = solver_pair.first;
            auto& name = solver_pair.second;

            if (verbose) cout << line << name << " solver:\n";
            if (verbose) cout << "  setup time: " << setup_time[solver] << "s.\n";

            const Vector& rhs = darcy.GetRHS();
            Vector sol = darcy.GetBC();
            ResetTimer();
            solver->Mult(rhs, sol);

            if (verbose) cout << "  solve time: " << chrono.RealTime() << "s.\n";
            if (verbose) cout << "  iteration count: "
                              << solver->GetNumIterations() <<"\n";
            if (show_error) darcy.ShowError(sol, spacetime, verbose);
        }
    }
    if (verbose) cout << line << "\n";

    MPI_Finalize();
    return 0;
}


void uFun_ex(const Vector & x, Vector & u)
{
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);
    if (x.Size() == 3)
    {
        zi = x(2);
    }

    u(0) = - exp(xi)*sin(yi)*cos(zi);
    u(1) = - exp(xi)*cos(yi)*cos(zi);

    if (x.Size() == 3)
    {
        u(2) = exp(xi)*sin(yi)*sin(zi);
    }
}

// Change if needed
double pFun_ex(const Vector & x)
{
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);

    if (x.Size() == 3)
    {
        zi = x(2);
    }

    return exp(xi)*sin(yi)*cos(zi);
}

void fFun(const Vector & x, Vector & f)
{
    f = 0.0;
}

double gFun(const Vector & x)
{
    if (x.Size() == 3)
    {
        return -pFun_ex(x);
    }
    else
    {
        return 0;
    }
}

double f_natural(const Vector & x)
{
    return (-pFun_ex(x));
}

void bFun(const Vector& xt, Vector& b)
{
    b.SetSize(xt.Size());
    b = 0.;

    b(0) = sin(xt(0)*M_PI)*cos(xt(1)*M_PI);
    b(1) = -sin(xt(1)*M_PI)*cos(xt(0)*M_PI);
    b(2) = 1.;
}

void sigmaFun_ex(const Vector& xt, Vector& sigma)
{
    Vector b;
    bFun(xt, b);
    sigma.SetSize(xt.Size());
    sigma(xt.Size()-1) = sFun_ex(xt);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = b(i) * sigma(xt.Size()-1);
}

double sFun_ex(const Vector& xt)
{
    double t = xt(xt.Size()-1);
    return sin(t)*exp(t)*xt(0)*xt(1);
}

void kFun(const Vector& xt, DenseMatrix& k)
{
    int nDim = xt.Size();
    Vector b;
    bFun(xt,b);

    double bTbInv = (-1.0/(b*b));
    k.Diag(1.0,nDim);
    AddMult_a_VVt(bTbInv,b,k);
}

double spacetime_fFun(const Vector& xt)
{
    double t = xt(xt.Size()-1);
    Vector b;
    bFun(xt, b);
    return (cos(t)*exp(t)+sin(t)*exp(t))*xt(0)*xt(1)
           +b(0)*sin(t)*exp(t)*xt(1) + b(1)*sin(t)*exp(t)*xt(0);
}
