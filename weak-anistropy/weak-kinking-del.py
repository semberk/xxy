#
# =============================================================================
# FEnics code  Variational Fracture Mechanics
# =============================================================================
#
# A static solution of the variational fracture mechanics problems
# using the regularization two-fold anisotropic damage model
#
# author: bin.li@upmc.fr
#
# date: 10/10/2017
#
# ----------------------------------------------------------------------------
# runing: python3 second-orderAniso_crack_direction.py 1.5 0.5 0.0 0.0 0.84
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
from __future__ import division
from dolfin import *
from mshr import *

import argparse
import math
import os
import shutil
import sympy
import sys
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Parameters for DOLFIN and SOLVER
# ----------------------------------------------------------------------------
set_log_level(50)  # log level
# set some dolfin specific parameters
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"

# -----------------------------------------------------------------------------
# parameters of the nonlinear solver used for the alpha-problem
solver_alpha_parameters = {"nonlinear_solver": "snes",
                           "snes_solver": {"linear_solver": "mumps",
                                           "method": "vinewtonssls",
                                           "line_search": "cp",  # "nleqerr",
                                           "preconditioner": "hypre_amg",
                                           "maximum_iterations": 40,
                                           "solution_tolerance": 1e-6,
                                           "relative_tolerance": 1e-6,
                                           "absolute_tolerance": 1e-6,
                                           "report": False,
                                           "error_on_nonconvergence": False,
                                           "krylov_solver": {
                                               "report": False,
                                               "monitor_convergence": False,
                                               "relative_tolerance": 1e-8}}}

solver_u_parameters = {"linear_solver": "mumps",
                       "symmetric": True,
                       "preconditioner": "hypre_amg",
                       "krylov_solver": {
                           "report": False,
                           "monitor_convergence": False,
                           "relative_tolerance": 1e-8}}
# -----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Parameters for ANISOTROPIC surface energy and materials
# ----------------------------------------------------------------------------
parser  = argparse.ArgumentParser(description='The second-order anisotropic surface energy damage model')
parser.add_argument('BMat',   type=float, nargs=2, help="input $2$ components of Tensor B")
parser.add_argument('theta0', type=float, nargs=1, help="input rotation angle $theta0(degree)$")
parser.add_argument('KI',     type=float, nargs=1, help="input mode I loading KI")
parser.add_argument('KII',    type=float, nargs=1, help="input mode II loading")
args    = parser.parse_args()

B11     = args.BMat[0]
B22     = args.BMat[1]
theta0  = args.theta0[0]*np.pi/180.0
KI      = args.KI[0]
KII     = args.KII[0]

B_mat   = [[B11, 0.0], [0.0, B22]]
Q       = [[np.cos(theta0), -np.sin(theta0)],\
          [np.sin(theta0), np.cos(theta0) ]]
B_mat   = np.matmul(np.matmul(Q,B_mat), np.transpose(Q))

B_11    = B_mat[0,0]
B_12    = B_mat[0,1]
B_22    = B_mat[1,1]

# Material constant
E       = Constant(7.0e12)
nu      = Constant(0.3)
Gc      = Constant(1.0)
k_ell   = Constant(1.e-6)  # residual stiffness

# -----------------------------------------------------------------------------
# Loading Parameters
# -----------------------------------------------------------------------------
ut           = 1.0   # reference value for the loading (imposed displacement)
load_min     = 0.95  # load multiplier min value
load_max     = 1.20  # load multiplier max value
load_steps   = 21    # number of time steps

# Numerical parameters of the alternate minimization
maxiteration = 2000
AM_tolerance = 1e-4

# ----------------------------------------------------------------------------
# Geometry and mesh generation and damage paramaters
# ----------------------------------------------------------------------------
L         = 0.1
N         = 300
hsize     = float(L/N)
cra_angle = float(0.5*np.pi/180.0)
cra_w     = 0.05*L*tan(cra_angle)
ell       = Constant(7.0*hsize) # damage paramaters

meshname  = "second-orderAniso_crack_direction.xdmf"
modelname = "second-orderAniso_crack_direction"
simulation_params = "B11_%.4f_B22_%.4f_theta0_%.4f_KI_%.4f_KII%.4f_h_%.4f" % (B11, B22, theta0, KI, KII, hsize)
savedir   = modelname+"/"+simulation_params+"/"

if MPI.rank(mpi_comm_world()) == 0:
    if os.path.isdir(savedir):
        shutil.rmtree(savedir)

#crack geometry
P1 = Point(0., -0.5*cra_w)
P2 = Point(0.45*L, -0.5*cra_w)
P4 = Point(0.45*L, 0.5*cra_w)
P5 = Point(0., 0.5*cra_w)
P3 = Point(0.5*L, 0.)
geometry = Rectangle(Point(0., -0.5*L), Point(L, 0.5*L)) - Polygon([P1,P2,P3,P4,P5])

# Mesh generation using cgal
mesh      = generate_mesh(geometry, N, 'cgal')
geo_mesh  = XDMFFile(mpi_comm_world(), savedir+meshname)
geo_mesh.write(mesh)

ndim = mesh.geometry().dim()  # get number of space dimensions
if MPI.rank(mpi_comm_world()) == 0:
    print ("the dimension of mesh: {0:2d}".format(ndim))

# ----------------------------------------------------------------------------
# Strain and stress and Constitutive functions of the damage model
# ----------------------------------------------------------------------------
# Strain and stress
def eps(v):
    return sym(grad(v))

def sigma_0(v):
    mu = E/(2.0*(1.0+nu))
    lmbda = E*nu/(1.0-nu**2)  # plane stress
    return 2.0*mu*(eps(v))+lmbda*tr(eps(v))*Identity(ndim)

# Constitutive functions of the damage model
def w(alpha):
    return alpha

def a(alpha):
    return (1.0-alpha)**2

# ----------------------------------------------------------------------------
# Define boundary sets for boundary conditions
# Impose the displacements field given by asymptotic expansion of crack tip
# ----------------------------------------------------------------------------
def boundaries(x):
    return near(x[1], 0.5*L, 0.1*hsize) or near(x[1], -0.5*L, 0.1*hsize) \
        or near(x[0], 0.0, 0.1*hsize) or near(x[0], L, 0.1 * hsize)

# ----------------------------------------------------------------------------
# Variational formulation
# ----------------------------------------------------------------------------
# Create function space for 2D elasticity + Damage
V_u     = VectorFunctionSpace(mesh, "Lagrange", 1)
V_alpha = FunctionSpace(mesh, "Lagrange", 1)

# Define the function, test and trial fields
u       = Function(V_u, name="Displacement")
du      = TrialFunction(V_u)
v       = TestFunction(V_u)
alpha   = Function(V_alpha, name="Damage")
dalpha  = TrialFunction(V_alpha)
beta    = TestFunction(V_alpha)

# --------------------------------------------------------------------
# Dirichlet boundary condition
# Impose the displacements field given by asymptotic expansion of crack tip
# --------------------------------------------------------------------
mu    = float(E/(2.0*(1.0+nu)))
kappa = float((3.0-nu)/(1.0+nu))
nKI   = float(sqrt(E*Gc)) #non-dimensional KI_C
u_U   = Expression(["t*KI*nKI/(2*mu)*sqrt(sqrt((x[0]-lc)*(x[0]-lc)+x[1]*x[1])/(2*pi))*(kappa-cos(atan2(x[1], x[0]-lc)))*cos(atan2(x[1], x[0]-lc)/2) + \
                   	t*KII*nKI/(2*mu)*sqrt(sqrt((x[0]-lc)*(x[0]-lc)+x[1]*x[1])/(2*pi))*(2.0+kappa+cos(atan2(x[1], x[0]-lc)))*sin(atan2(x[1], x[0]-lc)/2)",
                  	"t*KI*nKI/(2*mu)*sqrt(sqrt((x[0]-lc)*(x[0]-lc)+x[1]*x[1])/(2*pi))*(kappa-cos(atan2(x[1], x[0]-lc)))*sin(atan2(x[1], x[0]-lc)/2) + \
                   	t*KII*nKI/(2*mu)*sqrt(sqrt((x[0]-lc)*(x[0]-lc)+x[1]*x[1])/(2*pi))*(2.0-kappa-cos(atan2(x[1], x[0]-lc)))*cos(atan2(x[1], x[0]-lc)/2)"],
                  	degree=2, mu=mu, kappa=kappa, nKI=nKI, KI=KI, KII=KII, lc=0.5*L, t=0.0)

# bc - u (imposed displacement)
Gamma_u_0     = DirichletBC(V_u, u_U, boundaries)
bc_u          = [Gamma_u_0]
# bc - alpha (zero damage)
Gamma_alpha_0 = DirichletBC(V_alpha, 0.0, boundaries)
bc_alpha      = [Gamma_alpha_0]


# --------------------------------------------------------------------
# Define the energy functional of damage problem
# --------------------------------------------------------------------
# Fenics forms for the energies
def sigma(u, alpha):
    return (a(alpha)+k_ell)*sigma_0(u)

body_force        = Constant((0., 0.))
elastic_energy    = 0.5*inner(sigma(u, alpha), eps(u))*dx
external_work     = dot(body_force, u)*dx
elastic_potential = elastic_energy-external_work

# Weak form of elasticity problem
E_u  = derivative(elastic_potential, u, v)
# Writing tangent problems in term of test and trial functions for matrix assembly
E_du = replace(E_u, {u: du})

# Variational problem for the displacement
problem_u = LinearVariationalProblem(lhs(E_du), rhs(E_du), u, bc_u)
# Set up the solvers
solver_u  = LinearVariationalSolver(problem_u)
solver_u.parameters.update(solver_u_parameters)
# info(solver_u.parameters, True)

# --------------------------------------------------------------------
# Define the energy functional of damage problem
# --------------------------------------------------------------------
# Initialize damage field
alpha_0 = Function(V_alpha)
alpha_0 = interpolate(Expression("0.", degree=0), V_alpha)  # initial (known) alpha
#alpha_0 = interpolate(Expression("x[0]<=0.5*L & near(x[1], 0.0, tol) ? 1.0 : 0.0", \
#                                 degree=0, L= L, tol=1.2*cra_w), V_alpha)  # initial (known) alpha

# matrix notation for second-order tensor B
BMatrix   = as_matrix([[B_11, B_12], [B_12, B_22]])
gra_alpha = as_vector([alpha.dx(0), alpha.dx(1)])
z = sympy.Symbol("z", positive=True)
c_w = float(4 * sympy.integrate(sympy.sqrt(w(z)), (z, 0, 1)))
dissipated_energy = Gc/float(c_w)*(w(alpha)/ell+ell*dot(gra_alpha, BMatrix*gra_alpha))*dx
damage_functional = elastic_potential + dissipated_energy

# Compute directional derivative about alpha in the direction of beta (Gradient)
E_alpha       = derivative(damage_functional, alpha, beta)
# Compute directional derivative about alpha in the direction of dalpha (Hessian)
E_alpha_alpha = derivative(E_alpha, alpha, dalpha)

# --------------------------------------------------------------------
# Implement the box constraints for damage field
# --------------------------------------------------------------------
# Variational problem for the damage (non-linear to use variational inequality solvers of petsc)
problem_alpha = NonlinearVariationalProblem(E_alpha, alpha, bc_alpha, J=E_alpha_alpha)
# Set up the solvers
solver_alpha  = NonlinearVariationalSolver(problem_alpha)
solver_alpha.parameters.update(solver_alpha_parameters)
# info(solver_alpha.parameters,True) # uncomment to see available parameters

alpha_lb = interpolate(Expression("0.", degree=0), V_alpha)  # lower bound, set to 0
alpha_ub = interpolate(Expression("1.", degree=0), V_alpha)  # upper bound, set to 1
problem_alpha.set_bounds(alpha_lb, alpha_ub)  # set box constraints
# problem_alpha.set_bounds(alpha_0, alpha_ub)  # set box constraints

# loading and initialization of vectors to store time datas
load_multipliers  = np.linspace(load_min, load_max, load_steps)
energies          = np.zeros((len(load_multipliers), 4))
iterations        = np.zeros((len(load_multipliers), 2))

# set the saved data file name
file_u      = XDMFFile(mpi_comm_world(), savedir + "/u.xdmf")
file_u.parameters["flush_output"]     = True
file_alpha  = XDMFFile(mpi_comm_world(), savedir + "/alpha.xdmf")
file_alpha.parameters["flush_output"] = True

# ----------------------------------------------------------------------------
# Solving at each timestep
# ----------------------------------------------------------------------------
for (i_t, t) in enumerate(load_multipliers):
    u_U.t = t * ut
    if MPI.rank(mpi_comm_world()) == 0:
        print("\033[1;32m--- Starting of Time step {0:2d}: t = {1:4f} ---\033[1;m".format(i_t, t))
    # Alternate Mininimization
    # Initialization
    iteration = 1
    err_alpha = 1.0
    # Iterations
    while err_alpha > AM_tolerance and iteration < maxiteration:
        # solve elastic problem
        solver_u.solve()

        # solve damage problem
        solver_alpha.solve()

        # test error
        alpha_error = alpha.vector() - alpha_0.vector()
        err_alpha = alpha_error.norm('linf')

        # monitor the results
        if MPI.rank(mpi_comm_world()) == 0:
          print ("AM Iteration: {0:3d},  alpha_error: {1:>14.8f}".format(iteration, err_alpha))

        # update iteration
        alpha_0.assign(alpha)
        iteration = iteration + 1
    # updating the lower bound to account for the irreversibility
    alpha_lb.vector()[:] = alpha.vector()

    # Dump solution to file
    file_alpha.write(alpha, t)
    file_u.write(u, t)

    # ----------------------------------------
    # Some post-processing
    # ----------------------------------------
    # Save number of iterations for the time step
    iterations[i_t] = np.array([t, iteration])

    # Calculate the energies
    elastic_energy_value = assemble(elastic_energy)
    surface_energy_value = assemble(dissipated_energy)
    energies[i_t] = np.array([t, elastic_energy_value, surface_energy_value, elastic_energy_value+surface_energy_value])

    if MPI.rank(mpi_comm_world()) == 0:
        print("\nEnd of timestep {0:3d} with load multiplier {1:4f}".format(i_t, t))
        print("\nElastic and Surface Energies: [{0:6f},{0:6f}]".format(elastic_energy_value, surface_energy_value))
        print("-----------------------------------------")
        # Save some global quantities as a function of the time
        np.savetxt(savedir + '/energies.txt', energies)
        np.savetxt(savedir + '/iterations.txt', iterations)
# ----------------------------------------------------------------------------

# Plot energy and stresses
if MPI.rank(mpi_comm_world()) == 0:
    p1, = plt.plot(energies[:, 0], energies[:, 1])
    p2, = plt.plot(energies[:, 0], energies[:, 2])
    p3, = plt.plot(energies[:, 0], energies[:, 3])
    plt.legend([p1, p2, p3], ["Elastic", "Dissipated", "Total"], loc="best", frameon=False)
    plt.xlabel('Displacement')
    plt.ylabel('Energies')
    plt.savefig(savedir + '/energies.pdf', transparent=True)
    plt.close()