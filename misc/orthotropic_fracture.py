# FEnics code  Variational Fracture Mechanics
#
# A static solution of the variational fracture mechanics problems using 
# the regularization two-fold anisotropic damage model
# ----------------------------------------------------------------------------
# author: Bin LI
# email: bin.li@upmc.fr 
# date: 10/10/2017
# ----------------------------------------------------------------------------

from __future__ import division

from dolfin import *
from mshr import *
from utils import *

import argparse
import math
import os
import shutil
import sympy
import sys

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
set_log_level(WARNING)  # log level
# parameters.parse()   # read paramaters from command line
# set some dolfin specific parameters
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"

# parameters of the nonlinear solver used for the alpha-problem
solver_alpha_parameters = {"nonlinear_solver": "snes",
                           "snes_solver": {"linear_solver": "mumps",
                                           "method": "vinewtonrsls",
                                           "line_search": "cp",  # "nleqerr",
                                           "preconditioner": "hypre_amg",
                                           "maximum_iterations": 20,
                                           "report": False,
                                           "error_on_nonconvergence": False}}

solver_u_parameters = {"linear_solver": "mumps",
                       "symmetric": True,
                       "preconditioner": "hypre_amg",
                       "krylov_solver": {
                           "report": False,
                           "monitor_convergence": False,
                           "relative_tolerance": 1e-8}}

# set the parameters to have target anisotropic surface energy
parser = argparse.ArgumentParser(description='The simple (second-order) anisotropic surface energy damage model')
parser.add_argument('BMat', type=float, nargs=3, help="input $3$ components of Tensor B")
parser.add_argument('KI_C', type=float, nargs=1, help="input critical loading KI_C")
parser.add_argument('psi',  type=float, nargs=1, help="input mixed mode ratio")
args = parser.parse_args()

B_11, B_12, B_22 = args.BMat[0], args.BMat[1], args.BMat[2]
KI, psi = args.KI_C[0], args.psi[0]

# Material constant
E, nu = Constant(72000), Constant(0.33)
Gc = Constant(10.209)
k_ell = Constant(1.e-6)  # residual stiffness

# Loading
ut = 1.0  # reference value for the loading (imposed displacement)
load_min = 0.00   # load multiplier min value
load_max = 0.5    # load multiplier max value
load_steps = 101  # number of time steps

# Numerical parameters of the alternate minimization
maxiteration = 2000
AM_tolerance = 1e-4

# mesh and damage paramaters
hsize = 0.05
ell = Constant(5.0 * hsize)

modelname = "orthotropic_fractureA1"
savedir = "orthotropic_fracture_paths/%s-B12_%.4f-S%.4f" % (modelname, B_12, hsize)

if mpi_comm_world().rank == 0:
    if os.path.isdir(savedir):
        shutil.rmtree(savedir)

# ----------------------------------------------------------------------------
# Geometry and mesh generation
# ----------------------------------------------------------------------------
# crack geometry

def define_mesh(mesh_size):
    geofile = \
    """
    lc = DefineNumber[ %g, Name "Parameters/lc" ];
    // orthotropic fracture toughness simulations
    e0 = 16.0; 
    Point(1) = { 0.0,   0.0,  0.0};
    Point(2) = {70.0,   0.0,  0.0};
    Point(3) = {70.0, 140.0,  0.0};
    Point(4) = { 0.0, 140.0,  0.0};
    Point(5) = { 0.0,  70.0,  0.0};
    Line(1)  = {1,2};
    Line(2)  = {2,3};
    Line(3)  = {3,4};
    Line(4)  = {4,5};
    Line(5)  = {5,1};

    // smeared crack
    Point(6) = {13.0,  70.0,  0.0};
    Line(6)  = {5,6};
    // Circle
    Point(7) = {32.0, 70.0+e0, 0.0};
    Point(8) = {40.0, 70.0+e0, 0.0};
    Point(9) = {48.0, 70.0+e0, 0.0};

    Circle(7) = {7,8,9};
    Circle(8) = {9,8,7};

    Line Loop(1) = {1,2,3,4,5,7,8};
    Plane Surface(1) = {1};
    Line {6} In Surface {1};
    Physical Surface(0) = {1};

    // use a Box field to impose a element sizes inside a box
    Field[1] = Box;
    Field[1].VIn = lc;
    Field[1].VOut = 10.0*lc;
    Field[1].XMin =  0.0;
    Field[1].XMax = 55.0;
    Field[1].YMin = 68.0;
    Field[1].YMax = 70.0+1.6*e0;
    Background Field = 1;
    Mesh.Algorithm = 2;
    """ % mesh_size
    # Generate XML files
    return mesher(geofile, "orthotropic_fracture%g" % mesh_size)

mesh = define_mesh(hsize)

#mesh = Mesh("orthotropic_fractureA1.xml")
ndim = mesh.geometry().dim()  # get number of space dimensions
if mpi_comm_world().rank == 0:
    print ("the dimension of mesh: {0:2d}".format(ndim))


# ----------------------------------------------------------------------------
# Strain and stress and Constitutive functions of the damage model
# ----------------------------------------------------------------------------

# Strain and stress
def eps(v):
    return sym(grad(v))


def sigma_0(v):
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / (1.0 - nu ** 2)  # plane stress
    return 2.0 * mu * (eps(v)) + lmbda * tr(eps(v)) * Identity(ndim)


# Constitutive functions of the damage model
def w(alpha):
    return alpha


def a(alpha):
    return (1 - alpha) ** 2


# ----------------------------------------------------------------------------
# Define boundary sets for boundary conditions
# ----------------------------------------------------------------------------
def upper_boundary(x, on_boundary):
    return on_boundary and near(x[1], 140.0, 0.1 * hsize)

def lower_boundary(x, on_boundary):
    return on_boundary and near(x[1], 0.0, 0.1 * hsize)

# ----------------------------------------------------------------------------
# Variational formulation 
# ----------------------------------------------------------------------------
# Create function space for 2D elasticity + Damage
V_u = VectorFunctionSpace(mesh, "Lagrange", 1)

V_alpha = FunctionSpace(mesh, "Lagrange", 1)

# Define the function, test and trial fields
u, du, v = Function(V_u, name="Displacement"), TrialFunction(V_u), TestFunction(V_u)
alpha, dalpha, beta = Function(V_alpha, name="Damage"), TrialFunction(V_alpha), TestFunction(V_alpha),

u_U = Expression(["0.0","t"], degree=0, t=0.0)
# bc - u (imposed displacement)
Gamma_u_0 = DirichletBC(V_u, Expression(["0.0", "0.0"], degree=0), lower_boundary)
Gamma_u_1 = DirichletBC(V_u, u_U, upper_boundary)
bc_u = [Gamma_u_0, Gamma_u_1]

# bc - alpha (zero damage)
Gamma_alpha_0 = DirichletBC(V_alpha, 0.0, lower_boundary)
Gamma_alpha_1 = DirichletBC(V_alpha, 0.0, upper_boundary)
bc_alpha = [Gamma_alpha_0, Gamma_alpha_1]


# --------------------------------------------------------------------
# Define the energy functional of damage problem
# --------------------------------------------------------------------

# Fenics forms for the energies
def sigma(u, alpha):
    return (a(alpha) + k_ell) * sigma_0(u)


body_force = Constant((0., 0.))

elastic_energy = 0.5 * inner(sigma(u, alpha), eps(u)) * dx
external_work = dot(body_force, u) * dx
elastic_potential = elastic_energy - external_work

# Weak form of elasticity problem
E_u = derivative(elastic_potential, u, v)
# Writing tangent problems in term of test and trial functions for matrix assembly
E_du = replace(E_u, {u: du})

# Variational problem for the displacement
problem_u = LinearVariationalProblem(lhs(E_du), rhs(E_du), u, bc_u)

# --------------------------------------------------------------------
# Define the energy functional of damage problem
# --------------------------------------------------------------------
# Initialize damage field
alpha_0 = Function(V_alpha)
alpha_0 = interpolate(Expression('0.', degree=0), V_alpha)  # initial (known) alpha
alpha_0 = interpolate(Expression('x[0]<= 13.0 & near(x[1], 70.0, tol) ? 1.0 : 0.0', degree=0, tol=0.5*hsize), V_alpha) # initial (known) alpha

# matrix notation for second-order tensor B
BMatrix = as_matrix([[B_11, B_12], [B_12, B_22]])
gra_alpha = as_vector([alpha.dx(0), alpha.dx(1)])
z = sympy.Symbol("z", positive=True)
c_w = float(4 * sympy.integrate(sympy.sqrt(w(z)), (z, 0, 1)))
dissipated_energy = Gc / float(c_w) * (w(alpha) / ell + ell * dot(gra_alpha, BMatrix * gra_alpha)) * dx

damage_functional = elastic_potential + dissipated_energy

# Compute directional derivative about alpha in the direction of beta (Gradient)
E_alpha = derivative(damage_functional, alpha, beta)
# Compute directional derivative about alpha in the direction of dalpha (Hessian)
E_alpha_alpha = derivative(E_alpha, alpha, dalpha)

# --------------------------------------------------------------------
# Implement the box constraints for damage field
# --------------------------------------------------------------------
# Variational problem for the damage (non-linear to use variational inequality solvers of petsc)
problem_alpha = NonlinearVariationalProblem(E_alpha, alpha, bc_alpha, J=E_alpha_alpha)

# Set up the solvers                                        
solver_u = LinearVariationalSolver(problem_u)
solver_u.parameters.update(solver_u_parameters)
# info(solver_u.parameters, True)
solver_alpha = NonlinearVariationalSolver(problem_alpha)
solver_alpha.parameters.update(solver_alpha_parameters)
# info(solver_alpha.parameters,True) # uncomment to see available parameters

alpha_lb = interpolate(Expression("0.", degree=0), V_alpha)  # lower bound, set to 0
alpha_ub = interpolate(Expression("1.", degree=0), V_alpha)  # upper bound, set to 1
# problem_alpha.set_bounds(alpha_lb, alpha_ub)  # set box constraints
problem_alpha.set_bounds(alpha_0, alpha_ub)  # set box constraints

#  loading and initialization of vectors to store time datas
load_multipliers = np.linspace(load_min, load_max, load_steps)
energies = np.zeros((len(load_multipliers), 4))
iterations = np.zeros((len(load_multipliers), 2))
forces = np.zeros((len(load_multipliers), 2))

file_u = XDMFFile(mpi_comm_world(), savedir + "/u.xdmf")
file_u.parameters["flush_output"] = True
file_alpha = XDMFFile(mpi_comm_world(), savedir + "/alpha.xdmf")
file_alpha.parameters["flush_output"] = True

# Solving at each timestep

for (i_t, t) in enumerate(load_multipliers):
    u_U.t = t * ut
    if mpi_comm_world().rank == 0:
        print("\033[1;32m--- Starting of Time step %d: t = %f ---\033[1;m" % (i_t, t))
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
        if mpi_comm_world().rank == 0:
            print ("AM Iteration: {0:5d},  alpha_error: {1:>14.8f}".format(iteration, err_alpha))

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
    energies[i_t] = np.array(
        [t, elastic_energy_value, surface_energy_value, elastic_energy_value + surface_energy_value])

    if mpi_comm_world().rank == 0:
        print("\nEnd of timestep %d with load multiplier %f" % (i_t, t))
        print("\nElastic and Surface Energies: (%g,%g)" % (elastic_energy_value, surface_energy_value))
        print("-----------------------------------------")
        # Save some global quantities as a function of the time
        np.savetxt(savedir + '/energies.txt', energies)
        np.savetxt(savedir + '/iterations.txt', iterations)

# Plot energy and stresses
if mpi_comm_world().rank == 0:
    p1, = plt.plot(energies[:, 0], energies[:, 1])
    p2, = plt.plot(energies[:, 0], energies[:, 2])
    p3, = plt.plot(energies[:, 0], energies[:, 3])
    plt.legend([p1, p2, p3], ["Elastic", "Dissipated", "Total"], loc="best", frameon=False)
    plt.xlabel('Displacement')
    plt.ylabel('Energies')
    plt.savefig(savedir + '/energies.pdf', transparent=True)
    plt.close()
