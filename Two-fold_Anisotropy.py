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

import argparse
import math
import os
import shutil
import sympy
import sys

import numpy as np
import matplotlib.pyplot as plt

from dolfin import *
from mshr import *
from utils import *

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

solver_u_parameters = {"linear_solver": "cg",
                       "symmetric": True,
                       "preconditioner": "hypre_amg",
                       "krylov_solver": {
                           "report": False,
                           "monitor_convergence": False,
                           "relative_tolerance": 1e-8}}

# set the parameters to have target anisotropic surface energy
parser = argparse.ArgumentParser(description='Anisotropic Surface Energy Damage Model')
parser.add_argument('BMat', type=float, nargs=5, help="input components of Tensor B and KI_C & KII=\psi*KI_C")
args = parser.parse_args()

B_11, B_12, B_22 = Constant(args.BMat[0]), Constant(args.BMat[1]), Constant(args.BMat[2])
KI, psi = Constant(args.BMat[3]), Constant(args.BMat[4])

# Material constant
E, nu = Constant(1.0), Constant(0.3)
Gc = Constant(1.0)
k_ell = Constant(1.e-6)  # residual stiffness

# Loading
ut = 1.0  # reference value for the loading (imposed displacement)
load_min = 0.95  # load multiplier min value
load_max = 1.25  # load multiplier max value
load_steps = 31  # number of time steps

# Numerical parameters of the alternate minimization
maxiteration = 2000
AM_tolerance = 1e-4

# Geometry paramaters
L = 0.2
hsize = 1.0e-3
N = int(L / hsize)
cra_w = 0.5 * hsize
cra_angle = 1.0 * pi / 180.
ell = Constant(5 * hsize)

modelname = "anisosurfenergy"
meshname = "meshes/square-L%s-S%.4f.xdmf" % (L, hsize)
savedir = "twofold_crack_path/%s-B12_%.4f-L%s-S%.4f-l%.4f" % (modelname, B_12, L, hsize, ell)

if mpi_comm_world().rank == 0:
    if os.path.isdir(savedir):
        shutil.rmtree(savedir)

# ----------------------------------------------------------------------------
# Geometry and mesh generation
# ----------------------------------------------------------------------------
# crack geometry

P1, P2 = Point(0., -0.5 * cra_w), Point(0.5 * (L - cra_w / tan(cra_angle)), -0.5 * cra_w)
P4, P5 = Point(0.5 * (L - cra_w / tan(cra_angle)), 0.5 * cra_w), Point(0., 0.5 * cra_w)
P3 = Point(0.5 * L, 0.)

geometry = Rectangle(Point(0., -0.5 * L), Point(L, 0.5 * L)) - Polygon([P1, P2, P3, P4, P5])
'''
P1, P2, P3 = Point(0., -0.1*hsize), Point(0.5*L, 0.), Point(0., 0.1*hsize)

geometry = Rectangle(Point(0., -0.5*L), Point(L, 0.5*L)) - Polygon([P1,P2,P3])
'''
# Mesh generation using cgal
mesh = generate_mesh(geometry, N, 'cgal')
geo_mesh = XDMFFile(mpi_comm_world(), meshname)
geo_mesh.write(mesh)

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
# Impose the displacements field given by asymptotic expansion of crack tip
# ----------------------------------------------------------------------------
def boundaries(x, on_boundary):
    return on_boundary and (
        near(x[1], 0.5 * L, 0.1 * hsize) or near(x[1], -0.5 * L, 0.1 * hsize) or near(x[0], 0.0, 0.1 * hsize) or near(
            x[0],
            L,
            0.1 * hsize))


# ----------------------------------------------------------------------------
# Variational formulation 
# ----------------------------------------------------------------------------
# Create function space for 2D elasticity + Damage
V_u = VectorFunctionSpace(mesh, "Lagrange", 1)

V_alpha = FunctionSpace(mesh, "Lagrange", 1)

# Define the function, test and trial fields
u, du, v = Function(V_u, name="Displacement"), TrialFunction(V_u), TestFunction(V_u)
alpha, dalpha, beta = Function(V_alpha, name="Damage"), TrialFunction(V_alpha), TestFunction(V_alpha),

# --------------------------------------------------------------------
# Dirichlet boundary condition
# Impose the displacements field given by asymptotic expansion of crack tip
# --------------------------------------------------------------------
mu = float(E / (2.0 * (1.0 + nu)))
# KI = float(sqrt(E*Gc))
kappa = float((3.0 - nu) / (1.0 + nu))

u_U = Expression(["t*KI/(2*mu)*sqrt(sqrt((x[0]-lc)*(x[0]-lc)+x[1]*x[1])/(2*pi))*(kappa-cos(atan2(x[1], x[0]-lc)))*cos(atan2(x[1], x[0]-lc)/2) + \
                   t*psi*KI/(2*mu)*sqrt(sqrt((x[0]-lc)*(x[0]-lc)+x[1]*x[1])/(2*pi))*(2.0+kappa+cos(atan2(x[1], x[0]-lc)))*sin(atan2(x[1], x[0]-lc)/2)",
                  "t*KI/(2*mu)*sqrt(sqrt((x[0]-lc)*(x[0]-lc)+x[1]*x[1])/(2*pi))*(kappa-cos(atan2(x[1], x[0]-lc)))*sin(atan2(x[1], x[0]-lc)/2) + \
                   t*psi*KI/(2*mu)*sqrt(sqrt((x[0]-lc)*(x[0]-lc)+x[1]*x[1])/(2*pi))*(2.0-kappa-cos(atan2(x[1], x[0]-lc)))*cos(atan2(x[1], x[0]-lc)/2)"],
                 degree=2, mu=mu, kappa=kappa, KI=KI, psi=psi, lc=0.5 * L, t=0.0)

# bc - u (imposed displacement)
Gamma_u_0 = DirichletBC(V_u, u_U, boundaries)
bc_u = [Gamma_u_0]

# bc - alpha (zero damage)
Gamma_alpha_0 = DirichletBC(V_alpha, 0.0, boundaries)
bc_alpha = [Gamma_alpha_0]


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
alpha_0 = interpolate(Expression("0.", degree=0), V_alpha)  # initial (known) alpha

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
problem_alpha.set_bounds(alpha_lb, alpha_ub)  # set box constraints

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
