#
# =============================================================================
# FEniCS code  Variational Fracture Mechanics
# =============================================================================
#
# A static solution of the variational fracture mechanics problems
# using the regularization strongly anisotropic damage model
#
# author: Bin Li (bl736@cornell.edu), Corrado Maurini (corrado.maurini@upmc.fr)
#
# date: 10/10/2017
# --------------


# -----------------------------------------------------------------------------
from __future__ import division
import sys, petsc4py
petsc4py.init(sys.argv)
from dolfin import *
from mshr import *

import fem
from fem.utils import inner_e
from fem.functionspace import *

import argparse
import math
import os
import shutil
import sympy
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../")
from utils import save_timings
# -----------------------------------------------------------------------------
# Parameters for DOLFIN and SOLVER
# -----------------------------------------------------------------------------
set_log_level(20)  # log level
# set some dolfin specific parameters
parameters.use_petsc_signal_handler = True
parameters["ghost_mode"] = "shared_facet"
parameters.form_compiler.update({"representation": "uflacs", "optimize": True, "cpp_optimize": True, "quadrature_degree": 2})
info(parameters,True)
# -----------------------------------------------------------------------------
# parameters of the solvers
solver_u_parameters = {"linear_solver": "cg", "symmetric": True,"preconditioner": "hypre_amg",
                       "krylov_solver": {"report": False,"monitor_convergence": False,"relative_tolerance": 1e-8}}
petscop = PETScOptions()
#petscop.set("help")
petscop.set("snes_type","vinewtonssls")
petscop.set("snes_converged_reason")
petscop.set("snes_linesearch_type","basic") #shell basic l2 bt nleqerr cp
petscop.set("pc_type","lu")
petscop.set("pc_factor_mat_solver_package","mumps")
petscop.set("snes_monitor")
petscop.set("snes_vi_zero_tolerance",1.e-6)
petscop.set("snes_stol",1.e-6)
petscop.set("snes_atol",1.e-8)
petscop.set("snes_rtol",1.e-6)
petscop.set("snes_max_it",100)
petscop.set("snes_error_if_not_converged",1)
petscop.set("snes_force_iteration",1)

# set the user parameters
parameters.parse()
userpar = Parameters("user")
userpar.add("C11",1.8)
userpar.add("C12",-1.7)
userpar.add("C44",0.15)
userpar.add("theta0",26.0)
userpar.add("KI",1.) # mode I loading
userpar.add("KII",0.) # mode II loading
userpar.add("meshsize",25)
userpar.add("load_min",0.)
userpar.add("load_max",1.5)
userpar.add("load_steps",10)
userpar.add("MITC","project",["project","full"])
userpar.parse()

theta0 = userpar.theta0*np.pi/180.0
KI = userpar.KI
KII = userpar.KII

# Constitutive matrix Cmat for the fourth order phase-field and its rotated matrix Cmatr
Cmat = [[userpar.C11, userpar.C12, 0], [userpar.C12, userpar.C11, 0], [0, 0, userpar.C44]]
K = [[np.cos(theta0)**2, np.sin(theta0)**2 ,  2.0*np.cos(theta0)*np.sin(theta0)], \
            [np.sin(theta0)**2, np.cos(theta0)**2 , -2.0*np.cos(theta0)*np.sin(theta0)], \
            [-np.cos(theta0)*np.sin(theta0), np.cos(theta0)*np.sin(theta0) , np.cos(theta0)**2-np.sin(theta0)**2]]
Cmatr = np.matmul(np.matmul(K,Cmat), np.transpose(K))

#Rotated constitutive matrix
Cr11 = Cmatr[0,0]
Cr12 = Cmatr[0,1]
Cr14 = Cmatr[0,2]
Cr44 = Cmatr[2,2]

# Material constant
E = Constant(7.0e0)
nu = Constant(0.3)
Gc = Constant(1.0)
k_ell = Constant(1.e-6)  # residual stiffness

# Loading
ut = 1.0   # reference value for the loading (imposed displacement)

# Numerical parameters of the alternate minimization
maxiteration = 2000
AM_tolerance = 1e-4

# Geometry paramaters
L = 0.1
N = userpar.meshsize
hsize = float(L/N)
cra_angle = float(.5*np.pi/180.0)
cra_w = 0.05*L*tan(cra_angle)
ell = Constant(2.0*hsize)

modelname = "strong-kinking"
meshname  = modelname+"-mesh.xdmf"
simulation_params = "C11_%.4f_C12_%.4f_C44_%.4f_theta0_%.4f_KI_%.4f_KII_%.4f_h_%.4f_%s" %(userpar.C11, userpar.C12, userpar.C44, theta0, KI, KII, hsize, userpar.MITC)
savedir   = "output/"+modelname+"/"+simulation_params+"/"

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
mesh     = generate_mesh(geometry, N, 'cgal')
geo_mesh = XDMFFile(mpi_comm_world(), meshname)
geo_mesh.write(mesh)

mesh.init()
ndim = mesh.geometry().dim() # get number of space dimensions
if MPI.rank(mpi_comm_world()) == 0:
    print ("the dimension of mesh: {0:2d}".format(ndim))

# -----------------------------------------------------------------------------
# Strain and stress and Constitutive functions of the damage model
# -----------------------------------------------------------------------------
# Strain and stress
def eps(v):
    return sym(grad(v))

def sigma_0(v):
    mu    = E/(2.0*(1.0 + nu))
    lmbda = E*nu/(1.0 - nu**2) # plane stress
    return 2.0*mu*(eps(v)) + lmbda*tr(eps(v))*Identity(ndim)

# Constitutive functions of the damage model
def w(alpha):
    return 9.0*alpha

def aa(alpha):
    return (1-alpha)**2

# -----------------------------------------------------------------------------
# Define boundary sets for boundary conditions
# Impose the displacements field given by asymptotic expansion of crack tip
#----------------------------------------------------------------------------
def boundaries(x):
    return near(x[1], 0.5*L, 0.1*hsize) or near(x[1], -0.5*L, 0.1*hsize) \
        or near(x[0], 0.0, 0.1*hsize) or near(x[0], L, 0.1*hsize)

# -----------------------------------------------------------------------------
# Variational formulation
# -----------------------------------------------------------------------------
# Create function space for 2D elasticity
V_u = VectorFunctionSpace(mesh, "Lagrange", 1)

# Create function space for damage using mixed formulation
element_alpha = FiniteElement("Lagrange", triangle, 1)
element_a = VectorElement("Lagrange", triangle, 2)
element_s = FiniteElement("N1curl", triangle, 1)
element_p = RestrictedElement(FiniteElement("N1curl", triangle, 1), "edge")
element = MixedElement([element_alpha,element_a,element_s,element_p])
V_alpha = FunctionSpace(mesh,element_alpha)
V_a = FunctionSpace(mesh,element_a)
V_s = FunctionSpace(mesh,element_s)
V_p = FunctionSpace(mesh,element_p)
V_damage = ProjectedFunctionSpace(mesh, element, num_projected_subspaces=2)
V_damage_F = V_damage.full_space
V_damage_P = V_damage.projected_space
assigner_F = FunctionAssigner(V_damage_F,[V_alpha,V_a,V_s,V_p])
assigner_P = FunctionAssigner(V_damage_P,[V_alpha,V_a])

# Define the function, test and trial fields
u = Function(V_u, name="Displacement")
du = TrialFunction(V_u)
v = TestFunction(V_u)
damage = Function(V_damage_F, name="Damage")
damage_trial = TrialFunction(V_damage_F)
damage_test = TestFunction(V_damage_F),
alpha,a,s,p = split(damage)
damage_p = Function(V_damage_P, name="Damage")

# Define the bounds for the damage field
alpha_lb = Function(V_alpha); a_lb = Function(V_a); s_lb = Function(V_s); p_lb = Function(V_p);
alpha_ub = Function(V_alpha); a_ub = Function(V_a); s_ub = Function(V_s); p_ub = Function(V_p);
alpha_ub.vector()[:] = 1.; a_ub.vector()[:] = np.infty; s_ub.vector()[:] = np.infty; p_ub.vector()[:] = np.infty;
alpha_lb.vector()[:] = 0.; a_lb.vector()[:] = -np.infty; s_lb.vector()[:] = -np.infty; p_lb.vector()[:] = -np.infty;
if userpar.MITC == "project":
    damage_lb = Function(V_damage_P); damage_ub = Function(V_damage_P)
    alpha_ub.vector()[:] = 1.; a_ub.vector()[:] = np.infty;
    alpha_lb.vector()[:] = 0.; a_lb.vector()[:] = -np.infty;
    assigner_P.assign(damage_ub,[alpha_ub, a_ub])
    assigner_P.assign(damage_lb,[alpha_lb, a_lb])
else:
    damage_lb = Function(V_damage_F); damage_ub = Function(V_damage_F)
    assigner_F.assign(damage_ub,[alpha_ub,a_ub,s_ub,p_ub])
    assigner_F.assign(damage_lb,[alpha_lb,a_lb,s_lb,p_lb])

# -----------------------------------------------------------------------------
# Dirichlet boundary condition
# Impose the displacements field given by asymptotic expansion of crack tip
# -----------------------------------------------------------------------------
mu    = float(E/(2.0*(1.0 + nu)))
kappav = float((3.0-nu)/(1.0+nu))
nKI   = float(sqrt(E*Gc))
u_U   = Expression(["t*KI*nKI/(2*mu)*sqrt(sqrt((x[0]-lc)*(x[0]-lc)+x[1]*x[1])/(2*pi))*(kappa-cos(atan2(x[1], x[0]-lc)))*cos(atan2(x[1], x[0]-lc)/2) + \
                    t*KII*nKI/(2*mu)*sqrt(sqrt((x[0]-lc)*(x[0]-lc)+x[1]*x[1])/(2*pi))*(2.0+kappa+cos(atan2(x[1], x[0]-lc)))*sin(atan2(x[1], x[0]-lc)/2)",
                    "t*KI*nKI/(2*mu)*sqrt(sqrt((x[0]-lc)*(x[0]-lc)+x[1]*x[1])/(2*pi))*(kappa-cos(atan2(x[1], x[0]-lc)))*sin(atan2(x[1], x[0]-lc)/2) + \
                    t*KII*nKI/(2*mu)*sqrt(sqrt((x[0]-lc)*(x[0]-lc)+x[1]*x[1])/(2*pi))*(2.0-kappa-cos(atan2(x[1], x[0]-lc)))*cos(atan2(x[1], x[0]-lc)/2)"],
                    degree=2, mu=mu, kappa=kappav, nKI=nKI, KI=KI, KII=KII, lc=0.5*L, t=0.0)

# Boundary conditions
bc_u = [DirichletBC(V_u, u_U, boundaries)]
if userpar.MITC == "project":
    bc_damage = [DirichletBC(V_damage_P.sub(0), 0.0, boundaries)]
else:
    bc_damage = [DirichletBC(V_damage_F.sub(0), 0.0, boundaries)]

#--------------------------------------------------------------------
# Define the variational problem
#--------------------------------------------------------------------
# Displacement subproblem
#--------------------------------------------------------------------
def sigma(u, alpha):
    return (aa(alpha)+k_ell)*sigma_0(u)

body_force = Constant((0.,0.))
elastic_energy = 0.5*inner(sigma(u, alpha), eps(u))*dx
external_work = dot(body_force, u)*dx
elastic_potential = elastic_energy - external_work

# Weak form of elasticity problem
E_u  = derivative(elastic_potential,u,v)
# Writing tangent problems in term of test and trial functions for matrix assembly
E_du = replace(E_u,{u:du})
# Variational problem for the displacement
problem_u = LinearVariationalProblem(lhs(E_du), rhs(E_du), u, bc_u)
# Set up the solver
solver_u = LinearVariationalSolver(problem_u)
solver_u.parameters.update(solver_u_parameters)
#--------------------------------------------------------------------
# Damage subproblem
#--------------------------------------------------------------------
kappa_tensor = sym(grad(a)) # Hessian matrix of damage field
kappa = as_vector([kappa_tensor[0,0], kappa_tensor[1,1], kappa_tensor[0,1]])
# Voigt notation for fourth-order tensor Cr
Crv = as_matrix([[Cr11, Cr12, 2.0*Cr14], \
                    [Cr12, Cr11, -2.0*Cr14], \
                    [2.0*Cr14, -2.0*Cr14, 4.0*Cr44]])
dissipated_energy = Constant(5.0/96.0)*Gc*(w(alpha)/ell+pow(ell,3)*dot(kappa, Crv*kappa))*dx
penalty_energy = Constant(1.0e+3)*inner(s, s)*dx
# Here we show another way to apply the Duran-Liberman reduction operator,
# through constructing a Lagrangian term L_R.
# -----------------------------------------------------------------------------
# Impose the constraint that s=(grad(w)-theta) in a weak form
constraint = inner_e(grad(alpha)-a-s, p, False)
damage_functional = elastic_energy + dissipated_energy + penalty_energy + constraint
# Compute directional derivative about alpha in the test direction (Gradient)
F = derivative(damage_functional, damage, damage_test)
# Compute directional derivative about alpha in the trial direction (Hessian)
J = derivative(F, damage, damage_trial)
# =============================================================================

# loading and initialization of vectors to store time datas
load_multipliers = np.linspace(userpar.load_min,userpar.load_max,userpar.load_steps)
energies = np.zeros((len(load_multipliers),4))
iterations = np.zeros((len(load_multipliers),2))

file_u = XDMFFile(mpi_comm_world(), savedir+"/u.xdmf")
file_alpha = XDMFFile(mpi_comm_world(), savedir+"/alpha.xdmf")
file_a = XDMFFile(mpi_comm_world(), savedir+"/a.xdmf")
file_alpha.parameters.update({"flush_output": True, "rewrite_function_mesh" : False})
file_u.parameters.update({"flush_output": True, "rewrite_function_mesh" : False})
file_a.parameters.update({"flush_output": True, "rewrite_function_mesh" : False})

# -----------------------------------------------------------------------------
# Solving at each timestep
# -----------------------------------------------------------------------------
(alpha_0, a_0, s_0, p_0) = damage.split(deepcopy=True)
if userpar.MITC == "project":
    problem_damage = fem.ProjectedNonlinearProblem(V_damage_P, F, damage, damage_p, bcs=bc_damage, J=J)
else:
    problem_damage = fem.FullNonlinearProblem(V_damage_F, F, damage, bcs=bc_damage, J=J)
solver_damage = PETScSNESSolver("vinewtonssls")
snes = solver_damage.snes()
snes.setFromOptions()

for (i_t, t) in enumerate(load_multipliers):
    u_U.t = t*ut
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
        if userpar.MITC == "project":
            solver_damage.solve(problem_damage,damage_p.vector(),damage_lb.vector(),damage_ub.vector())
        else:
            solver_damage.solve(problem_damage,damage.vector(),damage_lb.vector(),damage_ub.vector())
        # check error
        (alpha_1, a_1, s_1, p_1) = damage.split(deepcopy=True)
        alpha_error = alpha_1.vector() - alpha_0.vector()
        err_alpha   = alpha_error.norm('linf')
        # monitor the results
        if MPI.rank(mpi_comm_world()) == 0:
            print ("AM Iteration: {0:3d},  alpha_error: {1:>14.8f}".format(iteration, err_alpha))
        # update iterations
        alpha_0.assign(alpha_1)
        iteration = iteration+1

    # updating the lower bound to account for the irreversibility
    if userpar.MITC == "project":
        assigner_P.assign(damage_lb,[project(alpha_1,V_alpha),a_lb])
    else:
        assigner_F.assign(damage_lb,[project(alpha_1,V_alpha),a_lb,s_lb,p_lb])

    # ----------------------------------------
    # Some post-processing
    # ----------------------------------------
    # Dump solution to file
    file_a.write(damage.split()[1],t)
    file_alpha.write(damage.split()[0],t)
    file_u.write(u,t)
    iterations[i_t] = np.array([t,iteration])
    elastic_energy_value = assemble(elastic_energy)
    surface_energy_value = assemble(dissipated_energy)
    energies[i_t] = np.array([t, elastic_energy_value, surface_energy_value, elastic_energy_value+surface_energy_value])

    if MPI.rank(mpi_comm_world()) == 0:
        print("\nEnd of timestep {0:3d} with load multiplier {1:4f}".format(i_t, t))
        print("\nElastic and Surface Energies: [{0:6f},{1:6f}]".format(elastic_energy_value, surface_energy_value))
        print("\nElastic and Surface Energies: [%s,%s]"%(elastic_energy_value, surface_energy_value))
        print("-----------------------------------------")
        # Save some global quantities as a function of the time
        np.savetxt(savedir+'/energies.txt', energies)
        np.savetxt(savedir+'/iterations.txt', iterations)
        print("Results saved in ", savedir)
    save_timings(savedir)

# Plot energy and stresses
if MPI.rank(mpi_comm_world()) == 0:
    p1, = plt.plot(energies[:,0], energies[:,1])
    p2, = plt.plot(energies[:,0], energies[:,2])
    p3, = plt.plot(energies[:,0], energies[:,3])
    plt.legend([p1, p2, p3], ["Elastic","Dissipated","Total"],loc = "best", frameon=False)
    plt.xlabel('Displacement')
    plt.ylabel('Energies')
    plt.savefig(savedir+'/energies.pdf', transparent=True)
    plt.close()

