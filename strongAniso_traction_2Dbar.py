#  
# =============================================================================
# FEnics code  Variational Fracture Mechanics
# =============================================================================
# 
# A static solution of the variational fracture mechanics problems  
# using the regularization strongly anisotropic damage model
#
# author: bin.li@upmc.fr 
#
# date: 10/10/2017
#
# -----------------------------------------------------------------------------
# runing: python3 strongAniso_traction_2Dbar.py 1.8 -1.7 0.15 2.5
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
from __future__ import division
from dolfin import *
from mshr import *
from fenics_shells import *  # Duran-Liberman reduction operator

import argparse
import math
import os
import shutil
import sympy
import sys
import numpy as np
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Parameters for DOLFIN and SOLVER 
# -----------------------------------------------------------------------------
#set_log_level(WARNING)  # log level
set_log_level(ERROR)  # log level
# set some dolfin specific parameters
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"

# -----------------------------------------------------------------------------
# parameters of the nonlinear solver used for the alpha-problem
solver_alpha_parameters = {"nonlinear_solver": "snes",
                           "snes_solver": {"linear_solver": "mumps",
                                           "method": "vinewtonssls",
                                           "line_search": "cp",#"nleqerr", #"cp",#
                                           #"preconditioner": "hypre_amg",
                                           "maximum_iterations": 20,
                                           "solution_tolerance": 1e-3,
                                           "relative_tolerance": 1e-4,
                                           "absolute_tolerance": 1e-4,
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
parser   = argparse.ArgumentParser(description='Anisotropic Surface Energy Damage Model')
parser.add_argument('Cmat',   type=float, nargs=3, help="input $3$ components of Tensor beta ")
parser.add_argument('theta0', type=float, nargs=1, help="input rotation angle $theta0(degree)$")

args     = parser.parse_args()

C11      = args.Cmat[0]
C12      = args.Cmat[1]
C44      = args.Cmat[2]
theta0   = args.theta0[0]*np.pi/180.0

C_mat    = [[C11, C12, 0], [C12, C11, 0], [0, 0, C44]]
K        = [[np.cos(theta0)**2, np.sin(theta0)**2 ,  2.0*np.cos(theta0)*np.sin(theta0)], \
            [np.sin(theta0)**2, np.cos(theta0)**2 , -2.0*np.cos(theta0)*np.sin(theta0)], \
            [-np.cos(theta0)*np.sin(theta0), np.cos(theta0)*np.sin(theta0) , np.cos(theta0)**2-np.sin(theta0)**2]]
beta_mat = np.matmul(np.matmul(K,C_mat), np.transpose(K))

beta_11  = beta_mat[0,0]
beta_12  = beta_mat[0,1]
beta_14  = beta_mat[0,2]
beta_44  = beta_mat[2,2]

# ----------------------------------------------------------------------------
# Geometry and mesh generation and damage paramaters
# ----------------------------------------------------------------------------

# Geometry paramaters
L         = 1.0
H         = 0.4
N         = 250
hsize     = float(L/N)
ell       = Constant(6.0*hsize)

# Material constant
E        = Constant(1.0)
nu       = Constant(0.3)
Gc       = Constant(1.0)/(1.+45./96.*hsize/ell)
k_ell    = Constant(1.e-6)  # residual stiffness

# -----------------------------------------------------------------------------
# Loading Parameters
# -----------------------------------------------------------------------------
ut           = float(sqrt(45.*Gc*E/96./ell)) # reference value for the loading (imposed displacement)
load_min     = 0.0   # load multiplier min value
load_max     = 1.2   # load multiplier max value
load_steps   = 120.  # number of time steps

# Numerical parameters of the alternate minimization
maxiteration = 2000 
AM_tolerance = 1e-4

meshname  = "strongAniso_traction_2Dbar.xdmf"
modelname = "strongAniso_traction_2Dbar"
simulation_params = "C11_%.4f_C12_%.4f_C44_%.4f_theta0_%.1f_h_%.4f" %(C11, C12, C44, args.theta0[0], hsize)
savedir   = modelname+"/"+simulation_params+"/"

if MPI.rank(mpi_comm_world()) == 0: 
    if os.path.isdir(savedir):
        shutil.rmtree(savedir)

# Mesh generation 
#mesh     = RectangleMesh(Point(0., -0.5*H), Point(L, 0.5*H), int(N), int(float(H/hsize)), "right/left") 
geometry = Rectangle(Point(0., -0.5*H), Point(L, 0.5*H))
mesh     = generate_mesh(geometry, int(N), 'cgal')
geo_mesh = XDMFFile(mpi_comm_world(), savedir+meshname)
geo_mesh.write(mesh)

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

def a(alpha):
    return (1-alpha)**2
    
# ----------------------------------------------------------------------------
# Define boundary sets for boundary conditions
# Impose the displacements field 
# ----------------------------------------------------------------------------
def left_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0.0, 0.1 * hsize)

def right_boundary(x, on_boundary):
    return on_boundary and near(x[0], L, 0.1 * hsize)
              
# -----------------------------------------------------------------------------
# Variational formulation 
# -----------------------------------------------------------------------------
# Create function space for 2D elasticity + Damage
V_u       = VectorFunctionSpace(mesh, "Lagrange", 1)

# gradient of damage field, damage field, "shear strains" and Lagrange multiplier field
element   = MixedElement([VectorElement("Lagrange", triangle, 2),
                          FiniteElement("Lagrange", triangle, 1),
                          FiniteElement("N1curl", triangle, 1),
                          RestrictedElement(FiniteElement("N1curl", triangle, 1), "edge")])

V_alpha   = ProjectedFunctionSpace(mesh, element, num_projected_subspaces=2)
V_alpha_F = V_alpha.full_space
#V_alpha_P = V_alpha.projected_space

# Define the function, test and trial fields
u         = Function(V_u, name="Displacement")
du        =  TrialFunction(V_u)
v         =  TestFunction(V_u)
alpha     = Function(V_alpha_F, name="Damage")
dalpha    = TrialFunction(V_alpha_F)
omega     = TestFunction(V_alpha_F),  
theta_, alpha_, R_gamma_, p_ = split(alpha)

# --------------------------------------------------------------------
# Dirichlet boundary condition
# Impose the displacements field 
# --------------------------------------------------------------------
u_UL = Expression(["0.0", "0.0"], degree=0)
#u_UR = Expression(["t",   "0.0"], t=0.0, degree=0)
u_UR = Expression("t", t=0.0, degree=0) # slide Dirichlet BCs

# bc - u (imposed displacement)
Gamma_u_0 = DirichletBC(V_u, u_UL, left_boundary)
Gamma_u_1 = DirichletBC(V_u.sub(0), u_UR, right_boundary) # slide Dirichlet boundary condition
#Gamma_u_1 = DirichletBC(V_u, u_UR, right_boundary) # non-slide Dirichlet boundary condition
bc_u = [Gamma_u_0, Gamma_u_1]

# bc - alpha (zero damage)
Gamma_alpha_0 = DirichletBC(V_alpha_F.sub(1), 0.0, left_boundary)
Gamma_alpha_1 = DirichletBC(V_alpha_F.sub(1), 0.0, right_boundary)
bc_alpha      = [Gamma_alpha_0, Gamma_alpha_1]

#--------------------------------------------------------------------
# Define the energy functional of damage problem
#--------------------------------------------------------------------

# Fenics forms for the energies
def sigma(u, alpha):
    return (a(alpha)+k_ell)*sigma_0(u)
    
body_force        = Constant((0.,0.))
elastic_energy    = 0.5*inner(sigma(u, alpha_), eps(u))*dx
external_work     = dot(body_force, u)*dx
elastic_potential = elastic_energy - external_work

# Weak form of elasticity problem
E_u  = derivative(elastic_potential,u,v)
# Writing tangent problems in term of test and trial functions for matrix assembly
E_du = replace(E_u,{u:du})

# Variational problem for the displacement
problem_u = LinearVariationalProblem(lhs(E_du), rhs(E_du), u, bc_u)
# Set up the solvers                                        
solver_u  = LinearVariationalSolver(problem_u)
solver_u.parameters.update(solver_u_parameters)
#info(solver_u.parameters, True)

# -----------------------------------------------------------------------------
# Define the energy functional of damage problem
# -----------------------------------------------------------------------------
kappa  = sym(grad(theta_)) #Hessian matrix of damage field
kappa_ = as_vector([kappa[0,0], kappa[1,1], kappa[0,1]])
# matrix notation for fourth-order tensor beta
beta_M = as_matrix([[beta_11, beta_12, 2.0*beta_14], \
                    [beta_12, beta_11, -2.0*beta_14], \
                    [2.0*beta_14, -2.0*beta_14, 4.0*beta_44]])

dissipated_energy = Constant(5.0/96.0)*Gc*(w(alpha_)/ell+pow(ell,3)*dot(kappa_, beta_M*kappa_))*dx 
shear_energy      = Gc*Constant(1.0e3)*inner(R_gamma_, R_gamma_)*dx

# Here we show another way to apply the Duran-Liberman reduction operator,
# through constructing a Lagrangian term L_R.
# -----------------------------------------------------------------------------
# Return shear strain vector calculated from primal variables
def gamma(theta, w):
    return grad(w)-theta

DL_reduction      = inner_e(gamma(theta_, alpha_)-R_gamma_, p_, True)
damage_functional = elastic_energy+dissipated_energy+shear_energy+DL_reduction

# Compute directional derivative about alpha in the direction of omega (Gradient)
F = derivative(damage_functional, alpha, omega)
# Compute directional derivative about alpha in the direction of dalpha (Hessian)
J = derivative(F, alpha, dalpha)
problem_alpha     = NonlinearVariationalProblem(F, alpha, bc_alpha, J=J)

# =============================================================================
# Implement the box constraints for damage field
# -----------------------------------------------------------------------------
alpha_lb      = Function(V_alpha_F)
alpha_ub      = Function(V_alpha_F)
V_theta_lub   = VectorFunctionSpace(mesh, "Lagrange", 2)
V_alpha_lub   = FunctionSpace(mesh, "Lagrange", 1)
V_R_gamma_lub = FunctionSpace(mesh, "N1curl", 1)
V_p_lub       = FunctionSpace(mesh, RestrictedElement(FiniteElement("N1curl", triangle, 1), "edge"))

# -----------------------------------------------------------------------------
# BCs for damage field
alpha_0 = Function(V_alpha_lub);  
alpha_0 = interpolate(Constant(0.0), V_alpha_lub); # Initialize damage field        
#alpha_0 = interpolate(Expression("near(x[0], 0.5*L, tol) & near(x[1], -0.5*H, tol) ? 0.5 : 0.0", \
#                                  degree=0, L= L, H=H, tol=0.5*hsize), V_alpha_lub) 

# -----------------------------------------------------------------------------
theta_n   = Function(V_theta_lub);    # current solution gradient of damage field
alpha_n   = Function(V_alpha_lub);    # current solution damage field
R_gamma_n = Function(V_R_gamma_lub);  # current solution "shear strain" field
p_n       = Function(V_p_lub);        # current solution Lagrange multiplier field
# -----------------------------------------------------------------------------
theta_lb  = Function(V_theta_lub)
theta_ub  = Function(V_theta_lub)
# -----------------------------------------------------------------------------
V_alpha_lub_       = FunctionSpace(mesh, "Lagrange", 2)
ninfty             = Function(V_alpha_lub_); 
ninfty.vector()[:] = -np.infty
pinfty             = Function(V_alpha_lub_); 
pinfty.vector()[:] =  np.infty
# -----------------------------------------------------------------------------
ninfty_R_gamma             = Function(V_R_gamma_lub); 
ninfty_R_gamma.vector()[:] = -np.infty
pinfty_R_gamma             = Function(V_R_gamma_lub); 
pinfty_R_gamma.vector()[:] =  np.infty
# -----------------------------------------------------------------------------
ninfty_p_                  = Function(V_p_lub); 
ninfty_p_.vector()[:]      = -np.infty
pinfty_p_                  = Function(V_p_lub); 
pinfty_p_.vector()[:]      =  np.infty
# -----------------------------------------------------------------------------
theta_lub = FunctionAssigner(V_theta_lub, [V_alpha_lub_, V_alpha_lub_])
theta_lub.assign(theta_lb, [ninfty, ninfty])
theta_lub.assign(theta_ub, [pinfty, pinfty])
# -----------------------------------------------------------------------------
assigner_lub   = FunctionAssigner(V_alpha_F, [V_theta_lub, V_alpha_lub, V_R_gamma_lub, V_p_lub])
# lower bound, set to 0 or intial alpha 
assigner_lub.assign(alpha_lb, [theta_lb, alpha_0, ninfty_R_gamma, ninfty_p_]) 
# upper bound, set to 1
assigner_lub.assign(alpha_ub, [theta_ub, interpolate(Expression("1.0", degree=0), V_alpha_lub), \
                    pinfty_R_gamma, pinfty_p_]) 
assigner_alpha = FunctionAssigner([V_theta_lub, V_alpha_lub, V_R_gamma_lub, V_p_lub], V_alpha_F)
problem_alpha.set_bounds(alpha_lb, alpha_ub) # set box constraints
#problem_alpha.set_bounds(alpha_0, alpha_ub) # set box constraints
# =============================================================================

solver_alpha = NonlinearVariationalSolver(problem_alpha)
solver_alpha.parameters.update(solver_alpha_parameters)
#info(solver_alpha.parameters,True) # uncomment to see available parameters

# loading and initialization of vectors to store time datas
load_multipliers = np.linspace(load_min,load_max,load_steps)
energies         = np.zeros((len(load_multipliers),4))
iterations       = np.zeros((len(load_multipliers),2))

file_u       = XDMFFile(mpi_comm_world(), savedir+"/u.xdmf")
file_u.parameters["flush_output"]       = True
file_alpha   = XDMFFile(mpi_comm_world(), savedir+"/alpha.xdmf")
file_alpha.parameters["flush_output"]   = True
file_R_gamma = XDMFFile(mpi_comm_world(), savedir+"/R_gamma.xdmf")
file_R_gamma.parameters["flush_output"] = True

# -----------------------------------------------------------------------------
# Solving at each timestep
# -----------------------------------------------------------------------------
for (i_t, t) in enumerate(load_multipliers):
    u_UR.t = t*ut
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
        assigner_alpha.assign([theta_n, alpha_n, R_gamma_n, p_n], alpha)
        (theta_1, alpha_1, R_gamma_1, p_1) =  alpha.split(deepcopy=True) 
        alpha_error = alpha_1.vector() - alpha_0.vector()
        err_alpha   = alpha_error.norm('linf')

        # monitor the results
        if MPI.rank(mpi_comm_world()) == 0:
            print ("AM Iteration: {0:3d},  alpha_error: {1:>14.8f}".format(iteration, err_alpha))

        # update iterations
        alpha_0.assign(alpha_1)
        iteration = iteration+1

    # updating the lower bound to account for the irreversibility
    assigner_lub.assign(alpha_lb, [theta_lb, alpha_n, ninfty_R_gamma, ninfty_p_]) # lower bound

    # Dump solution to file 
    file_R_gamma.write(alpha.split()[2],t) 
    file_alpha.write(alpha.split()[1],t) 
    file_u.write(u,t) 

    # ----------------------------------------
    # Some post-processing
    # ----------------------------------------
    # Save number of iterations for the time step    
    iterations[i_t] = np.array([t,iteration])

    # Calculate the energies
    elastic_energy_value = assemble(elastic_energy)
    surface_energy_value = assemble(dissipated_energy)
    energies[i_t] = np.array([t, elastic_energy_value, surface_energy_value, elastic_energy_value+surface_energy_value])

    if MPI.rank(mpi_comm_world()) == 0:
        print("\nEnd of timestep {0:3d} with load multiplier {1:4f}".format(i_t, t))
        print("\nElastic and Surface Energies: [{0:6f},{1:6f}]".format(elastic_energy_value, surface_energy_value))
        print("-----------------------------------------")
        # Save some global quantities as a function of the time
        np.savetxt(savedir+'/energies.txt', energies)
        np.savetxt(savedir+'/iterations.txt', iterations)

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
