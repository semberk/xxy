# FEnics code  Variational Fracture Mechanics
#
# A static solution of the variational fracture mechanics problems using the regularization fourth-order damage model
#
# author: bin.li@upmc.fr 
#
# date: 05/12/2016
#
from __future__ import division
from dolfin import *
from fenics_shells import *
from mshr import *
from utils import *
import sys, os, sympy, shutil, math, argparse
import numpy as np

#----------------------------------------------------------------------------
# Parameters
#----------------------------------------------------------------------------
set_log_level(WARNING) # log level
#parameters.parse()   # read paramaters from command line
# set some dolfin specific parameters
#parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"

# parameters of the nonlinear solver used for the alpha-problem
solver_alpha_parameters = {"nonlinear_solver": "snes",
                                  "snes_solver": {"linear_solver": "mumps",
                                            "method" : "vinewtonrsls",
                                            "line_search": "cp",#"nleqerr",
                                            "preconditioner" : "hypre_amg", 
                                          "maximum_iterations": 20,
                                          "report": False,
                                          "error_on_nonconvergence": False}}

solver_u_parameters =  {"linear_solver" : "cg", 
                            "symmetric" : True, 
                            "preconditioner" : "hypre_amg", 
                            "krylov_solver" : {
                                "report" : False,
                                "monitor_convergence" : False,
                                "relative_tolerance" : 1e-8 }}

# set the parameters to have target anisotropic surface energy
parser = argparse.ArgumentParser(description='Anisotropic Surface Energy Damage Model')
parser.add_argument('Cmat',   type=float, nargs=3, help="input $3$ components of Tensor beta ")
parser.add_argument('theta0', type=float, nargs=1, help="input rotation angle $theta0(degree)$")
parser.add_argument('KI_C',   type=float, nargs=1, help="input critical loading KI_C")
parser.add_argument('psi',    type=float, nargs=1, help="input mixed mode ratio")
args = parser.parse_args()
C11, C12, C44 = args.Cmat[0], args.Cmat[1], args.Cmat[2]
theta0, KI, psi = args.theta0[0]*np.pi/180.0, args.KI_C[0], args.psi[0]

C_mat = [[C11, C12, 0], [C12, C11, 0], [0, 0, C44]]
K = [[np.cos(theta0)**2, np.sin(theta0)**2 ,  2.0*np.cos(theta0)*np.sin(theta0)],\
     [np.sin(theta0)**2, np.cos(theta0)**2 , -2.0*np.cos(theta0)*np.sin(theta0)],\
     [-np.cos(theta0)*np.sin(theta0), np.cos(theta0)*np.sin(theta0) , np.cos(theta0)**2-np.sin(theta0)**2]]
beta_mat = np.matmul(np.matmul(K,C_mat), np.transpose(K))

beta_11 = beta_mat[0,0]
beta_12 = beta_mat[0,1]
beta_14 = beta_mat[0,2]
beta_44 = beta_mat[2,2]

# Material constant
E, nu = Constant(7.0e12), Constant(0.3)
Gc = Constant(1.0)
k_ell = Constant(1.e-6)  # residual stiffness

# Loading
ut = 1.0 # reference value for the loading (imposed displacement)
load_min = 1.10  # load multiplier min value
load_max = 1.40  # load multiplier max value
load_steps = 31  # number of time steps

# Numerical parameters of the alternate minimization
maxiteration = 2000 
AM_tolerance = 1e-4

# Geometry paramaters
L, N = 0.1, 250.0;
hsize, cra_angle = float(L/N), float(.5*np.pi/180.0)
cra_w, ell = 0.05*L*tan(cra_angle), Constant(8.0*hsize)

modelname = "strongAniso_crack_direction"
meshname = modelname+"/"+"strongAniso_crack_direction.xdmf"
savedir = "strongAniso_crack_direction/C11-%.4f-C12%.4f-C44-%.4f-theta0-%.4f-psi-%.4f-h%.4f" %(C11, C12, C44, args.theta0[0], psi, hsize)

if mpi_comm_world().rank == 0:
    if os.path.isdir(savedir):
        shutil.rmtree(savedir)

#----------------------------------------------------------------------------
# Geometry and mesh generation
#----------------------------------------------------------------------------
#crack geometry
P1, P2 = Point(0., -0.5*cra_w), Point(0.45*L, -0.5*cra_w)
P4, P5 = Point(0.45*L, 0.5*cra_w), Point(0., 0.5*cra_w) 
P3 = Point(0.5*L, 0.)
geometry = Rectangle(Point(0., -0.5*L), Point(L, 0.5*L)) - Polygon([P1,P2,P3,P4,P5])
# Mesh generation using cgal
mesh = generate_mesh(geometry, N, 'cgal')
geo_mesh = XDMFFile(mpi_comm_world(), meshname)
geo_mesh.write(mesh)

ndim = mesh.geometry().dim() # get number of space dimensions
if mpi_comm_world().rank == 0:
    print "the dimension of mesh: %.8g" %ndim

#----------------------------------------------------------------------------
# Strain and stress and Constitutive functions of the damage model
#----------------------------------------------------------------------------

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
    
#----------------------------------------------------------------------------
# Define boundary sets for boundary conditions
# Impose the displacements field given by asymptotic expansion of crack tip
#----------------------------------------------------------------------------
def boundaries(x, on_boundary):
    return on_boundary and (near(x[1], 0.5*L, 0.1*hsize) or near(x[1], -0.5*L, 0.1*hsize) or near(x[0], 0.0, 0.1*hsize) or near(x[0], L, 0.1*hsize))
              
#----------------------------------------------------------------------------
# Variational formulation 
#----------------------------------------------------------------------------
# Create function space for 2D elasticity + Damage
V_u = VectorFunctionSpace(mesh, "Lagrange", 1)

# gradient of damage field, damage field, "shear strains" and Lagrange multiplier field
element = MixedElement([VectorElement("Lagrange", triangle, 2),
                        FiniteElement("Lagrange", triangle, 1),
                        FiniteElement("N1curl", triangle, 1),
                        RestrictedElement(FiniteElement("N1curl", triangle, 1), "edge")])

V_alpha = ProjectedFunctionSpace(mesh, element, num_projected_subspaces=2)
V_alpha_F = V_alpha.full_space
V_alpha_P = V_alpha.projected_space

# Define the function, test and trial fields
u, du, v = Function(V_u, name="Displacement"), TrialFunction(V_u), TestFunction(V_u)
alpha, dalpha, omega = Function(V_alpha_F, name="Damage"), TrialFunction(V_alpha_F), TestFunction(V_alpha_F),  

theta_, alpha_, R_gamma_, p_ = split(alpha)

#--------------------------------------------------------------------
# Dirichlet boundary condition
# Impose the displacements field given by asymptotic expansion of crack tip
#--------------------------------------------------------------------
mu = float(E/(2.0*(1.0 + nu)))
#KI = float(sqrt(E*Gc))
kappa = float((3.0-nu)/(1.0+nu))
KI_ = float(sqrt(E*Gc))

u_U = Expression(["t*KI*KI_/(2*mu)*sqrt(sqrt((x[0]-lc)*(x[0]-lc)+x[1]*x[1])/(2*pi))*(kappa-cos(atan2(x[1], x[0]-lc)))*cos(atan2(x[1], x[0]-lc)/2) + \
                   t*psi*KI*KI_/(2*mu)*sqrt(sqrt((x[0]-lc)*(x[0]-lc)+x[1]*x[1])/(2*pi))*(2.0+kappa+cos(atan2(x[1], x[0]-lc)))*sin(atan2(x[1], x[0]-lc)/2)",
                  "t*KI*KI_/(2*mu)*sqrt(sqrt((x[0]-lc)*(x[0]-lc)+x[1]*x[1])/(2*pi))*(kappa-cos(atan2(x[1], x[0]-lc)))*sin(atan2(x[1], x[0]-lc)/2) + \
                   t*psi*KI*KI_/(2*mu)*sqrt(sqrt((x[0]-lc)*(x[0]-lc)+x[1]*x[1])/(2*pi))*(2.0-kappa-cos(atan2(x[1], x[0]-lc)))*cos(atan2(x[1], x[0]-lc)/2)"],
                 degree=2, mu=mu, kappa=kappa, KI=KI, KI_=KI_, psi=psi, lc=0.5 * L, t=0.0)

# bc - u (imposed displacement)
Gamma_u_0 = DirichletBC(V_u, u_U, boundaries)
bc_u = [Gamma_u_0]

# bc - alpha (zero damage)
Gamma_alpha_0 = DirichletBC(V_alpha_F.sub(1), 0.0, boundaries)
bc_alpha = [Gamma_alpha_0]

#--------------------------------------------------------------------
# Define the energy functional of damage problem
#--------------------------------------------------------------------

# Fenics forms for the energies
def sigma(u, alpha):
    return (a(alpha)+k_ell)*sigma_0(u)
    
body_force = Constant((0.,0.))

elastic_energy = 0.5*inner(sigma(u, alpha_), eps(u))*dx
external_work = dot(body_force, u)*dx
elastic_potential = elastic_energy - external_work

# Weak form of elasticity problem
E_u = derivative(elastic_potential,u,v)

# Writing tangent problems in term of test and trial functions for matrix assembly
E_du = replace(E_u,{u:du})

# Variational problem for the displacement
problem_u = LinearVariationalProblem(lhs(E_du), rhs(E_du), u, bc_u)

#--------------------------------------------------------------------
# Define the energy functional of damage problem
#--------------------------------------------------------------------
kappa  = sym(grad(theta_)) #Hessian matrix of damage field
kappa_ = as_vector([kappa[0,0], kappa[1,1], kappa[0,1]])

# matrix notation for fourth-order tensor beta
beta_M = as_matrix([[beta_11, beta_12, 2.0*beta_14],[beta_12, beta_11, -2.0*beta_14], [2.0*beta_14, -2.0*beta_14, 4.0*beta_44]])

dissipated_energy = Constant(5.0/96.0)*Gc*(w(alpha_)/ell + pow(ell,3)*dot(kappa_, beta_M*kappa_))*dx 

shear_energy = Constant(5.0/96.0)*Gc*Constant(1.0e5)*inner(R_gamma_, R_gamma_)*dx

# Here we show another way to apply the Duran-Liberman reduction operator,
# through constructing a Lagrangian term L_R.

# Return shear strain vector calculated from primal variables
def gamma(theta, w):
    return grad(w) - theta

DL_reduction = inner_e(gamma(theta_, alpha_) - R_gamma_, p_, True)

damage_functional = elastic_energy + dissipated_energy + shear_energy + DL_reduction

# Compute directional derivative about alpha in the direction of omega (Gradient)
F = derivative(damage_functional, alpha, omega)

# Compute directional derivative about alpha in the direction of dalpha (Hessian)
J = derivative(F, alpha, dalpha)

problem_alpha = NonlinearVariationalProblem(F, alpha, bc_alpha, J=J)

#--------------------------------------------------------------------
# Implement the box constraints for damage field
#--------------------------------------------------------------------

alpha_lb = Function(V_alpha_F)
alpha_ub = Function(V_alpha_F)

V_theta_lub = VectorFunctionSpace(mesh, "Lagrange", 2)
V_alpha_lub = FunctionSpace(mesh, "Lagrange", 1)
V_R_gamma_lub = FunctionSpace(mesh, "N1curl", 1)
V_p_lub = FunctionSpace(mesh, RestrictedElement(FiniteElement("N1curl", triangle, 1), "edge"))

#--------------------------------------------------------------------
# BCs for damage field
# Initialize damage field
alpha_0 = Function(V_alpha_lub);  
alpha_0 = interpolate(Constant(0.0), V_alpha_lub); # initial (known) alpha        
# initial (known) alpha
#alpha_0 = interpolate(Expression("near(x[0], 0.5*L, tol) & near(x[1], 0.0, tol) ? 1.0 : 0.0", degree=0, L= L,tol=0.5*hsize), V_alpha_lub) 

#--------------------------------------------------------------------

theta_n = Function(V_theta_lub);          # current solution gradient of damage field
alpha_n = Function(V_alpha_lub);          # current solution damage field
R_gamma_n = Function(V_R_gamma_lub);    # current solution "shear strain" field
p_n = Function(V_p_lub);                # current solution Lagrange multiplier field

theta_lb = Function(V_theta_lub)
theta_ub = Function(V_theta_lub)

V_alpha_lub_ = FunctionSpace(mesh, "Lagrange", 2)
ninfty = Function(V_alpha_lub_); 
ninfty.vector()[:] = -np.infty
pinfty = Function(V_alpha_lub_); 
pinfty.vector()[:] =  np.infty

ninfty_R_gamma = Function(V_R_gamma_lub); 
ninfty_R_gamma.vector()[:] = -np.infty
pinfty_R_gamma = Function(V_R_gamma_lub); 
pinfty_R_gamma.vector()[:] =  np.infty

ninfty_p_ = Function(V_p_lub); 
ninfty_p_.vector()[:] = -np.infty
pinfty_p_ = Function(V_p_lub); 
pinfty_p_.vector()[:] =  np.infty

theta_lub = FunctionAssigner(V_theta_lub, [V_alpha_lub_, V_alpha_lub_])
theta_lub.assign(theta_lb, [ninfty, ninfty])
theta_lub.assign(theta_ub, [pinfty, pinfty])

assigner_lub = FunctionAssigner(V_alpha_F, [V_theta_lub, V_alpha_lub, V_R_gamma_lub, V_p_lub])
assigner_lub.assign(alpha_lb, [theta_lb, alpha_0, ninfty_R_gamma, ninfty_p_]) # lower bound, set to 0 or intial alpha 
assigner_lub.assign(alpha_ub, [theta_ub, interpolate(Expression("1.0", degree=0), V_alpha_lub), pinfty_R_gamma, pinfty_p_]) # upper bound, set to 1
assigner_alpha = FunctionAssigner([V_theta_lub, V_alpha_lub, V_R_gamma_lub, V_p_lub], V_alpha_F)

problem_alpha.set_bounds(alpha_lb, alpha_ub) # set box constraints
#problem_alpha.set_bounds(alpha_0, alpha_ub)  # set box constraints

# Set up the solvers                                        
solver_u = LinearVariationalSolver(problem_u)
solver_u.parameters.update(solver_u_parameters)
#info(solver_u.parameters, True)

solver_alpha = NonlinearVariationalSolver(problem_alpha)
solver_alpha.parameters.update(solver_alpha_parameters)
#info(solver_alpha.parameters,True) # uncomment to see available parameters

#  loading and initialization of vectors to store time datas
load_multipliers = np.linspace(load_min,load_max,load_steps)
energies = np.zeros((len(load_multipliers),4))
iterations = np.zeros((len(load_multipliers),2))
forces = np.zeros((len(load_multipliers),2))

file_u = XDMFFile(mpi_comm_world(), savedir+"/u.xdmf")
file_u.parameters["flush_output"] = True
file_alpha = XDMFFile(mpi_comm_world(), savedir+"/alpha.xdmf")
file_alpha.parameters["flush_output"] = True

file_R_gamma = XDMFFile(mpi_comm_world(), savedir+"/R_gamma.xdmf")
file_R_gamma.parameters["flush_output"] = True

# Solving at each timestep

for (i_t, t) in enumerate(load_multipliers):
    u_U.t = t*ut
    if mpi_comm_world().rank == 0:
        print("\033[1;32m--- Starting of Time step %d: t = %f ---\033[1;m" % (i_t, t))
    # Alternate Mininimization 
    # Initialization
    iteration = 1; err_alpha = 1.0;
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
        if mpi_comm_world().rank == 0:
            print "AM Iteration: {0:5d},  alpha_error: {1:>14.8f}".format(iteration, err_alpha)

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

    if mpi_comm_world().rank == 0:
        print("\nEnd of timestep %d with load multiplier %f"%(i_t, t))
        print("\nElastic and Surface Energies: (%g,%g)" %(elastic_energy_value, surface_energy_value))
        print "-----------------------------------------"
        # Save some global quantities as a function of the time
        np.savetxt(savedir+'/energies.txt', energies)
        np.savetxt(savedir+'/iterations.txt', iterations)

# Plot energy and stresses
import matplotlib.pyplot as plt
if mpi_comm_world().rank == 0:
    p1, = plt.plot(energies[:,0], energies[:,1])
    p2, = plt.plot(energies[:,0], energies[:,2])
    p3, = plt.plot(energies[:,0], energies[:,3])
    plt.legend([p1, p2, p3], ["Elastic","Dissipated","Total"],loc = "best",frameon=False)
    plt.xlabel('Displacement')
    plt.ylabel('Energies')
    plt.savefig(savedir+'/energies.pdf',transparent=True)
    plt.close()
