from fenics import *
from ufl import nabla_div

# Variables
mu = 3e10
rho = 3e10
width = 10e3
lambda_ = mu
g = 9.81
tol = 1E-14 # Tolerance for boundary condition

# Calculate strain
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

# Calculate stress
def sigma(u):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

# Boundary conditions
def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < tol

# Create mesh and define function space
mesh = RectangleMesh(Point((-width, 0)), Point((width, 2 * width)), 10, 10)
V = VectorFunctionSpace(mesh, 'P', 1)
bc = DirichletBC(V, Constant((0, 0)), clamped_boundary)

# Define variational problem
u = TrialFunction(V)
d = u.geometric_dimension()  # space dimension
v = TestFunction(V)
f = Constant((0, -rho*g))
T = Constant((0, 0))
a = inner(sigma(u), epsilon(v))*dx
L = dot(f, v)*dx + dot(T, v)*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Evaluate the solution at a an arbitrary new point
u_eval = u(1, 1)
print(u_eval)
