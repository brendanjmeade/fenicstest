using Revise
using FEniCS
using PyPlot
using PyCall


# Calculate strain
function strain(u)
    0.5 * (nabla_grad(u) + Transpose(nabla_grad(u)))
end


# Calculate stress
function stress(u, lambda, mu)
     lambda * nabla_div(u) * Identity(2) + 2 * mu * strain(u)
end

function interior_eval(u, x, y)
    py"""
    def G(u, x, y):
        return u(x, y)
    """
    py"G"(u.pyobject, 1, 1)
end 


function elastic()
    mu = 3e10
    lambda = mu
    rho = 2700
    g = 9.81
    width = 10e3

    # Create mesh and define function space
    mesh = RectangleMesh(Point((-width, 0)), Point((width, 2 * width)), 10, 10)
    V = VectorFunctionSpace(mesh, "P", 1)
    bc = DirichletBC(V, Constant((0, 0)), "on_boundary && x[1]<1E-14") # What BCs are bing set???
    
    # Solve
    u = TrialFunction(V)
    d = geometric_dimension(u) # space dimension
    v = TestFunction(V)
    f = Constant((0, -rho * g)) # Vector of uniform body force
    T = Constant((0, 0))
    a = inner(stress(u, lambda, mu), strain(v)) * dx
    L = dot(f, v) * dx + dot(T, v) * ds
    u = FeFunction(V)
    lvsolve(a, L, u, bc)
    ueval = interior_eval(u, 1, 1)
    @show ueval
end
elastic()
