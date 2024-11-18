from mpi4py import MPI
import numpy as np
from dolfinx import mesh, fem, plot, io
from dolfinx.fem.petsc import LinearProblem
from ufl import SpatialCoordinate, TrialFunction, TestFunction, inner, grad, dx

# Create mesh and define function space
msh = mesh.create_unit_square(
    comm=MPI.COMM_WORLD,
    nx=8,
    ny=8
)

V = fem.functionspace(
    mesh=msh,
    element=("P", 1)
)

# Define boundary condition
def on_boundary(x):
    return np.isclose(x[0], 0) | np.isclose(x[0], 1) | np.isclose(x[1], 0) | np.isclose(x[1], 1)

boundary_dofs = fem.locate_dofs_geometrical(V=V, marker=on_boundary)

def manufactured_solution(x):
    return 1 + x[0]**2 + 2 * x[1]**2

uD = fem.Function(V)
uD.interpolate(manufactured_solution)

bc = fem.dirichletbc(value=uD, dofs=boundary_dofs)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = fem.Constant(msh, -6.) + uD
a = inner(grad(u), grad(v)) * dx + inner (u,v) * dx
L = f * v * dx

# Compute solution
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# Plot solution and mesh
import pyvista

cells, types, x = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["u"] = uh.x.array.real
grid.set_active_scalars("u")
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
warped = grid.warp_by_scalar()
plotter.add_mesh(warped)
plotter.show()

# Save solution to file in VTX format
with io.VTXWriter(msh.comm, "results/poisson.bp", [uh]) as vtx:
    vtx.write(0.0)