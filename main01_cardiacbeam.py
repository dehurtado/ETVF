import meshio
from dolfin import *
import time
import ufl
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import numpy as np
from scipy.sparse import csr_matrix
import scipy


class Beam():
    """
    Following a benchmark study on cardiac mechanics models reported in the literature [Land 2015], we considered the
    case of a cardiac beam with domain [0; 10]x[0; 1]x[0; 1] mm3. The beam is fully clamped at x=0. A constant pressure
    of 0.004 kPa was applied on the bottom surface of the beam, and the beam assumed the transversely isotropic consti-
    tutive law presented by Guccione with fibers oriented along the axial direction (1; 0; 0). The domain was discreti-
    zed using regular hexahedral elements, with different degrees of interpolation for the unknown fields.

    material parameters for transversely isotropic : C=2kPa, bf =8, bt =2, bfs =4

    References:
        - Guccione, J. M., McCulloch, A. D., Waldman, L. K., 1991. Passive material properties of intact ventricular
        myocardium determined from a cylindrical model. Journal of biomechanical engineering 113 (February), 42–55.

        - Land, S., Gurev, V., Arens, S., Augustin, C. M., Baron, L., Blake, R., Bradley, C., Castro, S., Crozier, A.,
        Favino, M., Fastl, T. E., Fritz, T., Gao, H., Gizzi, A., Griffith, B. E., Hurtado, D. E., Krause, R., Luo, X.,
        Nash, M. P., Pezzuto, S., Plank, G., Rossi, S., Ruprecht, D., Seemann, G., Smith, N. P., Sundnes, J., Rice, J.
        J., Trayanova, N. A., Wang, D., Jenny Wang, Z., Niederer, S. A., 2015. Verification of cardiac mechanics softwa-
        re: benchmark problems and solutions for testing active and passive material behaviour. Proceedings. Mathemati-
        cal, physical, and engineering sciences / the Royal Society 471 (2184), 20150641.
    """

    def __init__(self):

        # mesh
        self.mesh = None
        self.nx = None
        self.ny = None
        self.nz = None

        # spaces
        self.V0 = None
        self.T = None
        self.V = None
        self.u = None
        self.Qm = None
        self.m = None
        self.Qi = None
        self.inc = None
        self.ETVF = None  # space for etvf auxiliary fields
        self.INC = None  # space for incompressibility fields
        self.W = None  # mixed space

        # elements
        self.u_element = None
        self.inc_element = None
        self.etvf_element = None

        # fields
        self.theta = None
        self.p = None
        self.phi = None
        self.xi = None
        self.w = None  # mixed field

        # quantities
        self.E = None

        # fibers
        self.fiber_dir = None

        # border conditions
        self._pressure = 0.004  # kPa
        self.left = None
        self.bottom = None

        # model
        self.formulation = None
        self.strain_energy_density = None
        self.psi_inc = None
        self.problem = None
        self.solver = None
        self.load_steps = None
        self.integrals_n = None
        self.bcs = None
        self.c = None
        self.bf = None
        self.bt = None
        self.bfs = None
        self.incompressibility_model = None

        # results
        self.pvd = None
        self.cost = {'pressure_lv': [],
                     'pressure_rv': [],
                     'iterations_to_convergence': [],
                     'time_to_convergence': [],
                     'mean_time_iteration': 0.0}
        self.save_pvd = None
        self.total_time = 0.0
        self.total_iterations = 0.0

    def generate_mesh(self):
        """
        this routine generate the undeformed mesh
        :param : self
        :return: None
        """

        # mesh is created from the paper
        self.mesh = BoxMesh.create([Point(0.0, 0.0, 0.0), Point(10.0, 1.0, 1.0)], [self.nx, self.ny, self.nz],
                                   CellType.Type.hexahedron)
        # self.mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(10.0, 1.0, 1.0), self.nx, self.ny, self.nz)

    def set_border_conditions(self, _pressure):
        """
        this routine incorporates the natural and essentials border conditions
        :param _pressure: pressure applied to the beam[kPa]
        :return:
        """

        # mark boundary subdomains
        self.left = LeftBorder(self.mesh)
        self.bottom = BottomBorder(self.mesh)

        boundary_conditions = {0: {'Dirichlet': Constant((0.0, 0.0, 0.0))},
                               1: {'Neumann': _pressure}}

        boundary_markers = MeshFunction('size_t', self.mesh, 2, self.mesh.domains())
        boundary_markers.set_all(2)
        self.left.mark(boundary_markers, 0)
        self.bottom.mark(boundary_markers, 1)

        # essential boundary conditions
        self.bcs = []
        for i in boundary_conditions:
            if 'Dirichlet' in boundary_conditions[i]:
                bc = DirichletBC(self.W.sub(0), boundary_conditions[i]['Dirichlet'], boundary_markers, i)
                self.bcs.append(bc)

        # natural boundary conditions
        f = grad(self.u) + Identity(3)
        v = split(TestFunction(self.W))[0]
        ds = Measure('ds', domain=self.mesh, subdomain_data=boundary_markers)
        self.integrals_n = []
        for i in boundary_conditions:
            if 'Neumann' in boundary_conditions[i]:
                N = FacetNormal(self.mesh)
                n = ufl.cofac(f) * N
                pressure = boundary_conditions[i]['Neumann']
                self.integrals_n.append(pressure * inner(v, n) * ds(i))

    def set_model_parameters(self, parameters):
        """
        this routine assign the settings to the object
        :param parameters:  settings for model (dictionary)
        :return: None
        """
        self.c = Constant(parameters['c'], name='c')
        self.bf = Constant(parameters['bf'], name='bf')
        self.bt = Constant(parameters['bt'], name='bt')
        self.bfs = Constant(parameters['bfs'], name='bfs')
        self.u_element = parameters['u_element']
        self.inc_element = parameters['inc_element']
        self.etvf_element = parameters['etvf_element']
        self.nx = parameters['nx']
        self.ny = parameters['ny']
        self.nz = parameters['nz']
        self.formulation = parameters['formulation']
        self.save_pvd = parameters['save_pvd']
        self.incompressibility_model = parameters['incompressibility_model']
        self.set_load_steps(parameters['number_of_steps'])

    def set_spaces(self):
        """
        This routine generate the spaces and initial condition for each field
        :return: None
        """
        self.V = VectorElement(self.u_element[0], self.mesh.ufl_cell(), self.u_element[1])
        self.INC = FiniteElement(self.inc_element[0], self.mesh.ufl_cell(), self.inc_element[1])
        self.ETVF = FiniteElement(self.etvf_element[0], self.mesh.ufl_cell(), self.etvf_element[1])

        if self.formulation == 'primal':
            if self.incompressibility_model == 'full-incompressible':
                self.W = FunctionSpace(self.mesh, MixedElement([self.V, self.INC]))
                self.w = Function(self.W)
                self.u, self.p = split(self.w)
            elif self.incompressibility_model == 'quasi-incompressible':
                self.W = FunctionSpace(self.mesh, MixedElement([self.V, self.INC, self.INC]))
                self.w = Function(self.W)
                self.u, self.theta, self.p = split(self.w)

        elif self.formulation == 'etvf':
            if self.incompressibility_model == 'full-incompressible':
                self.W = FunctionSpace(self.mesh, MixedElement([self.V, self.INC, self.ETVF, self.ETVF]))
                self.w = Function(self.W)
                u_0 = Expression(('0.0', '0.0', '0.0'), degree=0)
                p_0 = Expression('0.0', degree=0)
                u1_0 = interpolate(u_0, self.W.sub(0).collapse())
                p1_0 = interpolate(p_0, self.W.sub(1).collapse())
                phi_0 = Expression('1.0', degree=0)
                xi_0 = Expression('1.0', degree=0)
                phi1_0 = interpolate(phi_0, self.W.sub(2).collapse())
                xi1_0 = interpolate(xi_0, self.W.sub(3).collapse())
                assign(self.w, [u1_0, p1_0, phi1_0, xi1_0])
                self.u, self.p, self.phi, self.xi = split(self.w)

            elif self.incompressibility_model == 'quasi-incompressible':
                self.W = FunctionSpace(self.mesh, MixedElement([self.V, self.INC, self.INC, self.ETVF, self.ETVF]))
                self.w = Function(self.W)
                self.u, self.theta, self.p, self.phi, self.xi = split(self.w)
                u_0 = Expression(('0.0', '0.0', '0.0'), degree=0)
                theta_0 = Expression('0.0', degree=0)
                p_0 = Expression('0.0', degree=0)
                phi_0 = Expression('1.0', degree=0)
                xi_0 = Expression('1.0', degree=0)
                u1_0 = interpolate(u_0, self.W.sub(0).collapse())
                theta1_0 = interpolate(theta_0, self.W.sub(1).collapse())
                p1_0 = interpolate(p_0, self.W.sub(2).collapse())
                phi1_0 = interpolate(phi_0, self.W.sub(3).collapse())
                xi1_0 = interpolate(xi_0, self.W.sub(4).collapse())

                assign(self.w, [u1_0, theta1_0, p1_0, phi1_0, xi1_0])
                self.u, self.theta, self.p, self.phi, self.xi = split(self.w)

    def w_split(self):
        """
        this routine do the split according the chosen model
        """
        if self.incompressibility_model == 'full-incompressible':
            if self.formulation == 'primal':
                self.u, self.p = split(self.w)
            elif self.formulation == 'etvf':
                self.u, self.p, self.phi, self.xi = split(self.w)

        elif self.incompressibility_model == 'quasi-incompressible':
            if self.formulation == 'primal':
                self.u, self.theta, self.p = split(self.w)
            elif self.formulation == 'etvf':
                self.u, self.theta, self.p, self.phi, self.xi = split(self.w)

    def incompressibility(self):
        """
        this routine calculate the incompressibility energy
        """
        if self.incompressibility_model == 'full-incompressible':
            f = grad(self.u) + Identity(3)
            j = det(f)
            self.E = 0.5 * (f.T * f - Identity(3))
            u_p = - self.p * (j - 1)
            self.psi_inc = u_p

        elif self.incompressibility_model == 'quasi-incompressible':
            f = grad(self.u) + Identity(3)
            j = det(f)
            f_vol = pow(j, float(1) / 3) * Identity(3)
            f_bar = pow(j, -float(1) / 3) * f
            j = det(f_vol)
            f = f_bar
            self.E = 0.5 * (f.T * f - Identity(3))
            kappa = Constant(1000.0)
            u_j = float(1) / 2 * kappa * (self.theta - 1) ** 2
            u_p = self.p * (j - self.theta)
            self.psi_inc = (u_j + u_p)

    def get_strain_energy_density(self):
        """
        this routine calculates the strain energy density for Guccione type material
        """
        self.w_split()
        self.incompressibility()

        e1 = as_vector([1.0, 0.0, 0.0])
        e2 = as_vector([0.0, 1.0, 0.0])
        e3 = as_vector([0.0, 0.0, 1.0])

        E11, E12, E13 = inner(self.E * e1, e1), inner(self.E * e1, e2), inner(self.E * e1, e3)
        E21, E22, E23 = inner(self.E * e2, e1), inner(self.E * e2, e2), inner(self.E * e2, e3)
        E31, E32, E33 = inner(self.E * e3, e1), inner(self.E * e3, e2), inner(self.E * e3, e3)

        Q = self.bf * E11 ** 2 + self.bt * (E22 ** 2 + E33 ** 2 + E23 ** 2 + E32 ** 2) + self.bfs * (
                E12 ** 2 + E21 ** 2 + E13 ** 2 + E31 ** 2)

        # the strain energy density for the body is calculated according to the model
        if self.formulation == 'primal':
            psi_primal = self.c / 2.0 * (exp(Q) - 1)
            self.strain_energy_density = psi_primal + self.psi_inc
        else:
            psi_ = self.c / 2.0 * (self.phi ** 2 - 1)
            psi_lagrange = self.xi * (2 * ln(self.phi) - Q)
            self.strain_energy_density = psi_ - psi_lagrange + self.psi_inc

    def set_solver(self, solver_parameters):
        """
        this routine create a dolfin.cpp.fem.NonlinearVariationalSolver and set the parameters from user
        :param solver_parameters:  settings for dolfin.cpp.fem.NonlinearVariationalSolver (dictionary)
        :return: None
        """

        self.solver = NonlinearVariationalSolver(self.problem)
        prm = self.solver.parameters
        # newton solver settings
        prm['newton_solver']['convergence_criterion'] = 'incremental'  # poner en el diccionario
        prm['newton_solver']['absolute_tolerance'] = solver_parameters['newton_solver']['absolute_tolerance']
        prm['newton_solver']['relative_tolerance'] = solver_parameters['newton_solver']['relative_tolerance']
        prm['newton_solver']['maximum_iterations'] = solver_parameters['newton_solver']['maximum_iterations']
        prm['newton_solver']['report'] = solver_parameters['newton_solver']['report']
        prm['newton_solver']['error_on_nonconvergence'] = solver_parameters['newton_solver']['error_on_nonconvergence']
        prm['newton_solver']['linear_solver'] = solver_parameters['newton_solver']['linear_solver']
        if prm['newton_solver']['linear_solver'] == 'gmres':
            prm['newton_solver']['preconditioner'] = 'icc'
        prm['newton_solver']['krylov_solver']['absolute_tolerance'] = solver_parameters['krylov_solver'][
            'absolute_tolerance']
        prm['newton_solver']['krylov_solver']['maximum_iterations'] = 10000
        prm['newton_solver']['krylov_solver']['relative_tolerance'] = solver_parameters['krylov_solver'][
            'relative_tolerance']
        prm['newton_solver']['krylov_solver']['nonzero_initial_guess'] = solver_parameters['krylov_solver'][
            'nonzero_initial_guess']
        prm['newton_solver']['krylov_solver']['error_on_nonconvergence'] = solver_parameters['krylov_solver'][
            'error_on_nonconvergence']
        prm['newton_solver']['krylov_solver']['monitor_convergence'] = solver_parameters['krylov_solver'][
            'monitor_convergence']
        prm['newton_solver']['krylov_solver']['report'] = solver_parameters['krylov_solver']['report']
        prm['newton_solver']['krylov_solver']['divergence_limit'] = solver_parameters['krylov_solver'][
            'divergence_limit']

    def set_load_steps(self, ls):
        """
        this routine set the values for load steps in the simulation. In this case, the pattern is linear
        :param ls:  number of load steps for the simulation (integer)
        :return: None
        """
        self.load_steps = [0.0]
        p0 = 4 * 1E-3
        for i in range(ls):
            self.load_steps.append(self._pressure - (self._pressure - p0) / ls * (ls - i))
        self.load_steps.append(self._pressure)

    def solve(self, solver_parameters):
        """
        this routine solves the problem
        :param solver_parameters: parameters for FEniCS solver
        :return:
        """

        # the strain energy density is computed for the actual state and the problem is defined
        self.get_strain_energy_density()
        # displacement problem
        ru = derivative(self.strain_energy_density * dx, self.w, TestFunction(self.W)) + sum(self.integrals_n)
        tu = derivative(ru, self.w, TrialFunction(self.W))

        if solver_parameters['condition_number']:
            self.get_condition_number(tu)

        self.problem = NonlinearVariationalProblem(ru, self.w, self.bcs, J=tu)
        self.set_solver(solver_parameters=solver_parameters)

        # the problem is solved
        time_0 = time.time()
        info = self.solver.solve()
        time_f = time.time()

        self.total_time = self.total_time + time_f - time_0
        self.total_iterations = self.total_iterations + info[0]

        print('Computing time : {} [segs]'.format(time_f - time_0))
        print('iterations to convergence : {}'.format(info[0]))

    def simulation(self, model_parameters, solver_parameters):
        """
        this routine do the simulation and record results
        :param model_parameters: parameters for mechanical model (dictionary)
        :param solver_parameters: parameters for FEniCS solver (dictionary)
        :return:
        """

        self.set_model_parameters(parameters=model_parameters)
        self.generate_mesh()
        self.set_spaces()

        if self.save_pvd:
            self.pvd = File('paraview/result{}.pvd'.format(model_parameters['formulation']))
        for i in range(len(self.load_steps)):
            print('load step : {} kPa'.format(self.load_steps[i]))
            self.w_split()

            # the border conditions are set
            self.set_border_conditions(_pressure=self.load_steps[i])
            if model_parameters['formulation'] == 'primal':
                # the problem is solved
                self.solve(solver_parameters=solver_parameters)
            elif model_parameters['formulation'] == 'etvf':
                # the problem is solved
                self.solve(solver_parameters=solver_parameters)

            # save vtk
            if self.save_pvd:
                self.u = self.w.split()[0]
                self.u.rename('displacement', 'u')
                self.pvd << (self.u, float(i))

        print('Simulation finish')
        print('total iterations : {}'.format(self.total_iterations))
        print('simulation time : {} '.format(self.total_time))

    def spy_form_matrix(self, a, l=None, pattern_only=True):
        """
        this routine transform a dolfin matrix to sparse coordinate matrix for analysis
        :param a: (dolfin.fem.formmanipulations.derivative) matrix to manipulate
        :param l:
        :param pattern_only:
        :return: sp_mat : (scipy.sparse.csr_matrix) with the A matrix elements in a new sparse matrix
        """

        assert isinstance(a, ufl.form.Form)
        if l: assert isinstance(l, ufl.form.Form)
        eigen_matrix = EigenMatrix()
        if not l:
            assemble(a, tensor=eigen_matrix)
            if not self.bcs is None:
                for bc in self.bcs: bc.apply(eigen_matrix)
        else:
            SystemAssembler(a, l, self.bcs).assemble(eigen_matrix)
        a = eigen_matrix

        row, col, data = a.data()
        if pattern_only:
            data[:] = 1.0

        sp_mat = sp.csr_matrix((data, col, row), dtype='float')
        show = False
        marker_size = 1
        if show:
            import matplotlib.pyplot as plt
            plt.spy(sp_mat, markersize=marker_size, precision=0)
            plt.show()
        return sp_mat

    def get_condition_number(self, mat):
        """
        this routine calculates the condition number for an input matrix. This procedure starts solving the eigenvalue
        problem for the matrix and then doing \kappa (A)={\frac {\left|\lambda {\max }(A)\right|}{\left|\lambda
        _{\min }(A)\right|}}

        References:
        Scalable Library for Eigenvalue Problem Computations
        https://fenicsproject.org/docs/dolfin/1.3.0/python/demo/documented/eigenvalue/python/documentation.html
        https://fenicsproject.org/docs/dolfin/1.3.0/python/programmers-reference/cpp/la/SLEPcEigenSolver.html
        """

        # assemble matrices
        a = PETScMatrix()
        assemble(mat, tensor=a)
        [bc.apply(a) for bc in self.bcs]
        # create FEniCS eigensolver
        eigensolver = SLEPcEigenSolver(a)

        # compute all eigenvalues of A x = \lambda x
        print('Computing eigenvalues. This can take a minute.')

        # parameters for solver are set
        eigensolver.parameters['problem_type'] = 'gen_hermitian'
        eigensolver.parameters['verbose'] = True
        eigensolver.parameters['tolerance'] = 1E-3
        eigensolver.parameters['maximum_iterations'] = 1000
        eigensolver.parameters['solver'] = 'krylov-schur'
        eigensolver.parameters['verbose'] = True

        # largest eigenvalue
        eigensolver.parameters['spectrum'] = 'largest real'
        eigensolver.solve(1)
        r, c, rx, cx = eigensolver.get_eigenpair(0)

        # smallest eigenvalue
        eigensolver.parameters['spectrum'] = 'smallest real'
        eigensolver.solve(1)
        r_, c_, rx_, cx_ = eigensolver.get_eigenpair(0)

        # condition number
        lamb_a = abs(r) / abs(r_)
        print('condition number: {}'.format(lamb_a))


class LeftBorder(SubDomain, Beam):

    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)

    def get_nodes(self):
        nodes_inside = []
        for i, v in enumerate(vertices(self.mesh)):
            if self.inside(v.point(), True):
                x = v.point().x()
                y = v.point().y()
                z = v.point().z()
                nodes_inside.append(i)
        return nodes_inside


class BottomBorder(SubDomain, Beam):

    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh

    def inside(self, x, on_boundary):
        return on_boundary and near(x[2], 0.0)

    def get_nodes(self):
        nodes_inside = []
        for i, v in enumerate(vertices(self.mesh)):
            if self.inside(v.point(), True):
                x = v.point().x()
                y = v.point().y()
                z = v.point().z()
                nodes_inside.append(i)
        return nodes_inside


def l2_error(mesh, u1, u2):
    """
    static method to compare two solutions using l2-norm
    :param mesh: geometric domain
    :param u1: solution 1
    :param u2: solution 2
    :return: None
    """
    v = VectorFunctionSpace(mesh, 'CG', 1)
    project(u2, v)
    project(u1, v)
    delta = project(u2 - u1, v)
    delta = norm(delta, 'l2')
    print(delta)


"""
Execution
"""

q_degree = 4
dx = dx(metadata={'quadrature_degree': q_degree})
parameters['allow_extrapolation'] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = q_degree

ffc_options = {"optimize": True,
               "eliminate_zeros": True,
               "precompute_basis_const": True,
               "precompute_ip_const": True}
form_compiler_parameters = {"keep_diagonal": True}

newton_solver_parameters = {'absolute_tolerance': 1E-5,
                            'relative_tolerance': 1E-5,
                            'maximum_iterations': 50,
                            'report': True,
                            'error_on_nonconvergence': False,
                            'linear_solver': 'mumps'}
krylov_solver_parameters = {'absolute_tolerance': 9E-5,
                            'relative_tolerance': 9E-5,
                            'nonzero_initial_guess': True,
                            'error_on_nonconvergence': True,
                            'monitor_convergence': True,
                            'report': True,
                            'divergence_limit': 1E+12}

solver_parameters = {'newton_solver': newton_solver_parameters,
                     'krylov_solver': krylov_solver_parameters,
                     'condition_number': False}

model_parameters = {'c': 2.0,
                    'bf': 8.0,
                    'bt': 2.0,
                    'bfs': 4.0,
                    'u_element': ('CG', 1),
                    'inc_element': ('DG', 0),
                    'etvf_element': ('CG', 1),
                    'formulation': 'primal',
                    'nx': 12,
                    'ny': 3,
                    'nz': 3,
                    'save_pvd': True,
                    'incompressibility_model': 'quasi-incompressible',
                    'number_of_steps': 0}

print('Cardiac beam benchmark ')

beam = Beam()
beam.simulation(model_parameters=model_parameters, solver_parameters=solver_parameters)

# results
print('mesh information : {}'.format(beam.mesh))
print('formulation : {}'.format(beam.formulation))
print('incompressibility model : {}'.format(beam.incompressibility_model))
print('deformed location of the point (10.0, 0.5, 0.5) : {} [mm]'.format(0.5 + beam.w.split()[0](10.0, 0.5, 0.5)[2]))
print('maximal deflection [deformed location of the point (10.0, 0.5, 1.0)] : {} [mm]'.format(
    1.0 + beam.w.split()[0](10.0, 0.5, 1.0)[2]))
