from dolfin import *
import numpy as np
import ufl
import meshio
import time
import scipy.sparse as sp


class Heart():

    def __init__(self):


        # model parameters
        self.model_parameters = None
        self.mesh = None
        self.mesh_name = None
        self.fiber_dir = None
        self.sheet_dir = None

        # save results
        self.save_pvd = None
        self.pvd = None
        self.pvd_stress = None

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

        # parameters
        self.a1 = None
        self.b1 = None
        self.a4f = None
        self.b4f = None
        self.a4s = None
        self.b4s = None
        self.a8fs = None
        self.b8fs = None
        self.lv_pressure = None

        # fields
        self.theta = None
        self.p = None
        self.phi = None
        self.xi = None
        self.w = None  # mixed field

        # subdomains initialization
        self.phi_lv = None
        self.phi_rv = None
        self.right_ventricle = None
        self.left_ventricle = None
        self.top = None

        # load steps setting
        self.load_steps = None

        # model
        self.formulation = None
        self.strain_energy_density = None
        self.psi_inc = None
        self.problem = None
        self.solver = None
        self.load_steps = None
        self.number_of_steps = None
        self.integrals_n = None
        self.bcs = None
        self.c = None
        self.bf = None
        self.bt = None
        self.bfs = None
        self.incompressibility_model = None
        self.right_ventricle_cavity = None
        self.left_ventricle_cavity = None

        # results
        self.lv_residual_volume = 0
        self.rv_residual_volume = 0
        self.total_time = 0
        self.total_iterations = 0

    def import_mesh(self):
        """
        importation of mesh
        :param :
        :return: None
        """
        self.mesh = Mesh(self.mesh_name + '.xml')

    def set_model_parameters(self, parameters):

        # model parameters
        self.mesh_name = parameters['mesh_name']
        self.import_mesh()
        self.a1 = Constant(parameters['a1'], name='a1')
        self.b1 = Constant(parameters['b1'], name='b1')
        self.a4f = Constant(parameters['a4f'], name='a4f')
        self.b4f = Constant(parameters['b4f'], name='b4f')
        self.a4s = Constant(parameters['a4s'], name='a4s')
        self.b4s = Constant(parameters['b4s'], name='b4s')
        self.a8fs = Constant(parameters['a8fs'], name='a8fs')
        self.b8fs = Constant(parameters['b8fs'], name='b8fs')
        self.u_element = parameters['u_element']
        self.inc_element = parameters['inc_element']
        self.etvf_element = parameters['etvf_element']
        self.number_of_steps = parameters['number_of_steps']
        self.formulation = parameters['formulation']
        self.incompressibility_model = parameters['incompressibility_model']
        self.lv_pressure = Constant(parameters['lv_pressure'], name='lv_pressure')
        self.load_steps = np.linspace((0.005 * self.lv_pressure, 0.005 * self.lv_pressure / 5),
                                      (self.lv_pressure, self.lv_pressure / 5),
                                      self.number_of_steps)
        self.save_pvd = parameters['save_pvd']
        self.vtk_data()

    def vtk_data(self):
        """
        ventricles generation
        :return:
        """
        # we mark each ventricle from .vtk file
        heart_mesh_vtk = meshio.read(self.mesh_name + '.vtk')

        # we read the fiber and myocyte sheet direction at each point
        phi_id = FunctionSpace(self.mesh, 'CG', 1)
        self.phi_lv = Function(phi_id)
        self.phi_rv = Function(phi_id)
        self.phi_lv.vector()[:] = heart_mesh_vtk.point_data['Phi_lv'].flatten()[dof_to_vertex_map(phi_id)]
        self.phi_rv.vector()[:] = heart_mesh_vtk.point_data['Phi_rv'].flatten()[dof_to_vertex_map(phi_id)]
        # fiber and sheet directions
        fib = VectorFunctionSpace(self.mesh, 'CG', 1)
        self.fiber_dir = Function(fib)
        self.sheet_dir = Function(fib)
        self.fiber_dir.vector()[:] = heart_mesh_vtk.point_data['fiber_F'].flatten()[dof_to_vertex_map(fib)]
        self.sheet_dir.vector()[:] = heart_mesh_vtk.point_data['fiber_S'].flatten()[dof_to_vertex_map(fib)]
        self.right_ventricle = Ventricle(self.phi_rv)
        self.left_ventricle = Ventricle(self.phi_lv)
        self.top = TopBorder()

    def set_border_conditions(self, _pressure):
        """
        incorporation of natural and essentials border conditions
        :param _pressure:
        :return:
        """

        # boundary condition are identified for this simulation
        boundary_conditions = {0: {'Dirichlet': Constant((0.0, 0.0, 0.0))},
                               1: {'Neumann_rv': _pressure[1]},
                               2: {'Neumann_lv': _pressure[0]}}

        boundary_markers = MeshFunction('size_t', self.mesh, 2, self.mesh.domains())
        boundary_markers.set_all(3)
        self.top.mark(boundary_markers, 0)
        self.right_ventricle.mark(boundary_markers, 1)
        self.left_ventricle.mark(boundary_markers, 2)

        # save subdomains in .pvd format
        if self.save_pvd:
            file = File('paraview/subdomains.pvd')
            file << boundary_markers

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

        # we calculate the surface integral applying the internal pressure
        for i in boundary_conditions:
            N = FacetNormal(self.mesh)
            n = ufl.cofac(f) * N
            if 'Neumann_rv' in boundary_conditions[i]:
                pressure = boundary_conditions[i]['Neumann_rv']
                self.integrals_n.append(pressure * inner(v, n) * ds(i))
            elif 'Neumann_lv' in boundary_conditions[i]:
                pressure = boundary_conditions[i]['Neumann_lv']
                self.integrals_n.append(pressure * inner(v, n) * ds(i))

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
                u_0 = Expression(('0.0', '0.0', '0.0'), degree=0)
                p_0 = Expression('0.0', degree=0)
                u1_0 = interpolate(u_0, self.W.sub(0).collapse())
                p1_0 = interpolate(p_0, self.W.sub(1).collapse())
                assign(self.w, [u1_0, p1_0])
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
            u_p = - self.p * (j - 1)
            self.psi_inc = u_p

        elif self.incompressibility_model == 'quasi-incompressible':
            f = grad(self.u) + Identity(3)
            j = det(f)
            f_vol = pow(j, float(1) / 3) * Identity(3)
            j = det(f_vol)
            kappa = Constant(1000.0)
            u_j = float(1) / 2 * kappa * (self.theta - 1) ** 2
            u_p = self.p * (j - self.theta)
            self.psi_inc = (u_j + u_p)

    def get_strain_energy(self):
        """
        fiber reinforced law for passive myocardium by Holzapfel and Ogden
        :param :
        :return: None
        """
        self.w_split()
        self.incompressibility()
        # kinematic quantities
        f = grad(self.u) + Identity(3)
        c = f.T * f

        # calculation of invariants for the model
        i1 = tr(c)
        i4f = inner(as_vector(self.fiber_dir), c * as_vector(self.fiber_dir))
        i4s = inner(as_vector(self.sheet_dir), c * as_vector(self.sheet_dir))
        i8fs = inner(as_vector(self.fiber_dir), c * as_vector(self.sheet_dir))

        # we consider that fibers and miocite sheet take only tractions, so the invariants are strictly higher than one
        i4f_ = project(i4f, FunctionSpace(self.mesh, 'CG', 1))
        i4s_ = project(i4s, FunctionSpace(self.mesh, 'CG', 1))
        i8fs_ = project(i8fs, FunctionSpace(self.mesh, 'CG', 1))

        for i in range(len(self.mesh.coordinates())):
            i4f_.vector()[i] = max(i4f_.vector()[i], 1.0)
            i4s_.vector()[i] = max(i4s_.vector()[i], 1.0)
            i8fs_.vector()[i] = max(i8fs_.vector()[i], 1.0)

        i4f = i4f_
        i4s = i4s_
        i8fs = i8fs_

        if self.formulation == 'primal':
            psi_primal = self.a1 / (2 * self.b1) * exp(self.b1 * (i1 - 3)) + self.a4f / (2 * self.b4f) * (
                    exp(self.b4f * (i4f - 1) ** 2) - 1) + self.a4s / (2 * self.b4s) * (
                                 exp(self.b4s * (i4s - 1) ** 2) - 1) + self.a8fs / (2 * self.b8fs) * (
                                 exp(self.b8fs * i8fs ** 2) - 1)
            self.strain_energy_density = psi_primal + self.psi_inc
        else:
            psi_ = self.a1 / (2 * self.b1) * exp(self.b1 * (i1 - 3)) + self.a4f / (2 * self.b4f) * (
                    self.phi ** 2 - 1) + self.a4s / (2 * self.b4s) * (
                           exp(self.b4s * (i4s - 1) ** 2) - 1) + self.a8fs / (2 * self.b8fs) * (
                           exp(self.b8fs * i8fs ** 2) - 1) - self.xi * (
                           self.phi**2 - exp(self.b4f * (i4f - 1) ** 2))
            self.strain_energy_density = psi_ + self.psi_inc

    def set_solver(self, solver_parameters):
        """
        this routine create a dolfin.cpp.fem.NonlinearVariationalSolver and set the parameters from user
        :param solver_parameters:  settings for dolfin.cpp.fem.NonlinearVariationalSolver (dictionary)
        :return: None
        """

        self.solver = NonlinearVariationalSolver(self.problem)
        prm = self.solver.parameters
        # newton solver settings
        prm['newton_solver']['convergence_criterion'] = 'incremental'
        prm['newton_solver']['absolute_tolerance'] = solver_parameters['newton_solver']['absolute_tolerance']
        prm['newton_solver']['relative_tolerance'] = solver_parameters['newton_solver']['relative_tolerance']
        prm['newton_solver']['maximum_iterations'] = solver_parameters['newton_solver']['maximum_iterations']
        prm['newton_solver']['report'] = solver_parameters['newton_solver']['report']
        prm['newton_solver']['error_on_nonconvergence'] = solver_parameters['newton_solver']['error_on_nonconvergence']
        prm['newton_solver']['linear_solver'] = solver_parameters['newton_solver']['linear_solver']
        #if prm['newton_solver']['linear_solver'] == 'gmres':
        #    prm['newton_solver']['preconditioner'] = 'icc'
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

    def solve(self, solver_parameters):
        """
        this routine solves the problem
        :param solver_parameters: parameters for FEniCS solver
        :return:
        """

        # the strain energy density is computed for the actual state and the problem is defined
        self.get_strain_energy()
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
        simulation of the passive inflation of the heart (at the end diastolic internal pressure)
        :return:
        """
        self.set_model_parameters(parameters=model_parameters)
        self.set_spaces()
        self.aggregate_subdomains()

        if self.save_pvd:
            self.pvd = File('paraview/inflation.pvd')

        print('load step : {} kPa'.format((0.0, 0.0)))
        self.w_split()
        # initial border conditions
        self.set_border_conditions(_pressure=(0.0, 0.0))
        self.solve(solver_parameters=solver_parameters)
        # save vtk
        if self.save_pvd:
            u = self.w.split()[0]
            u.rename('displacement', 'u')
            self.pvd << (u, 0)
        for i in range(len(self.load_steps)):
            print('load step : {} kPa'.format(self.load_steps[i]))
            self.w_split()
            # border conditions for this load step
            self.set_border_conditions(_pressure=self.load_steps[i])
            self.solve(solver_parameters=solver_parameters)

            # uncomment to calculation of stresses
            # self.pvd_stress = File('paraview/stress.pvd')
            # self.stress_cardiac_holzapfel()

            # save vtk
            if self.save_pvd:
                u = self.w.split()[0]
                u.rename('displacement', 'u')
                self.pvd << (u, float(i) + 1)
        print('Simulation finished')

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

    # postprocess

    def stress_cardiac_holzapfel(self):
        """

        :return:
        """

        self.u = split(self.w)[0]
        # kinematic quantities
        I = Identity(3)
        f = grad(self.u) + I
        c = f.T * f
        C = variable(c)
        S = 2.0 * diff(self.strain_energy_density, C)
        pk2_tensor = project(S, TensorFunctionSpace(self.mesh, 'DG', 0))
        pk2_tensor.rename('stress', 'stress')
        self.pvd_stress << pk2_tensor

    def aggregate_ventricles_cavities(self, rv_name, lv_name):
        """
        This routine aggregate the file of each ventricle and generate a mesh from it
        :param rv_name: (str) name of .vtk file which contains the right ventricle info
        :param lv_name: (str) name of .vtk file which contains the left ventricle info
        :return: None
        """

        # data is read from .vtu file for right ventricle
        rv_mesh = meshio.read(rv_name + '.vtk')
        rv_mesh = meshio.Mesh(rv_mesh.points, rv_mesh.cells)
        meshio.write(rv_name + '.xml', rv_mesh)
        self.right_ventricle_cavity = Mesh(rv_name + '.xml')
        # data is read from .vtu file for left ventricle
        lv_mesh = meshio.read(lv_name + '.vtk')
        lv_mesh = meshio.Mesh(lv_mesh.points, lv_mesh.cells)
        meshio.write(lv_name + '.xml', lv_mesh)
        self.left_ventricle_cavity = Mesh(lv_name + '.xml')

    def aggregate_subdomains(self):
        """
        this routine aggregate the subdomains identified in the mesh
        :return: None
        """

        self.right_ventricle = Ventricle(self.phi_rv)
        self.left_ventricle = Ventricle(self.phi_lv)
        self.top = TopBorder()

    def warped_mesh(self, vtu):
        """
        this routine maps the deformation of the heart, contained in .vtu file , to each ventricle cavity
        :param vtu: (str) name of the .vtu file
        :return: None
        """

        self.import_mesh()
        self.aggregate_subdomains()

        # data is loaded from .vtu file
        heart_mesh_vtu = meshio.read(vtu + '.vtu')
        heart_mesh = meshio.Mesh(heart_mesh_vtu.points, heart_mesh_vtu.cells)
        meshio.write(vtu + '.xml', heart_mesh)
        vtu_mesh = Mesh(vtu + '.xml')

        # displacement are assigned to whole heart
        V = VectorFunctionSpace(vtu_mesh, 'CG', 1)
        u = Function(V)
        u.vector()[:] = heart_mesh_vtu.point_data['displacement'].flatten()[dof_to_vertex_map(V)]

        # the parameters for elasticity problem in the cavities are set
        E = 1e9
        nu = 0.3
        mu = E / (2.0 * (1.0 + nu))
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # the displacement in .vtu file is mapped to left ventricle cavity mesh
        lv_sp = VectorFunctionSpace(self.left_ventricle_cavity, 'CG', 1)
        u_lv = TrialFunction(lv_sp)
        v_lv = TestFunction(lv_sp)
        f = Constant((0.0, 0.0, 0.0))
        a = inner(2.0 * mu * sym(grad(u_lv)) + lmbda * tr(sym(grad(u_lv))) * Identity(len(u_lv)), grad(v_lv)) * dx
        L = inner(f, v_lv) * dx
        border_lv = self.left_ventricle
        displacement = interpolate(u, lv_sp)
        bc_lv = DirichletBC(lv_sp, displacement, border_lv)
        top = CavityTop(self.left_ventricle_cavity)
        bc_top = DirichletBC(lv_sp, Constant((0.0, 0.0, 0.0)), top)
        A, b = assemble_system(a, L, [bc_lv, bc_top])
        sol_lv = Function(lv_sp)
        pc = PETScPreconditioner("petsc_amg")
        solver = PETScKrylovSolver("cg", pc)
        solver.set_operator(A)
        solver.solve(sol_lv.vector(), b)
        for i, v in enumerate(self.left_ventricle_cavity.coordinates()):
            self.left_ventricle_cavity.coordinates()[i] = self.left_ventricle_cavity.coordinates()[i] + sol_lv(v)

        # the displacement in .vtu file is mapped to left ventricle cavity mesh
        rv_sp = VectorFunctionSpace(self.right_ventricle_cavity, 'CG', 1)  #
        u_rv = TrialFunction(rv_sp)
        v_rv = TestFunction(rv_sp)
        f = Constant((0.0, 0.0, 0.0))
        a = inner(2.0 * mu * sym(grad(u_rv)) + lmbda * tr(sym(grad(u_rv))) * Identity(len(u_rv)), grad(v_rv)) * dx
        L = inner(f, v_rv) * dx
        border_rv = self.right_ventricle
        displacement = interpolate(u, rv_sp)
        bc_rv = DirichletBC(rv_sp, displacement, border_rv)
        top = CavityTop(self.right_ventricle_cavity)
        bc_top = DirichletBC(rv_sp, Constant((0.0, 0.0, 0.0)), top)
        A, b = assemble_system(a, L, [bc_rv, bc_top])
        sol_rv = Function(rv_sp)
        pc = PETScPreconditioner("petsc_amg")
        solver = PETScKrylovSolver("cg", pc)
        solver.set_operator(A);
        solver.solve(sol_rv.vector(), b);
        # the mesh coordinates are warped from the solution of the system
        for i, v in enumerate(self.right_ventricle_cavity.coordinates()):
            self.right_ventricle_cavity.coordinates()[i] = self.right_ventricle_cavity.coordinates()[i] + sol_rv(v)

    def get_residual_volume(self):
        """
        residual volume for ventricles is calculated
        :return: None
        """

        if self.left_ventricle_cavity is not None and self.right_ventricle_cavity is not None:
            self.lv_residual_volume = 0
            self.rv_residual_volume = 0
            # the volume of each tetrahedron is computed and adding for each ventricle
            for e in self.left_ventricle_cavity.cells():
                x1, y1, z1 = self.left_ventricle_cavity.coordinates()[e][0]
                x2, y2, z2 = self.left_ventricle_cavity.coordinates()[e][1]
                x3, y3, z3 = self.left_ventricle_cavity.coordinates()[e][2]
                x4, y4, z4 = self.left_ventricle_cavity.coordinates()[e][3]
                v14 = np.array([x1 - x4, y1 - y4, z1 - z4])
                v24 = np.array([x2 - x4, y2 - y4, z2 - z4])
                v34 = np.array([x3 - x4, y3 - y4, z3 - z4])
                ve = 1 / 6 * abs(np.dot(v14, np.cross(v24, v34)))
                self.lv_residual_volume = self.lv_residual_volume + ve
            for e in self.right_ventricle_cavity.cells():
                x1, y1, z1 = self.right_ventricle_cavity.coordinates()[e][0]
                x2, y2, z2 = self.right_ventricle_cavity.coordinates()[e][1]
                x3, y3, z3 = self.right_ventricle_cavity.coordinates()[e][2]
                x4, y4, z4 = self.right_ventricle_cavity.coordinates()[e][3]
                v14 = np.array([x1 - x4, y1 - y4, z1 - z4])
                v24 = np.array([x2 - x4, y2 - y4, z2 - z4])
                v34 = np.array([x3 - x4, y3 - y4, z3 - z4])
                ve = 1 / 6 * abs(np.dot(v14, np.cross(v24, v34)))
                self.rv_residual_volume = self.rv_residual_volume + ve

            print('The residual volume for left ventricle is  : {} [mm3]'.format(self.lv_residual_volume.round(0)))
            print('The residual volume for right ventricle is : {} [mm3]'.format(self.rv_residual_volume.round(0)))
            print('The ventricular residual volume is : {} [mm3]'.format(
                (self.lv_residual_volume + self.rv_residual_volume).round(0)))
        else:
            print('.vtu file must be added for left and right ventricle')

    def get_volume(self, vtu):
        """
        this routine is for calculate the volume of the ventricular cavities during the simulation
        :param vtu: (str) name of .vtu file with the displacement field stored
        :return: None
        """

        self.warped_mesh(vtu)
        self.left_ventricle_volume = 0
        for e in self.left_ventricle_cavity.cells():
            x1, y1, z1 = self.left_ventricle_cavity.coordinates()[e][0]
            x2, y2, z2 = self.left_ventricle_cavity.coordinates()[e][1]
            x3, y3, z3 = self.left_ventricle_cavity.coordinates()[e][2]
            x4, y4, z4 = self.left_ventricle_cavity.coordinates()[e][3]
            v14 = np.array([x1 - x4, y1 - y4, z1 - z4])
            v24 = np.array([x2 - x4, y2 - y4, z2 - z4])
            v34 = np.array([x3 - x4, y3 - y4, z3 - z4])
            ve = 1 / 6 * abs(np.dot(v14, np.cross(v24, v34)))
            self.left_ventricle_volume = self.left_ventricle_volume + ve

        self.right_ventricle_volume = 0
        for e in self.right_ventricle_cavity.cells():
            x1, y1, z1 = self.right_ventricle_cavity.coordinates()[e][0]
            x2, y2, z2 = self.right_ventricle_cavity.coordinates()[e][1]
            x3, y3, z3 = self.right_ventricle_cavity.coordinates()[e][2]
            x4, y4, z4 = self.right_ventricle_cavity.coordinates()[e][3]
            v14 = np.array([x1 - x4, y1 - y4, z1 - z4])
            v24 = np.array([x2 - x4, y2 - y4, z2 - z4])
            v34 = np.array([x3 - x4, y3 - y4, z3 - z4])
            ve = 1 / 6 * abs(np.dot(v14, np.cross(v24, v34)))
            self.right_ventricle_volume = self.right_ventricle_volume + ve

        print('left ventricular volume  : {} [mm3]'.format(self.left_ventricle_volume.round(0)))
        print('right ventricular volume : {} [mm3]'.format(self.right_ventricle_volume.round(0)))
        print(
            'ventricular volume : {} [mm3]'.format((self.left_ventricle_volume + self.right_ventricle_volume).round(0)))
        return self.left_ventricle_volume

    def pv_curve(self, ls, first_ls, formulation, ventricles, csv):
        """
        this routine calculate the pv curve from a list of vtu
        """
        edp = [0.0]
        p0_lv = first_ls
        for i in range(ls):
            edp.append((self.lv_pressure - p0_lv) * (i / ls) ** 2 + p0_lv)

        edp.append(self.lv_pressure)
        edv = []

        for i in range(len(edp)):
            self.aggregate_ventricles_cavities(ventricles['right_ventricle_cavity'],
                                               ventricles['left_ventricle_cavity'])
            if i < 10:
                edv.append(self.get_volume(vtu='result{}00000{}'.format(formulation, i)))
            else:
                edv.append(self.get_volume(vtu='result{}0000{}'.format(formulation, i)))

        csv_file = (edp, edv)
        if csv:
            np.savetxt('results{}'.format(formulation) + '.csv', csv_file, delimiter=",")
        return edv, edp


class Ventricle(SubDomain, Heart):

    def __init__(self, id_):
        super().__init__()
        self.ventricle_id = id_

    def inside(self, x, on_boundary):
        return True if on_boundary and self.ventricle_id(x[0], x[1], x[2]) >= 0.9 else False


class TopBorder(SubDomain, Heart):

    def __init__(self):
        super().__init__()

    def inside(self, x, on_boundary):
        return on_boundary and x[2] <= 3.5


class CavityTop(SubDomain, Heart):

    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh

    def inside(self, x, on_boundary):
        return True if on_boundary and x[2] <= 1.1 * min(self.mesh.coordinates()[:, 2]) else False


"""
Simulation of the passive inflation of the heart

We will simulate the passive mechanical behavior of the cardiac tissue during passive inflation. The units of measureme-
nt are [mm], [kPa]
"""
# quadrature points and settings to numerical integration
q_degree = 2
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

# in this example we will use a newton scheme to solve the nonlinear problem
newton_solver_parameters = {'absolute_tolerance': 1E-3,
                            'relative_tolerance': 1E-3,
                            'maximum_iterations': 50,
                            'report': True,
                            'error_on_nonconvergence': False,
                            'linear_solver': 'mumps'}
# the linearized problem is solved using a krylov solver
krylov_solver_parameters = {'absolute_tolerance': 1E-3,
                            'relative_tolerance': 1E-3,
                            'nonzero_initial_guess': True,
                            'error_on_nonconvergence': True,
                            'monitor_convergence': True,
                            'report': True,
                            'divergence_limit': 1E+12}

solver_parameters = {'newton_solver': newton_solver_parameters,
                     'krylov_solver': krylov_solver_parameters,
                     'condition_number': False}

# model parameters settings
model_parameters = {'a1': 0.24,
                    'b1': 5.08,
                    'a4f': 1.46,
                    'b4f': 4.15,
                    'a4s': 0.87,
                    'b4s': 1.6,
                    'a8fs': 0.3,
                    'b8fs': 1.3,
                    'u_element': ('CG', 2),
                    'inc_element': ('DG', 0),
                    'etvf_element': ('CG', 1),
                    'formulation': 'primal',
                    'lv_pressure': 1.59987,
                    'incompressibility_model': 'full-incompressible',
                    'mesh_name': 'HeartMeshFibers_37k',
                    'save_pvd': True,
                    'number_of_steps': 30}

parameters_ = {'model_parameters': model_parameters,
               'solver_parameters': solver_parameters}

print('Passive inflation of the heart')
heart = Heart()
heart.simulation(model_parameters=model_parameters, solver_parameters=solver_parameters)
