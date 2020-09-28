from dolfin import *
import time
import ufl
import scipy.sparse as sp
import numpy as np


class Artery():

    """
    Here we simulate the passive mechanical response of a cylindrical artery during inflation, following a study pre-
    viously reported in the literature [Holzapfel, 2000]. The loading protocol seeks to simulate the deformation occu-
    rring in healthy young arterial segment under pressure levels that vary from 0 to 21.33 kPa, along with an axial
    extension between 50% and 90%. We discretized the domain using tetrahedral elements, and considered simulations with
    and without residual stresses.

    References:
        Holzapfel, G. A., Gasser, T. C., Ogden, R. W., 2000. A new constitutive framework for arterial wall mechanics
        and a comparative study of material models. Journal of Elasticity 61 (1-3), 1–48.
    """

    def __init__(self):

        self.mesh = None
        self.name = None

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

        # model
        self.formulation = None
        self.strain_energy_density = None
        self.psi_inc = None
        self.problem = None
        self.solver = None
        self.load_steps = None
        self.integrals_n = None
        self.bcs = None
        self.incompressibility_model = None
        self.current = None  # current phase of the simulation
        self.u_element = None
        self.inc_element = None
        self.etvf_element = None

        # quantities
        self.k1 = None
        self.k2 = None
        self.c1 = None
        self.c2 = None

        # multi layers
        self.a_01 = None
        self.a_02 = None

        # parameters
        self.model_parameters = None
        self.solver_parameters = None
        self.name = None
        self.radius_min = None
        self.radius_max = None
        self.length = None
        self.alpha_media = None
        self.alpha_adventitia = None
        self.c1_media = None
        self.c1_adventitia = None
        self.k1_media = None
        self.k1_adventitia = None
        self.k2_media = None
        self.k2_adventitia = None
        self.theta_media = None
        self.theta_adventitia = None
        self.current = None
        self.steps_for_inflation = None
        self.steps_for_extension = None
        self.steps_for_residual_stress = None
        self.i_ = None
        self.save_pvd = None

        # model
        self.formulation = None
        self.strain_energy_density = None
        self.psi_inc = None
        self.problem = None
        self.solver = None
        self.load_steps = None
        self.integrals_n = None
        self.bcs = None
        self.incompressibility_model = None

        # subdomains
        self.bottom = None
        self.top = None
        self.intima = None
        self.boundary_markers = None

    def import_mesh(self):
        """
        this routine import the mesh and mark all subdomains
        :param :
        :return: None
        """
        self.mesh = Mesh(self.name + '.xml')
        # subdomains
        self.bottom = BottomBorder()
        self.top = TopBorder()
        self.intima = Intima()
        self.boundary_markers = MeshFunction('size_t', self.mesh, 2, self.mesh.domains())
        self.boundary_markers.set_all(3)
        self.bottom.mark(self.boundary_markers, 0)
        self.top.mark(self.boundary_markers, 1)
        self.intima.mark(self.boundary_markers, 2)

    def set_model_parameters(self, parameters):
        """
        this routine assign the settings to the object
        :param parameters:  settings for model (dictionary)
        :return: None
        """

        self.name = parameters['mesh_name']
        self.radius_min = parameters['radius_min']
        self.radius_max = parameters['radius_max']
        self.length = 10.0
        self.alpha_media = parameters['alpha_media']
        self.alpha_adventitia = parameters['alpha_adventitia']
        self.c1_media = parameters['c1_media']
        self.c1_adventitia = parameters['c1_adventitia']
        self.k1_media = parameters['k1_media']
        self.k1_adventitia = parameters['k1_adventitia']
        self.k2_media = parameters['k2_media']
        self.k2_adventitia = parameters['k2_adventitia']
        self.theta_media = parameters['theta_media']
        self.theta_adventitia = parameters['theta_adventitia']
        self.steps_for_inflation = parameters['steps_for_inflation']
        self.steps_for_extension = parameters['steps_for_extension']
        self.steps_for_residual_stress = parameters['steps_for_residual_stress']
        self.save_pvd = parameters['save_pvd']
        self.formulation = parameters['formulation']
        self.incompressibility_model = parameters['incompressibility_model']
        self.u_element = parameters['u_']
        self.inc_element = parameters['inc_']
        self.etvf_element = parameters['etvf_']
        self.model_parameters = parameters

    def set_spaces(self):
        """
        this routine generate the spaces and initial condition for each field
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

    def multilayers(self):
        """
        this routine assign a preferred orientation for collagen fibers and mechanical parameters in each layer
        return : None
        """

        v1 = FunctionSpace(self.mesh, 'CG', 1)
        f_ = VectorFunctionSpace(self.mesh, 'CG', 1)
        xyz = v1.tabulate_dof_coordinates()
        vector = (np.array([xyz[:, 0], xyz[:, 1]])).T

        dz_a01 = []
        dz_a02 = []
        r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
        c1 = []
        k1 = []
        k2 = []
        angle1 = []
        angle2 = []

        # model parameters are assigned to each point in the domain
        for i in range(len(r)):
            if r[i] <= self.radius_min + 2 / 3 * (self.radius_max - self.radius_min):
                dz_a01.append(np.sin(self.theta_media))
                dz_a02.append(np.sin(-self.theta_media))
                c1.append(self.c1_media)
                k1.append(self.k1_media)
                k2.append(self.k2_media)
                angle1.append(np.cos(self.theta_media))
                angle2.append(np.cos(-self.theta_media))

            else:
                dz_a01.append(np.sin(self.theta_adventitia))
                dz_a02.append(np.sin(-self.theta_adventitia))
                c1.append(self.c1_adventitia)
                k1.append(self.k1_adventitia)
                k2.append(self.k2_adventitia)
                angle1.append(np.cos(self.theta_adventitia))
                angle2.append(np.cos(-self.theta_adventitia))

        # fibers
        n_01 = []
        n_02 = []
        for i in range(len(vector)):
            # fibre a01
            vec1 = np.array([-vector[i, 1], vector[i, 0]])
            vec1 = vec1 / np.linalg.norm(vec1) * angle1[i]
            n_01.append([vec1[0], vec1[1], dz_a01[i]])
            # fibre a02
            vec2 = np.array([-vector[i, 1], vector[i, 0]])
            vec2 = vec2 / np.linalg.norm(vec2) * angle2[i]
            n_02.append([vec2[0], vec2[1], dz_a02[i]])

        n_01 = np.array(n_01)
        n_02 = np.array(n_02)

        self.a_01 = Function(f_)
        self.a_01.vector().set_local(n_01.flatten())

        self.a_02 = Function(f_)
        self.a_02.vector().set_local(n_02.flatten())

        # material parameters
        self.c1 = Function(v1)
        self.c1.vector().set_local(c1)
        self.k1 = Function(v1)
        self.k1.vector().set_local(k1)
        self.k2 = Function(v1)
        self.k2.vector().set_local(k2)

    def w_split(self):
        """
        this routine do the split according the chosen model
        :return:
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
        this routine calculate the volumetric energy
        :return:
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
            kappa = Constant(10.0)
            u_j = float(1) / 2 * kappa * (self.theta - 1) ** 2
            u_p = self.p * (j - self.theta)
            self.psi_inc = (u_j + u_p)

    def get_strain_energy_density(self):
        """
        multilayer fiber reinforced constitutive law by Holzapel
        :return:
        """
        self.w_split()
        self.incompressibility()
        f = grad(self.u) + Identity(3)
        if self.model_parameters['residual_stress']:
            f = f * self.residual_stress(i_=self.i_)
        c = f.T * f
        i1 = tr(c)
        i4 = inner(as_vector(self.a_01), c * as_vector(self.a_01))
        i6 = inner(as_vector(self.a_02), c * as_vector(self.a_02))

        # the strain energy density for the body is calculated according to the model
        if self.formulation == 'primal':
            internal_energy = self.c1 / 2.0 * (i1 - 3) + self.k1 / (2.0 * self.k2) * (
                    exp(self.k2 * (i6 - 1.0) ** 2) - 1.0) + self.k1 / (2.0 * self.k2) * (
                                      exp(self.k2 * (i4 - 1.0) ** 2) - 1.0)
            self.strain_energy_density = internal_energy + self.psi_inc
        else:
            internal_energy = self.c1 / 2.0 * (i1 - 3) + self.k1 / (2.0 * self.k2) * (
                    exp(self.k2 * (i6 - 1) ** 2) - 1) + self.k1 / (2.0 * self.k2) * (self.phi ** 2 - 1) - self.xi * (
                                      2 * ln(self.phi) - self.k2 * (i4 - 1) ** 2)
            self.strain_energy_density = internal_energy + self.psi_inc

    def inflation(self, pressure):
        """
        this function calculates the energy associated with boundary loads
        :param pressure: internal pressure [kPa]
        :return:
        """

        v = split(TestFunction(self.W))[0]
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundary_markers)
        N = FacetNormal(self.mesh)
        f = grad(self.u) + Identity(3)
        n = ufl.cofac(f) * N
        self.integrals_n = pressure * inner(v, n) * ds(2)

    def residual_stress(self, i_):
        """
        residual stress calculation
        :param i_:
        :return:
        """
        v1 = FunctionSpace(self.mesh, 'CG', 1)
        xyz = v1.tabulate_dof_coordinates()
        r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
        k_alpha = []

        # model variables are assigned to each point in the domain
        for i in range(len(r)):
            if r[i] <= self.radius_min + 2 / 3 * (self.radius_max - self.radius_min):
                k_alpha.append(2 * np.pi / (2 * np.pi - i_ * self.alpha_media))
            else:
                k_alpha.append(2 * np.pi / (2 * np.pi - i_ * self.alpha_adventitia))
        k_alpha_ = Function(v1)
        k_alpha_.vector().set_local(k_alpha)

        phi_ = Expression('atan(x[1]/x[0])', degree=1, domain=self.mesh)
        r = as_matrix(((cos(phi_), -sin(phi_), 0.0), (sin(phi_), cos(phi_), 0.0), (0.0, 0.0, 1.0)))

        f = as_matrix(((1.0, 0.0, 0.0), (0.0, k_alpha_, 0.0), (0.0, 0.0, 1.0)))
        f = r * f * r.T
        return f

    def set_solver(self):
        """
        this routine create a dolfin.cpp.fem.NonlinearVariationalSolver and set the parameters from user
        :param:
        :return: None
        """
        self.solver = NonlinearVariationalSolver(self.problem)
        prm = self.solver.parameters
        # newton solver settings
        prm['newton_solver']['convergence_criterion'] = 'incremental'
        prm['newton_solver']['absolute_tolerance'] = self.solver_parameters['newton_solver']['absolute_tolerance']
        prm['newton_solver']['relative_tolerance'] = self.solver_parameters['newton_solver']['relative_tolerance']
        prm['newton_solver']['maximum_iterations'] = self.solver_parameters['newton_solver']['maximum_iterations']
        prm['newton_solver']['report'] = self.solver_parameters['newton_solver']['report']
        prm['newton_solver']['error_on_nonconvergence'] = self.solver_parameters['newton_solver'][
            'error_on_nonconvergence']
        prm['newton_solver']['linear_solver'] = self.solver_parameters['newton_solver']['linear_solver']
        if prm['newton_solver']['linear_solver'] == 'gmres':
            prm['newton_solver']['preconditioner'] = 'icc'  # 'hypre_euclid' #icc''#hypre_amg
        prm['newton_solver']['krylov_solver']['absolute_tolerance'] = solver_parameters['krylov_solver'][
            'absolute_tolerance']
        prm['newton_solver']['krylov_solver']['maximum_iterations'] = 700000
        prm['newton_solver']['krylov_solver']['relative_tolerance'] = self.solver_parameters['krylov_solver'][
            'relative_tolerance']
        prm['newton_solver']['krylov_solver']['nonzero_initial_guess'] = self.solver_parameters['krylov_solver'][
            'nonzero_initial_guess']
        prm['newton_solver']['krylov_solver']['error_on_nonconvergence'] = self.solver_parameters['krylov_solver'][
            'error_on_nonconvergence']
        prm['newton_solver']['krylov_solver']['monitor_convergence'] = self.solver_parameters['krylov_solver'][
            'monitor_convergence']
        prm['newton_solver']['krylov_solver']['report'] = self.solver_parameters['krylov_solver']['report']
        prm['newton_solver']['krylov_solver']['divergence_limit'] = self.solver_parameters['krylov_solver'][
            'divergence_limit']

    def solve(self, solver_parameters):
        """
        this routine solves the problem
        :param solver_parameters: parameters for FEniCS solver
        :return:
        """
        self.solver_parameters = solver_parameters

        # the strain energy density is computed for the actual state and the problem is defined
        self.get_strain_energy_density()
        # displacement problem
        if self.current == 'inflation':
            ru = derivative(self.strain_energy_density * dx, self.w, TestFunction(self.W)) + self.integrals_n
        else:
            ru = derivative(self.strain_energy_density * dx, self.w, TestFunction(self.W))
        tu = derivative(ru, self.w, TrialFunction(self.W))

        if solver_parameters['condition_number']:
            self.get_condition_number(tu)

        self.problem = NonlinearVariationalProblem(ru, self.w, self.bcs, J=tu)
        self.set_solver()

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
        simulation of inflation of an artery
        :param model_parameters: parameters for modeling the artery (dictionary)
        :param solver_parameters: parameters for set the FEniCS solver (dictionary)
        :return:
        """
        self.set_model_parameters(parameters=model_parameters)
        self.import_mesh()
        self.set_spaces()
        self.multilayers()
        # residual stress calculation
        if self.save_pvd:
            self.pvd = File('paraview/residual_stress_result{}.pvd'.format(model_parameters['formulation']))
        if self.model_parameters['residual_stress']:
            print('residual stress calculation')
            self.current = 'residual_stress'
            # boundary conditions to calculation of residual stresses
            bctop = DirichletBC(self.W.sub(0), Constant((0.0, 0.0, 0.0)), self.top)
            bcbottom = DirichletBC(self.W.sub(0), Constant((0.0, 0.0, 0.0)), self.bottom)
            self.bcs = [bctop, bcbottom]
            self.integrals_n = None
            steps = np.linspace(0.0, 1.0, self.model_parameters['steps_for_residual_stress'])
            for i in steps:
                self.i_ = i
                self.solve(solver_parameters=solver_parameters)
                # save vtk
                if self.save_pvd:
                    self.w_split()
                    if self.formulation == 'primal' and self.incompressibility_model == 'mean-dilatation':
                        u = self.u
                        u.rename('displacement', 'u')
                    else:
                        u = self.w.split()[0]
                        u.rename('displacement', 'u')
                    self.pvd << (u, float(i))

        # extension
        if self.save_pvd:
            self.pvd = File('paraview/extension_result{}.pvd'.format(model_parameters['formulation']))
        if self.model_parameters['extension']:
            print('extension')
            self.current = 'extension'
            extension = np.linspace(1.0, self.model_parameters['lambda_z'],
                                    self.model_parameters['steps_for_extension'])
            self.integrals_n = None
            for i in extension:
                self.i_ = i
                # boundary conditions for extension phase
                bctop = DirichletBC(self.W.sub(0), Constant((0.0, 0.0, (i - 1.0) * self.length)), self.top)
                bcbottom = DirichletBC(self.W.sub(0), Constant((0.0, 0.0, 0.0)), self.bottom)
                self.bcs = [bctop, bcbottom]
                self.solve(solver_parameters=solver_parameters)
                self.inner_radius()
                # save vtk
                if self.save_pvd:
                    self.w_split()
                    if self.formulation == 'primal' and self.incompressibility_model == 'mean-dilatation':
                        u = self.u
                        u.rename('displacement', 'u')
                    else:
                        u = self.w.split()[0]
                        u.rename('displacement', 'u')
                    self.pvd << (u, float(i))

        # inflation
        if self.save_pvd:
            self.pvd = File('paraview/inflation_result{}.pvd'.format(model_parameters['formulation']))
        if self.model_parameters['inflation']:
            print('inflation')
            self.current = 'inflation'
            internal_pressure = np.linspace(0.0, self.model_parameters['final_pressure'],
                                            self.model_parameters['steps_for_inflation'])
            for i in internal_pressure:
                print('current pressure : {} kPa'.format(i))
                self.i_ = i
                # boundary conditions for inflation phase
                if self.model_parameters['extension']:
                    bctop = DirichletBC(self.W.sub(0),
                                        Constant(
                                            (0.0, 0.0, (self.model_parameters['lambda_z'] - 1.0) * self.length)),
                                        self.top)
                    bcbottom = DirichletBC(self.W.sub(0), Constant((0.0, 0.0, 0.0)), self.bottom)
                    self.bcs = [bctop, bcbottom]

                self.inflation(pressure=i)
                self.inner_radius()
                # save vtk
                if self.save_pvd:
                    self.w_split()
                    if self.formulation == 'primal' and self.incompressibility_model == 'mean-dilatation':
                        u = self.u
                        u.rename('displacement', 'u')
                    else:
                        u = self.w.split()[0]
                        u.rename('displacement', 'u')
                    self.pvd << (u, float(i))

        print('Simulation finished')
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

    def inner_radius(self):
        """
        this routine calculates the actual internal diameter after a load step
        :return:
        """
        u_ = project(self.u, VectorFunctionSpace(self.mesh, 'CG', 1))
        ri = (self.radius_min + (u_(self.radius_min, 0.0, 5.0)[0]))
        print('inner radius : {} [mm]'.format(ri))


class TopBorder(SubDomain):

    def inside(self, x, on_boundary):
        return on_boundary and near(x[2], 10.0)


class BottomBorder(SubDomain):

    def inside(self, x, on_boundary):
        return on_boundary and near(x[2], 0.0)


class Intima(SubDomain):

    def inside(self, x, on_boundary):
        return True if on_boundary and sqrt(x[0] ** 2 + x[1] ** 2) <= 0.73 else False


"""
Execution
"""
q_degree = 4
dx = dx(metadata={'quadrature_degree': q_degree})
parameters['allow_extrapolation'] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = q_degree

model_parameters = {'radius_min': 0.71,
                    'radius_max': 1.10,
                    'theta_media': 29.0 / 90.0 * np.pi / 2.0,
                    'theta_adventitia': 62.0 / 90.0 * np.pi / 2.0,
                    'c1_media': 3.0,
                    'c1_adventitia': 0.3,
                    'k1_media': 2.3632,
                    'k1_adventitia': 0.5620,
                    'k2_media': 0.8393,
                    'k2_adventitia': 0.7112,
                    'alpha_media': 160.0 / 90.0 * np.pi / 2.0,
                    'alpha_adventitia': 160.0 / 90.0 * np.pi / 2.0,
                    'save_pvd': False,
                    'mesh_name': 'cylinder_fine_no_residual',
                    'formulation': 'primal',
                    'incompressibility_model': 'full-incompressible',
                    'u_': ('CG', 2),
                    'inc_': ('DG', 0),
                    'etvf_': ('DG', 0),
                    'inflation': True,
                    'extension': True,
                    'residual_stress': False,
                    'steps_for_inflation': 50,
                    'steps_for_residual_stress': 2,
                    'steps_for_extension': 10,
                    'final_pressure': 21.33,
                    'lambda_z': 1.9,
                    'condition_number': False}

parameters['allow_extrapolation'] = True
parameters["form_compiler"]["cpp_optimize"] = True

ffc_options = {"optimize": True,
               "eliminate_zeros": True,
               "precompute_basis_const": True,
               "precompute_ip_const": True}

form_compiler_parameters = {"keep_diagonal": True}

newton_solver_parameters = {'absolute_tolerance': 1E-8,
                            'relative_tolerance': 1E-8,
                            'maximum_iterations': 50,
                            'report': True,
                            'error_on_nonconvergence': False,
                            'linear_solver': 'mumps'}
krylov_solver_parameters = {'absolute_tolerance': 1E-5,
                            'relative_tolerance': 1E-5,
                            'nonzero_initial_guess': True,
                            'error_on_nonconvergence': True,
                            'monitor_convergence': True,
                            'report': True,
                            'divergence_limit': 1E+15}

solver_parameters = {'newton_solver': newton_solver_parameters,
                     'krylov_solver': krylov_solver_parameters,
                     'condition_number': False}

parameters_ = {'model_parameters': model_parameters,
               'solver_parameters': solver_parameters}

print('Benchmark - Problem')
artery = Artery()

artery.simulation(model_parameters=model_parameters, solver_parameters=solver_parameters)

# Results
print('mesh information : {}'.format(artery.mesh))
print('number of cells : {}'.format(artery.mesh.num_cells()))
print('formulation : {}'.format(artery.formulation))
print('incompressibility model : {}'.format(artery.incompressibility_model))
print('number of degrees of freedom for displacement: {}'.format(artery.w.vector()[:].shape[0]))
