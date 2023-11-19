import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import legendre

from BoundaryConditions import DirichletBC
from EquationModels.EquationBaseClass import EquationBaseClass
from SquareDomain import SquareDomain


class EquationClass(EquationBaseClass):

    def __init__(self, network_hyperparameters, additional_hyper_parameters):
        super().__init__(network_hyperparameters, additional_hyper_parameters)

        self.type_of_points = additional_hyper_parameters["point"]
        self.output_dimension = 1
        self.space_dimensions = 1
        self.time_dimensions = 0
        self.parameters_values = torch.tensor([[-1, 1]])
        self.parameter_dimensions = 1
        self.extrema_values = torch.tensor([[0.0, 1.0]])

        self.extrema_values = torch.cat([self.extrema_values, self.parameters_values], 0)
        self.input_dimensions = self.extrema_values.shape[0]
        self.list_of_BC = list([[self.ub0x, self.ub1x]])

        self.square_domain = SquareDomain(self.output_dimension,
                                          self.time_dimensions,
                                          self.space_dimensions,
                                          self.list_of_BC,
                                          self.extrema_values,
                                          self.type_of_points)
        self.ub_0 = 1.
        self.ub_1 = 0.

        assert self.input_dimensions is not None, "The variable input_dimensions has not been initialized in the child class."

        self.init_model()
        self.init_params()

    def add_collocation_points(self, n_coll, random_seed):
        x_coll, y_coll = self.square_domain.add_collocation_points(n_coll, random_seed)

        return x_coll, y_coll

    def add_boundary_points(self, n_boundary, random_seed):
        return self.square_domain.add_boundary_points(n_boundary, random_seed)

    def add_initial_points(self, n_initial, random_seed):
        x_time_0 = torch.rand((n_initial, self.space_dimensions))
        y_time_0 = torch.rand((n_initial, self.output_dimension))

        return x_time_0, y_time_0

    def apply_bc(self, x_b_train, u_b_train, u_pred_var_list, u_train_var_list):
        x = x_b_train[:, 0]
        mu = x_b_train[:, 1]

        x0 = x[x == self.extrema_values[0, 0]]
        mu0 = mu[x == self.extrema_values[0, 0]]
        x1 = x[x == self.extrema_values[0, 1]]
        mu1 = mu[x == self.extrema_values[0, 1]]

        n0_len = x0.shape[0]
        n1_len = x1.shape[0]

        n0 = torch.full(size=(n0_len,), fill_value=-1.0)
        n1 = torch.full(size=(n1_len,), fill_value=1.0)
        n = torch.cat([n0, n1], 0).to(self.device)
        mu_ = torch.cat([mu0, mu1], 0).to(self.device)

        tmp_x_b_train = torch.clone(x_b_train)
        tmp_x_b_train[:x0.shape[0]] = x_b_train[x == self.extrema_values[0, 0]]
        tmp_x_b_train[x0.shape[0]:] = x_b_train[x == self.extrema_values[0, 1]]
        x_b_train = torch.clone(tmp_x_b_train)

        tmp_u_b_train = torch.clone(u_b_train)
        tmp_u_b_train[:x0.shape[0]] = u_b_train[x == self.extrema_values[0, 0]]
        tmp_u_b_train[x0.shape[0]:] = u_b_train[x == self.extrema_values[0, 1]]
        u_b_train = torch.clone(tmp_u_b_train)

        scalar = n * mu_ < 0

        x_boundary_masked = x_b_train[scalar, :]
        u_boundary_masked = u_b_train[scalar, :]
        u_pred = self.model(x_boundary_masked)

        u_pred_var_list.append(u_pred)
        u_train_var_list.append(u_boundary_masked)

    def sigma(self, inputs):
        x = inputs[:, 0]

        return x / 2

    def kernel(self, mu, mu_prime):
        d = [1.0, 1.98398, 1.50823, 0.70075, 0.23489, 0.05133, 0.00760, 0.00048]
        k = torch.full(size=(mu.shape[0], mu_prime.shape[0]), fill_value=0.0).to(self.device)

        for p in range(len(d)):
            pn_mu = torch.from_numpy(legendre(p)(mu.detach().cpu().numpy()).reshape(-1, 1)).type(torch.FloatTensor).to(self.device)
            pn_mu_prime = torch.from_numpy(legendre(p)(mu_prime.detach().cpu().numpy()).reshape(-1, 1).T).type(torch.FloatTensor).to(self.device)
            kn = torch.matmul(pn_mu, pn_mu_prime)
            k = k + d[p] * kn

        return k

    def compute_scattering(self, x, mu):
        n_quad = 10
        mu_prime, w = np.polynomial.legendre.leggauss(n_quad)
        w = torch.from_numpy(w).type(torch.FloatTensor).to(self.device)
        mu_prime = torch.from_numpy(mu_prime).type(torch.FloatTensor).to(self.device)

        inputs = torch.cartesian_prod(x, mu_prime)
        u = self.model(inputs)
        u = u.reshape(x.shape[0], mu_prime.shape[0])

        kern = self.kernel(mu, mu_prime)

        scatter_values = torch.matmul(kern * u, w)
        return scatter_values

    def compute_res(self, x_f_train):
        x_f_train.requires_grad = True
        u = self.model(x_f_train).reshape(-1, )
        scatter = self.sigma(x_f_train)

        s = self.compute_scattering(x_f_train[:, 0], x_f_train[:, 1])
        mu = x_f_train[:, 1]

        grad_u_x = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0]

        res = mu * grad_u_x + u - scatter * s

        return res

    def ub0x(self, t):
        type_BC = [DirichletBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=self.ub_0)
        return u, type_BC

    def ub1x(self, t):
        type_BC = [DirichletBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=self.ub_1)
        return u, type_BC

    def u0(self, inputs):
        u0 = torch.zeros_like(inputs[:, 0])
        return u0.reshape(-1, 1)

    def compute_generalization_error(self, folder_path):
        model = self.model
        model.cpu()
        model = model.eval()
        n = 100
        m = 100

        x = torch.linspace(0, 1, n)
        mu, _ = torch.from_numpy(np.array(np.polynomial.legendre.leggauss(m))).type(torch.FloatTensor)

        inputs = torch.cartesian_prod(x, mu)
        sol = model(inputs)
        sol = sol.reshape(x.shape[0], mu.shape[0])

        exact_sol = np.loadtxt("Data/RadSolution.txt")

        L2_test = np.mean(abs(sol.detach().numpy() - exact_sol))  # / np.mean(abs(exact_sol))
        rel_L2_test = L2_test / np.mean(abs(exact_sol))

        print("Error Test:", L2_test)
        print("Relative Error Test:", rel_L2_test)

        return L2_test, rel_L2_test

    def plotting(self, images_path):
        model = self.model
        model.cpu()
        model = model.eval()
        n = 100
        m = 100

        x = torch.linspace(0, 1, n)
        mu, _ = torch.from_numpy(np.array(np.polynomial.legendre.leggauss(m))).type(torch.FloatTensor)

        inputs = torch.cartesian_prod(x, mu)
        sol = model(inputs)
        sol = sol.reshape(x.shape[0], mu.shape[0])

        exact_sol = np.loadtxt("Data/RadSolution.txt")

        err_tot = np.mean(abs(sol.detach().numpy() - exact_sol) / np.mean(abs(exact_sol)))

        levels = [0.00, 0.006, 0.013, 0.021, 0.029, 0.04, 0.047, 0.06, 0.071, 0.099, 0.143, 0.214, 0.286, 0.357, 0.429, 0.5, 0.571, 0.643, 0.714, 0.786, 0.857, 0.929, 1]
        norml = matplotlib.colors.BoundaryNorm(levels, 256)
        fig, axes = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 0.1]})
        im1 = axes[0].contourf(x.reshape(-1, ), mu.reshape(-1, ), sol.detach().numpy().T, cmap='jet', levels=levels, norm=norml, vmin=0, vmax=1)
        axes[0].tick_params(axis='x', labelsize=8)
        axes[0].tick_params(axis='y', labelsize=8)
        # plt.axes().set_aspect('equal')
        axes[0].set_xlabel(r'$x$')
        axes[0].set_ylabel(r'$\mu$')
        axes[0].set_title(r'$u^\ast$', fontsize=10)
        im2 = axes[1].contourf(x.reshape(-1, ), mu.reshape(-1, ), exact_sol.T, cmap='jet', levels=levels, norm=norml, vmin=0, vmax=1)
        axes[1].tick_params(axis='x', labelsize=8)
        axes[1].tick_params(axis='y', labelsize=8)
        # plt.axes().set_aspect('equal')
        axes[1].set_xlabel(r'$x$')
        axes[1].set_title(r'$u$', fontsize=10)

        # Create the colorbar subplot
        cbar = plt.colorbar(im2, cax=axes[2])
        axes[2].tick_params(axis='x', labelsize=8)
        axes[2].tick_params(axis='y', labelsize=8)

        # Adjust the spacing between the subplots and the colorbar
        fig.subplots_adjust(wspace=0.3)

        plt.savefig(images_path + "/net_sol.png", dpi=400)

        x = np.linspace(0, 0, 1)
        mu = np.linspace(-1, 1, n)
        theta_ex = [0 + np.pi, np.pi / 12 + np.pi, np.pi / 6 + np.pi, np.pi / 4 + np.pi, np.pi / 3 + np.pi, 5 * np.pi / 12 + np.pi, 0, np.pi / 12, np.pi / 6, np.pi / 4, np.pi / 3, 5 * np.pi / 12, np.pi / 2]
        mu_ex = np.cos(theta_ex)
        sol_ex = np.array([0.0079, 0.0089, 0.0123, 0.0189, 0.0297, 0.0385, 1, 1, 1, 1, 1, 1, 1])
        inputs = torch.from_numpy(np.transpose([np.repeat(x, len(mu)), np.tile(mu, len(x))])).type(torch.FloatTensor)
        sol = model(inputs).detach().numpy()

        inputs_err = torch.from_numpy(np.concatenate([np.linspace(0, 0, len(mu_ex)).reshape(-1, 1), mu_ex.reshape(-1, 1)], 1)).type(torch.FloatTensor)
        sol_err = model(inputs_err).detach().numpy()

        err_1 = np.sqrt(np.mean((sol_ex.reshape(-1, ) - sol_err.reshape(-1, )) ** 2) / np.mean(sol_ex.reshape(-1, ) ** 2))

        plt.figure()
        plt.grid(True, which="both", ls=":")
        plt.plot(mu, sol, color="grey", lw=2, label=r'Learned Solution')
        plt.scatter(mu_ex, sol_ex, label=r'Exact Solution')
        plt.xlabel(r'$\mu$')
        plt.ylabel(r'$u^-(x=0)$')
        plt.legend()
        plt.savefig(images_path + "/u0.png", dpi=400)

        x = np.linspace(1, 1, 1)
        mu = np.linspace(-1, 1, n)
        theta_ex = [0 + np.pi, np.pi / 12 + np.pi, np.pi / 6 + np.pi, np.pi / 4 + np.pi, np.pi / 3 + np.pi, 5 * np.pi / 12 + np.pi, np.pi / 2 + np.pi, 0, np.pi / 12, np.pi / 6, np.pi / 4, np.pi / 3, 5 * np.pi / 12]
        mu_ex = np.cos(theta_ex)
        sol_ex = np.array([0, 0, 0, 0, 0, 0, 0, 0.5363, 0.5234, 0.4830, 0.4104, 0.3020, 0.1848])
        inputs = torch.from_numpy(np.transpose([np.repeat(x, len(mu)), np.tile(mu, len(x))])).type(torch.FloatTensor)
        sol = model(inputs).detach().numpy()

        inputs_err = torch.from_numpy(np.concatenate([np.linspace(1, 1, len(mu_ex)).reshape(-1, 1), mu_ex.reshape(-1, 1)], 1)).type(torch.FloatTensor)
        sol_err = model(inputs_err).detach().numpy()

        err_2 = np.sqrt(np.mean((sol_ex.reshape(-1, ) - sol_err.reshape(-1, )) ** 2) / np.mean(sol_ex.reshape(-1, ) ** 2))

        plt.figure()
        plt.grid(True, which="both", ls=":")
        plt.plot(mu, sol, color="grey", lw=2, label=r'Learned Solution')
        plt.scatter(mu_ex, sol_ex, label=r'Exact Solution')
        plt.xlabel(r'$\mu$')
        plt.ylabel(r'$u^+(x=1)$')
        plt.legend()
        plt.savefig(images_path + "/u1.png", dpi=400)
