import matplotlib.pyplot as plt
import numpy as np
import sobol_seq
import torch

from EquationModels.EquationBaseClass import EquationBaseClass
from SquareDomain import SquareDomain


class EquationClass(EquationBaseClass):

    def __init__(self, network_hyperparameters, additional_hyper_parameters):
        super().__init__(network_hyperparameters, additional_hyper_parameters)

        self.type_of_points = additional_hyper_parameters["point"]
        self.output_dimension = 1
        self.space_dimensions = 2
        self.time_dimensions = 0
        self.input_dimensions = self.space_dimensions + self.time_dimensions
        self.extrema_values = torch.tensor([[0, 1],
                                            [0, 1]])

        self.square_domain = SquareDomain(self.output_dimension,
                                          self.time_dimensions,
                                          self.space_dimensions,
                                          list(),
                                          self.extrema_values,
                                          self.type_of_points)
        self.inner_type_p = "grid"
        self.espilon = 0

        assert self.input_dimensions is not None, "The variable input_dimensions has not been initialized in the child class."

        self.init_model()
        self.init_params()

    def add_collocation_points(self, n_coll, random_seed):
        return self.square_domain.add_collocation_points(n_coll, random_seed)

    def add_boundary_points(self, n_boundary, random_seed):
        x_boundary = torch.rand((n_boundary, self.space_dimensions))
        y_boundary = torch.rand((n_boundary, self.output_dimension))
        return x_boundary, y_boundary

    def add_initial_points(self, n_initial, random_seed):
        x_time_0 = torch.rand((n_initial, self.input_dimensions))
        y_time_0 = torch.rand((n_initial, self.output_dimension))

        return x_time_0, y_time_0

    def add_internal_points(self, n_internal, random_seed):
        # Grid Points
        if self.inner_type_p == "grid":
            n_internal_x = int(np.sqrt(n_internal))
            x1 = np.linspace(0.125, 0.875, n_internal_x)
            inputs = torch.from_numpy(np.transpose([np.repeat(x1, len(x1)), np.tile(x1, len(x1))])).type(torch.FloatTensor)
        # Sobol Points
        elif self.inner_type_p == "sobol":
            x = torch.from_numpy(sobol_seq.i4_sobol_generate(2, n_internal)).type(torch.FloatTensor)
            extrema_inner = torch.tensor([[0.125, 0.875],
                                          [0.125, 0.875]])
            inputs = self.convert(x, extrema_inner)
        # Random Uniform Points
        elif self.inner_type_p == "uniform":
            x = torch.rand([n_internal, self.extrema_values.shape[0]])
            extrema_inner = torch.tensor([[0.125, 0.875],
                                          [0.125, 0.875]])
            inputs = self.convert(x, extrema_inner)
        else:
            print("Choice of inner_type_p not supported")
        exact = self.exact(inputs)
        outputs = exact * (1 + self.espilon * torch.randn(exact.shape))
        return inputs, outputs

    def apply_bc(self, x_b_train, u_b_train, u_pred_var_list, u_train_var_list):
        self.square_domain.apply_boundary_conditions(self.model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list)

    def apply_ic(self, x_u_train_init, u_train_init, u_pred_var_list, u_train_var_list):
        for j in range(self.output_dimension):
            u = self.model(x_u_train_init)[:, j]
            u_pred_var_list.append(u)
            u_train_var_list.append(u_train_init[:, j])

    def source(self, inputs):
        x_, y_ = inputs[:, 0], inputs[:, 1]
        f = 2 * (30 * y_ * (1 - y_) + 30 * x_ * (1 - x_))
        return f

    def compute_res(self, x_f_train):
        x_f_train.requires_grad = True
        output = self.model(x_f_train)
        u = output[:, 0]

        grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        grad_u_x = grad_u[:, 0]
        grad_u_y = grad_u[:, 1]
        grad_u_xx = torch.autograd.grad(grad_u_x, x_f_train, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0]
        grad_u_yy = torch.autograd.grad(grad_u_y, x_f_train, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 1]
        lap_u_real = grad_u_xx + grad_u_yy

        f = self.source(x_f_train)
        residual = lap_u_real + f

        return residual

    @staticmethod
    def exact(inputs):
        x = inputs[:, 0]
        y = inputs[:, 1]
        u = 30 * x * (1 - x) * y * (1 - y)

        return u.reshape(-1, 1)

    @staticmethod
    def grad_exact(inputs):
        x = inputs[:, 0]
        y = inputs[:, 1]
        der_x = 30 * (1 - 2 * x) * y * (1 - y)
        der_y = 30 * x * (1 - x) * (1 - 2 * y)

        return der_x, der_y

    def compute_generalization_error(self, folder_path):
        model = self.model
        model.eval()
        test_inp = self.convert(torch.rand([100000, self.extrema_values.shape[0]]), self.extrema_values)
        Exact = (self.exact(test_inp)).numpy()
        test_out = model(test_inp).detach().numpy()
        assert (Exact.shape[1] == test_out.shape[1])
        L2_test = np.sqrt(np.mean((Exact - test_out) ** 2))
        print("Error Test:", L2_test)

        rel_L2_test = L2_test / np.sqrt(np.mean(Exact ** 2))
        print("Relative Error Test:", rel_L2_test)

        return L2_test, rel_L2_test

    def plotting(self, images_path):
        model = self.model
        model.cpu()
        model.eval()
        x = torch.linspace(0., 1., 400).reshape(-1, 1)
        y = torch.linspace(0., 1., 400).reshape(-1, 1)
        xy = torch.tensor([[x_i, y_i] for x_i in x for y_i in y]).reshape(x.shape[0] * y.shape[0], 2)

        xy.requires_grad = True

        output = model(xy)

        u = output[:, 0]
        uex = self.exact(xy)[:, 0]

        output_grad = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        output_grad_x = output_grad[:, 0]
        output_grad_y = output_grad[:, 1]

        ex_grad = torch.autograd.grad(uex, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        exact_grad_x = ex_grad[:, 0]
        exact_grad_y = ex_grad[:, 1]

        l2_glob = torch.sqrt(torch.mean((u - uex) ** 2))
        l2_glob_rel = l2_glob / torch.sqrt(torch.mean(uex ** 2))

        h1_glob = torch.sqrt(torch.mean((u - uex) ** 2) + torch.mean((exact_grad_x - output_grad_x) ** 2 + (exact_grad_y - output_grad_y) ** 2))
        h1_glob_rel = h1_glob / torch.sqrt(torch.mean(uex ** 2) + torch.mean(exact_grad_x ** 2 + exact_grad_y ** 2))

        u = u.reshape(x.shape[0], y.shape[0])
        u = u.detach().numpy()

        uex = uex.reshape(x.shape[0], y.shape[0])
        uex = uex.detach().numpy()
        err = np.sqrt((u - uex) ** 2) / np.sqrt(np.mean(uex ** 2))

        output_grad_x = output_grad_x.reshape(x.shape[0], y.shape[0])
        output_grad_x = output_grad_x.detach().numpy()

        exact_grad_x = exact_grad_x.reshape(x.shape[0], y.shape[0])
        exact_grad_x = exact_grad_x.detach().numpy()

        vmin_sol = min(np.min(u), np.min(uex))
        vmax_sol = max(np.max(u), np.max(uex))

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax1 = ax[0]
        ax2 = ax[1]
        ax1.set_title(r'$u^\ast(x,y)$')
        ttl = ax1.title
        ttl.set_position([.5, 1.035])
        im1 = ax1.contourf(x.reshape(-1, ), y.reshape(-1, ), u.T, 20, cmap='Spectral', vmin=vmin_sol, vmax=vmax_sol)
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$y$')

        ax2.set_title(r'$u(x,y)$')
        ttl = ax2.title
        ttl.set_position([.5, 1.035])
        ax2.contourf(x.reshape(-1, ), y.reshape(-1, ), uex.T, 20, cmap='Spectral', vmin=vmin_sol, vmax=vmax_sol)
        ax2.set_xlabel(r'$x$')
        ax2.set_ylabel(r'$y$')
        fig.colorbar(im1, ax=ax, orientation='vertical')
        plt.savefig(images_path + "/Poiss_u.png", dpi=400)

        plt.figure()
        plt.title(r'$||u(x,y) - u^\ast(x,y)||$')
        ax = plt.gca()
        ttl = ax.title
        ttl.set_position([.5, 1.035])
        plt.contourf(x.reshape(-1, ), y.reshape(-1, ), err.T, 20, cmap='Spectral', vmin=vmin_sol, vmax=vmax_sol)
        plt.colorbar()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.savefig(images_path + "/Poiss_err.png", dpi=400)

        #############################################################
