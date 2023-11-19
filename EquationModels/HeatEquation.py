import matplotlib.pyplot as plt
import numpy as np
import torch

from BoundaryConditions import DirichletBC
from EquationModels.EquationBaseClass import EquationBaseClass


class EquationClass(EquationBaseClass):

    def __init__(self, network_hyperparameters, additional_hyper_parameters):
        super().__init__(network_hyperparameters, additional_hyper_parameters)
        """
        Initializes the EquationClass with specific network and additional hyperparameters. 
        Sets the output, space, and time dimensions, as well as the extrema values for the problem domain. 
        It also initializes the model and parameters.

        Args:
          network_hyperparameters: A dictionary containing hyperparameters for the network.
          additional_hyper_parameters: A dictionary containing additional hyperparameters, including types of points for the problem.
        """

        self.type_of_points = additional_hyper_parameters["point"]
        self.output_dimension = 1
        self.space_dimensions = 1
        self.time_dimensions = 1
        self.extrema_values = torch.tensor([[0, 1],
                                            [-1, 1]])

        self.input_dimensions = self.extrema_values.shape[0]

        self.T = self.extrema_values[0, 1]
        self.x0 = self.extrema_values[1, 0]
        self.xL = self.extrema_values[1, 1]
        self.k = 0.2

        assert self.input_dimensions is not None, "The variable input_dimensions has not been initialized in the child class."

        self.init_model()
        self.init_params()

    def add_collocation_points(self, n_coll, random_seed):
        """
        Generates collocation points for the domain.

        Args:
           n_coll: Number of collocation points to generate.
           random_seed: Seed for random number generator to ensure reproducibility.

        Returns:
           A tuple of tensors representing the input and output for the collocation points.
       """

        n = int(n_coll ** 0.5) + 1
        x = torch.linspace(self.x0, self.xL, n)
        dx = (x[1] - x[0]) / 2
        t = torch.linspace(0, self.T, n)
        dt = (t[1] - t[0]) / 2
        x = x[:-1] + dx
        t = t[:-1] + dt

        inputs = torch.cartesian_prod(t, x)
        outputs = torch.zeros((inputs.shape[0], 1))

        return inputs, outputs

    def add_boundary_points(self, n_boundary, random_seed):
        """
        Generates boundary points for the domain.

        Args:
          n_boundary: Number of boundary points to generate.
          random_seed: Seed for random number generator to ensure reproducibility.

        Returns:
          A tuple of tensors representing the input and output for the boundary points.
        """
        t = torch.linspace(0, self.T, n_boundary + 1).reshape(-1, 1)
        dt = (t[1] - t[0]) / 2
        t = t[:-1] + dt

        x0 = torch.full_like(t, self.x0)
        xL = torch.full_like(t, self.xL)

        inputs_0 = torch.cat((t, x0), -1)
        inputs_L = torch.cat((t, xL), -1)

        outputs_0, _ = self.ub0(t)
        outputs_L, _ = self.ub1(t)
        inputs = torch.cat((inputs_0, inputs_L))

        outputs = torch.cat((outputs_0, outputs_L))

        return inputs, outputs

    def add_initial_points(self, n_initial, random_seed):
        """
        Generates initial points for the problem.

        Args:
          n_initial: Number of initial points to generate.
          random_seed: Seed for random number generator to ensure reproducibility.

        Returns:
          A tuple of tensors representing the input and output for the initial points.
        """
        x_vec = torch.linspace(self.x0, self.xL, n_initial + 1).reshape(-1, 1)

        dx = (x_vec[1] - x_vec[0]) / 2
        x_vec = x_vec[:-1] + dx
        inputs = torch.cat((torch.full(size=(n_initial, 1), fill_value=0.0), x_vec), -1)
        outputs = self.u0(x_vec)

        return inputs, outputs

    def apply_bc(self, x_b_train, u_b_train, u_pred_var_list, u_train_var_list):
        """
        Applies boundary conditions during training.

        Args:
          x_b_train: Tensor containing the boundary training data.
          u_b_train: Tensor containing the boundary values.
          u_pred_var_list: List to store predicted boundary variables.
          u_train_var_list: List to store training boundary variables.
        """
        u_pred_b = self.model(x_b_train)
        u_pred_var_list.append(u_pred_b[:, 0])
        u_train_var_list.append(u_b_train[:, 0])

    def apply_ic(self, x_u_train, u_train, u_pred_var_list, u_train_var_list):
        """
        Applies initial conditions during training.

        Args:
          x_u_train: Tensor containing the initial condition training data.
          u_train: Tensor containing the initial condition values.
          u_pred_var_list: List to store predicted initial condition variables.
          u_train_var_list: List to store training initial condition variables.
        """
        for j in range(self.output_dimension):
            if x_u_train.shape[0] != 0:
                u_pred_var_list.append(self.model(x_u_train)[:, j])
                u_train_var_list.append(u_train[:, j])

    def compute_res(self, x_f_train):
        """
        Computes the residual in the PINN model for given training data.

        Args:
          x_f_train: Tensor containing training data for residual computation.

        Returns:
          Tensor containing computed residuals.
        """
        x_f_train.requires_grad = True
        u = self.model(x_f_train).reshape(-1, )
        inputs = torch.ones((x_f_train.shape[0],))

        grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=inputs.to(self.device), create_graph=True)[0]

        grad_u_t = grad_u[:, 0]

        grad_u_x = grad_u[:, 1]
        grad_grad_u_x = torch.autograd.grad(grad_u_x, x_f_train, grad_outputs=inputs, create_graph=True)[0]
        grad_u_xx = grad_grad_u_x[:, 1]

        residual = grad_u_t.reshape(-1, ) - self.k * grad_u_xx.reshape(-1, )
        return residual

    def ub0(self, t):
        """
        Boundary condition at one edge of the domain.

        Args:
          t: Tensor representing time points at the boundary.

        Returns:
          A tuple containing the boundary values and the type of boundary condition.
        """
        type_BC = [DirichletBC()]
        out = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return out.reshape(-1, 1), type_BC

    def ub1(self, t):
        """
        Boundary condition at the other edge of the domain.

        Args:
          t: Tensor representing time points at the boundary.

        Returns:
          A tuple containing the boundary values and the type of boundary condition.
        """

        type_BC = [DirichletBC()]
        out = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return out.reshape(-1, 1), type_BC

    def u0(self, x):
        """
        Initial condition for the problem.

        Args:
          x: Tensor representing spatial points at the initial time.

        Returns:
          Tensor containing the initial condition values.
        """
        u = -torch.sin(np.pi * x)
        return u.reshape(-1, 1)

    def exact(self, inputs):
        """
        Computes the exact solution for given input points.

        Args:
          inputs: Tensor containing input points in space and time.

        Returns:
          Tensor containing the exact solution values at the given points.
        """

        u = - torch.exp(-self.k * np.pi ** 2 * inputs[:, 0]) * torch.sin(np.pi * inputs[:, 1])

        return u.reshape(-1, 1)

    def compute_generalization_error(self, folder_path):
        """
        Computes the generalization error of the model by comparing the model's output to the exact solution.

        Args:
          folder_path: Path to the folder containing the model data.

        Returns:
          A tuple containing the L2 error and the relative L2 error.
        """
        model = self.model
        extrema = self.extrema_values
        model.eval()
        test_inp = self.convert(torch.rand([100000, extrema.shape[0]]), extrema)
        Exact = (self.exact(test_inp)).numpy()
        test_out = model(test_inp).detach().numpy()
        assert (Exact.shape[1] == test_out.shape[1])
        L2_test = np.sqrt(np.mean((Exact - test_out) ** 2))
        print("Error Test:", L2_test)

        rel_L2_test = L2_test / np.sqrt(np.mean(Exact ** 2))
        print("Relative Error Test:", rel_L2_test)

        return L2_test, rel_L2_test

    def plotting(self, images_path):
        """
        Plots the exact and predicted solutions at different time steps.

        Args:
          images_path: Path to the folder where the plots will be saved.
        """
        model = self.model
        model.cpu()
        model = model.eval()
        x = torch.reshape(torch.linspace(self.extrema_values[1, 0], self.extrema_values[1, 1], 100), [100, 1])
        time_steps = [0.0, 0.25, 0.5, 0.75, 1]
        scale_vec = np.linspace(0.65, 1.55, len(time_steps))

        fig = plt.figure()
        plt.grid(True, which="both", ls=":")
        for val, scale in zip(time_steps, scale_vec):
            plot_var = torch.cat([torch.full(size=(100, 1), fill_value=val), x], 1)
            plt.plot(x, self.exact(plot_var), linewidth=2, label=r'Source, $t=$' + str(val) + r'$s$', color=self.lighten_color('grey', scale), zorder=0)
            plt.scatter(plot_var[:, 1].detach().numpy(), model(plot_var).detach().numpy(), label=r'Predicted, $t=$' + str(val) + r'$s$', marker="o", s=14,
                        color=self.lighten_color('C0', scale), zorder=10)

        plt.xlabel(r'$x$')
        plt.ylabel(r'$u$')
        plt.legend()
        plt.savefig(images_path + "/Samples.png", dpi=500)
