import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from BoundaryConditions import DirichletBC
from EquationModels.EquationBaseClass import EquationBaseClass
from GeneratorPoints import generator_points


class EquationClass(EquationBaseClass):

    def __init__(self, network_hyperparameters, additional_hyper_parameters):
        super().__init__(network_hyperparameters, additional_hyper_parameters)

        self.type_of_points = additional_hyper_parameters["point"]
        self.output_dimension = 1
        self.extrema_values = None
        self.time_dimensions = 0
        self.space_dimensions = 2
        self.input_dimensions = self.space_dimensions + self.time_dimensions
        #self.list_of_BC = [[self.ubx0, self.ubx1], [self.uby0, self.uby1]]

        self.extrema_values_0 = torch.tensor([[-1, 0],
                                              [-1, 0]])

        self.extrema_values_1 = torch.tensor([[-1, 0],
                                              [0, 1]])

        self.extrema_values_2 = torch.tensor([[0, 1],
                                              [0, 1]])

        self.extrema_values_squares = list([self.extrema_values_0, self.extrema_values_1, self.extrema_values_2])

        self.extrema_values_b0 = torch.tensor([[-1, -1],
                                               [-1, 1]])

        self.extrema_values_b1 = torch.tensor([[-1, 1],
                                               [1, 1]])

        self.extrema_values_b2 = torch.tensor([[1, 1],
                                               [0, 1]])
        self.extrema_values_b3 = torch.tensor([[0, 1],
                                               [0, 0]])

        self.extrema_values_b4 = torch.tensor([[0, 0],
                                               [-1, 0]])

        self.extrema_values_b5 = torch.tensor([[-1, 0],
                                               [-1, -1]])

        self.extrema_values_boundaries = list([self.extrema_values_b0, self.extrema_values_b1, self.extrema_values_b2,
                                               self.extrema_values_b3, self.extrema_values_b4, self.extrema_values_b5])

        assert self.input_dimensions is not None, "The variable input_dimensions has not been initialized in the child class."

        self.init_model()
        self.init_params()

    def add_collocation_points(self, n_coll, random_seed):

        x_coll = list()
        n_coll_i = int(n_coll / 3)

        for i in range(len(self.extrema_values_squares)):
            x_coll_i = generator_points(n_coll_i, self.space_dimensions + self.time_dimensions, random_seed, self.type_of_points, False)
            x_coll_i = self.convert(x_coll_i, self.extrema_values_squares[i])
            x_coll.append(x_coll_i)

        x_coll = torch.cat(x_coll)
        y_coll = torch.full((x_coll.shape[0], self.output_dimension), np.nan)

        return x_coll, y_coll

    def add_boundary_points(self, n_boundary, random_seed):
        x_boundary = list()
        n_boundary_list = [n_boundary, n_boundary, int(n_boundary / 2), int(n_boundary / 2), int(n_boundary / 2), int(n_boundary / 2)]

        for i in range(len(self.extrema_values_boundaries)):
            x_b_i = generator_points(n_boundary_list[i], self.space_dimensions + self.time_dimensions, random_seed, self.type_of_points, False)
            x_b_i = self.convert(x_b_i, self.extrema_values_boundaries[i])
            x_boundary.append(x_b_i)

        x_boundary = torch.cat(x_boundary)

        y_boundary = torch.full((x_boundary.shape[0], self.output_dimension), 0.0)

        return x_boundary, y_boundary

    def add_initial_points(self, n_initial, random_seed):
        x_initial = torch.zeros((0, self.space_dimensions + self.time_dimensions))
        y_initial = torch.zeros((0, self.output_dimension))
        return x_initial, y_initial

    def apply_bc(self,  x_b_train, u_b_train, u_pred_var_list, u_train_var_list):
        bc = DirichletBC()
        bc.apply(self.model, x_b_train, u_b_train, 0, u_pred_var_list, u_train_var_list)

    def compute_res(self, x_f_train):

        x_f_train.requires_grad = True
        u = self.model(x_f_train).reshape(-1, )
        grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ), create_graph=True)[0]

        grad_u_x = grad_u[:, 0]
        grad_u_y = grad_u[:, 1]
        grad_u_xx = torch.autograd.grad(grad_u_x, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0]), create_graph=True)[0][:, 0]
        grad_u_yy = torch.autograd.grad(grad_u_y, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0]), create_graph=True)[0][:, 1]

        f = self.source(x_f_train)
        residual = grad_u_xx.reshape(-1, ) + grad_u_yy.reshape(-1, ) + f.reshape(-1, )

        return residual

    def source(self, inputs):
        return torch.ones_like(inputs[:, 0])

    def compute_generalization_error(self, folder_path):
        model = self.model
        model.eval()
        basename = "Data/Lshape_5"
        vertices = np.loadtxt("%s_vertices.txt" % basename)
        Exact = np.loadtxt("%s_values.txt" % basename).reshape(-1, 1)
        test_out = (model(torch.from_numpy(vertices[:, :2]).type(torch.FloatTensor))).detach().numpy()
        assert (Exact.shape[1] == test_out.shape[1])
        L2_test = np.sqrt(np.mean((Exact - test_out) ** 2))
        print("Error Test:", L2_test)

        rel_L2_test = L2_test / np.sqrt(np.mean(Exact ** 2))
        print("Relative Error Test:", rel_L2_test)

        return L2_test, rel_L2_test

    def plotting(self, images_path):
        model = self.model
        model.cpu()
        model = model.eval()

        basename = "Data/Lshape_5"
        vertices = np.loadtxt("%s_vertices.txt" % basename)
        indices = np.loadtxt("%s_triangles.txt" % basename)
        ex_sol = np.loadtxt("%s_values.txt" % basename)
        pinns_sol = (model(torch.from_numpy(vertices[:, :2]).type(torch.FloatTensor))[:, 0]).detach().numpy()
        fi, ax = plt.subplots(1, 2, figsize=(15, 5))

        grid = matplotlib.tri.Triangulation(vertices[:, 0], vertices[:, 1], indices)

        vmin = min(np.min(ex_sol), np.min(pinns_sol))
        vmax = max(np.max(ex_sol), np.max(pinns_sol))

        # Create both plots with the same vmin and vmax
        tpc1 = ax[0].tripcolor(grid, ex_sol, cmap='Spectral', vmin=vmin, vmax=vmax)
        tpc2 = ax[1].tripcolor(grid, pinns_sol, cmap='Spectral', vmin=vmin, vmax=vmax)
        ax[0].set_title(r'$u$', fontsize=10)
        ax[1].set_title(r'$u^\ast$', fontsize=10)

        # Create a colorbar that applies to both axes
        fi.colorbar(tpc1, ax=ax, orientation='vertical')

        plt.savefig(images_path + "/sol.png", dpi=200)
