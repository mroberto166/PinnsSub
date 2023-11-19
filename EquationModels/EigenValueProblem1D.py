import copy

import matplotlib.pyplot as plt
import numpy as np
import torch.nn

from BoundaryConditions import DirichletBC
from EquationModels.EquationBaseClass import EquationBaseClass
from ModelClass import init_xavier, init_xavier_eigen
from SquareDomain import SquareDomain


class EquationClass(EquationBaseClass):

    def __init__(self, network_hyperparameters, additional_hyper_parameters):
        super().__init__(network_hyperparameters, additional_hyper_parameters)

        self.type_of_points = additional_hyper_parameters["point"]

        self.output_dimension = 1
        self.space_dimensions = 1
        self.time_dimensions = 0
        self.parameter_dimensions = 1
        self.extrema_values = torch.tensor([[0, 1]])

        self.input_dimensions = self.extrema_values.shape[0]

        self.list_of_BC = list([[self.ub0x, self.ub1x]])

        self.square_domain = SquareDomain(self.output_dimension,
                                          self.time_dimensions,
                                          self.space_dimensions,
                                          self.list_of_BC,
                                          self.extrema_values,
                                          self.type_of_points)

        self.tol_start = 1e-4
        self.tol_final = 1e-3
        if self.input_dimensions is None:
            raise ValueError("The variable input_dimensions han not been initialized in the child class.")

        self.model_lists = list()
        self.lambda_lists = list()
        self.eigenvalue = torch.nn.Sequential(torch.nn.Linear(self.extrema_values.shape[0], 1)).to(self.device)
        self.init_model()

        torch.manual_seed(network_hyperparameters["retrain"])
        init_xavier_eigen(self.eigenvalue)
        init_xavier(self.model)

        self.params = list(self.model.parameters()) + list(self.eigenvalue.parameters())

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
        self.square_domain.apply_boundary_conditions(self.model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list)

    def compute_res(self, x_f_train):
        x_f_train.requires_grad = True
        u = self.model(x_f_train).reshape(-1, )
        lambda_ = self.eigenvalue(torch.ones_like(x_f_train)).reshape(-1, )
        A_square = self.input_dimensions * lambda_ ** 2

        grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        grad_u_x = grad_u[:, 0]

        grad_u_xx = torch.autograd.grad(grad_u_x, x_f_train, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0]
        lap = grad_u_xx / (2 * np.pi) ** 2

        res = lap + A_square * u

        return res

    def fit(self, training_set_class, verbose=False, writer=None, folder=None, p=2):
        model_id = list([0])
        iteration = list([0])
        tol = list([self.tol_start])
        converged = list([False])
        num_epochs = self.additional_hyper_parameters["epochs_LBFGS"] + self.additional_hyper_parameters["epochs_adam"]

        freq = 10

        training_coll = training_set_class.data_coll
        training_boundary = training_set_class.data_boundary

        self.model.train()

        def closure():
            optimizer.zero_grad()
            loss_f, loss_vars, loss_pde, loss_ortho, norm_loss = self.loss(x_u_train_, u_train_, x_b_train_, u_b_train_, x_coll_train_, iteration[0], writer, p=p)

            total_loss = loss_vars + loss_pde + norm_loss + loss_ortho

            train_losses[0] = loss_f
            train_losses[1] = loss_vars
            train_losses[2] = loss_pde
            train_losses[3] = norm_loss
            train_losses[4] = loss_ortho

            loss_f.backward()
            lambda_ = self.eigenvalue(torch.ones_like(x_coll_train_)).reshape(-1, )[0]
            writer.add_scalar("Total Loss", total_loss, iteration[0])
            writer.add_scalar("Train/Norm Loss", norm_loss, iteration[0])
            writer.add_scalar("Train/PDE Residual", loss_pde, iteration[0])
            writer.add_scalar("Train/BC Residual", loss_vars, iteration[0])
            writer.add_scalar("Train/Orthogonal Residual", loss_ortho, iteration[0])
            writer.add_scalar("Lambda", lambda_, iteration[0])
            writer.add_scalar("#Basis", len(self.model_lists), iteration[0])
            if iteration[0] % 500 == 0:
                torch.save(self.model, folder + "/model" + str(model_id[0]) + ".pkl")
            iteration[0] = iteration[0] + 1

            if total_loss < tol[0]:
                print("Restarting")
                torch.save(self.model, folder + "/model" + str(model_id[0]) + ".pkl")
                self.model_lists.append(copy.deepcopy(self.model))
                self.lambda_lists.append(lambda_)
                optimizer.load_state_dict(org_state_dict)
                converged[0] = True
                model_id[0] = model_id[0] + 1
                raise StopIteration("Achieve search tolerance")
            else:
                converged[0] = False

            return loss_f

        optimizer = self.model.optimizer_lbfgs
        org_state_dict = optimizer.state_dict()
        train_losses = list([torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)])

        for epoch in range(num_epochs):
            log_tol = (np.log10(self.tol_final) - np.log10(self.tol_start)) / (num_epochs - 1 + 1e-16) * epoch + np.log10(self.tol_start)
            tol[0] = np.power(10, log_tol)

            if verbose and epoch % freq == 0:
                print("################################ ", epoch, " ################################")

            for step, ((x_coll_train_, u_coll_train_), (x_b_train_, u_b_train_)) in enumerate(zip(training_coll, training_boundary)):

                x_u_train_ = torch.full((0, 1), 0)
                u_train_ = torch.full((0, 1), 0)

                x_coll_train_ = x_coll_train_.to(self.device)
                x_b_train_ = x_b_train_.to(self.device)
                u_b_train_ = u_b_train_.to(self.device)
                x_u_train_ = x_u_train_.to(self.device)
                u_train_ = u_train_.to(self.device)

                try:
                    optimizer.step(closure=closure)
                except StopIteration as e:
                    print(e)

            print("Resetting parameters")
            retrain = np.random.randint(0, 10000)
            torch.manual_seed(retrain)
            init_xavier(self.model)
            init_xavier_eigen(self.eigenvalue)
            if not converged[0]:
                print("Not Converged")
            else:
                self.plotting(folder)
                print("Converged")
        writer.flush()
        writer.close()
        final_errors = dict({
            "final_error_train": (train_losses[0] ** (1 / p)).item(),
            "error_vars": (train_losses[1] ** (1 / p)).item(),
            "error_pde": (train_losses[2] ** (1 / p)).item(),
            "error_norm": (train_losses[3] ** (1 / p)).item(),
            "error_orth": (train_losses[4] ** (1 / p)).item(),
        })
        return final_errors

    def loss(self, x_u_train, u_train, x_b_train, u_b_train, x_f_train, it, writer, p=2):
        lambda_residual = self.additional_hyper_parameters["residual_parameter"]
        lambda_norm = self.additional_hyper_parameters["lambda_norm"]
        lambda_orth = self.additional_hyper_parameters["lambda_orth"]
        lambda_reg = self.additional_hyper_parameters["regularization_parameter"]
        order_regularizer = self.additional_hyper_parameters["p_regularization"]

        u_pred_var_list = list()
        u_train_var_list = list()

        if x_b_train.shape[0] != 0:
            self.apply_bc(x_b_train, u_b_train, u_pred_var_list, u_train_var_list)
        if x_u_train.shape[0] != 0:
            self.apply_ic(x_u_train, u_train, u_pred_var_list, u_train_var_list)

        if u_train_var_list:
            u_pred_tot_vars = torch.cat(u_pred_var_list, 0).to(self.device)
            u_train_tot_vars = torch.cat(u_train_var_list, 0).to(self.device)

            assert not torch.isnan(u_pred_tot_vars).any()
        else:
            u_pred_tot_vars = torch.tensor(0.).to(self.device)
            u_train_tot_vars = torch.tensor(0.).to(self.device)

        res = self.compute_res(x_f_train).to(self.device)

        loss_res = (torch.mean(abs(res) ** p))
        loss_vars = (torch.mean(abs(u_pred_tot_vars - u_train_tot_vars) ** p))
        loss_reg = self.regularization_lp(order_regularizer)
        norm_loss = self.additional_losses(x_f_train, p=p)
        ortho_loss = self.orth_loss(x_f_train, p=p)
        loss_v = torch.log10(lambda_residual * loss_vars +
                             loss_res +
                             lambda_orth * ortho_loss +
                             lambda_norm * norm_loss +
                             lambda_reg * loss_reg)
        print("Total Loss:", loss_v.detach().cpu().numpy().round(5),
              "| PDE Loss:", loss_res.detach().cpu().numpy().round(5),
              "| Var Loss:", loss_vars.detach().cpu().numpy().round(5),
              "| Norm Loss:", norm_loss.detach().cpu().numpy().round(5),
              "| Orth Loss:", ortho_loss.detach().cpu().numpy().round(5), "\n")
        return loss_v, loss_vars, loss_res, ortho_loss, norm_loss

    def additional_losses(self, x_f_train, p=2):
        u = self.model(x_f_train).reshape(-1, )
        l2_norm_u = torch.mean(u ** 2)

        loss_norm = abs(l2_norm_u - 1. / 2)

        return loss_norm ** p

    def orth_loss(self, x_f_train, p=2):
        loss_ortho = torch.tensor(0.)
        if self.model_lists:
            output_current_model = self.model(x_f_train).reshape(-1, )
            for k, old_model in enumerate(self.model_lists):
                # Compute scalar product
                output_old_model = old_model(x_f_train).reshape(-1, )
                scalar_prod = abs(torch.dot(output_current_model, output_old_model) / output_current_model.shape[0]) ** p

                loss_ortho = loss_ortho + scalar_prod

        return loss_ortho

    def ub0x(self, inputs):
        type_BC = [DirichletBC()]
        u = torch.full(size=(inputs.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def ub1x(self, inputs):
        type_BC = [DirichletBC()]
        u = torch.full(size=(inputs.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def compute_generalization_error(self, images_path):
        L2_test = 0.
        rel_L2_test = 0.

        print("Error Test:", L2_test)
        print("Relative Error Test:", rel_L2_test)

        return L2_test, rel_L2_test

    def plotting(self, images_path):

        nx = 100
        x = torch.linspace(0, 1, nx).unsqueeze(-1)

        plt.figure()
        plt.grid(True, which="both", ls=":")

        for k, model in enumerate(self.model_lists):
            model = model.eval()
            u = model(x).cpu()
            plt.plot(x.reshape(-1, ), u.detach().numpy(), label=f"Eigen Function {k + 1}, " + r"$\lambda=$" + str(round(self.lambda_lists[k].item(), 3)))
            plt.xlabel(r'$x$')
            plt.ylabel(r'$u$')
        plt.legend()
        plt.savefig(images_path + "/u.png", dpi=400)
