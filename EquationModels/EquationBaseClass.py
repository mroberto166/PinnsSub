import colorsys

import matplotlib.colors as mc
import torch

from DataClass import DefineDataset
from ModelClass import MLP, init_xavier


class EquationBaseClass:
    def __init__(self, network_hyperparameters, additional_hyper_parameters):
        """
        Initializes the EquationBaseClass with network and additional hyperparameters.
        Sets up the device (CPU/GPU) and initializes various model parameters and hyperparameters.

        Args:
          network_hyperparameters: A dictionary containing hyperparameters for the network.
          additional_hyper_parameters: A dictionary containing additional hyperparameters not directly related to the network structure.
        """
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
        self.additional_hyper_parameters = additional_hyper_parameters
        self.network_hyperparameters = network_hyperparameters

        self.input_dimensions = None
        self.output_dimension = None
        self.model = None
        self.params = None

    def init_model(self):
        """
        Initializes the neural network model by creating an instance of MLP with the specified input and output dimensions and hyperparameters.
        """
        # #####################################################################################################################################
        # Model Creation
        print("Model Creation")
        self.model = MLP(input_dimension=self.input_dimensions, output_dimension=self.output_dimension, hyper_parameters=self.network_hyperparameters)
        self.model.to(self.device)

    def init_params(self):
        """
        Initializes the parameters of the model by setting a random seed for reproducibility, applying Xavier initialization to the model parameters, and storing these parameters.
        """
        # #####################################################################################################################################
        # Weights Initialization
        torch.manual_seed(self.network_hyperparameters["retrain"])
        init_xavier(self.model)
        self.params = list(self.model.parameters())

    def apply_bc(self, x_b_train, u_b_train, u_pred_var_list, u_train_var_list):
        """
        Placeholder method for applying boundary conditions. To be implemented in subclasses.

        Args:
          x_b_train: Tensor containing the boundary training data.
          u_b_train: Tensor containing the boundary values.
          u_pred_var_list: List to store predicted boundary variables.
          u_train_var_list: List to store training boundary variables.

        Raises:
          NotImplemented: Indicates that the function needs to be implemented in a subclass.
        """
        NotImplemented("The function apply_bc is not implemented. Are you trying to solve a problem with no BC?")

    def apply_ic(self, x_u_train, u_train, u_pred_var_list, u_train_var_list):
        """
        Placeholder method for applying initial conditions. To be implemented in subclasses.

        Args:
          x_u_train: Tensor containing the initial condition training data.
          u_train: Tensor containing the initial condition values.
          u_pred_var_list: List to store predicted initial condition variables.
          u_train_var_list: List to store training initial condition variables.

        Raises:
          NotImplemented: Indicates that the function needs to be implemented in a subclass.
        """
        NotImplemented("The function apply_ic is not implemented. Are you trying to solve a problem with no IC?")

    def compute_res(self, x_f_train):
        """
        Placeholder method for computing residuals in the PINN model. To be implemented in subclasses.

        Args:
          x_f_train: Tensor containing training data for residual computation.

        Returns:
          Tensor of zeros as a placeholder for actual residual values.
        """
        NotImplemented("The function compute_res is not implemented. Is this not a PINN?")
        return torch.zeros((x_f_train.shape[0],))

    def loss(self, x_u_train, u_train, x_b_train, u_b_train, x_f_train, it, writer, p=2):
        """
        Compute the loss for the model, including contributions from boundary conditions, initial conditions, residuals, and regularization.

        Args:
          x_u_train: Tensor containing training data for initial conditions.
          u_train: Tensor containing values for initial conditions.
          x_b_train: Tensor containing training data for boundary conditions.
          u_b_train: Tensor containing values for boundary conditions.
          x_f_train: Tensor containing training data for residuals.
          it: Current iteration number.
          writer: Writer object for logging.
          p: Order of the norm for the loss calculation.

        Returns:
          A tuple containing total loss, loss from variables, and loss from PDE residuals.
        """
        self.model.train()
        lambda_residual = self.additional_hyper_parameters["residual_parameter"]
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

        res = self.compute_res(x_f_train)

        loss_res = (torch.mean(abs(res) ** p))
        loss_vars = (torch.mean(abs(u_pred_tot_vars - u_train_tot_vars) ** p))
        loss_vars_rel = loss_vars / (torch.mean(abs(u_train_tot_vars) ** p))
        loss_reg = lambda_reg * self.regularization_lp(order_regularizer)

        if lambda_residual >= 1:
            loss_v = (lambda_residual * loss_vars + loss_res + loss_reg)
        else:
            loss_v = (loss_vars + loss_res / lambda_residual + loss_reg)
        print("Total Loss:", loss_v.detach().cpu().numpy().round(5),
              "| Function Loss:", loss_vars.detach().cpu().numpy().round(5),
              "| PDE Loss:", loss_res.detach().cpu().numpy().round(5), "\n")
        writer.add_scalar("Total Loss", loss_vars + loss_res, it)
        writer.add_scalar("Total Loss with Reg", loss_v, it)
        writer.add_scalar("Train Components/PDE Residual", loss_res, it)
        writer.add_scalar("Train Components/BC Residual", loss_vars, it)
        writer.add_scalar("Train Components/BC Rel Residual", loss_vars_rel, it)
        return loss_v, loss_vars, loss_res

    def fit(self, training_set_class: DefineDataset, verbose=False, writer=None, folder=None, p=2) -> dict:
        """
        Trains the model using the provided dataset. Iterates through epochs, applies optimizer steps, and logs training progress and losses.

        Args:
          training_set_class: An instance of DefineDataset containing training data.
          verbose: Boolean flag to control the verbosity of the training process.
          writer: Writer object for logging training metrics.
          folder: Path to the folder for saving the model periodically.
          p: Order of the norm for the loss calculation.

        Returns:
          A dictionary containing final training errors including 'final_error_train', 'error_vars', and 'error_pde'.
        """
        num_epochs = self.additional_hyper_parameters["epochs_LBFGS"] + self.additional_hyper_parameters["epochs_adam"]

        freq = 10

        training_coll = training_set_class.data_coll
        training_boundary = training_set_class.data_boundary
        training_initial_internal = training_set_class.data_initial_internal

        self.model.train()

        def closure():
            optimizer.zero_grad()
            loss_tot, loss_vars, loss_pde = self.loss(x_u_train_, u_train_, x_b_train_, u_b_train_, x_coll_train_, iteration[0], writer, p)
            loss_f = torch.log10(loss_tot)
            loss_f.backward()

            train_losses[0] = loss_tot
            train_losses[1] = loss_vars
            train_losses[2] = loss_pde
            if iteration[0] % 500 == 0:
                self.save_model(folder)
            iteration[0] = iteration[0] + 1

            return loss_f

        train_losses = list([torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)])
        iteration = list([0])
        for epoch in range(num_epochs):
            if epoch < self.additional_hyper_parameters["epochs_adam"]:
                optimizer = self.model.optimizer_adam
            else:
                print("Setting full batch default option for LBFG. Minibatch and LBFGS together do not work")
                optimizer = self.model.optimizer_lbfgs
                training_set_class.batch_dim = training_set_class.n_samples
                training_set_class.assemble_dataset()
                training_coll = training_set_class.data_coll
                training_boundary = training_set_class.data_boundary
                training_initial_internal = training_set_class.data_initial_internal

            if verbose and epoch % freq == 0:
                print("################################ ", epoch, " ################################")

            if len(training_boundary) != 0 and len(training_initial_internal) != 0:
                for step, ((x_coll_train_, u_coll_train_), (x_b_train_, u_b_train_), (x_u_train_, u_train_)) in enumerate(zip(training_coll, training_boundary, training_initial_internal)):
                    x_coll_train_ = x_coll_train_.to(self.device)
                    x_b_train_ = x_b_train_.to(self.device)
                    u_b_train_ = u_b_train_.to(self.device)
                    x_u_train_ = x_u_train_.to(self.device)
                    u_train_ = u_train_.to(self.device)

                    optimizer.step(closure=closure)

            elif len(training_boundary) == 0 and len(training_initial_internal) != 0:
                for step, ((x_coll_train_, u_coll_train_), (x_u_train_, u_train_)) in enumerate(zip(training_coll, training_initial_internal)):
                    x_b_train_ = torch.full((0, x_u_train_.shape[1]), 0)
                    u_b_train_ = torch.full((0, x_u_train_.shape[1]), 0)

                    x_coll_train_ = x_coll_train_.to(self.device)
                    x_b_train_ = x_b_train_.to(self.device)
                    u_b_train_ = u_b_train_.to(self.device)
                    x_u_train_ = x_u_train_.to(self.device)
                    u_train_ = u_train_.to(self.device)

                    optimizer.step(closure=closure)

            elif len(training_boundary) != 0 and len(training_initial_internal) == 0:
                for step, ((x_coll_train_, u_coll_train_), (x_b_train_, u_b_train_)) in enumerate(zip(training_coll, training_boundary)):
                    x_u_train_ = torch.full((0, 1), 0)
                    u_train_ = torch.full((0, 1), 0)

                    x_coll_train_ = x_coll_train_.to(self.device)
                    x_b_train_ = x_b_train_.to(self.device)
                    u_b_train_ = u_b_train_.to(self.device)
                    x_u_train_ = x_u_train_.to(self.device)
                    u_train_ = u_train_.to(self.device)

                    optimizer.step(closure=closure)

            elif len(training_boundary) == 0 and len(training_initial_internal) == 0:
                for step, (x_coll_train_, u_coll_train_) in enumerate(training_coll):
                    x_u_train_ = torch.full((0, 1), 0)
                    u_train_ = torch.full((0, 1), 0)

                    x_b_train_ = torch.full((0, x_u_train_.shape[1]), 0)
                    u_b_train_ = torch.full((0, x_u_train_.shape[1]), 0)

                    x_coll_train_ = x_coll_train_.to(self.device)
                    x_b_train_ = x_b_train_.to(self.device)
                    u_b_train_ = u_b_train_.to(self.device)
                    x_u_train_ = x_u_train_.to(self.device)
                    u_train_ = u_train_.to(self.device)

                    optimizer.step(closure=closure)

                if epoch % freq == 0:
                    print("################################ ", epoch, " ################################")
                    print("PDE Residual: ", train_losses[2].detach().cpu().numpy().round(4))

        writer.flush()
        writer.close()
        final_errors = dict({
            "final_error_train": (train_losses[0] ** (1 / p)).item(),
            "error_vars": (train_losses[1] ** (1 / p)).item(),
            "error_pde": (train_losses[2] ** (1 / p)).item(),
        })
        return final_errors

    def compute_generalization_error(self, folder_path):
        """
        Placeholder method for computing the generalization error. To be implemented in subclasses.

        Args:
          folder_path: Path to the folder containing model data.

        Returns:
          Generalization error (not implemented, returns 0 by default).
        """
        print("compute_generalization_error not implemented")
        return 0, 0

    def plotting(self, folder_path):
        """
        Placeholder for plotting functionality. To be implemented in subclasses.

        Args:
          folder_path: Path to the folder where plots will be saved.
        """
        print("plotting not implemented")

    def save_model(self, folder_path):
        """
       Saves the model and its state to the specified folder.

       Args:
         folder_path: Path to the folder where the model and its state dictionary will be saved.
       """
        torch.save(self.model, folder_path + "/model.pkl")
        torch.save(self.model.state_dict(), folder_path + "/model_state_dict.pkl")

    def regularization_lp(self, p):
        """
        Computes the Lp regularization loss.

        Args:
          p: Order of the norm for regularization calculation.

        Returns:
          The regularization loss calculated as the Lp norm of model parameters.
        """
        reg_loss = 0
        for name, param in self.model.named_parameters():
            reg_loss = reg_loss + torch.norm(param, p)
        return reg_loss

    @staticmethod
    def convert(vector, extrema_values):
        """
        Converts a vector using given extrema values by scaling and shifting.

        Args:
          vector: The vector to be converted.
          extrema_values: Tensor containing the minimum and maximum values for scaling and shifting.

        Returns:
          The converted vector after scaling and shifting.
        """
        max_val, _ = torch.max(extrema_values, dim=1)
        min_val, _ = torch.min(extrema_values, dim=1)
        vector = vector * (max_val - min_val) + min_val
        return vector

    @staticmethod
    def lighten_color(color, amount=0.5):
        """
        Lightens a given color by a specified amount.

        Args:
          color: The color to be lightened. Can be a named color or RGB tuple.
          amount: The amount by which to lighten the color (0-1 range).

        Returns:
          The lightened color as an RGB tuple.
        """
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
