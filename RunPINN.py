import os
import pprint
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from Config import initialize_inputs
from DataClass import DefineDataset
from utils import dump_dict_to_file

torch.manual_seed(42)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

sampling_seed, N_int_train, N_u_train, N_ob_train, folder_path, equation_class = initialize_inputs(len(sys.argv))

additional_hyper_parameters = equation_class.additional_hyper_parameters
network_hyperparameters = equation_class.network_hyperparameters

num_samples = {"N_int": N_int_train,
               "N_u": N_u_train,
               "N_ob": N_ob_train}

dump_dict_to_file(folder_path + os.sep + "network_hyperparameters.csv", network_hyperparameters)
dump_dict_to_file(folder_path + os.sep + "additional_hyper_parameters.csv", additional_hyper_parameters)
dump_dict_to_file(folder_path + os.sep + "num_samples.csv", num_samples)

if equation_class.extrema_values is not None:
    extrema = equation_class.extrema_values
    space_dimensions = equation_class.space_dimensions
    time_dimension = equation_class.time_dimensions
    if hasattr(equation_class, "parameter_dimensions"):
        parameter_dimensions = equation_class.parameter_dimensions
    else:
        parameter_dimensions = 0

    print(space_dimensions, time_dimension, parameter_dimensions)
else:
    print("Using free shape. Make sure you have the functions:")
    print("     - add_boundary(n_samples)")
    print("     - add_collocation(n_samples)")
    print("in the Equation file")

    extrema = None
    space_dimensions = equation_class.space_dimensions
    time_dimension = equation_class.time_dimensions
try:
    parameters_values = equation_class.parameters_values
    parameter_dimensions = parameters_values.shape[0]
except AttributeError:
    print("No additional parameter found")
    parameters_values = None
    parameter_dimensions = 0

input_dimensions = parameter_dimensions + time_dimension + space_dimensions
output_dimension = equation_class.output_dimension
max_iter = additional_hyper_parameters["max_iter_LBFGS"]

N_train = N_u_train + N_int_train + N_ob_train

if space_dimensions > 0:
    N_bc_train = int(N_u_train / (4 * space_dimensions))
else:
    N_bc_train = 0
if time_dimension == 1:
    N_ic_train = N_u_train - 2 * space_dimensions * N_bc_train
elif time_dimension == 0:
    N_bc_train = int(N_u_train / (2 * space_dimensions))
    N_ic_train = 0
else:
    raise ValueError()

print("\n######################################")
print("*******Domain Properties********")
print(extrema)

print("\n######################################")
print("*******Info Training Points********")
print("Number of train collocation points: ", N_int_train)
print("Number of initial and boundary points: ", N_u_train, N_ic_train, N_bc_train)
print("Number of internal points: ", N_ob_train)
print("Total number of training points: ", N_train)

print("\n######################################")
print("******* Network Hyper-parameters ********")
pprint.pprint(network_hyperparameters)
print("\n######################################")
print("******* Additional Hyper-parameters ********")
pprint.pprint(additional_hyper_parameters)

print("\n######################################")
print("*******Dimensions********")
print("Space Dimensions", space_dimensions)
print("Time Dimension", time_dimension)
print("Parameter Dimensions", parameter_dimensions)
print("\n######################################")
batch_dim = additional_hyper_parameters["batch_size"]
if additional_hyper_parameters["epochs_LBFGS"] != 1 and additional_hyper_parameters["max_iter_LBFGS"] == 1 and (batch_dim == "full" or batch_dim == N_train):
    print(f"WARNING: you set max_iter=1 and epochs={additional_hyper_parameters['epochs_LBFGS']}  with a LBFGS optimizer.\n "
          f"This will work but it is not efficient in full batch mode. Set max_iter = {additional_hyper_parameters['epochs_LBFGS']} and epochs_LBFGS=1. instead")

############################################################################################################################################################
# Dataset Creation
print("Dataset Creation")
training_set_class = DefineDataset(equation_class, N_int_train, N_bc_train, N_ic_train, N_ob_train, batch_dim=batch_dim, random_seed=sampling_seed, shuffle=True)

# #############################################################################################################################################################
# Optimizers Creation
optimizer_LBFGS = optim.LBFGS(equation_class.params,
                              lr=float(additional_hyper_parameters["learning_rate"]),
                              max_iter=max_iter,
                              max_eval=50000,
                              history_size=100,
                              line_search_fn="strong_wolfe",
                              tolerance_grad=1.0 * np.finfo(float).eps,
                              tolerance_change=1.0 * np.finfo(float).eps
                              )

optimizer_ADAM = optim.Adam(equation_class.params,
                            lr=float(additional_hyper_parameters["learning_rate_adam"]),
                            amsgrad=True)

equation_class.model.optimizer_lbfgs = optimizer_LBFGS
equation_class.model.optimizer_adam = optimizer_ADAM

# #############################################################################################################################################################
# Model Training
start = time.time()
print("Fitting Model")
writer = SummaryWriter(log_dir=folder_path)
errors = equation_class.fit(training_set_class, verbose=False, writer=writer, folder=folder_path)

end = time.time() - start
print(f"Training Time: {end}")

model = equation_class.model.eval()
print("\n################################################")
print("Final Training Loss:", errors["final_error_train"])
print("################################################")

# #############################################################################################################################################################
# Plotting ang Assessing Performance
equation_class.save_model(folder_path + "/model_final.pkl")
equation_class.plotting(folder_path)
tot_error, rel_tot_error = equation_class.compute_generalization_error(folder_path)
errors["tot_error"] = tot_error
errors["rel_tot_error"] = rel_tot_error

dump_dict_to_file(folder_path + os.sep + "errors.csv", errors)