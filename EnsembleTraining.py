import os
import sys
import itertools
import subprocess
from ImportFile import *

rs = 0
# Number of collocation points (where the PDE is enforced)
N_coll = int(sys.argv[1])
# Number of initial and boundary points
N_u = int(sys.argv[2])
# Number of internal points (neither boundaries or initial points) where the solution of the PDE is known
N_int = int(sys.argv[3])

# TO DO: remove n_time_steps (not used)
n_time_steps = 0

# TO DO: Add or not an object in the domain (only circles and squares)
ob = "None"
# number of points at the surface of an object in the domain
n_object = 0

time_dimensions = 0
# Dimensions of parameter space (for UQ). Not used in the experiments of the paper
parameter_dimensions = 0
# Number of output dimension of the newtwork
n_out = 1
folder_name = sys.argv[4]
# Type of points (random or sobol)
point = "sobol"
validation_size = 0.0

# Hyperparameters configurations for ensamble training
network_properties = {
    "hidden_layers": [4, 8, 10],
    "neurons": [16, 20, 24],
    "residual_parameter": [0.001, 0.01, 0.1, 1],
    "kernel_regularizer": [2],
    "regularization_parameter": [0, 1e-6],
    "batch_size": [(N_coll + N_u + N_int)],
    "epochs": [1],
    "activation": ["tanh"],
}

shuffle = "false"
# Run on cluster
cluster = sys.argv[5]

if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
settings = list(itertools.product(*network_properties.values()))

i = 0
for setup in settings:
    print(setup)

    folder_path = folder_name + "/Setup_" + str(i)
    print("###################################")
    setup_properties = {
        "hidden_layers": setup[0],
        "neurons": setup[1],
        "residual_parameter": setup[2],
        "kernel_regularizer": setup[3],
        "regularization_parameter": setup[4],
        "batch_size": setup[5],
        "epochs": setup[6],
        "activation": setup[7]
    }

    arguments = list()
    arguments.append(str(rs))
    arguments.append(str(N_coll))
    arguments.append(str(N_u))
    arguments.append(str(N_int))
    arguments.append(str(n_time_steps))
    arguments.append(str(n_object))
    arguments.append(str(ob))
    arguments.append(str(time_dimensions))
    arguments.append(str(parameter_dimensions))
    arguments.append(str(n_out))
    arguments.append(str(folder_path))
    arguments.append(str(point))
    arguments.append(str(validation_size))
    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        arguments.append("\'" + str(setup_properties).replace("\'", "\"") + "\'")
    else:
        arguments.append(str(setup_properties).replace("\'", "\""))
    arguments.append(str(shuffle))
    arguments.append(str(cluster))

    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        if cluster == "true":
            string_to_exec = "bsub python3 single_retraining.py "
        else:
            string_to_exec = "python3 single_retraining.py "
        for arg in arguments:
            string_to_exec = string_to_exec + " " + arg
        print(string_to_exec)
        os.system(string_to_exec)
    i = i + 1
