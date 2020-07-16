import os
import sys
import itertools
import subprocess
from ImportFile import *

###############################################
# Convergence Analysis for PINNS
# N_coll = [1000, 2000, 4000, 8000, 16000]
# N_u = [16, 32, 64, 128, 512]
# N_int = [0]

###############################################
# Resampling for Inverse Problem
# TO DO: merge the two
N_coll = np.array([int(sys.argv[1])])
N_u = np.array([int(sys.argv[2])])
N_int = np.array([int(sys.argv[3])])
print(N_coll)
print(N_int)

n_time_steps = 0
n_object = 0
ob = "None"
time_dimensions = 0
parameter_dimensions = 0
n_out = 3
folder_name = sys.argv[4]
point = "random"
validation_size = 0.0
network_properties = {
    "hidden_layers": 4,
    "neurons": 24,
    "residual_parameter": 0.001,
    "kernel_regularizer": 2,
    "regularization_parameter": 0,
    "batch_size": "full",
    "epochs": 1,
    "activation": "tanh",
}
shuffle = "false"
cluster = sys.argv[5]

'''

N_coll = [int(500*(1/validation_size)), int(1000*(1/validation_size)), int(2000*(1/validation_size)), int(4000*(1/validation_size)), int(8000*(1/validation_size))]
N_u = [int(25*(1/validation_size)),  int(50*(1/validation_size)), int(200*(1/validation_size)), int(400*(1/validation_size)), int(800*(1/validation_size))]
N_int = [0]

N_coll = [200, 800, 3200, 12800, 25600]
N_u = [100, 200, 400, 800, 1600]
N_int = [0]
'''

# N_coll = [2000]
# N_u = [200]
# N_int = [0]

if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
'''settings = list(itertools.product(N_coll, N_u, N_int))
'''

# for setup in settings:
for i in range(N_coll.shape[0]):

    '''N_coll_set = setup[0]
    N_u_set = setup[1]
    N_int_set = setup[2]'''
    N_coll_set = N_coll[i]
    N_u_set = N_u[i]
    N_int_set = N_int[i]

    print("\n")
    print("##########################################")
    print("Number of samples:")
    print(" - Collocation points:", N_coll_set)
    print(" - Initial and boundary points:", N_u_set)
    print(" - Internal points:", N_int_set)
    print("\n")
    folder_path = folder_name + "/" + str(int(N_u_set)) + "_" + str(int(N_coll_set)) + "_" + str(int(N_int_set))

    arguments = list()

    arguments.append(str(N_coll_set))
    arguments.append(str(N_u_set))
    arguments.append(str(N_int_set))
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
        arguments.append("\'" + str(network_properties).replace("\'", "\"") + "\'")
    else:
        arguments.append(str(network_properties).replace("\'", "\""))

    arguments.append(str(shuffle))
    arguments.append(str(cluster))

    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        if cluster == "true":
            string_to_exec = "bsub -W 1:00 python3 single_training.py "
        else:
            string_to_exec = "python3 single_training.py "
        for arg in arguments:
            string_to_exec = string_to_exec + " " + str(arg)
        os.system(string_to_exec)
    else:
        python = os.environ['PYTHON36']
        p = subprocess.Popen([python, "single_training.py"] + arguments)
        p.wait()
