import json
import sys
import os
import importlib


def initialize_inputs(len_sys_argv):
    # Define the module name
    # Choose one among the files in the EquationModels folder:
    #    - HeatEquation
    #    - PoissonDomainL
    #    - RadiativeTransfer1D
    #    - PoissonDAP
    #    - EigenValueProblem1D
    module_name = "EigenValueProblem1D"

    # Dynamic import of EquationClass
    EquationClass = getattr(importlib.import_module(f"EquationModels.{module_name}"), "EquationClass")

    if len_sys_argv == 1:

        use_default = True

        if use_default:
            # Construct the configuration file name and load it
            config_file_name = f"DefaultConfig/{module_name}.json"

            with open(config_file_name, 'r') as file:
                config = json.load(file)

            # Extract variables
            sampling_seed_ = config['sampling_seed']
            n_int_ = config['n_int']
            n_u_ = config['n_u']
            n_ob_ = config['n_ob']
            folder_path_ = config['folder_path']
            network_hyperparameters = config['network_hyperparameters']
            additional_hyper_parameters = config['additional_hyper_parameters']

            # Evaluate the 'batch_size' expression
            additional_hyper_parameters['batch_size'] = eval(additional_hyper_parameters['batch_size'], {'n_int': n_int_, 'n_u': n_u_, 'n_ob': n_ob_})
        else:
            # Random Seed for sampling the dataset
            sampling_seed_ = 128

            # Number of training
            n_int_ = 8192
            n_u_ = 2048
            n_ob_ = 0

            # Additional Info
            folder_path_ = "EigenTest"
            network_hyperparameters = {
                "hidden_layers": 4,
                "neurons": 20,
                "activation": "tanh",
                "adaptive": 0,
                "n": 1,
                "retrain": 4,
            }
            additional_hyper_parameters = {
                "residual_parameter": 10,
                "p_regularization": 2,
                "regularization_parameter": 0,
                "batch_size": (n_int_ + n_u_ + n_ob_),
                "epochs_LBFGS": 1,
                "max_iter_LBFGS": 5000,
                "learning_rate": 0.1,
                "epochs_adam": 0,
                "learning_rate_adam": 1e-3,
                "point": "sobol",
                "lambda_orth": 1000,
                "lambda_norm": 100,
            }

    else:
        # Random Seed for sampling the dataset
        sampling_seed_ = int(sys.argv[1])

        # Number of training+validation points
        n_int_ = int(sys.argv[2])
        n_u_ = int(sys.argv[3])
        n_ob_ = int(sys.argv[4])

        # Additional Info
        folder_path_ = sys.argv[5]

        network_hyperparameters = json.loads(sys.argv[6].replace("\'", "\""))
        additional_hyper_parameters = json.loads(sys.argv[7].replace("\'", "\""))

    property_types = {
        "hidden_layers": int,
        "neurons": int,
        "residual_parameter": float,
        "p_regularization": int,
        "regularization_parameter": float,
        "batch_size": int,
        "epochs_LBFGS": int,
        "max_iter_LBFGS": int,
        "activation": str,
        "learning_rate": float,  # ADAM,
        "adaptive": bool,
        "n": int,
        "epochs_adam": int,
        "learning_rate_adam": float,
        "retrain": int,
        "point": str,
        "lambda_orth": float,
        "lambda_norm": float,
    }

    # Convert the properties to their respective types
    for name, _ in network_hyperparameters.items():
        network_hyperparameters[name] = property_types[name](network_hyperparameters[name])

    for name, _ in additional_hyper_parameters.items():
        additional_hyper_parameters[name] = property_types[name](additional_hyper_parameters[name])

    if not os.path.isdir(folder_path_):
        os.mkdir(folder_path_)

    equation_class = EquationClass(network_hyperparameters, additional_hyper_parameters)

    return sampling_seed_, n_int_, n_u_, n_ob_, folder_path_, equation_class
