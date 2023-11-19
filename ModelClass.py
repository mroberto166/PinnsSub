import math
import torch
import torch.nn as nn

pi = math.pi


class Swish(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Snake(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.5

    def forward(self, x):
        return x + torch.sin(self.alpha * x) ** 2 / self.alpha


class Sin(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


def activation(name, alpha=0.5):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    elif name in ['celu', 'CeLU']:
        return nn.CELU()
    elif name in ['swish']:
        return Swish()
    elif name in ['sin']:
        return Sin()
    elif name in ['snake']:
        return Snake()
    else:
        raise ValueError('Unknown activation function')


class MLP(nn.Module):

    def __init__(self, input_dimension, output_dimension, hyper_parameters):
        super(MLP, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_hidden_layers = int(hyper_parameters["hidden_layers"])
        self.neurons = int(hyper_parameters["neurons"])
        # self.lambda_residual = float(hyper_parameters["residual_parameter"])
        # self.lambda_pde = float(hyper_parameters["lambda_pde"])
        # self.kernel_regularizer = int(hyper_parameters["kernel_regularizer"])
        # self.regularization_param = float(hyper_parameters["regularization_parameter"])
        # self.num_epochs_opt_LBFGS = int(hyper_parameters["epochs_LBFGS"])
        # self.num_epochs_opt_adam = int(hyper_parameters["epochs_adam"])
        self.act_string = str(hyper_parameters["activation"])
        # self.optimizer = hyper_parameters["optimizer"]
        # self.adaptive = hyper_parameters["adaptive"]
        # self.lambda_inverse = float(hyper_parameters["lambda_inverse"])
        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(self.n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        self.activation = activation(self.act_string)

    def forward(self, x):
        u = self.singe_forward(x)
        return u

    def singe_forward(self, x):
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        u = self.output_layer(x)
        return u

    def compute_l2_norm(self):
        u = self.singe_forward(self.x_inner)
        l2_norm = torch.mean(u ** 2) ** 0.5

        return l2_norm


def init_xavier(model):
    # torch.nn.init.uniform_(model.a, 0, 2)

    def init_weights(m):
        '''for coeff in model.coeff_list:
            torch.nn.init.uniform_(coeff, 1, 10)'''
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            if model.act_string == "sin" or model.act_string == "swish" or model.act_string == "celu" or model.act_string == "snake":
                gain = 1
            else:
                gain = nn.init.calculate_gain(model.act_string)
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0)

    model.apply(init_weights)


def init_xavier_eigen(model):
    # torch.nn.init.uniform_(model.a, 0, 2)

    def init_weights(m):
        '''for coeff in model.coeff_list:
            torch.nn.init.uniform_(coeff, 1, 10)'''
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            gain = nn.init.calculate_gain("relu")
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0)

    model.apply(init_weights)


def init_uniform(model):
    # torch.nn.init.uniform_(model.a, 0, 2)

    def init_weights(m):
        '''for coeff in model.coeff_list:
            torch.nn.init.uniform_(coeff, 1, 10)'''
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            m.weight.data.uniform_(-1.0, 1.0)
            m.bias.data.uniform_(-1.0, 1.0)

    model.apply(init_weights)


def init_normal(model):
    # torch.nn.init.uniform_(model.a, 0, 2)

    def init_weights(m):
        '''for coeff in model.coeff_list:
            torch.nn.init.uniform_(coeff, 1, 10)'''
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            m.weight.data.normal_(0, 1.0)
            m.bias.data.normal_(0, 1.0)

    model.apply(init_weights)
