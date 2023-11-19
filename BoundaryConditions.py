import torch
import numpy as np
import matplotlib.pyplot as plt


class DirichletBC:
    def __init__(self):
        self.name = "DirichletBC"

    def apply(self, model, x_boundary, u_boundary, n_out, u_pred_var_list, u_train_var_list, space_dim=None, time_dim=None, x_boundary_sym=None, boundary=None, vel_wave=None, inverse=False):
        u_boundary_pred = model(x_boundary)
        u_pred_var_list.append(u_boundary_pred[:, n_out])
        u_train_var_list.append(u_boundary[:, n_out])

        return boundary


class NeumannBC:
    def __init__(self):
        self.name = "NeumannBC"

    def apply(self, model, x_boundary, u_boundary, n_out, u_pred_var_list, u_train_var_list, space_dim=None, time_dim=None, x_boundary_sym=None, boundary=None, vel_wave=None, inverse=False):
        x_boundary.requires_grad = True
        u_boundary_pred = model(x_boundary)[:, n_out]
        grad_u_x = torch.autograd.grad(u_boundary_pred, x_boundary, grad_outputs=torch.ones_like(u_boundary_pred), create_graph=True)[0][:, time_dim + space_dim]
        u_pred_var_list.append(grad_u_x)
        u_train_var_list.append(u_boundary[:, n_out])
        return boundary


class PeriodicBC:
    def __init__(self):
        self.name = "PeriodicBC"

    def apply(self, model, x_boundary, u_boundary, n_out, u_pred_var_list, u_train_var_list, space_dim=None, time_dim=None, x_boundary_sym=None, boundary=None, vel_wave=None, inverse=False):
        x_boundary.requires_grad = True
        x_boundary_sym.requires_grad = True
        u_boundary_pred = model(x_boundary)
        u_boundary_pred_sym = model(x_boundary_sym)
        u_pred_var_list.append(u_boundary_pred[:, n_out])
        u_train_var_list.append(u_boundary_pred_sym[:, n_out])

        grad_u_x = torch.autograd.grad(u_boundary_pred, x_boundary, grad_outputs=torch.ones_like(u_boundary_pred), create_graph=True)[0][:, time_dim + space_dim]
        grad_u_x_sym = torch.autograd.grad(u_boundary_pred_sym, x_boundary_sym, grad_outputs=torch.ones_like(u_boundary_pred), create_graph=True)[0][:, time_dim + space_dim]
        u_pred_var_list.append(grad_u_x)
        u_train_var_list.append(grad_u_x_sym)
        boundary = boundary + 1
        return boundary


class AbsorbingBC:
    def __init__(self):
        self.name = "AbsorbingBC"

    def apply(self, model, x_boundary, u_boundary, n_out, u_pred_var_list, u_train_var_list, space_dim=None, time_dim=None, x_boundary_sym=None, boundary=None, vel_wave=None, inverse=False):
        x_boundary.requires_grad = True
        u_boundary_pred = model(x_boundary)
        grad_u = torch.autograd.grad(u_boundary_pred, x_boundary, grad_outputs=torch.ones_like(u_boundary_pred), create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_n = grad_u[:, 1 + space_dim]

        if boundary == 1:
            n = torch.ones_like(grad_u_n)
        else:
            n = -torch.ones_like(grad_u_n)

        if not inverse:
            vel = vel_wave(x_boundary[:, 1:3]).reshape(-1, )
            absorb = grad_u_t + n * vel * grad_u_n
            # absorb = grad_u_t + n*vel_wave(x_boundary).reshape(-1,)*grad_u_n
        else:
            vel_w = vel_wave(x_boundary).reshape(-1, )
            # vel_w = model.a
            absorb = grad_u_t + n * vel_w * grad_u_n
        # print(torch.mean(abs(absorb)))
        u_pred_var_list.append(absorb)
        u_train_var_list.append(torch.zeros_like(absorb))
        print((torch.mean(abs(absorb) ** 2)))
        return boundary


class NoneBC:
    def __init__(self):
        self.name = "NoneBC"

    def apply(self, model, x_boundary, u_boundary, n_out, u_pred_var_list, u_train_var_list, space_dim=None, time_dim=None, x_boundary_sym=None, boundary=None, vel_wave=None, inverse=False):
        return boundary
