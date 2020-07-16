from ImportFile import *

pi = math.pi
T = 10

a = [4, 0]

extrema_values = torch.tensor([[0, 1],
                               [-8, 8],
                               [-8, 8]])


def compute_res(network, x_f_train, space_dimensions, solid, computing_error=False):
    x_f_train.requires_grad = True
    u = (network(x_f_train))[:, 0].reshape(-1, )
    v = (network(x_f_train))[:, 1].reshape(-1, )
    p = (network(x_f_train))[:, 2].reshape(-1, )

    inputs = torch.ones(x_f_train.shape[0], )

    if not computing_error and torch.cuda.is_available():
        inputs = inputs.cuda()

    grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=inputs, create_graph=True)[0]
    grad_u_t = grad_u[:, 0].reshape(-1, )
    grad_u_x = grad_u[:, 1].reshape(-1, )
    grad_u_y = grad_u[:, 2].reshape(-1, )

    grad_v = torch.autograd.grad(v, x_f_train, grad_outputs=inputs, create_graph=True)[0]
    grad_v_t = grad_v[:, 0].reshape(-1, )
    grad_v_x = grad_v[:, 1].reshape(-1, )
    grad_v_y = grad_v[:, 2].reshape(-1, )

    grad_p = torch.autograd.grad(p, x_f_train, grad_outputs=inputs, create_graph=True)[0]
    grad_p_x = grad_p[:, 1].reshape(-1, )
    grad_p_y = grad_p[:, 2].reshape(-1, )

    res_u = grad_u_t + u * grad_u_x + v * grad_u_y + grad_p_x
    res_v = grad_v_t + u * grad_v_x + v * grad_v_y + grad_p_y
    res_d = grad_u_x + grad_v_y

    mean_P = torch.mean(p).reshape(-1, )

    res = torch.cat([mean_P, res_u, res_v, res_d], 0)

    if torch.cuda.is_available():
        del inputs
        torch.cuda.empty_cache()

    return res


def exact(inputs):
    t = inputs[:, 0]
    x = inputs[:, 1] - a[0] * t
    y = inputs[:, 2] - a[1] * t

    u_0 = (-y * torch.exp(0.5 * (1 - x ** 2 - y ** 2)) + a[0]).reshape(-1, 1)
    v_0 = (x * torch.exp(0.5 * (1 - x ** 2 - y ** 2)) + a[1]).reshape(-1, 1)
    return torch.cat([u_0, v_0], 1)


def convert(vector, extrema_values):
    vector = np.array(vector)
    max_val = np.max(np.array(extrema_values), axis=1)
    min_val = np.min(np.array(extrema_values), axis=1)
    vector = vector * (max_val - min_val) + min_val
    return torch.from_numpy(vector).type(torch.FloatTensor)


def compute_generalization_error(model, extrema, images_path=None):
    model.eval()
    test_inp = convert(torch.rand([100000, extrema.shape[0]]), extrema)
    Exact = exact(test_inp).detach().numpy()
    test_out = model(test_inp).detach().numpy()

    u_exact = (Exact[:, 0].reshape(-1, 1))
    u = test_out[:, 0].reshape(-1, 1)

    v_exact = (Exact[:, 1].reshape(-1, 1))
    v = test_out[:, 1].reshape(-1, 1)

    assert (v_exact.shape[1] == v.shape[1])
    assert (u_exact.shape[1] == u.shape[1])

    L2_test = np.sqrt(np.mean((u_exact - u) ** 2 + (v_exact - v) ** 2))
    print("Error Test:", L2_test)
    rel_L2_test = L2_test / np.sqrt(np.mean(u_exact ** 2 + v_exact ** 2))
    print("Relative Error Test:", rel_L2_test)

    if images_path is not None:
        plt.figure()
        plt.grid(True, which="both", ls=":")
        plt.scatter(u_exact, u)
        plt.xlabel(r'Exact Values')
        plt.ylabel(r'Predicted Values')
        plt.savefig(images_path + "/TV_Score_u.png", dpi=400)
    if images_path is not None:
        plt.figure()
        plt.grid(True, which="both", ls=":")
        plt.scatter(v_exact, v)
        plt.xlabel(r'Exact Values')
        plt.ylabel(r'Predicted Values')
        plt.savefig(images_path + "/TV_Score_v.png", dpi=400)
    return L2_test, rel_L2_test


def ub0(y):
    type_BC = ["periodic", "periodic", "periodic"]
    u = torch.tensor(()).new_full(size=(y.shape[0], 1), fill_value=0.0)
    v = torch.tensor(()).new_full(size=(y.shape[0], 1), fill_value=0.0)
    p = torch.tensor(()).new_full(size=(y.shape[0], 1), fill_value=0.0)
    return torch.cat([u, v, p], 1), type_BC


def ub1(y):
    type_BC = ["periodic", "periodic", "periodic"]
    u = torch.tensor(()).new_full(size=(y.shape[0], 1), fill_value=0.0)
    v = torch.tensor(()).new_full(size=(y.shape[0], 1), fill_value=0.0)
    p = torch.tensor(()).new_full(size=(y.shape[0], 1), fill_value=0.0)
    return torch.cat([u, v, p], 1), type_BC


def ub0y(x):
    type_BC = ["periodic", "periodic", "periodic"]
    u = torch.tensor(()).new_full(size=(x.shape[0], 1), fill_value=0.0)
    v = torch.tensor(()).new_full(size=(x.shape[0], 1), fill_value=0.0)
    p = torch.tensor(()).new_full(size=(x.shape[0], 1), fill_value=0.0)
    return torch.cat([u, v, p], 1), type_BC


def ub1y(x):
    type_BC = ["periodic", "periodic", "periodic"]
    u = torch.tensor(()).new_full(size=(x.shape[0], 1), fill_value=0.0)
    v = torch.tensor(()).new_full(size=(x.shape[0], 1), fill_value=0.0)
    p = torch.tensor(()).new_full(size=(x.shape[0], 1), fill_value=0.0)
    return torch.cat([u, v, p], 1), type_BC


list_of_BC = [[ub0, ub1], [ub0y, ub1y]]


def u0(input):
    x = input[:, 0]
    y = input[:, 1]
    u_0 = (-y * torch.exp(0.5 * (1 - x ** 2 - y ** 2)) + a[0]).reshape(-1, 1)
    v_0 = (x * torch.exp(0.5 * (1 - x ** 2 - y ** 2)) + a[1]).reshape(-1, 1)
    p_0 = torch.tensor(()).new_full(size=(x.shape[0], 1), fill_value=0.0)

    return torch.cat([u_0, v_0, p_0], 1)


def plotting(model, images_path, extrema, solid):
    x = torch.linspace(extrema[1, 0], extrema[1, 1], 400).reshape(-1, 1)
    y = torch.linspace(extrema[2, 0], extrema[2, 1], 400).reshape(-1, 1)
    xy = torch.from_numpy(np.array([[x_i, y_i] for x_i in x for y_i in y]).reshape(x.shape[0] * y.shape[0], 2)).type(torch.FloatTensor)

    for val in [0, 1]:
        t = torch.tensor(()).new_full(size=(xy.shape[0], 1), fill_value=val)
        input_vals = torch.cat([t, xy], 1)
        input_vals.requires_grad = True
        output = model(input_vals)

        exact_solution = exact(input_vals)
        u = output[:, 0]
        v = output[:, 1]

        grad_u = torch.autograd.grad(u, input_vals, grad_outputs=torch.ones(input_vals.shape[0], ), create_graph=True)[0]
        grad_u_y = grad_u[:, 2].reshape(-1, )

        grad_v = torch.autograd.grad(v, input_vals, grad_outputs=torch.ones(input_vals.shape[0], ), create_graph=True)[0]
        grad_v_x = grad_v[:, 1].reshape(-1, )

        w = -grad_u_y + grad_v_x
        w = w.reshape(x.shape[0], y.shape[0])
        w = w.detach().numpy()

        u_ex = exact_solution[:, 0]
        v_ex = exact_solution[:, 1]

        grad_u_ex = torch.autograd.grad(u_ex, input_vals, grad_outputs=torch.ones(input_vals.shape[0], ), create_graph=True)[0]
        grad_u_ex_y = grad_u_ex[:, 2].reshape(-1, )

        grad_v_ex = torch.autograd.grad(v_ex, input_vals, grad_outputs=torch.ones(input_vals.shape[0], ), create_graph=True)[0]
        grad_v_ex_x = grad_v_ex[:, 1].reshape(-1, )

        w_ex = -grad_u_ex_y + grad_v_ex_x
        w_ex = w_ex.reshape(x.shape[0], y.shape[0])
        w_ex = w_ex.detach().numpy()

        u = u.reshape(x.shape[0], y.shape[0])
        u = u.detach().numpy()

        v = v.reshape(x.shape[0], y.shape[0])
        v = v.detach().numpy()

        u_ex = u_ex.reshape(x.shape[0], y.shape[0])
        u_ex = u_ex.detach().numpy()

        v_ex = v_ex.reshape(x.shape[0], y.shape[0])
        v_ex = v_ex.detach().numpy()

        plt.figure()
        plt.contourf(x.reshape(-1, ), y.reshape(-1, ), w.T, 40, cmap='Spectral')
        plt.colorbar()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(r'$\omega(x,y)$,\quad t=' + str(val))
        plt.savefig(images_path + "/TV_Samples_w_" + str(val) + ".png", dpi=400)
        plt.close()

        plt.figure()
        plt.contourf(x.reshape(-1, ), y.reshape(-1, ), w_ex.T, 40, cmap='Spectral')
        plt.colorbar()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(r'$\omega(x,y)$,\quad t=' + str(val))
        plt.savefig(images_path + "/TV_Samples_w_ex_" + str(val) + ".png", dpi=400)
        plt.close()

        plt.figure()
        plt.contourf(x.reshape(-1, ), y.reshape(-1, ), u.T, 40, cmap='Spectral')
        plt.colorbar()
        plt.savefig(images_path + "/TV_Samples_u_" + str(val) + ".png", dpi=400)
        plt.close()

        plt.figure()
        plt.contourf(x.reshape(-1, ), y.reshape(-1, ), v.T, 40, cmap='Spectral')
        plt.colorbar()
        plt.savefig(images_path + "/TV_Samples_v_" + str(val) + ".png", dpi=400)
        plt.close()

        plt.figure()
        plt.contourf(x.reshape(-1, ), y.reshape(-1, ), u_ex.T, 40, cmap='Spectral')
        plt.colorbar()
        plt.savefig(images_path + "/TV_Samples_u_ex_" + str(val) + ".png", dpi=400)
        plt.close()

        plt.figure()
        plt.contourf(x.reshape(-1, ), y.reshape(-1, ), v_ex.T, 40, cmap='Spectral')
        plt.colorbar()
        plt.savefig(images_path + "/TV_Samples_v_ex_" + str(val) + ".png", dpi=400)
        plt.close()
