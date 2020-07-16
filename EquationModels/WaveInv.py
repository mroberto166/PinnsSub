from ImportFile import *

pi = math.pi
extrema_values = torch.tensor([[0, 1],
                               [0, 1]])

inner_type_p = "uniform"
omega_1 = torch.tensor([[0, 1],
                        [0., 0.2]])
omega_2 = torch.tensor([[0, 1],
                        [0.8, 1.0]])

domain = "GC"


def compute_res(network, x_f_train, space_dimensions, solid, computing_error):
    x_f_train.requires_grad = True

    u = (network(x_f_train)).reshape(-1, )
    inputs = torch.ones(x_f_train.shape[0], )

    if torch.cuda.is_available():
        inputs = inputs.cuda()

    grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=inputs, create_graph=True)[0]

    grad_u_t = grad_u[:, 0].reshape(-1, )
    grad_u_x = grad_u[:, 1].reshape(-1, )

    grad_u_xx = torch.autograd.grad(grad_u_x, x_f_train, grad_outputs=inputs, create_graph=True)[0][:, 1]
    grad_u_tt = torch.autograd.grad(grad_u_t, x_f_train, grad_outputs=inputs, create_graph=True)[0][:, 0]

    res_u = grad_u_tt - grad_u_xx
    return res_u


def exact(inputs):
    t = inputs[:, 0]
    x = inputs[:, 1]
    u = torch.cos(2 * pi * t) * torch.sin(2 * pi * x)

    return u.reshape(-1, 1)


def convert(vector, extrema_values):
    vector = np.array(vector)
    max_val = np.max(np.array(extrema_values), axis=1)
    min_val = np.min(np.array(extrema_values), axis=1)
    vector = vector * (max_val - min_val) + min_val
    return torch.from_numpy(vector).type(torch.FloatTensor)


def compute_generalization_error(model, extrema, images_path=None):
    model.eval()
    test_inp = convert(torch.rand([100000, extrema.shape[0]]), extrema)
    Exact = (exact(test_inp)).numpy()
    test_out = model(test_inp).detach().numpy()
    assert (Exact.shape[1] == test_out.shape[1])
    L2_test = np.sqrt(np.mean((Exact - test_out) ** 2))
    print("Error Test:", L2_test)

    rel_L2_test = L2_test / np.sqrt(np.mean(Exact ** 2))
    print("Relative Error Test:", rel_L2_test)

    if images_path is not None:
        plt.figure()
        plt.grid(True, which="both", ls=":")
        plt.scatter(Exact, test_out)
        plt.xlabel(r'Exact Values')
        plt.ylabel(r'Predicted Values')
        plt.savefig(images_path + "/Score.png", dpi=400)
    return L2_test, rel_L2_test


def add_internal_points(n_internal):
    # Grid Points

    if inner_type_p == "grid":
        if domain == "GC":

            nx = int(np.sqrt(n_internal / 0.4))
            nx_int = int(0.4 * nx) - 2
            print(nx_int)
            nt_int = nx
            n_bound = nt_int
            dx1 = (omega_1[1, 1] - omega_1[1, 0]) / (nx_int / 2)
            dx2 = (omega_2[1, 1] - omega_2[1, 0]) / (nx_int / 2)
            x1 = np.linspace(omega_1[1, 0] + dx1, omega_1[1, 1], int(nx_int / 2))
            x2 = np.linspace(omega_2[1, 0], omega_2[1, 1] - dx2, int(nx_int / 2))
            x = np.concatenate([x1, x2], 0)
        else:
            nx = int(np.sqrt(n_internal / 0.2))
            nx_int = int(0.2 * nx) - 2
            print(nx_int)
            nt_int = nx
            n_bound = nt_int
            dx = (omega_1[1, 1] - omega_1[1, 0]) / nx_int
            x = np.linspace(omega_1[1, 0] + dx, omega_1[1, 1], int(nx_int))
        # x = x1

        t = np.linspace(omega_1[0, 0], omega_1[0, 1], nt_int)
        inputs = torch.from_numpy(np.transpose([np.repeat(t, len(x)), np.tile(x, len(t))])).type(torch.FloatTensor)

    if inner_type_p == "uniform":
        if domain == "GC":

            nx = int(np.sqrt(n_internal / 0.4))
            nx_int = int(0.4 * nx) - 2
            nt_int = nx
            n_bound = nt_int
            n_internal_new = n_internal - 2 * n_bound
            inputs = torch.rand((n_internal_new, 2))
            inputs[:int(n_internal_new / 2), :] = convert(inputs[:int(n_internal_new / 2), :], omega_1)
            inputs[int(n_internal_new / 2):, :] = convert(inputs[int(n_internal_new / 2):, :], omega_2)
        else:
            nx = int(np.sqrt(n_internal / 0.2))
            nx_int = int(0.2 * nx) - 2
            nt_int = nx
            n_bound = nt_int
            n_internal_new = n_internal - 2 * n_bound
            inputs = torch.rand((n_internal_new, 2))
            inputs = convert(inputs, omega_1)

    inputs_b0 = torch.cat([torch.linspace(omega_1[0, 0], omega_1[0, 1], int(n_bound)).reshape(-1, 1), torch.full((n_bound, 1), 0)], 1)
    inputs_b1 = torch.cat([torch.linspace(omega_1[0, 0], omega_1[0, 1], int(n_bound)).reshape(-1, 1), torch.full((n_bound, 1), 1)], 1)
    inputs = torch.cat([inputs, inputs_b0, inputs_b1])

    u = exact(inputs)
    return inputs, u


def ub0(y):
    type_BC = ["func"]
    u = torch.tensor(()).new_full(size=(y.shape[0], 1), fill_value=0.0)
    return u, type_BC


def ub1(y):
    type_BC = ["func"]
    u = torch.tensor(()).new_full(size=(y.shape[0], 1), fill_value=0.0)
    return u, type_BC


def ub0y(x):
    type_BC = ["func"]
    u = torch.tensor(()).new_full(size=(x.shape[0], 1), fill_value=0.0)
    return u, type_BC


def ub1y(x):
    type_BC = ["func"]
    u = torch.tensor(()).new_full(size=(x.shape[0], 1), fill_value=0.0)
    return u, type_BC


def u0(x):
    u = torch.tensor(()).new_full(size=(0, 1), fill_value=0.0)
    return u


list_of_BC = [[ub0, ub1], [ub0y, ub1y]]


def plotting(model, images_path, extrema, solid):
    model.eval()
    t = torch.linspace(0, 1, 400).reshape(-1, 1)
    x = torch.linspace(0., 1., 400).reshape(-1, 1)
    tx = torch.from_numpy(np.array([[t_i, x_i] for t_i in t for x_i in x]).reshape(t.shape[0] * x.shape[0], 2)).type(torch.FloatTensor)

    tx.requires_grad = True

    output = model(tx)

    u = output[:, 0]
    uex = exact(tx)[:, 0]

    output_grad = torch.autograd.grad(u, tx, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    output_grad_x = output_grad[:, 1].reshape(t.shape[0], x.shape[0]).detach().numpy()

    ex_grad = torch.autograd.grad(uex, tx, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    exact_grad_x = ex_grad[:, 1].reshape(t.shape[0], x.shape[0]).detach().numpy()

    u = u.reshape(t.shape[0], x.shape[0])
    u = u.detach().numpy()

    uex = uex.reshape(t.shape[0], x.shape[0])
    uex = uex.detach().numpy()

    L2_err = np.sqrt(np.mean((u - uex) ** 2, 1))
    H1_err = np.sqrt(np.mean((u - uex) ** 2, 1) + np.mean((output_grad_x - exact_grad_x) ** 2, 1))

    L2_ex = np.sqrt(np.mean(uex ** 2, 1))
    H1_ex = np.sqrt(np.mean(uex ** 2, 1) + np.mean(exact_grad_x ** 2, 1))

    L2_rel = L2_err / L2_ex
    H1_rel = H1_err / H1_ex

    l2_glob = np.max(L2_err)
    h1_glob = np.max(H1_err)
    l2_glob_rel = np.max(L2_err) / np.max(L2_ex)
    h1_glob_rel = np.max(H1_err) / np.max(H1_ex)

    err = np.sqrt((u - uex) ** 2) / np.sqrt(np.mean(uex ** 2))

    plt.figure()
    plt.title(r'$u^\ast(t,x)$')
    ax = plt.gca()
    ttl = ax.title
    ttl.set_position([.5, 1.035])
    plt.contourf(t.reshape(-1, ), x.reshape(-1, ), u.T, 20, cmap='Spectral')
    plt.colorbar()
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.savefig(images_path + "/Wave_u.png", dpi=400)

    plt.figure()
    plt.title(r'$u(t,x)$')
    ax = plt.gca()
    ttl = ax.title
    ttl.set_position([.5, 1.035])
    plt.contourf(t.reshape(-1, ), x.reshape(-1, ), uex.T, 20, cmap='Spectral')
    plt.colorbar()
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.savefig(images_path + "/Wave_uex.png", dpi=400)

    plt.figure()
    plt.title(r'$||u(t,x) - u^\ast(t,x)||$')
    ax = plt.gca()
    ttl = ax.title
    ttl.set_position([.5, 1.035])
    plt.contourf(t.reshape(-1, ), x.reshape(-1, ), err.T, 20, cmap='Spectral')
    plt.colorbar()
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.savefig(images_path + "/Wave_err.png", dpi=400)

    model.eval()
    t = torch.linspace(0., 1, 400).reshape(-1, 1)

    if domain == "GC":
        x1 = np.linspace(omega_1[1, 0], omega_1[1, 1], 200)
        x2 = np.linspace(omega_2[1, 0], omega_2[1, 1], 200)
        x = np.concatenate([x1, x2], 0)
    else:
        x = torch.linspace(omega_1[1, 0], omega_1[1, 1], 400).reshape(-1, 1)

    tx = torch.from_numpy(np.array([[t_i, x_i] for t_i in t for x_i in x]).reshape(t.shape[0] * x.shape[0], 2)).type(torch.FloatTensor)

    output = model(tx)

    u = output[:, 0]
    uex = exact(tx)[:, 0]

    u = u.reshape(t.shape[0], x.shape[0])
    u = u.detach().numpy()

    uex = uex.reshape(t.shape[0], x.shape[0])
    uex = uex.detach().numpy()

    l2_om_big = np.max((np.sqrt(np.mean((u - uex) ** 2)), 0))
    l2_om_big_rel = l2_om_big / np.max((np.sqrt(np.mean(uex ** 2)), 0))

    print(l2_om_big, l2_om_big_rel)
    print(l2_glob, l2_glob_rel)
    print(h1_glob, h1_glob_rel)

    with open(images_path + '/errors.txt', 'w') as file:
        file.write("l2_glob,"
                   "l2_glob_rel,"
                   "h1_glob,"
                   "h1_glob_rel,"
                   "l2_om_big,"
                   "l2_om_big_rel\n")
        file.write(str(float(l2_glob)) + "," +
                   str(float(l2_glob_rel)) + "," +
                   str(float(h1_glob)) + "," +
                   str(float(h1_glob_rel)) + "," +
                   str(float(l2_om_big)) + "," +
                   str(float(l2_om_big_rel)))
