from ImportFile import *

pi = math.pi
T = 0.02
k = 2
a = 0.2
extrema_values = torch.tensor([[0, T],
                               [0, 1]])
inner_type_p = "grid"


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

    res_u = grad_u_t - grad_u_xx
    return res_u


def exact(inputs):
    t = inputs[:, 0]
    x = inputs[:, 1]
    u = torch.exp(-k ** 2 * pi ** 2 * t) * torch.sin(pi * k * x)

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
    # Uniform Points
    # x = torch.rand([n_internal, extrema_values.shape[0]])

    # Grid Points
    nt_int = 16
    nx_int = int(n_internal / nt_int) - 2
    n_bound = nt_int

    if inner_type_p == "grid":
        x = np.linspace(a, 1 - a, nx_int)
        t = np.linspace(0, T, nt_int)
        inputs = torch.from_numpy(np.transpose([np.repeat(t, len(x)), np.tile(x, len(t))])).type(torch.FloatTensor)
        print(n_internal, inputs.shape)
    # Sobol Points
    if inner_type_p == "sobol":
        inputs = torch.from_numpy(sobol_seq.i4_sobol_generate(2, n_internal)).type(torch.FloatTensor)
        extrema_inner = torch.tensor([[0, 0.02],
                                      [a, 1 - a]])
        inputs = convert(inputs, extrema_inner)

    if inner_type_p == "uniform":
        inputs = torch.rand([int(nt_int * nx_int), extrema_values.shape[0]])
        extrema_inner = torch.tensor([[0, 0.02],
                                      [a, 1 - a]])
        inputs = convert(inputs, extrema_inner)

    inputs_b0 = torch.cat([torch.linspace(0, T, int(n_bound)).reshape(-1, 1), torch.full((n_bound, 1), 0)], 1)
    inputs_b1 = torch.cat([torch.linspace(0, T, int(n_bound)).reshape(-1, 1), torch.full((n_bound, 1), 1)], 1)
    inputs = torch.cat([inputs, inputs_b0, inputs_b1])

    u = exact(inputs)
    return inputs, u


def ub0(y):
    type_BC = ["func"]
    u = torch.sin(y).reshape(-1, 1)
    u = torch.tensor(()).new_full(size=(y.shape[0], 1), fill_value=0.0)
    return u, type_BC


def ub1(y):
    type_BC = ["func"]
    u = torch.sin(y).reshape(-1, 1)
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
    t = torch.linspace(0, T, 1).reshape(-1, 1)
    x = torch.linspace(0., 1., 400).reshape(-1, 1)
    tx = torch.from_numpy(np.array([[t_i, x_i] for t_i in t for x_i in x]).reshape(t.shape[0] * x.shape[0], 2)).type(torch.FloatTensor)
    tx.requires_grad = True

    output = model(tx)

    u = output[:, 0]
    uex = exact(tx)[:, 0]

    output_grad = torch.autograd.grad(u, tx, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    output_grad_x = output_grad[:, 1]

    ex_grad = torch.autograd.grad(uex, tx, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    exact_grad_x = ex_grad[:, 1]

    l2_glob = torch.sqrt(torch.mean((u - uex) ** 2))
    l2_glob_rel = l2_glob / torch.sqrt(torch.mean(uex ** 2))

    h1_glob = torch.sqrt(torch.mean((u - uex) ** 2) + torch.mean((exact_grad_x - output_grad_x) ** 2))
    h1_glob_rel = h1_glob / (torch.sqrt(torch.mean(uex ** 2) + torch.mean(exact_grad_x ** 2)))

    t = torch.linspace(0, T, 400).reshape(-1, 1)
    x = torch.linspace(0., 1., 400).reshape(-1, 1)
    tx = torch.from_numpy(np.array([[t_i, x_i] for t_i in t for x_i in x]).reshape(t.shape[0] * x.shape[0], 2)).type(torch.FloatTensor)

    output = model(tx)

    u = output[:, 0]
    uex = exact(tx)[:, 0]

    u = u.reshape(t.shape[0], x.shape[0])
    u = u.detach().numpy()

    uex = uex.reshape(t.shape[0], x.shape[0])
    uex = uex.detach().numpy()
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
    plt.savefig(images_path + "/Heat_u.png", dpi=400)

    plt.figure()
    plt.title(r'$u(t,x)$')
    ax = plt.gca()
    ttl = ax.title
    ttl.set_position([.5, 1.035])
    plt.contourf(t.reshape(-1, ), x.reshape(-1, ), uex.T, 20, cmap='Spectral')
    plt.colorbar()
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.savefig(images_path + "/Heat_uex.png", dpi=400)

    plt.figure()
    plt.title(r'$||u(t,x) - u^\ast(t,x)||$')
    ax = plt.gca()
    ttl = ax.title
    ttl.set_position([.5, 1.035])
    plt.contourf(t.reshape(-1, ), x.reshape(-1, ), err.T, 20, cmap='Spectral')
    plt.colorbar()
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.savefig(images_path + "/Heat_err.png", dpi=400)

    model.eval()
    t = torch.linspace(T, T, 1).reshape(-1, 1)
    x = torch.linspace(a, 1 - a, 400).reshape(-1, 1)
    tx = torch.from_numpy(np.array([[t_i, x_i] for t_i in t for x_i in x]).reshape(t.shape[0] * x.shape[0], 2)).type(torch.FloatTensor)

    output = model(tx)

    u = output[:, 0]
    uex = exact(tx)[:, 0]

    l2_om_big = torch.sqrt(torch.mean((u - uex) ** 2))
    l2_om_big_rel = l2_om_big / torch.sqrt(torch.mean(uex ** 2))

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
        file.write(str(float(l2_glob.detach().numpy())) + "," +
                   str(float(l2_glob_rel.detach().numpy())) + "," +
                   str(float(h1_glob.detach().numpy())) + "," +
                   str(float(h1_glob_rel.detach().numpy())) + "," +
                   str(float(l2_om_big.detach().numpy())) + "," +
                   str(float(l2_om_big_rel.detach().numpy())))
