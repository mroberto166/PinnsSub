from ImportFile import *

pi = math.pi
T = 10
extrema_values = torch.tensor([[0, 1],
                               [0, 1]])

inner_type_p = "grid"
espilon = 0.0


def compute_res(network, x_f_train, space_dimensions, solid, computing_error):
    x_f_train.requires_grad = True
    u = (network(x_f_train)).reshape(-1, )
    inputs = torch.ones(x_f_train.shape[0], )

    if torch.cuda.is_available():
        inputs = inputs.cuda()

    grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=inputs, create_graph=True)[0]

    grad_u_x = grad_u[:, 0].reshape(-1, )
    grad_u_y = grad_u[:, 1].reshape(-1, )

    grad_u_xx = torch.autograd.grad(grad_u_x, x_f_train, grad_outputs=inputs, create_graph=True)[0][:, 0]
    grad_u_yy = torch.autograd.grad(grad_u_y, x_f_train, grad_outputs=inputs, create_graph=True)[0][:, 1]
    lap_u = grad_u_xx + grad_u_yy

    x = x_f_train[:, 0].reshape(-1, )
    y = x_f_train[:, 1].reshape(-1, )

    f = 2 * (30 * y * (1 - y) + 30 * x * (1 - x))
    # f_pert = f*(1 + espilon*torch.randn(f.shape))

    if torch.cuda.is_available():
        f = f.cuda()
    res_u = lap_u + f
    return res_u


def exact(inputs):
    x = inputs[:, 0]
    y = inputs[:, 1]
    u = 30 * x * (1 - x) * y * (1 - y)

    return u.reshape(-1, 1)


def grad_exact(inputs):
    x = inputs[:, 0]
    y = inputs[:, 1]
    der_x = 30 * (1 - 2 * x) * y * (1 - y)
    der_y = 30 * x * (1 - x) * (1 - 2 * y)

    return der_x, der_y


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

    if inner_type_p == "grid":
        n_internal_x = int(np.sqrt(n_internal))
        x1 = np.linspace(0.125, 0.875, n_internal_x)
        x = torch.from_numpy(np.transpose([np.repeat(x1, len(x1)), np.tile(x1, len(x1))])).type(torch.FloatTensor)

    # Sobol Points
    if inner_type_p == "sobol":
        x = torch.from_numpy(sobol_seq.i4_sobol_generate(2, n_internal)).type(torch.FloatTensor)
        extrema_inner = torch.tensor([[0.125, 0.875],
                                      [0.125, 0.875]])
        x = convert(x, extrema_inner)

    if inner_type_p == "uniform":
        x = torch.rand([n_internal, extrema_values.shape[0]])
        extrema_inner = torch.tensor([[0.125, 0.875],
                                      [0.125, 0.875]])
        x = convert(x, extrema_inner)
    u = exact(x) * (1 + espilon * torch.randn(exact(x).shape))
    return x, u


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
    u = torch.tensor(()).new_full(size=(x.shape[0], 1), fill_value=0.0)
    return u


list_of_BC = [[ub0, ub1], [ub0y, ub1y]]


def plotting(model, images_path, extrema, solid):
    model.eval()
    x = torch.linspace(0., 1., 400).reshape(-1, 1)
    y = torch.linspace(0., 1., 400).reshape(-1, 1)
    xy = torch.from_numpy(np.array([[x_i, y_i] for x_i in x for y_i in y]).reshape(x.shape[0] * y.shape[0], 2)).type(torch.FloatTensor)

    xy.requires_grad = True

    output = model(xy)

    u = output[:, 0]
    uex = exact(xy)[:, 0]

    output_grad = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    output_grad_x = output_grad[:, 0]
    output_grad_y = output_grad[:, 1]

    ex_grad = torch.autograd.grad(uex, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    exact_grad_x = ex_grad[:, 0]
    exact_grad_y = ex_grad[:, 1]

    l2_glob = torch.sqrt(torch.mean((u - uex) ** 2))
    l2_glob_rel = l2_glob / torch.sqrt(torch.mean(uex ** 2))

    h1_glob = torch.sqrt(torch.mean((u - uex) ** 2) + torch.mean((exact_grad_x - output_grad_x) ** 2 + (exact_grad_y - output_grad_y) ** 2))
    h1_glob_rel = h1_glob / torch.sqrt(torch.mean(uex ** 2) + torch.mean(exact_grad_x ** 2 + exact_grad_y ** 2))

    u = u.reshape(x.shape[0], y.shape[0])
    u = u.detach().numpy()

    uex = uex.reshape(x.shape[0], y.shape[0])
    uex = uex.detach().numpy()
    err = np.sqrt((u - uex) ** 2) / np.sqrt(np.mean(uex ** 2))

    output_grad_x = output_grad_x.reshape(x.shape[0], y.shape[0])
    output_grad_x = output_grad_x.detach().numpy()

    exact_grad_x = exact_grad_x.reshape(x.shape[0], y.shape[0])
    exact_grad_x = exact_grad_x.detach().numpy()

    plt.figure()
    plt.title(r'$u^\ast(x,y)$')
    ax = plt.gca()
    ttl = ax.title
    ttl.set_position([.5, 1.035])
    plt.contourf(x.reshape(-1, ), y.reshape(-1, ), u.T, 20, cmap='Spectral')
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.savefig(images_path + "/Poiss_u.png", dpi=400)

    plt.figure()
    plt.title(r'$u^\ast(x,y)$')
    ax = plt.gca()
    ttl = ax.title
    ttl.set_position([.5, 1.035])
    plt.contourf(x.reshape(-1, ), y.reshape(-1, ), output_grad_x.T, 20, cmap='Spectral')
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.savefig(images_path + "/Poiss_ux.png", dpi=400)

    plt.figure()
    plt.title(r'$u^\ast(x,y)$')
    ax = plt.gca()
    ttl = ax.title
    ttl.set_position([.5, 1.035])
    plt.contourf(x.reshape(-1, ), y.reshape(-1, ), exact_grad_x.T, 20, cmap='Spectral')
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.savefig(images_path + "/Poiss_ux_ex.png", dpi=400)

    plt.figure()
    plt.title(r'$u(x,y)$')
    ax = plt.gca()
    ttl = ax.title
    ttl.set_position([.5, 1.035])
    plt.contourf(x.reshape(-1, ), y.reshape(-1, ), uex.T, 20, cmap='Spectral')
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.savefig(images_path + "/Poiss_uex.png", dpi=400)

    plt.figure()
    plt.title(r'$||u(x,y) - u^\ast(x,y)||$')
    ax = plt.gca()
    ttl = ax.title
    ttl.set_position([.5, 1.035])
    plt.contourf(x.reshape(-1, ), y.reshape(-1, ), err.T, 20, cmap='Spectral')
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.savefig(images_path + "/Poiss_err.png", dpi=400)

    model.eval()
    x = torch.linspace(0.125, 0.875, 400).reshape(-1, 1)
    y = x
    xy = torch.from_numpy(np.array([[x_i, y_i] for x_i in x for y_i in y]).reshape(x.shape[0] * y.shape[0], 2)).type(torch.FloatTensor)

    output = model(xy)

    u = output[:, 0]
    uex = exact(xy)[:, 0]

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
