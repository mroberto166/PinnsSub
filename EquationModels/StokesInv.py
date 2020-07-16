from ImportFile import *

pi = math.pi
extrema_values = torch.tensor([[0, 1],
                               [0, 1]])

radius = 0.25
inner_type_p = "grid"
epsilon = 0


def compute_res(network, x_f_train, space_dimensions, solid, computing_error):
    # x_f_train = x_f_train[(x_f_train[:,0]-0.5)**2 + (x_f_train[:,1]-0.5)**2 <0.125**2,:]
    x_f_train.requires_grad = True

    u = (network(x_f_train))[:, 0].reshape(-1, )
    v = (network(x_f_train))[:, 1].reshape(-1, )
    p = (network(x_f_train))[:, 2].reshape(-1, )

    inputs = torch.ones(x_f_train.shape[0], )

    if not computing_error and torch.cuda.is_available():
        inputs = inputs.cuda()

    grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=inputs, create_graph=True)[0]
    grad_u_x = grad_u[:, 0].reshape(-1, )
    grad_u_y = grad_u[:, 1].reshape(-1, )

    grad_v = torch.autograd.grad(v, x_f_train, grad_outputs=inputs, create_graph=True)[0]
    grad_v_x = grad_v[:, 0].reshape(-1, )
    grad_v_y = grad_v[:, 1].reshape(-1, )

    grad_p = torch.autograd.grad(p, x_f_train, grad_outputs=inputs, create_graph=True)[0]
    grad_p_x = grad_p[:, 0].reshape(-1, )
    grad_p_y = grad_p[:, 1].reshape(-1, )

    grad_u_xx = torch.autograd.grad(grad_u_x, x_f_train, grad_outputs=inputs, create_graph=True)[0][:, 0]
    grad_u_yy = torch.autograd.grad(grad_u_y, x_f_train, grad_outputs=inputs, create_graph=True)[0][:, 1]
    lap_u = grad_u_xx + grad_u_yy

    grad_v_xx = torch.autograd.grad(grad_v_x, x_f_train, grad_outputs=inputs, create_graph=True)[0][:, 0]
    grad_v_yy = torch.autograd.grad(grad_v_y, x_f_train, grad_outputs=inputs, create_graph=True)[0][:, 1]
    lap_v = grad_v_xx + grad_v_yy

    mean_p = torch.mean(p).reshape(1, )

    res_u = grad_p_x - lap_u
    res_v = grad_p_y - lap_v
    res_d = grad_u_x + grad_v_y

    if torch.cuda.is_available():
        del inputs
        torch.cuda.empty_cache()

    res = torch.cat([res_u, res_v, res_d, mean_p], 0)
    return res


def exact(inputs):
    x = inputs[:, 0]
    y = inputs[:, 1]

    u = 4 * x * y ** 3
    v = x ** 4 - y ** 4
    p = 12 * x ** 2 * y - 4 * y ** 3 - 1

    u = u.reshape(-1, 1)
    v = v.reshape(-1, 1)
    p = p.reshape(-1, 1)

    out = torch.cat([u, v, p], 1)

    return out


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

    '''if images_path is not None:
        plt.figure()
        plt.grid(True, which="both", ls=":")
        plt.scatter(Exact, test_out)
        plt.xlabel(r'Exact Values')
        plt.ylabel(r'Predicted Values')
        plt.savefig(images_path + "/Score.png", dpi=400)'''
    return L2_test, rel_L2_test


def add_internal_points(n_internal):
    ny = int(np.sqrt(n_internal))

    nx = n_internal - ny ** 2 + ny
    if inner_type_p == "grid":
        theta = np.linspace(0, 2 * pi, nx)
        r = np.linspace(0, radius, ny)
        inputs_theta_r = torch.from_numpy(np.transpose([np.repeat(r, len(theta)), np.tile(theta, len(r))])).type(torch.FloatTensor)
        r_l = inputs_theta_r[:, 0]
        theta_l = inputs_theta_r[:, 1]
        x = r_l * torch.cos(theta_l) + 0.5
        y = r_l * torch.sin(theta_l) + 0.5
        inputs = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1)], 1)

    # Sobol Points
    if inner_type_p == "sobol":
        inputs = torch.from_numpy(sobol_seq.i4_sobol_generate(2, n_internal)).type(torch.FloatTensor)
        extrema_inner = torch.tensor([[0, 0.02],
                                      [a, 1 - a]])
        inputs = convert(inputs, extrema_inner)

    if inner_type_p == "uniform":
        inputs_theta_r = torch.rand([n_internal, extrema_values.shape[0]])
        extrema_inner = torch.tensor([[0, radius],
                                      [0, 2 * pi]])
        inputs_theta_r = convert(inputs_theta_r, extrema_inner)
        r_l = inputs_theta_r[:, 0]
        theta_l = inputs_theta_r[:, 1]
        x = r_l * torch.cos(theta_l) + 0.5
        y = r_l * torch.sin(theta_l) + 0.5
        inputs = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1)], 1)

    u = exact(inputs) * (1 + epsilon * torch.randn(exact(inputs).shape))
    return inputs, u


def ub0(y):
    type_BC = ["func"]
    u = torch.sin(y).reshape(-1, 1)
    u = torch.tensor(()).new_full(size=(y.shape[0], 3), fill_value=0.0)
    return u, type_BC


def ub1(y):
    type_BC = ["func"]
    u = torch.sin(y).reshape(-1, 1)
    u = torch.tensor(()).new_full(size=(y.shape[0], 3), fill_value=0.0)
    return u, type_BC


def ub0y(x):
    type_BC = ["func"]
    u = torch.tensor(()).new_full(size=(x.shape[0], 3), fill_value=0.0)
    return u, type_BC


def ub1y(x):
    type_BC = ["func"]
    u = torch.tensor(()).new_full(size=(x.shape[0], 3), fill_value=0.0)
    return u, type_BC


def u0(x):
    u = torch.tensor(()).new_full(size=(0, 3), fill_value=0.0)
    return u


list_of_BC = [[ub0, ub1], [ub0y, ub1y]]


def plotting(model, images_path, extrema, solid):
    model.eval()
    x = torch.linspace(0., 1, 400).reshape(-1, 1)
    y = torch.linspace(0., 1., 400).reshape(-1, 1)
    xy = torch.from_numpy(np.array([[x_i, y_i] for x_i in x for y_i in y]).reshape(x.shape[0] * y.shape[0], 2)).type(torch.FloatTensor)

    xy.requires_grad = True

    output = model(xy)

    u = output[:, 0]
    uex = exact(xy)[:, 0]

    v = output[:, 1]
    vex = exact(xy)[:, 1]

    p = output[:, 2]
    pex = exact(xy)[:, 2]

    u_grad = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_grad_x = u_grad[:, 0]
    u_grad_y = u_grad[:, 1]

    uex_grad = torch.autograd.grad(uex, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    uexact_grad_x = uex_grad[:, 0]
    uexact_grad_y = uex_grad[:, 1]

    v_grad = torch.autograd.grad(v, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_grad_x = v_grad[:, 0]
    v_grad_y = v_grad[:, 1]

    vex_grad = torch.autograd.grad(vex, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    vexact_grad_x = vex_grad[:, 0]
    vexact_grad_y = vex_grad[:, 1]

    p_grad = torch.autograd.grad(p, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    p_grad_x = p_grad[:, 0]
    p_grad_y = p_grad[:, 1]

    pex_grad = torch.autograd.grad(pex, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    pexact_grad_x = pex_grad[:, 0]
    pexact_grad_y = pex_grad[:, 1]

    l2_glob = torch.sqrt(torch.mean((u - uex) ** 2) + torch.mean((v - vex) ** 2))
    l2_glob_rel = l2_glob / torch.sqrt(torch.mean(uex ** 2) + torch.mean(vex ** 2))

    l2_p = torch.sqrt(torch.mean((p - pex) ** 2))
    l2_p_rel = l2_p / torch.sqrt(torch.mean(pex ** 2))

    h1_glob = torch.sqrt(torch.mean((u - uex) ** 2) + torch.mean((v - vex) ** 2) +
                         torch.mean((u_grad_x - uexact_grad_x) ** 2) + torch.mean((u_grad_y - uexact_grad_y) ** 2) +
                         torch.mean((v_grad_x - vexact_grad_x) ** 2) + torch.mean((v_grad_y - vexact_grad_y) ** 2))
    h1_glob_rel = h1_glob / torch.sqrt(torch.mean(uex ** 2) + torch.mean(vex ** 2) +
                                       torch.mean(uexact_grad_x ** 2) + torch.mean(uexact_grad_y ** 2) +
                                       torch.mean(vexact_grad_x ** 2) + torch.mean(vexact_grad_y ** 2))

    h1_p = torch.sqrt(torch.mean((p - pex) ** 2) +
                      torch.mean((p_grad_x - pexact_grad_x) ** 2) + torch.mean((p_grad_y - pexact_grad_y) ** 2))
    h1_p_rel = h1_p / torch.sqrt(torch.mean(pex ** 2) +
                                 torch.mean(pexact_grad_x ** 2) + torch.mean(pexact_grad_y ** 2))

    u = u.reshape(x.shape[0], y.shape[0])
    u = u.detach().numpy()

    v = v.reshape(x.shape[0], y.shape[0])
    v = v.detach().numpy()

    p = p.reshape(x.shape[0], y.shape[0])
    p = p.detach().numpy()

    uex = uex.reshape(x.shape[0], y.shape[0])
    uex = uex.detach().numpy()

    vex = vex.reshape(x.shape[0], y.shape[0])
    vex = vex.detach().numpy()

    pex = pex.reshape(x.shape[0], y.shape[0])
    pex = pex.detach().numpy()

    err = np.sqrt((u - uex) ** 2 + (v - vex) ** 2) / np.sqrt(np.mean(uex ** 2) + np.mean(vex ** 2))

    plt.figure()
    plt.title(r'$u_1^\ast(x_1,x_2)$')
    ax = plt.gca()
    ttl = ax.title
    ttl.set_position([.5, 1.035])
    plt.contourf(x.reshape(-1, ), y.reshape(-1, ), u.T, 20, cmap='Spectral')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.savefig(images_path + "/Stokes_u.png", dpi=400)

    plt.figure()
    plt.title(r'$u_1(x_1,x_2)$')
    ax = plt.gca()
    ttl = ax.title
    ttl.set_position([.5, 1.035])
    plt.contourf(x.reshape(-1, ), y.reshape(-1, ), uex.T, 20, cmap='Spectral')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.savefig(images_path + "/Stokes_uex.png", dpi=400)

    plt.figure()
    plt.title(r'$u_2^\ast(x_1,x_2)$')
    ax = plt.gca()
    ttl = ax.title
    ttl.set_position([.5, 1.035])
    plt.contourf(x.reshape(-1, ), y.reshape(-1, ), v.T, 20, cmap='Spectral')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.savefig(images_path + "/Stokes_v.png", dpi=400)

    plt.figure()
    plt.title(r'$u_2(x_1,x_2)$')
    ax = plt.gca()
    ttl = ax.title
    ttl.set_position([.5, 1.035])
    plt.contourf(x.reshape(-1, ), y.reshape(-1, ), vex.T, 20, cmap='Spectral')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.savefig(images_path + "/Stokes_vex.png", dpi=400)

    plt.figure()
    plt.title(r'$p^\ast(x_1,x_2)$')
    ax = plt.gca()
    ttl = ax.title
    ttl.set_position([.5, 1.035])
    plt.contourf(x.reshape(-1, ), y.reshape(-1, ), p.T, 20, cmap='Spectral')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.savefig(images_path + "/Stokes_p.png", dpi=400)

    plt.figure()
    plt.title(r'$p(x_1,x_2)$')
    ax = plt.gca()
    ttl = ax.title
    ttl.set_position([.5, 1.035])
    plt.contourf(x.reshape(-1, ), y.reshape(-1, ), pex.T, 20, cmap='Spectral')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.savefig(images_path + "/Stokes_pex.png", dpi=400)

    plt.figure()
    plt.title(r'$||\mathbf{u}(x_1,x_2) - \mathbf{u}^\ast(x_1,x_2)||$')
    ax = plt.gca()
    ttl = ax.title
    ttl.set_position([.5, 1.035])
    plt.contourf(x.reshape(-1, ), y.reshape(-1, ), err.T, 20, cmap='Spectral')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.savefig(images_path + "/Stokes_u_err.png", dpi=400)

    model.eval()
    theta = np.linspace(0, 2 * pi, 100)
    r = np.linspace(0, radius, 100)
    inputs_theta_r = torch.from_numpy(np.transpose([np.repeat(r, len(theta)), np.tile(theta, len(r))])).type(torch.FloatTensor)
    r_l = inputs_theta_r[:, 0]
    theta_l = inputs_theta_r[:, 1]
    x = r_l * torch.cos(theta_l) + 0.5
    y = r_l * torch.sin(theta_l) + 0.5

    xy = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1)], 1)

    output = model(xy)

    u = output[:, 0]
    uex = exact(xy)[:, 0]

    v = output[:, 1]
    vex = exact(xy)[:, 1]

    l2_om_big = torch.sqrt(torch.mean((u - uex) ** 2) + torch.mean((v - vex) ** 2))
    l2_om_big_rel = l2_om_big / torch.sqrt(torch.mean(uex ** 2) + torch.mean(vex ** 2))

    print(l2_om_big, l2_om_big_rel)
    print(l2_glob, l2_glob_rel)
    print(h1_glob, h1_glob_rel)

    with open(images_path + '/errors.txt', 'w') as file:
        file.write("l2_glob,"
                   "l2_glob_rel,"
                   "h1_glob,"
                   "h1_glob_rel,"
                   "l2_p_rel,"
                   "h1_p_rel,"
                   "l2_om_big,"
                   "l2_om_big_rel\n")
        file.write(str(float(l2_glob.detach().numpy())) + "," +
                   str(float(l2_glob_rel.detach().numpy())) + "," +
                   str(float(h1_glob.detach().numpy())) + "," +
                   str(float(h1_glob_rel.detach().numpy())) + "," +
                   str(float(l2_p_rel.detach().numpy())) + "," +
                   str(float(h1_p_rel.detach().numpy())) + "," +
                   str(float(l2_om_big.detach().numpy())) + "," +
                   str(float(l2_om_big_rel.detach().numpy())))
