from ImportFile import *

np.random.seed(42)
pi = math.pi
T = 10
extrema_values = torch.tensor([[0, 5],
                               [0, 2 * pi],
                               [0, 2 * pi]])


def compute_res(network, x_f_train, space_dimensions, solid):
    x_f_train.requires_grad = True
    u = (network(x_f_train))[:, 0].reshape(-1, )
    v = (network(x_f_train))[:, 1].reshape(-1, )
    p = (network(x_f_train))[:, 2].reshape(-1, )

    inputs = torch.ones(x_f_train.shape[0], )

    if torch.cuda.is_available():
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
    mean_P = torch.mean(p).reshape(1, )

    if torch.cuda.is_available():
        del inputs
        torch.cuda.empty_cache()

    res = torch.cat([mean_P, res_u, res_v, res_d], 0)
    return res


def exact(x):

    u_0 = torch.cos(x[:, 0]) * torch.sin(x[:, 1])
    v_0 = -torch.sin(x[:, 0]) * torch.cos(x[:, 1])
    p_0 = x[:, 0] * x[:, 1]

    u_0 = u_0.reshape(-1, 1)
    v_0 = v_0.reshape(-1, 1)
    p_0 = p_0.reshape(-1, 1)

    return torch.cat([u_0, v_0, p_0], 1)


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
    L2_test = np.sqrt(np.mean((Exact[:, :2] - test_out[:, :2]) ** 2))
    print("Error Test:", L2_test)

    if images_path is not None:
        for j in range(3):
            plt.figure()
            plt.scatter(test_out[:, j], Exact[:, j])
            plt.savefig(images_path + "/Score" + str(j) + ".png")
    return L2_test


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


def u0(x):
    output_u = np.loadtxt("Data/initial_u.txt")
    output_v = np.loadtxt("Data/initial_v.txt")
    random_index = np.random.choice(output_u.shape[0], x.shape[0], replace=False)
    part_out_u = output_u[random_index, :]
    part_out_u = torch.from_numpy(part_out_u).type(torch.FloatTensor)

    part_out_v = output_v[random_index, :]
    part_out_v = torch.from_numpy(part_out_v).type(torch.FloatTensor)

    xy_u = part_out_u[:, :2]
    xy_v = part_out_v[:, :2]

    val_0 = extrema_values[1:, 0]
    val_f = extrema_values[1:, 1]

    xy_u = (xy_u - val_0) / (val_f - val_0)
    xy_v = (xy_v - val_0) / (val_f - val_0)

    u_0 = part_out_u[:, 2].reshape(-1, 1)
    v_0 = part_out_v[:, 2].reshape(-1, 1)
    p_0 = torch.tensor(()).new_full(size=(xy_u.shape[0], 1), fill_value=0.0)

    return xy_u, torch.cat([u_0, v_0, p_0], 1)


def plotting(model, images_path, extrema, solid):
    x = torch.linspace(extrema[1, 0], extrema[1, 1], 400).reshape(-1, 1)
    y = torch.linspace(extrema[2, 0], extrema[2, 1], 400).reshape(-1, 1)
    xy = torch.from_numpy(np.array([[x_i, y_i] for x_i in x for y_i in y]).reshape(x.shape[0] * y.shape[0], 2)).type(torch.FloatTensor)

    for val in [extrema[0, 0].numpy(), extrema[0, 1].numpy()]:
        t = torch.tensor(()).new_full(size=(xy.shape[0], 1), fill_value=float(val))
        input_vals = torch.cat([t, xy], 1)
        input_vals.requires_grad = True
        output = model(input_vals)
        u = output[:, 0]
        v = output[:, 1]
        p = output[:, 2]
        grad_u = torch.autograd.grad(u, input_vals, grad_outputs=torch.ones(input_vals.shape[0], ), create_graph=True)[0]
        grad_u_y = grad_u[:, 2].reshape(-1, )

        grad_v = torch.autograd.grad(v, input_vals, grad_outputs=torch.ones(input_vals.shape[0], ), create_graph=True)[0]
        grad_v_x = grad_v[:, 1].reshape(-1, )

        w = -grad_u_y + grad_v_x
        w = w.reshape(x.shape[0], y.shape[0])
        w = w.detach().numpy()

        u = u.reshape(x.shape[0], y.shape[0])
        u = u.detach().numpy()

        v = v.reshape(x.shape[0], y.shape[0])
        v = v.detach().numpy()

        p = p.reshape(x.shape[0], y.shape[0])
        p = p.detach().numpy()
        plt.figure()
        plt.contourf(x.reshape(-1, ), y.reshape(-1, ), w.T, 20, cmap='Spectral')
        plt.colorbar()
        plt.savefig(images_path + "/DS_Samples_w_" + str(val) + ".png")

        plt.figure()
        plt.contourf(x.reshape(-1, ), y.reshape(-1, ), u.T, 20, cmap='Spectral')
        plt.colorbar()
        plt.savefig(images_path + "/DS_Samples_u_" + str(val) + ".png")

        plt.figure()
        plt.contourf(x.reshape(-1, ), y.reshape(-1, ), v.T, 20, cmap='Spectral')
        plt.colorbar()
        plt.savefig(images_path + "/DS_Samples_v_" + str(val) + ".png")

        plt.figure()
        plt.contourf(x.reshape(-1, ), y.reshape(-1, ), p.T, 20, cmap='Spectral')
        plt.colorbar()
        plt.savefig(images_path + "/DS_Samples_p_" + str(val) + ".png")
