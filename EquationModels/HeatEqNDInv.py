from ImportFile import *

n_int = 0
pi = math.pi
tot_dim = 100 + 1
d = tot_dim - 1

extrema_values = torch.tensor(()).new_full(size=(tot_dim, 2), fill_value=0.0)
for i in range(tot_dim):
    extrema_values[i, :] = torch.tensor([0, 1])


def compute_res(network, x_f_train, space_dimensions, solid, computing_error=False):
    # start =time.time()
    inputs = torch.ones(x_f_train.shape[0], )
    if not computing_error and torch.cuda.is_available():
        inputs = inputs.cuda()
    x_f_train.requires_grad = True
    u = network(x_f_train).reshape(-1, )
    grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=inputs, create_graph=True)[0]

    grad_u_t = grad_u[:, 0]
    res = grad_u_t
    for k in range(1, grad_u.shape[1]):
        rad_grad_u_xx = torch.autograd.grad(grad_u[:, k], x_f_train, grad_outputs=inputs, create_graph=True)[0][:, k]
        res = res - rad_grad_u_xx
    return res


def exact(inp):
    t = inp[:, 0]
    x = inp[:, 1:]
    return 0.5 / d * torch.sum(x ** 2, 1).reshape(-1, 1) + t.reshape(-1, 1)


def add_internal_points(n_internal):
    n_bound = int(n_internal / 4)
    n_bound_dim = int(n_bound / (2 * d))
    n_internal = n_internal - n_bound

    x_bound = list()
    y_bound = list()
    for l in range(1, d + 1):
        x_bound_0 = torch.rand((n_bound_dim, tot_dim))
        x_bound_0[:, l] = torch.full(x_bound_0[:, l].shape, 0.0)
        x_bound_1 = torch.rand((n_bound_dim, tot_dim))
        x_bound_1[:, l] = torch.full(x_bound_1[:, l].shape, 1.0)
        y_bound_0 = exact(x_bound_0)
        y_bound_1 = exact(x_bound_1)
        x_bound.append(x_bound_0)
        x_bound.append(x_bound_1)
        y_bound.append(y_bound_0)
        y_bound.append(y_bound_1)

    x_bound = torch.cat(x_bound, 0)
    y_bound = torch.cat(y_bound, 0)

    x_internal = torch.rand((n_internal, tot_dim)) * (0.6 - 0.4) + 0.4
    print(x_internal)
    y_internal = exact(x_internal)

    x = torch.cat([x_bound, x_internal])
    y = torch.cat([y_bound, y_internal])

    return x, y


def ub0(inp):
    type_BC = ["func"]
    if inp.shape[1] == 1:
        return inp.reshape(-1, 1), type_BC
    else:
        t = inp[:, 0]
        x = inp[:, 1:]
        return 0.5 / d * torch.sum(x ** 2, 1).reshape(-1, 1) + t.reshape(-1, 1), type_BC


def ub1(inp):
    type_BC = ["func"]
    if inp.shape[1] == 1:
        return inp.reshape(-1, 1) + 0.5, type_BC
    else:
        t = inp[:, 0]
        x = inp[:, 1:]
        return 0.5 / d * (torch.sum(x ** 2, 1).reshape(-1, 1) + 1) + t.reshape(-1, 1), type_BC


list_of_BC = list([ub0, ub1] for i in range(d))
print(list_of_BC)


def u0(x):
    return 0.5 / d * torch.sum(x ** 2, 1).reshape(-1, 1)


def plotting(model, images_path, extrema, solid):
    return


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
    rel_L2_test = L2_test / np.sqrt(np.mean((Exact) ** 2))
    print("Relative Error Test:", rel_L2_test)

    if images_path is not None:
        plt.figure()
        plt.grid(True, which="both", ls=":")
        plt.scatter(Exact, test_out)
        plt.xlabel(r'Exact Values')
        plt.ylabel(r'Predicted Values')
        plt.savefig(images_path + "/Score.png")
    return L2_test, rel_L2_test
