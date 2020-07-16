from ImportFile import *

# n_coll = 8192
# n_u = 1024
# n_int = 0
pi = math.pi
v = 0.0 / pi

extrema_values = torch.tensor([[0, 0.5],
                               [-1, 1]])


def compute_res(network, x_f_train, space_dimensions, solid_object, computing_error):
    x_f_train.requires_grad = True
    u = network(x_f_train).reshape(-1, )
    inputs = torch.ones(x_f_train.shape[0], )

    if not computing_error and torch.cuda.is_available():
        inputs = inputs.cuda()

    u_sq = 0.5 * u * u
    grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=inputs, create_graph=True)[0]
    grad_u_sq = torch.autograd.grad(u_sq, x_f_train, grad_outputs=inputs, create_graph=True)[0]

    grad_u_t = grad_u[:, 0]

    grad_u_x = grad_u[:, 1]
    grad_u_sq_x = grad_u_sq[:, 1]
    grad_grad_u_x = torch.autograd.grad(grad_u_x, x_f_train, grad_outputs=inputs, create_graph=True)[0]
    grad_u_xx = grad_grad_u_x[:, 1]

    if torch.cuda.is_available():
        del inputs
        torch.cuda.empty_cache()

    residual = grad_u_t.reshape(-1, ) + grad_u_sq_x - v * grad_u_xx.reshape(-1, )  # u * grad_u_x #- 0.01 / pi * grad_u_xx.reshape(-1, )

    return residual


def ub0(t):
    # Impose certain type of BC per variable
    # First element is the the index variable
    # Second element is the type of BC for the variable defined by the first element
    type_BC = ["func"]
    out = torch.tensor(()).new_full(size=(t.shape[0], 1), fill_value=0.0)
    return out.reshape(-1, 1), type_BC


def ub1(t):
    type_BC = ["func"]
    out = torch.tensor(()).new_full(size=(t.shape[0], 1), fill_value=1.0)
    return out.reshape(-1, 1), type_BC


list_of_BC = [[ub0, ub1]]


def u0(x):
    u_0 = torch.tensor(()).new_full(size=(x.shape[0], 1), fill_value=0.0)

    for i in range(u_0.shape[0]):
        if x[i] > 0:
            u_0[i, 0] = 1
    return u_0.reshape(-1, 1)


def exact(inputs):
    t = inputs[:, 0]
    x = inputs[:, 1]
    u = torch.tensor(()).new_full(size=(x.shape[0], 1), fill_value=0.0)

    for i in range(u.shape[0]):
        if 0 <= x[i] < t[i]:
            u[i, 0] = x[i] / t[i]
        elif x[i] >= t[i]:
            u[i, 0] = 1
    return u


def convert(vector, extrema_values):
    vector = np.array(vector)
    max_val = np.max(np.array(extrema_values), axis=1)
    min_val = np.min(np.array(extrema_values), axis=1)
    vector = vector * (max_val - min_val) + min_val
    return torch.from_numpy(vector).type(torch.FloatTensor)


def compute_generalization_error(model, extrema, images_path=None):
    model.eval()

    file_ex = "Data/BurgersRar.txt"
    exact_solution = np.loadtxt(file_ex)
    print(exact_solution)

    Exact = exact_solution[np.where((exact_solution[:, 2] == 0.0) & (exact_solution[:, 3] == v)), -1].reshape(-1, 1)
    test_inp = torch.from_numpy(exact_solution[np.where((exact_solution[:, 2] == 0.0) & (exact_solution[:, 3] == v)), :2]).type(torch.FloatTensor)
    print(test_inp.shape)
    if v == 0.:

        test_inp = torch.from_numpy(exact_solution[np.where((exact_solution[:, 2] == 0.0) & (exact_solution[:, 3] == 0.01 / pi)), :2]).type(torch.FloatTensor)
        test_inp = test_inp.reshape(test_inp.shape[1], 2)
        Exact = exact(test_inp).reshape(-1, 1).detach().numpy()
    else:
        test_inp = test_inp.reshape(Exact.shape[0], 2)
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


def plotting(model, images_path, extrema, solid):
    model.cpu()
    model = model.eval()

    file_ex = "Data/BurgersRar.txt"
    exact_solution = np.loadtxt(file_ex)

    time_steps = [0.0, 0.24, 0.5]
    scale_vec = np.linspace(0.65, 1.55, len(time_steps))

    fig = plt.figure()
    plt.grid(True, which="both", ls=":")
    for val, scale in zip(time_steps, scale_vec):
        ex = exact_solution[np.where((exact_solution[:, 0] == val) & (exact_solution[:, 2] == 0.0) & (exact_solution[:, 3] == v)), -1].reshape(-1, 1)

        inputs = torch.from_numpy(exact_solution[np.where((exact_solution[:, 0] == val) & (exact_solution[:, 2] == 0.0) & (exact_solution[:, 3] == v)), :2]).type(torch.FloatTensor)
        if v == 0.:
            inputs = torch.from_numpy(exact_solution[np.where((exact_solution[:, 0] == val) & (exact_solution[:, 2] == 0.0) & (exact_solution[:, 3] == 0.01 / pi)), :2]).type(
                torch.FloatTensor)
            inputs = inputs.reshape(inputs.shape[1], 2)
            ex = exact(inputs).reshape(-1, 1).detach().numpy()
        else:
            inputs = inputs.reshape(ex.shape[0], 2)
        x = torch.linspace(-1, 1, 100).reshape(-1, 1)
        t = torch.tensor(()).new_full(size=(x.shape[0], 1), fill_value=val)
        inputs_m = torch.cat([t, x], 1)
        x_plot = inputs[:, 1].reshape(-1, 1)
        plt.plot(x_plot.detach().numpy(), ex, linewidth=2, label=r'Exact, $t=$' + str(val), color=lighten_color('grey', scale), zorder=0)
        plt.scatter(x.detach().numpy(), model(inputs_m).detach().numpy(), label=r'Predicted, $t=$' + str(val), marker="o", s=14, color=lighten_color('C0', scale),
                    zorder=10)

    plt.xlabel(r'$x$')
    plt.ylabel(r'u')
    plt.legend()
    plt.savefig(images_path + "/Rar_Samples.png", dpi=500)
